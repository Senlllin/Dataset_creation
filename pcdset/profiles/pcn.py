"""PCN profile implementation."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from ..io import read_points, write_ply, LMDBWriter
from ..ops import (
    center as op_center,
    unit_sphere,
    bbox_scale,
    random_sample,
    farthest_point_sample,
    voxel_downsample,
    dedup as op_dedup,
)
from ..utils.logging import logger
from ..manifest import Entry
from .base import BaseProfile


@dataclass
class PCNProfile(BaseProfile):
    """Implements the PCN dataset layout."""

    name: str = "pcn"
    partial_n: int = 2048
    complete_n: int = 16384
    normalize: str = "none"
    center: bool = False
    dedup: bool = False
    fps: bool = False
    voxel: float = 0.0
    to_lmdb: bool = False
    lmdb_max_gb: int = 64
    save_attrs: bool = False
    overwrite: bool = False
    workers: int = 8
    split_strategy: str = "FILE"
    ratios: tuple = (0.9, 0.03, 0.07)

    def prepare(self, points: np.ndarray, role: str, _args: Optional[dict] = None) -> np.ndarray:
        if self.voxel > 0:
            points = voxel_downsample(points, self.voxel)
        if self.dedup:
            points = op_dedup(points)
        if self.center:
            points = op_center(points)
        if self.normalize == "unit":
            points = unit_sphere(points)
        elif self.normalize == "bbox":
            points = bbox_scale(points)
        n = self.partial_n if role == "partial" else self.complete_n
        if self.fps:
            sampled = farthest_point_sample(points, min(len(points), n))
        else:
            sampled = random_sample(points, min(len(points), n))
        if len(sampled) < n:
            extra = random_sample(points, n - len(sampled))
            sampled = np.concatenate([sampled, extra], axis=0)
        return sampled.astype(np.float32)

    # Internal worker processing
    def _process(self, entry: Entry, out_dir: Path, lmdb: Optional[LMDBWriter], failed: List[Entry]) -> None:
        try:
            points, attrs = read_points(entry.path)
            pts = self.prepare(points, entry.role)
            if entry.role == "partial":
                file = (
                    out_dir
                    / entry.split
                    / "partial"
                    / entry.category
                    / entry.model_id
                    / f"{entry.view_id}.ply"
                )
            else:  # complete
                file = (
                    out_dir
                    / entry.split
                    / "complete"
                    / entry.category
                    / f"{entry.model_id}.ply"
                )
            write_ply(file, pts)
            if self.save_attrs and attrs:
                np.savez(file.with_suffix(".npz"), **attrs)
            if lmdb is not None:
                key = f"{entry.role}/{entry.category}/{entry.model_id}/{entry.view_id or ''}"
                lmdb.put(key, pts)
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("Failed to process %s: %s", entry.path, exc)
            failed.append(entry)

    def convert(self, entries: Iterable[Entry], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        lmdb_writer: Optional[LMDBWriter] = None
        if self.to_lmdb:
            lmdb_writer = LMDBWriter(out_dir / "lmdb", map_size_gb=self.lmdb_max_gb, overwrite=self.overwrite)
        failed: List[Entry] = []
        entries_list = list(entries)
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futures = [ex.submit(self._process, e, out_dir, lmdb_writer, failed) for e in entries_list]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="convert"):
                pass
        if lmdb_writer is not None:
            meta = {"profile": self.name, "timestamp": time.time()}
            lmdb_writer.close(meta)
        if failed:
            import csv
            with (out_dir / "_failed.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["path", "role", "category", "model_id", "view_id", "split"])
                for e in failed:
                    writer.writerow([e.path, e.role, e.category, e.model_id, e.view_id or "", e.split])
            logger.warning("%d files failed. See _failed.csv", len(failed))

    def validate_structure(self, root: Path) -> None:
        missing = 0
        for split in ("train", "val", "test"):
            part_dir = root / split / "partial"
            comp_dir = root / split / "complete"
            if not part_dir.exists() or not comp_dir.exists():
                continue
            for part_file in part_dir.rglob("*.ply"):
                rel = part_file.relative_to(part_dir)
                parts = rel.parts
                if len(parts) < 3:
                    continue
                cat, model_id = parts[0], parts[1]
                comp_file = comp_dir / cat / f"{model_id}.ply"
                if not comp_file.exists():
                    logger.error("Missing complete for %s", part_file)
                    missing += 1
                else:
                    try:
                        pts, _ = read_points(part_file)
                        cpts, _ = read_points(comp_file)
                        if len(pts) != self.partial_n or len(cpts) != self.complete_n:
                            logger.error("Point count mismatch for %s", part_file)
                            missing += 1
                    except Exception as exc:
                        logger.error("Read error %s: %s", part_file, exc)
                        missing += 1
        if missing:
            raise SystemExit(1)
        logger.info("Validation passed")
