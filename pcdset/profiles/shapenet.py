"""ShapeNet profile implementation."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from ..io import read_points
from ..io.writer_ply import write_point_file
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
from ..utils.manifest import Entry
from ..utils import taxonomy
from .base import BaseProfile

# LMDB writer imported separately to avoid circular __all__
from ..io.writer_lmdb import LMDBWriter


def _sanitize(name: str) -> str:
    """Return a filesystem friendly ``name``."""
    allowed = "-_.0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join(ch for ch in name if ch in allowed)


@dataclass
class ShapeNetProfile(BaseProfile):
    """Convert point clouds into a ShapeNet style layout."""

    name: str = "shapenet"
    points_n: int = 2048
    file_ext: str = "ply"  # ply|pcd|npz
    basename: str = "points"
    normalize: str = "none"
    center: bool = False
    dedup: bool = False
    fps: bool = False
    voxel: float = 0.0
    to_lmdb: bool = False
    lmdb_max_gb: int = 64
    taxonomy_out: Optional[Path] = None
    save_meta: bool = False
    save_attrs: bool = False
    overwrite: bool = False
    workers: int = 8
    split_strategy: str = "FILE"
    ratios: tuple = (0.9, 0.05, 0.05)

    def prepare(self, points: np.ndarray, _role: str = "object", _args: Optional[dict] = None) -> np.ndarray:
        """Prepare ``points`` for output.

        Examples
        --------
        >>> import numpy as np
        >>> prof = ShapeNetProfile(points_n=4, center=True)
        >>> pts = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=float)
        >>> prof.prepare(pts).shape
        (4, 3)
        """
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
        n = self.points_n
        sample_n = min(len(points), n)
        if self.fps:
            sampled = farthest_point_sample(points, sample_n)
        else:
            sampled = random_sample(points, sample_n)
        if len(sampled) < n:
            logger.warning("upsampling %s from %d to %d", _role, len(sampled), n)
            extra = random_sample(points, n - len(sampled))
            sampled = np.concatenate([sampled, extra], axis=0)
        return sampled.astype(np.float32)

    def _process(self, entry: Entry, out_dir: Path, lmdb: Optional[LMDBWriter]) -> bool:
        """Process a single entry. Returns True on success."""
        try:
            points, attrs = read_points(entry.path)
            pts = self.prepare(points, "object")
            cat = _sanitize(entry.category)
            model = _sanitize(entry.model_id)
            file = (
                out_dir
                / entry.split
                / cat
                / model
                / f"{self.basename}_{self.points_n}.{self.file_ext}"
            )
            write_point_file(
                file,
                pts,
                attrs if self.file_ext == "npz" and self.save_attrs and attrs else None,
            )
            if self.save_attrs and self.file_ext != "npz" and attrs:
                np.savez(file.with_suffix(".npz"), **attrs)
            if self.save_meta:
                meta = {"category": entry.category, "model_id": entry.model_id, "source": str(entry.path)}
                with (file.parent / "meta.json").open("w", encoding="utf-8") as fh:
                    json.dump(meta, fh, indent=2)
            if lmdb is not None:
                key = f"object/{cat}/{model}"
                lmdb.put(key, pts)
            return True
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("Failed to process %s: %s", entry.path, exc)
            return False

    def convert(self, entries: Iterable[Entry], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        lmdb_writer: Optional[LMDBWriter] = None
        if self.to_lmdb:
            lmdb_writer = LMDBWriter(out_dir / "lmdb", map_size_gb=self.lmdb_max_gb, overwrite=self.overwrite)
        failed: List[Entry] = []
        entries_list = list(entries)
        splits: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
        cats: Set[str] = set()
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            future_map = {ex.submit(self._process, e, out_dir, lmdb_writer): e for e in entries_list}
            for fut in tqdm(as_completed(future_map), total=len(future_map), desc="convert"):
                e = future_map[fut]
                if fut.result():
                    splits[e.split].add(f"{_sanitize(e.category)}/{_sanitize(e.model_id)}")
                    cats.add(e.category)
                else:
                    failed.append(e)
        if lmdb_writer is not None:
            meta = {"profile": self.name, "timestamp": time.time()}
            lmdb_writer.close(meta)
        split_dir = out_dir / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        for split, items in splits.items():
            if not items:
                continue
            with (split_dir / f"{split}.txt").open("w", encoding="utf-8") as fh:
                for it in sorted(items):
                    fh.write(f"{it}\n")
        if self.taxonomy_out:
            if not self.taxonomy_out.exists():
                taxonomy.save_taxonomy(self.taxonomy_out, taxonomy.build_taxonomy(cats))
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
        expected = f"{self.basename}_{self.points_n}.{self.file_ext}"
        for split in ("train", "val", "test"):
            split_dir = root / split
            if not split_dir.exists():
                continue
            for cat_dir in split_dir.iterdir():
                if not cat_dir.is_dir():
                    continue
                for model_dir in cat_dir.iterdir():
                    if not model_dir.is_dir():
                        continue
                    file = model_dir / expected
                    if not file.exists():
                        logger.error("Missing %s", file)
                        missing += 1
                        continue
                    try:
                        pts, _ = read_points(file)
                        if len(pts) != self.points_n:
                            logger.error("Point count mismatch for %s", file)
                            missing += 1
                    except Exception as exc:
                        logger.error("Read error %s: %s", file, exc)
                        missing += 1
        split_dir = root / "splits"
        for split in ("train", "val", "test"):
            txt = split_dir / f"{split}.txt"
            if not txt.exists():
                continue
            with txt.open("r", encoding="utf-8") as fh:
                for line in fh:
                    rel = line.strip()
                    file = root / split / rel / expected
                    if not file.exists():
                        logger.error("Listed model missing: %s", file)
                        missing += 1
        lmdb_path = root / "lmdb"
        if lmdb_path.exists():
            try:  # pragma: no cover - best effort
                import lmdb, msgpack, msgpack_numpy as m
                env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False)
                with env.begin() as txn:
                    cursor = txn.cursor()
                    for key, val in cursor:
                        key = key.decode()
                        file = root / key.replace("object/", "") / expected
                        if not file.exists():
                            logger.error("LMDB key without file: %s", key)
                            missing += 1
                        data = msgpack.unpackb(val, object_hook=m.decode)
                        if np.asarray(data["points"]).shape != (self.points_n, 3):
                            logger.error("LMDB shape mismatch for %s", key)
                            missing += 1
                env.close()
            except Exception as exc:
                logger.error("LMDB validation failed: %s", exc)
                missing += 1
        if missing:
            raise SystemExit(1)
        logger.info("Validation passed")
