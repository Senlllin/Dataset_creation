"""ShapeNet profile implementation."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set
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
from ..utils import taxonomy
from .base import BaseProfile

_SAN = re.compile(r"[^-_.0-9a-zA-Z]+")


def _sanitize(text: str) -> str:
    """Sanitize category/model identifiers."""

    return _SAN.sub("_", text)


@dataclass
class ShapeNetProfile(BaseProfile):
    """Convert arbitrary point clouds into a ShapeNet style layout."""

    name: str = "shapenet"
    points_n: int = 2048
    file_ext: str = "ply"
    basename: str = "points"
    normalize: str = "none"
    center: bool = False
    dedup: bool = False
    fps: bool = False
    voxel: float = 0.0
    save_meta: bool = False
    save_attrs: bool = False
    to_lmdb: bool = False
    lmdb_max_gb: int = 64
    overwrite: bool = False
    workers: int = 8
    taxonomy_out: Optional[Path] = None

    def prepare(self, points: np.ndarray, role: str, _args: Optional[dict] = None) -> np.ndarray:  # noqa: D401 - see base class
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
        if self.fps:
            sampled = farthest_point_sample(points, min(len(points), n))
        else:
            sampled = random_sample(points, min(len(points), n))
        if len(sampled) < n:
            extra = random_sample(points, n - len(sampled))
            sampled = np.concatenate([sampled, extra], axis=0)
            logger.warning("Point cloud had fewer than %d points, sampling with replacement", n)
        return sampled.astype(np.float32)

    # Internal worker processing
    def _process(
        self,
        entry: Entry,
        out_dir: Path,
        lmdb: Optional[LMDBWriter],
        failed: List[Entry],
        splits: Dict[str, Set[str]],
        cats: Set[str],
    ) -> None:
        try:
            points, attrs = read_points(entry.path)
            pts = self.prepare(points, entry.role)
            cat = _sanitize(entry.category)
            model = _sanitize(entry.model_id)
            rel = f"{cat}/{model}"
            splits.setdefault(entry.split, set()).add(rel)
            cats.add(cat)
            model_dir = out_dir / entry.split / cat / model
            model_dir.mkdir(parents=True, exist_ok=True)
            file_name = f"{self.basename}_{self.points_n}.{self.file_ext}"
            file_path = model_dir / file_name
            write_ply(file_path, pts)
            if self.save_attrs and attrs:
                np.savez(file_path.with_suffix(".npz"), **attrs)
            if self.save_meta:
                meta = {"source": str(entry.path)}
                with (model_dir / "meta.json").open("w", encoding="utf-8") as fh:
                    json.dump(meta, fh, indent=2)
            if lmdb is not None:
                lmdb.put(f"object/{rel}", pts)
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("Failed to process %s: %s", entry.path, exc)
            failed.append(entry)

    def convert(self, entries: Iterable[Entry], out_dir: Path) -> None:  # noqa: D401 - see base class
        out_dir.mkdir(parents=True, exist_ok=True)
        lmdb_writer: Optional[LMDBWriter] = None
        if self.to_lmdb:
            lmdb_writer = LMDBWriter(out_dir / "lmdb", map_size_gb=self.lmdb_max_gb, overwrite=self.overwrite)
        failed: List[Entry] = []
        splits: Dict[str, Set[str]] = {}
        cats: Set[str] = set()
        entries_list = list(entries)
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futures = [
                ex.submit(self._process, e, out_dir, lmdb_writer, failed, splits, cats) for e in entries_list
            ]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="convert"):
                pass
        if lmdb_writer is not None:
            meta = {"profile": self.name, "timestamp": time.time()}
            lmdb_writer.close(meta)
        split_dir = out_dir / "splits"
        for split, items in splits.items():
            if not items:
                continue
            file = split_dir / f"{split}.txt"
            file.parent.mkdir(parents=True, exist_ok=True)
            with file.open("w", encoding="utf-8") as fh:
                for rel in sorted(items):
                    fh.write(rel + "\n")
        if self.taxonomy_out and not self.taxonomy_out.exists():
            tax = taxonomy.build_taxonomy(cats)
            taxonomy.save_taxonomy(tax, self.taxonomy_out)
        if failed:
            import csv
            with (out_dir / "_failed.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["path", "role", "category", "model_id", "view_id", "split"])
                for e in failed:
                    writer.writerow([e.path, e.role, e.category, e.model_id, e.view_id or "", e.split])
            logger.warning("%d files failed. See _failed.csv", len(failed))

    def validate_structure(self, root: Path) -> None:  # noqa: D401 - see base class
        missing = 0
        data_paths: Set[str] = set()
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
                    data_file = model_dir / f"{self.basename}_{self.points_n}.{self.file_ext}"
                    rel = f"{cat_dir.name}/{model_dir.name}"
                    data_paths.add(rel)
                    if not data_file.exists():
                        logger.error("Missing point cloud for %s", data_file)
                        missing += 1
        splits_dir = root / "splits"
        for split_file in splits_dir.glob("*.txt"):
            split_name = split_file.stem
            with split_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    rel = line.strip()
                    if rel and rel not in data_paths:
                        logger.error("Split %s references missing model %s", split_name, rel)
                        missing += 1
        lmdb_path = root / "lmdb"
        if lmdb_path.exists():
            import lmdb
            env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
            with env.begin() as txn:
                for rel in data_paths:
                    key = f"object/{rel}".encode("utf-8")
                    val = txn.get(key)
                    if val is None:
                        logger.error("LMDB missing key %s", key.decode())
                        missing += 1
            env.close()
        if missing:
            raise SystemExit(1)
        logger.info("Validation passed")


__all__ = ["ShapeNetProfile"]

