"""Manifest handling utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
import csv
import random

import pandas as pd

from .logging import logger

@dataclass
class Entry:
    path: Path
    role: str  # 'partial' or 'complete'
    category: str
    model_id: str
    view_id: Optional[str]
    split: str  # train/val/test


def load_category_map(path: Path) -> Dict[str, str]:
    """Load a category mapping CSV.

    CSV format: ``src,dst``.
    """
    mapping: Dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            mapping[row["src"].strip()] = row["dst"].strip()
    return mapping


def build_example_manifest(path: Path, profile: str = "pcn") -> None:
    """Write a small example manifest for ``profile``."""

    if profile == "pcn":
        rows = [
            {
                "path": "chair_0001.ply",
                "role": "partial",
                "category": "chair",
                "model_id": "0001",
                "view_id": "00",
                "split": "train",
            },
            {
                "path": "chair_0001_complete.ply",
                "role": "complete",
                "category": "chair",
                "model_id": "0001",
                "view_id": "",
                "split": "train",
            },
            {
                "path": "airplane_0001.ply",
                "role": "partial",
                "category": "airplane",
                "model_id": "0001",
                "view_id": "00",
                "split": "val",
            },
        ]
    else:  # shapenet
        rows = [
            {
                "path": "chair_0001.ply",
                "role": "object",
                "category": "chair",
                "model_id": "0001",
                "view_id": "",
                "split": "train",
            },
            {
                "path": "chair_0002.csv",
                "role": "object",
                "category": "chair",
                "model_id": "0002",
                "view_id": "",
                "split": "val",
            },
            {
                "path": "table_0001.txt",
                "role": "object",
                "category": "table",
                "model_id": "0001",
                "view_id": "",
                "split": "test",
            },
            {
                "path": "sofa_0003.pcd",
                "role": "object",
                "category": "sofa",
                "model_id": "0003",
                "view_id": "",
                "split": "train",
            },
        ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        if profile == "shapenet":
            fh.write("# path,role,category,model_id,view_id,split\n")
            fh.write("# role is ignored for shapenet; use 'object'\n")
        writer = csv.DictWriter(
            fh, fieldnames=["path", "role", "category", "model_id", "view_id", "split"]
        )
        writer.writeheader()
        writer.writerows(rows)


def load_entries(base: Path,
                 manifest: Optional[Path],
                 split_strategy: str = "FILE",
                 ratios: Tuple[float, float, float] = (0.9,0.03,0.07),
                 category_map: Optional[Dict[str,str]] = None,
                 profile: str = "pcn") -> List[Entry]:
    """Load manifest entries or infer from directory.

    Parameters
    ----------
    profile:
        Either ``"pcn"`` or ``"shapenet"`` determining directory
        inference strategy and how manifest rows are interpreted.
    """
    entries: List[Entry] = []
    if manifest:
        df = pd.read_csv(manifest)
        for row in df.itertuples(index=False):
            path = Path(row.path)
            if not path.is_absolute():
                path = base / path
            cat = row.category
            if category_map and cat in category_map:
                cat = category_map[cat]
            view_id = getattr(row, "view_id", None)
            view_id = None if (isinstance(view_id, float) and pd.isna(view_id)) else (str(view_id) if view_id else None)
            split = getattr(row, "split", "train")
            role = getattr(row, "role", "object")
            if profile == "shapenet":
                role = "object"
            entries.append(Entry(path, role, cat, str(row.model_id), view_id, split))
    else:
        allowed_ext = {".ply", ".pcd", ".txt", ".csv", ".npz"}
        if profile == "pcn":
            for role in ("partial", "complete"):
                role_dir = base / role
                if not role_dir.exists():
                    continue
                for cat_dir in role_dir.iterdir():
                    if not cat_dir.is_dir():
                        continue
                    cat = category_map.get(cat_dir.name, cat_dir.name) if category_map else cat_dir.name
                    for model_dir in cat_dir.iterdir():
                        if not model_dir.is_dir():
                            continue
                        model_id = model_dir.name
                        if role == "partial":
                            for file in model_dir.iterdir():
                                if file.suffix.lower() not in allowed_ext:
                                    continue
                                view_id = file.stem
                                entries.append(Entry(file, role, cat, model_id, view_id, "train"))
                        else:
                            for file in model_dir.iterdir():
                                if file.suffix.lower() not in allowed_ext:
                                    continue
                                entries.append(Entry(file, role, cat, model_id, None, "train"))
        else:  # shapenet
            for cat_dir in base.iterdir():
                if not cat_dir.is_dir():
                    continue
                cat = category_map.get(cat_dir.name, cat_dir.name) if category_map else cat_dir.name
                for model_dir in cat_dir.iterdir():
                    if not model_dir.is_dir():
                        continue
                    model_id = model_dir.name
                    file = None
                    for f in model_dir.iterdir():
                        if f.suffix.lower() in allowed_ext:
                            file = f
                            break
                    if file is None:
                        continue
                    entries.append(Entry(file, "object", cat, model_id, None, "train"))
    if split_strategy.upper() == "RATIO" and entries:
        random.shuffle(entries)
        n = len(entries)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        for i, e in enumerate(entries):
            if i < n_train:
                e.split = "train"
            elif i < n_train + n_val:
                e.split = "val"
            else:
                e.split = "test"
    return entries
