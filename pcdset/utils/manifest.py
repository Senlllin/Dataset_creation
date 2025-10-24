"""Manifest handling utilities."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import csv
import random
import re

import pandas as pd

from .logging import logger

@dataclass
class Entry:
    path: Path
    role: str  # e.g. 'partial', 'complete' or 'object'
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


_SAN = re.compile(r"[^-_.0-9a-zA-Z]+")


def _sanitize(text: str) -> str:
    """Return a filesystem friendly identifier."""

    cleaned = _SAN.sub("_", text.strip())
    return cleaned or "item"


def build_example_manifest(path: Path) -> None:
    """Write a small example manifest for the PCN profile."""
    rows = [
        {"path": "chair_0001.ply", "role": "partial", "category": "chair", "model_id": "0001", "view_id": "00", "split": "train"},
        {"path": "chair_0001_complete.ply", "role": "complete", "category": "chair", "model_id": "0001", "view_id": "", "split": "train"},
        {"path": "airplane_0001.ply", "role": "partial", "category": "airplane", "model_id": "0001", "view_id": "00", "split": "val"},
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "role", "category", "model_id", "view_id", "split"])
        writer.writeheader()
        writer.writerows(rows)


def build_example_manifest_shapenet(path: Path) -> None:
    """Write a small example manifest for the ShapeNet profile."""

    lines = [
        "# path,role,category,model_id,view_id,split",
        "# role is ignored; use 'object' or leave blank",
        "path,role,category,model_id,view_id,split",
        "chair_0001.txt,object,chair,0001,,train",
        "airplane_0002.csv,object,airplane,0002,,val",
        "car_0003.ply,object,car,0003,,test",
        "lamp_0004.npz,object,lamp,0004,,train",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_simple_entries(
    base: Path,
    *,
    allowed_ext: Optional[Iterable[str]] = None,
    default_category: str = "default",
    use_folder_category: bool = True,
) -> List[Entry]:
    """Infer entries from a directory of point cloud files."""

    allowed = {".ply", ".pcd", ".txt", ".csv", ".npz"}
    if allowed_ext:
        normalised = set()
        for ext in allowed_ext:
            ext = ext.strip().lower()
            if not ext:
                continue
            if not ext.startswith("."):
                ext = f".{ext}"
            normalised.add(ext)
        if normalised:
            allowed = normalised

    files: List[Path] = []
    for file in sorted(base.rglob("*")):
        if file.is_file() and file.suffix.lower() in allowed:
            files.append(file)

    if not files:
        return []

    categories_present = {
        file.relative_to(base).parts[0]
        for file in files
        if len(file.relative_to(base).parts) > 1
    }
    use_categories = use_folder_category and bool(categories_present)

    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    entries: List[Entry] = []
    for file in files:
        rel = file.relative_to(base)
        parts = rel.parts
        if use_categories and len(parts) > 1:
            category = parts[0]
        else:
            category = default_category
        category = _sanitize(category)
        stem = _sanitize(file.stem)
        counts[category][stem] += 1
        idx = counts[category][stem]
        model_id = stem if idx == 1 else f"{stem}_{idx}"
        entries.append(Entry(file, "object", category, model_id, None, "train"))
    return entries


def assign_splits(
    entries: List[Entry],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    *,
    seed: Optional[int] = None,
) -> None:
    """Shuffle and assign dataset splits in-place."""

    if not entries:
        return
    rng = random.Random(seed)
    rng.shuffle(entries)
    train_ratio, val_ratio, test_ratio = ratios
    n = len(entries)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    for i, entry in enumerate(entries):
        if i < n_train:
            entry.split = "train"
        elif i < n_train + n_val:
            entry.split = "val"
        else:
            entry.split = "test"


def write_manifest(entries: Iterable[Entry], path: Path, base: Optional[Path] = None) -> None:
    """Write manifest entries to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["path", "role", "category", "model_id", "view_id", "split"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            rel_path = entry.path
            if base:
                try:
                    rel_path = entry.path.relative_to(base)
                except ValueError:
                    rel_path = entry.path
            writer.writerow(
                {
                    "path": str(rel_path),
                    "role": entry.role,
                    "category": entry.category,
                    "model_id": entry.model_id,
                    "view_id": entry.view_id or "",
                    "split": entry.split,
                }
            )


def load_entries(base: Path, manifest: Optional[Path], split_strategy: str = "FILE", ratios: Tuple[float, float, float] = (0.9,0.03,0.07), category_map: Optional[Dict[str,str]] = None) -> List[Entry]:
    """Load manifest entries or infer from directory."""
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
            entries.append(Entry(path, row.role, cat, str(row.model_id), view_id, split))
    else:
        # Directory inference
        allowed_ext = {".ply", ".pcd", ".txt", ".csv", ".npz"}
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


def load_entries_shapenet(base: Path, manifest: Optional[Path], split_strategy: str = "FILE", ratios: Tuple[float, float, float] = (0.9,0.05,0.05), category_map: Optional[Dict[str, str]] = None) -> List[Entry]:
    """Load entries for the ShapeNet profile."""

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
            model_id = str(row.model_id)
            view_id = getattr(row, "view_id", None)
            view_id = None if (isinstance(view_id, float) and pd.isna(view_id)) else (str(view_id) if view_id else None)
            split = getattr(row, "split", "train")
            role = getattr(row, "role", "object")
            entries.append(Entry(path, role, cat, model_id, view_id, split))
    else:
        allowed_ext = {".ply", ".pcd", ".txt", ".csv", ".npz"}
        for cat_dir in base.iterdir():
            if not cat_dir.is_dir():
                continue
            cat = category_map.get(cat_dir.name, cat_dir.name) if category_map else cat_dir.name
            for model_dir in cat_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model_id = model_dir.name
                file: Optional[Path] = None
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
