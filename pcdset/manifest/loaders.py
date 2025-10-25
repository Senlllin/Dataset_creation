"""Load manifest entries from CSV files or folder structures."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .builders import _ALLOWED_EXT
from .models import Entry


def _normalise_view_id(value: object) -> Optional[str]:
    if isinstance(value, float) and pd.isna(value):
        return None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_entries(
    base: Path,
    manifest: Optional[Path],
    split_strategy: str = "FILE",
    ratios: Sequence[float] = (0.9, 0.03, 0.07),
    *,
    category_map: Optional[Dict[str, str]] = None,
) -> List[Entry]:
    """Load PCN profile entries from ``manifest`` or infer them from ``base``."""

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
            view_id = _normalise_view_id(getattr(row, "view_id", None))
            split = getattr(row, "split", "train")
            entries.append(Entry(path, row.role, cat, str(row.model_id), view_id, split))
    else:
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
                    for file in model_dir.iterdir():
                        if file.suffix.lower() not in _ALLOWED_EXT:
                            continue
                        if role == "partial":
                            view_id = file.stem
                        else:
                            view_id = None
                        entries.append(Entry(file, role, cat, model_id, view_id, "train"))

    if split_strategy.upper() == "RATIO" and entries:
        random.shuffle(entries)
        n = len(entries)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        for i, entry in enumerate(entries):
            if i < n_train:
                entry.split = "train"
            elif i < n_train + n_val:
                entry.split = "val"
            else:
                entry.split = "test"
    return entries


def load_entries_shapenet(
    base: Path,
    manifest: Optional[Path],
    split_strategy: str = "FILE",
    ratios: Sequence[float] = (0.9, 0.05, 0.05),
    *,
    category_map: Optional[Dict[str, str]] = None,
) -> List[Entry]:
    """Load ShapeNet profile entries from ``manifest`` or infer them from ``base``."""

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
            view_id = _normalise_view_id(getattr(row, "view_id", None))
            split = getattr(row, "split", "train")
            role = getattr(row, "role", "object")
            entries.append(Entry(path, role, cat, model_id, view_id, split))
    else:
        for cat_dir in base.iterdir():
            if not cat_dir.is_dir():
                continue
            cat = category_map.get(cat_dir.name, cat_dir.name) if category_map else cat_dir.name
            for model_dir in cat_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model_id = model_dir.name
                file: Optional[Path] = None
                for candidate in model_dir.iterdir():
                    if candidate.suffix.lower() in _ALLOWED_EXT:
                        file = candidate
                        break
                if file is None:
                    continue
                entries.append(Entry(file, "object", cat, model_id, None, "train"))

    if split_strategy.upper() == "RATIO" and entries:
        random.shuffle(entries)
        n = len(entries)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        for i, entry in enumerate(entries):
            if i < n_train:
                entry.split = "train"
            elif i < n_train + n_val:
                entry.split = "val"
            else:
                entry.split = "test"
    return entries


__all__ = ["load_entries", "load_entries_shapenet"]
