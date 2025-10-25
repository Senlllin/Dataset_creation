"""Helpers for constructing manifest entries."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
import random
import re

from .models import Entry

_ALLOWED_EXT = {".ply", ".pcd", ".txt", ".csv", ".npz"}
_SANITISE_PATTERN = re.compile(r"[^-_.0-9a-zA-Z]+")


def _sanitise(text: str) -> str:
    cleaned = _SANITISE_PATTERN.sub("_", text.strip())
    return cleaned or "item"


def build_simple_entries(
    base: Path,
    *,
    allowed_ext: Optional[Iterable[str]] = None,
    default_category: str = "default",
    use_folder_category: bool = True,
) -> List[Entry]:
    """Infer manifest entries from a directory of point cloud files."""

    allowed = set(_ALLOWED_EXT)
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

    files = [file for file in sorted(base.rglob("*")) if file.is_file() and file.suffix.lower() in allowed]
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
        category = _sanitise(category)
        stem = _sanitise(file.stem)
        counts[category][stem] += 1
        idx = counts[category][stem]
        model_id = stem if idx == 1 else f"{stem}_{idx}"
        entries.append(Entry(file, "object", category, model_id, None, "train"))
    return entries


def assign_splits(
    entries: List[Entry],
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    *,
    seed: Optional[int] = None,
) -> None:
    """Shuffle entries in-place and assign dataset split labels."""

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


__all__ = ["build_simple_entries", "assign_splits"]
