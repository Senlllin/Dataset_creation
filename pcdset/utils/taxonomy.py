"""Utilities for handling dataset taxonomy files.

The taxonomy maps category identifiers to human readable labels.
The helpers below work with both CSV (``synset,label``) and JSON
({"synset": "label"}) formats.

Example
-------
>>> from pathlib import Path
>>> mapping = build_taxonomy(["chair", "table"])
>>> save_taxonomy(Path('tax.csv'), mapping)
>>> load_taxonomy(Path('tax.csv')) == mapping
True
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable
import csv
import json

__all__ = ["load_taxonomy", "save_taxonomy", "build_taxonomy"]


def load_taxonomy(path: Path) -> Dict[str, str]:
    """Load a taxonomy mapping from ``path``.

    Parameters
    ----------
    path:
        File path pointing to ``.csv`` or ``.json`` taxonomy file.
    """

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    mapping: Dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            mapping[row["synset"].strip()] = row["label"].strip()
    return mapping


def save_taxonomy(path: Path, mapping: Dict[str, str]) -> None:
    """Save ``mapping`` to ``path`` as CSV or JSON.

    Parameters
    ----------
    path:
        Output path; extension decides the format.
    mapping:
        ``{synset: label}`` pairs to write.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        with path.open("w", encoding="utf-8") as fh:
            json.dump(mapping, fh, indent=2, sort_keys=True)
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["synset", "label"])
        for synset, label in mapping.items():
            writer.writerow([synset, label])


def build_taxonomy(categories: Iterable[str]) -> Dict[str, str]:
    """Create a trivial taxonomy mapping each category to itself.

    Useful when building a dataset from scratch without an existing
taxonomy file.
    """

    return {c: c for c in sorted(set(categories))}
