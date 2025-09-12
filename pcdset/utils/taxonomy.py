"""Taxonomy helper utilities.

This module provides small helpers used by the ShapeNet profile to build and
persist taxonomy files.  A taxonomy maps a category *synset* to a human readable
label.  Two file formats are supported: CSV and JSON.  CSV files are expected to
contain two columns named ``synset`` and ``label``.

The functions are intentionally lightweight and do not depend on external
libraries so they can be used in small conversion scripts as well.

Examples
--------
>>> from pathlib import Path
>>> save_taxonomy({'03001627': 'chair'}, Path('tax.csv'))
>>> load_taxonomy(Path('tax.csv'))['03001627']
'chair'
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable


def load_taxonomy(path: Path) -> Dict[str, str]:
    """Load a taxonomy mapping from ``path``.

    Parameters
    ----------
    path:
        Input file path.  The format is derived from the file extension.
    """

    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))

    mapping: Dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            mapping[row["synset"].strip()] = row.get("label", "").strip()
    return mapping


def save_taxonomy(taxonomy: Dict[str, str], path: Path) -> None:
    """Persist ``taxonomy`` to ``path``.

    ``path`` may end in ``.json`` or ``.csv``.  Parent directories are created
    automatically.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        with path.open("w", encoding="utf-8") as fh:
            json.dump(taxonomy, fh, indent=2, sort_keys=True)
        return

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["synset", "label"])
        for syn, label in taxonomy.items():
            writer.writerow([syn, label])


def build_taxonomy(categories: Iterable[str]) -> Dict[str, str]:
    """Create a simple taxonomy mapping from *categories*.

    The returned dictionary maps each category to itself.  It can later be
    modified manually to include human readable labels or synset ids.
    """

    return {c: c for c in sorted(set(categories))}


def load_category_map(path: Path) -> Dict[str, str]:
    """Load a category remapping CSV file.

    The CSV must contain two columns named ``src`` and ``dst``.  This function
    is shared between the PCN and ShapeNet profiles.
    """

    mapping: Dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            mapping[row["src"].strip()] = row["dst"].strip()
    return mapping


__all__ = [
    "load_taxonomy",
    "save_taxonomy",
    "build_taxonomy",
    "load_category_map",
]

