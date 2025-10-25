"""Persistence helpers for manifest entries."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Optional

from .models import Entry


def write_manifest(entries: Iterable[Entry], path: Path, base: Optional[Path] = None) -> None:
    """Write *entries* to ``path`` in CSV format."""

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


__all__ = ["write_manifest"]
