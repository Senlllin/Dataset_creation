"""Core manifest data structures."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Entry:
    """Description of a single dataset sample.

    Attributes
    ----------
    path:
        Absolute path to the source point cloud file.
    role:
        The semantic role of the entry (``partial``, ``complete`` or ``object``).
    category:
        Category or synset identifier for the sample.
    model_id:
        Unique identifier for the object instance.
    view_id:
        Optional view identifier for partial scans.
    split:
        Dataset split label (``train``, ``val`` or ``test``).
    """

    path: Path
    role: str
    category: str
    model_id: str
    view_id: Optional[str]
    split: str


__all__ = ["Entry"]
