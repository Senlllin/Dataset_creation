"""Base profile definition.

Profiles encapsulate dataset specific conversion logic.  Subclasses
should implement :meth:`prepare`, :meth:`writer` and
:meth:`validate_structure`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Dict, Any, Tuple

import numpy as np


class BaseProfile(ABC):
    """Abstract profile for dataset conversion."""

    name: str = "base"

    @abstractmethod
    def prepare(self, points: np.ndarray, role: str, args: Any) -> np.ndarray:
        """Return processed points for the given role."""

    @abstractmethod
    def convert(self, entries: Iterable[Dict[str, Any]], out_dir: Path) -> None:
        """Convert entries to dataset structure."""

    @abstractmethod
    def validate_structure(self, root: Path) -> None:
        """Validate the produced dataset."""
