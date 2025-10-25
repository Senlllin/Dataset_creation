"""Common helpers for dataset generation scripts."""
from __future__ import annotations

import math
from typing import Sequence


def validate_ratios(ratios: Sequence[float]) -> None:
    """Ensure train/val/test ratios sum to 1."""

    if len(ratios) != 3:
        raise ValueError("Expected three ratios for train/val/test splits")

    total = sum(ratios)
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("train, val and test ratios must sum to 1.0")


__all__ = ["validate_ratios"]
