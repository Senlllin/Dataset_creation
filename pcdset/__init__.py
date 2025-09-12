"""Top level package for pcdset.

Provides package metadata and convenience imports.
"""
from __future__ import annotations

from .profiles.shapenet import ShapeNetProfile

__all__ = ["__version__", "ShapeNetProfile"]

__version__ = "0.1.0"
