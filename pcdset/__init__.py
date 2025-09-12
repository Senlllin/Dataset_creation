"""Top level package for pcdset.

Provides package metadata and convenience imports.
"""
from __future__ import annotations

from .profiles.pcn import PCNProfile
from .profiles.shapenet import ShapeNetProfile

__all__ = ["__version__", "PCNProfile", "ShapeNetProfile"]

__version__ = "0.2.1"
