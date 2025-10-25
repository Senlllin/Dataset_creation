"""Top level package for pcdset.

Provides package metadata and convenience imports.
"""
from __future__ import annotations

from .profiles.shapenet import ShapeNetProfile
from .datasets import (
    AutoShapeNetConfig,
    PCNConversionConfig,
    ShapeNetConversionConfig,
    auto_shapenet_main,
    pcn_main,
    shapenet_main,
)

__all__ = [
    "__version__",
    "ShapeNetProfile",
    "PCNConversionConfig",
    "ShapeNetConversionConfig",
    "AutoShapeNetConfig",
    "pcn_main",
    "shapenet_main",
    "auto_shapenet_main",
]

__version__ = "0.1.0"
