"""Dataset generation entry points grouped by dataset family."""
from __future__ import annotations

from .auto_shapenet import AutoShapeNetConfig, main as auto_shapenet_main
from .pcn import PCNConversionConfig, main as pcn_main
from .shapenet import ShapeNetConversionConfig, main as shapenet_main

__all__ = [
    "PCNConversionConfig",
    "ShapeNetConversionConfig",
    "AutoShapeNetConfig",
    "pcn_main",
    "shapenet_main",
    "auto_shapenet_main",
]
