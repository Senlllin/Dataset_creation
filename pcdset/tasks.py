"""Backward-compatible re-exports for dataset generation configs."""
from __future__ import annotations

from .datasets import AutoShapeNetConfig, PCNConversionConfig, ShapeNetConversionConfig

__all__ = [
    "PCNConversionConfig",
    "ShapeNetConversionConfig",
    "AutoShapeNetConfig",
]
