"""Backward compatible shim for :mod:`pcdset.cli`."""
from __future__ import annotations

from .cli import app

__all__ = ["app"]
