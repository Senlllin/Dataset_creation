"""Command registration helpers for :mod:`pcdset.cli`."""
from __future__ import annotations

import typer

from . import auto, convert, example, list_profiles, validate


def register(app: typer.Typer) -> None:
    """Register all CLI commands with *app*."""

    for module in (list_profiles, example, validate, convert, auto):
        module.register(app)


__all__ = ["register"]
