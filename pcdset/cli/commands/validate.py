"""Implementation of the ``validate`` command."""
from __future__ import annotations

from pathlib import Path

import typer

from ..common import resolve_profile


def register(app: typer.Typer) -> None:
    """Register the command with *app*."""

    @app.command()
    def validate(
        profile: str = typer.Option("pcn"),
        root: Path = typer.Option(..., exists=True, file_okay=False, help="Root dataset directory"),
    ) -> None:
        """Validate that a converted dataset appears structurally sound."""

        profile_cls = resolve_profile(profile)
        profile_cls().validate_structure(root)


__all__ = ["register"]
