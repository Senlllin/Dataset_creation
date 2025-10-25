"""Implementation of the ``list-profiles`` command."""
from __future__ import annotations

import typer

from ..common import profile_descriptions


def register(app: typer.Typer) -> None:
    """Register the command with *app*."""

    @app.command("list-profiles")
    def list_profiles() -> None:
        """List available conversion profiles."""

        for name, description in profile_descriptions().items():
            typer.echo(f"{name} - {description}")


__all__ = ["register"]
