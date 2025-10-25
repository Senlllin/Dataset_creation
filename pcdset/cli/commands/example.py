"""Implementation of the ``example-manifest`` command."""
from __future__ import annotations

from pathlib import Path

import typer

from ...manifest import build_example_manifest, build_example_manifest_shapenet


def register(app: typer.Typer) -> None:
    """Register the command with *app*."""

    @app.command("example-manifest")
    def example_manifest(
        profile: str = typer.Option("pcn"),
        output: Path = typer.Option("manifest.csv", "-o", help="Output CSV path"),
    ) -> None:
        """Create an example manifest file for the selected profile."""

        if profile == "pcn":
            build_example_manifest(output)
        elif profile == "shapenet":
            build_example_manifest_shapenet(output)
        else:
            raise typer.BadParameter("Unknown profile")
        typer.echo(f"Wrote example manifest to {output}")


__all__ = ["register"]
