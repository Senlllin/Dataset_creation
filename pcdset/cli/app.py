"""Typer application wiring for :mod:`pcdset`."""
from __future__ import annotations

import typer

from ..utils import logging as log_utils
from . import commands


def create_app() -> typer.Typer:
    """Create and configure the Typer application."""

    app = typer.Typer(add_completion=False, help="Point cloud dataset tools")

    @app.callback()
    def main_callback(verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging")) -> None:
        """Configure global logging level."""

        log_utils.setup_logging(verbose)

    commands.register(app)
    return app


app = create_app()

__all__ = ["app", "create_app"]
