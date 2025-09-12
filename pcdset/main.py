"""Main entry point for the :mod:`pcdset` command line tool.

Exposes conversion profiles such as ``pcn`` and ``shapenet`` via the
CLI defined in :mod:`pcdset.cli`.
"""
from __future__ import annotations

from .cli import app


def main() -> None:
    """Run the :mod:`pcdset` CLI."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
