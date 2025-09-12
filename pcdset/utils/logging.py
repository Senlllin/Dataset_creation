"""Logging helpers for :mod:`pcdset`."""
from __future__ import annotations

import logging
from typing import Optional


def setup_logging(verbose: bool = False) -> None:
    """Configure root logging.

    Parameters
    ----------
    verbose: bool
        If ``True`` set level to ``DEBUG`` otherwise ``INFO``.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")


logger = logging.getLogger("pcdset")
