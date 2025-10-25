"""Shared helpers for CLI commands."""
from __future__ import annotations

from typing import Dict, Type

import typer

from ..profiles import BaseProfile, iter_profiles


def profile_descriptions() -> Dict[str, str]:
    """Return a mapping of profile names to human readable descriptions."""

    return {name: description for name, _cls, description in iter_profiles()}


def resolve_profile(name: str) -> Type[BaseProfile]:
    """Return the profile class registered under ``name`` or raise an error."""

    for registered, cls, _description in iter_profiles():
        if registered == name:
            return cls
    raise typer.BadParameter("Unknown profile")


__all__ = ["profile_descriptions", "resolve_profile"]
