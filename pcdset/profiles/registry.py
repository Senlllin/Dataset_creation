"""Registry of available dataset conversion profiles."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple, Type

from .base import BaseProfile
from .pcn import PCNProfile
from .shapenet import ShapeNetProfile

_ProfileInfo = Tuple[Type[BaseProfile], str]

_PROFILE_REGISTRY: Dict[str, _ProfileInfo] = {
    "pcn": (PCNProfile, "PCN dataset structure (partial_n=2048, complete_n=16384)"),
    "shapenet": (ShapeNetProfile, "ShapeNet dataset structure (points_n=2048)"),
}


def get_profile_class(name: str) -> Type[BaseProfile]:
    """Return the profile class registered under ``name``."""

    try:
        return _PROFILE_REGISTRY[name][0]
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise KeyError(f"Unknown profile '{name}'") from exc


def iter_profiles() -> Iterable[Tuple[str, Type[BaseProfile], str]]:
    """Yield ``(name, class, description)`` tuples for registered profiles."""

    for name, (cls, description) in _PROFILE_REGISTRY.items():
        yield name, cls, description


__all__ = ["get_profile_class", "iter_profiles"]
