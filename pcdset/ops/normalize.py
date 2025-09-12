"""Simple normalization utilities."""
from __future__ import annotations

import numpy as np


def center(points: np.ndarray) -> np.ndarray:
    """Center points around the origin.

    Examples
    --------
    >>> pts = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
    >>> centered = center(pts)
    >>> np.allclose(centered.mean(axis=0), 0)
    True
    """
    return points - points.mean(axis=0, keepdims=True)


def unit_sphere(points: np.ndarray) -> np.ndarray:
    """Scale points so the furthest point has radius 1."""
    radius = np.linalg.norm(points, axis=1).max()
    if radius == 0:
        return points
    return points / radius


def bbox_scale(points: np.ndarray) -> np.ndarray:
    """Scale points so that the longest bounding box edge is 1."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    size = (maxs - mins).max()
    if size == 0:
        return points
    return points / size
