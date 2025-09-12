"""Point cloud resampling utilities."""
from __future__ import annotations

from typing import Tuple

import numpy as np

try:  # pragma: no cover - open3d optional
    import open3d as o3d
except Exception:  # pragma: no cover
    o3d = None  # type: ignore


def random_sample(points: np.ndarray, n: int) -> np.ndarray:
    """Randomly sample ``n`` points."""
    if len(points) >= n:
        idx = np.random.choice(len(points), n, replace=False)
    else:
        idx = np.random.choice(len(points), n, replace=True)
    return points[idx]


def farthest_point_sample(points: np.ndarray, n: int) -> np.ndarray:
    """Naive farthest point sampling."""
    if o3d is not None:
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        ds = pc.farthest_point_down_sample(n)
        return np.asarray(ds.points, dtype=np.float32)
    pts = points
    N = len(pts)
    if N == 0:
        return pts
    farthest = np.zeros((n,), dtype=np.int64)
    distance = np.full((N,), np.inf)
    farthest[0] = np.random.randint(N)
    for i in range(1, n):
        dist = np.linalg.norm(pts - pts[farthest[i - 1]], axis=1)
        distance = np.minimum(distance, dist)
        farthest[i] = np.argmax(distance)
    return pts[farthest]


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Down sample using a voxel grid."""
    if voxel_size <= 0:
        return points
    if o3d is None:  # simple rounding fallback
        keys = np.floor(points / voxel_size)
        _, idx = np.unique(keys, axis=0, return_index=True)
        return points[idx]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    ds = pc.voxel_down_sample(voxel_size)
    return np.asarray(ds.points, dtype=np.float32)


def dedup(points: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Approximate duplicate removal using rounding."""
    keys = np.round(points / eps).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[idx]
