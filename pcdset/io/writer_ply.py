"""Point cloud writers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:  # pragma: no cover
    import open3d as o3d
except Exception:  # pragma: no cover
    o3d = None  # type: ignore

try:  # pragma: no cover
    from plyfile import PlyData, PlyElement
except Exception:  # pragma: no cover
    PlyData = None  # type: ignore

from ..utils.logging import logger


def _write_with_plyfile(path: Path, points: np.ndarray, attrs: Optional[Dict[str, np.ndarray]]) -> None:
    """Fallback PLY writer using :mod:`plyfile`.

    Parameters
    ----------
    path:
        Destination file path.
    points:
        ``(N,3)`` array of XYZ coordinates.
    attrs:
        Optional additional attributes, each ``(N,)`` array.
    """

    if PlyData is None:
        raise RuntimeError("plyfile library required to write PLY when open3d is missing")
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if attrs:
        for key, arr in attrs.items():
            dtype.append((key, arr.dtype.str))
    data = np.empty(len(points), dtype=dtype)
    data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
    if attrs:
        for key, arr in attrs.items():
            data[key] = arr
    el = PlyElement.describe(data, "vertex")
    PlyData([el]).write(str(path))


def write_point_file(path: Path, points: np.ndarray, attrs: Optional[Dict[str, np.ndarray]] = None) -> None:
    """Write ``points`` to ``path``.

    The format is inferred from ``path`` suffix and supports ``.ply``,
    ``.pcd`` and ``.npz``.  Extra attributes are stored in PLY/PCD if
    possible or alongside the ``points`` array in NPZ.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        np.savez(path, points=points.astype(np.float32), **(attrs or {}))
        return

    if o3d is not None:
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        if suffix == ".pcd":
            o3d.io.write_point_cloud(str(path), pc, write_ascii=True)
            return
        if suffix == ".ply":
            o3d.io.write_point_cloud(str(path), pc, write_ascii=True)
            return
        logger.warning("Unsupported extension %s for open3d writer", suffix)
        return

    if suffix == ".pcd":  # pragma: no cover - open3d typically available
        raise RuntimeError("PCD writing requires open3d")
    if suffix == ".ply":
        _write_with_plyfile(path, points, attrs)
        return
    raise RuntimeError(f"Unsupported extension {suffix}")


def write_ply(path: Path, points: np.ndarray, attrs: Optional[Dict[str, np.ndarray]] = None) -> None:
    """Write PLY file (compatibility helper)."""

    if path.suffix.lower() != ".ply":
        path = path.with_suffix(".ply")
    write_point_file(path, points, attrs)


def write_pcd(path: Path, points: np.ndarray, attrs: Optional[Dict[str, np.ndarray]] = None) -> None:
    """Write PCD file."""

    if path.suffix.lower() != ".pcd":
        path = path.with_suffix(".pcd")
    write_point_file(path, points, attrs)


def write_npz(path: Path, points: np.ndarray, attrs: Optional[Dict[str, np.ndarray]] = None) -> None:
    """Write NPZ file with a ``points`` array."""

    if path.suffix.lower() != ".npz":
        path = path.with_suffix(".npz")
    write_point_file(path, points, attrs)
