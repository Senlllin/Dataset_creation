"""Point cloud writers for simple formats.

This module primarily exposes :func:`write_ply` which historically only
handled the PLY format.  For the ShapeNet profile we also need support for
``.pcd`` and ``.npz`` outputs while keeping the same function signature.
The implementation therefore inspects the target file extension and selects
an appropriate backend.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import open3d as o3d
except Exception:  # pragma: no cover
    o3d = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from plyfile import PlyData, PlyElement
except Exception:  # pragma: no cover
    PlyData = None  # type: ignore

from ..utils.logging import logger


def write_ply(path: Path, points: np.ndarray, attrs: Optional[Dict[str, np.ndarray]] = None) -> None:
    """Write points to ``path``.

    The output format is determined by the file extension:

    ``.ply`` or ``.pcd``
        Written via :mod:`open3d` when available and falling back to
        :mod:`plyfile` for PLY.  Only the XYZ coordinates are persisted unless
        ``attrs`` is provided, in which case additional columns are written
        for the PLY backend.

    ``.npz``
        Stored using :func:`numpy.savez` with the key ``"points"`` and any
        optional attributes.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()

    if ext == ".npz":
        data: Dict[str, np.ndarray] = {"points": points.astype(np.float32)}
        if attrs:
            data.update(attrs)
        np.savez(path, **data)
        return

    if o3d is not None:
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        # open3d selects the format based on the extension (ply or pcd).
        o3d.io.write_point_cloud(str(path), pc, write_ascii=True)
        return

    if PlyData is None or ext != ".ply":
        raise RuntimeError("open3d required to write this file format")

    # Fallback PLY writer using plyfile
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
