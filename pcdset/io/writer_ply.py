"""PLY writer."""
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


def write_ply(path: Path, points: np.ndarray, attrs: Optional[Dict[str, np.ndarray]] = None) -> None:
    """Write points to ``path`` in PLY format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if o3d is not None:
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        o3d.io.write_point_cloud(str(path), pc, write_ascii=True)
        return
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
