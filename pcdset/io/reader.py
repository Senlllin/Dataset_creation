"""Unified point cloud reader."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import open3d as o3d
except Exception:  # pragma: no cover
    o3d = None  # type: ignore

try:  # pragma: no cover
    from plyfile import PlyData
except Exception:  # pragma: no cover
    PlyData = None  # type: ignore

from ..utils.logging import logger


NumericArray = np.ndarray


def _read_txt_csv(path: Path) -> Tuple[NumericArray, Optional[Dict[str, NumericArray]]]:
    with path.open("r", encoding="utf-8") as fh:
        sample = fh.read(1024)
        fh.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
        df = pd.read_csv(fh, sep=dialect.delimiter)
    numeric = df.select_dtypes(include=["number"])
    if numeric.shape[1] < 3:
        raise ValueError("File must contain at least three numeric columns")
    pts = numeric.iloc[:, :3].to_numpy(dtype=np.float32)
    attrs: Dict[str, NumericArray] = {}
    if numeric.shape[1] > 3:
        for col in numeric.columns[3:]:
            attrs[col] = numeric[col].to_numpy()
    return pts, attrs or None


def _read_ply(path: Path) -> Tuple[NumericArray, Optional[Dict[str, NumericArray]]]:
    if o3d is not None:
        pcd = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(pcd.points, dtype=np.float32)
        attrs: Dict[str, NumericArray] = {}
        if pcd.has_normals():
            attrs["normals"] = np.asarray(pcd.normals, dtype=np.float32)
        return pts, attrs or None
    if PlyData is None:
        raise RuntimeError("plyfile is required to read PLY without open3d")
    data = PlyData.read(path.as_posix())
    el = data["vertex"]
    pts = np.vstack([el["x"], el["y"], el["z"]]).T.astype(np.float32)
    attrs: Dict[str, NumericArray] = {}
    for name in el.data.dtype.names:
        if name in {"x", "y", "z"}:
            continue
        attrs[name] = np.asarray(el[name])
    return pts, attrs or None


def _read_pcd(path: Path) -> Tuple[NumericArray, Optional[Dict[str, NumericArray]]]:
    if o3d is None:
        raise RuntimeError("open3d is required to read PCD files")
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float32)
    attrs: Dict[str, NumericArray] = {}
    if pcd.has_normals():
        attrs["normals"] = np.asarray(pcd.normals, dtype=np.float32)
    return pts, attrs or None


def _read_npz(path: Path) -> Tuple[NumericArray, Optional[Dict[str, NumericArray]]]:
    data = np.load(path)
    if "points" not in data:
        raise ValueError("NPZ file must contain 'points' array")
    pts = np.asarray(data["points"], dtype=np.float32)
    attrs: Dict[str, NumericArray] = {}
    for k in data.files:
        if k == "points":
            continue
        attrs[k] = np.asarray(data[k])
    return pts, attrs or None


_READERS = {
    ".txt": _read_txt_csv,
    ".csv": _read_txt_csv,
    ".ply": _read_ply,
    ".pcd": _read_pcd,
    ".npz": _read_npz,
}


def read_points(path: Path) -> Tuple[NumericArray, Optional[Dict[str, NumericArray]]]:
    """Read a point cloud file.

    Parameters
    ----------
    path:
        Input file path.
    """
    reader = _READERS.get(path.suffix.lower())
    if not reader:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    points, attrs = reader(path)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Point array must be of shape (N,3)")
    if len(points) < 3:
        raise ValueError("Point cloud must contain at least 3 points")
    if not np.isfinite(points).all():
        raise ValueError("Point cloud contains NaN or Inf")
    return points.astype(np.float32), attrs
