"""LMDB writer utility."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import lmdb
import msgpack
import msgpack_numpy as m
import numpy as np

m.patch()


def _encode(obj: Dict[str, Any]) -> bytes:
    return msgpack.packb(obj, use_bin_type=True)


@dataclass
class LMDBWriter:
    path: Path
    map_size_gb: int = 64
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.overwrite and self.path.exists():
            for f in self.path.glob("*"):
                if f.is_file():
                    f.unlink()
        self.env = lmdb.open(
            str(self.path), map_size=self.map_size_gb * (1024 ** 3), subdir=True, lock=False, readahead=False
        )

    def put(self, key: str, points: np.ndarray) -> None:
        data = _encode({"points": points.astype(np.float32)})
        with self.env.begin(write=True) as txn:
            txn.put(key.encode("utf-8"), data)

    def close(self, meta: Dict[str, Any]) -> None:
        with self.env.begin(write=True) as txn:
            txn.put(b"__meta__", json.dumps(meta).encode("utf-8"))
        self.env.sync()
        self.env.close()
