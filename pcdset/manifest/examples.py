"""Manifest authoring helpers."""
from __future__ import annotations

from pathlib import Path


def build_example_manifest(path: Path) -> None:
    """Write a small example manifest for the PCN profile."""

    lines = [
        "path,role,category,model_id,view_id,split",
        "chair_0001.ply,partial,chair,0001,00,train",
        "chair_0001_complete.ply,complete,chair,0001,,train",
        "airplane_0001.ply,partial,airplane,0001,00,val",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_example_manifest_shapenet(path: Path) -> None:
    """Write a small example manifest for the ShapeNet profile."""

    lines = [
        "# path,role,category,model_id,view_id,split",
        "# role is ignored; use 'object' or leave blank",
        "path,role,category,model_id,view_id,split",
        "chair_0001.txt,object,chair,0001,,train",
        "airplane_0002.csv,object,airplane,0002,,val",
        "car_0003.ply,object,car,0003,,test",
        "lamp_0004.npz,object,lamp,0004,,train",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["build_example_manifest", "build_example_manifest_shapenet"]
