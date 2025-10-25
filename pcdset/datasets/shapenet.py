"""Tools for converting datasets to the ShapeNet style layout."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from ..profiles.shapenet import ShapeNetProfile
from ..utils.manifest import load_entries_shapenet
from ..utils.taxonomy import load_category_map
from ._common import validate_ratios


@dataclass
class ShapeNetConversionConfig:
    """Configuration for converting point clouds to the ShapeNet style format."""

    input: Path
    out: Path
    manifest: Optional[Path] = None
    split_strategy: str = "FILE"
    ratios: Sequence[float] = (0.9, 0.03, 0.07)
    points_n: int = 2048
    file_ext: str = "ply"
    basename: str = "points"
    normalize: str = "none"
    center: bool = False
    dedup: bool = False
    fps: bool = False
    voxel: float = 0.0
    to_lmdb: bool = False
    lmdb_max_gb: int = 64
    save_meta: bool = False
    save_attrs: bool = False
    overwrite: bool = False
    workers: int = 8
    taxonomy_out: Optional[Path] = None
    category_map: Optional[Path] = None

    def run(self) -> None:
        """Execute the conversion using :class:`~pcdset.profiles.shapenet.ShapeNetProfile`."""

        validate_ratios(self.ratios)

        profile = ShapeNetProfile(
            points_n=self.points_n,
            file_ext=self.file_ext,
            basename=self.basename,
            normalize=self.normalize,
            center=self.center,
            dedup=self.dedup,
            fps=self.fps,
            voxel=self.voxel,
            to_lmdb=self.to_lmdb,
            lmdb_max_gb=self.lmdb_max_gb,
            save_meta=self.save_meta,
            save_attrs=self.save_attrs,
            overwrite=self.overwrite,
            workers=self.workers,
            taxonomy_out=self.taxonomy_out,
        )

        cat_map = load_category_map(self.category_map) if self.category_map else None
        entries = load_entries_shapenet(
            self.input,
            self.manifest,
            self.split_strategy,
            ratios=self.ratios,
            category_map=cat_map,
        )

        profile.convert(entries, self.out)


def main() -> None:
    """Example entry point for generating a ShapeNet style dataset."""

    config = ShapeNetConversionConfig(
        input=Path("data/raw_shapenet"),
        out=Path("data/shapenet_dataset"),
        manifest=Path("data/shapenet_manifest.csv"),
    )
    config.run()


__all__ = ["ShapeNetConversionConfig", "main"]
