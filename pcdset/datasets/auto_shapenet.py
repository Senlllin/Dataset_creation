"""Automatically build manifests and convert to the ShapeNet layout."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from ..profiles.shapenet import ShapeNetProfile
from ..utils.manifest import assign_splits, build_simple_entries, write_manifest
from ._common import validate_ratios


@dataclass
class AutoShapeNetConfig:
    """Automatically generate a dataset manifest and convert to ShapeNet format."""

    input: Path
    out: Path
    ratios: Sequence[float] = (0.8, 0.1, 0.1)
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
    allowed_ext: Optional[Iterable[str]] = None
    default_category: str = "default"
    use_folder_category: bool = True
    manifest_out: Optional[Path] = None
    seed: Optional[int] = None

    def run(self) -> None:
        """Build entries from a folder structure and convert them."""

        validate_ratios(self.ratios)

        entries = build_simple_entries(
            self.input,
            allowed_ext=None if self.allowed_ext is None else list(self.allowed_ext),
            default_category=self.default_category,
            use_folder_category=self.use_folder_category,
        )
        if not entries:
            raise ValueError("No point cloud files found under the input directory")

        assign_splits(entries, self.ratios, seed=self.seed)

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

        profile.convert(entries, self.out)

        if self.manifest_out:
            write_manifest(entries, self.manifest_out, base=self.input)


def main() -> None:
    """Example entry point for discovering and exporting a dataset."""

    config = AutoShapeNetConfig(
        input=Path("data/raw_point_clouds"),
        out=Path("data/dataset_shapenet"),
        manifest_out=Path("data/manifest.csv"),
    )
    config.run()


__all__ = ["AutoShapeNetConfig", "main"]
