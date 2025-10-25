"""Implementation of the ``convert`` command."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import typer

from ...manifest import load_entries, load_entries_shapenet
from ...profiles import PCNProfile, ShapeNetProfile
from ...utils.taxonomy import load_category_map


def _build_pcn_profile(
    *,
    partial_n: int,
    complete_n: int,
    normalize: str,
    center: bool,
    dedup: bool,
    fps: bool,
    voxel: float,
    to_lmdb: bool,
    lmdb_max_gb: int,
    save_attrs: bool,
    overwrite: bool,
    workers: int,
    split_strategy: str,
    ratios: Sequence[float],
) -> PCNProfile:
    return PCNProfile(
        partial_n=partial_n,
        complete_n=complete_n,
        normalize=normalize,
        center=center,
        dedup=dedup,
        fps=fps,
        voxel=voxel,
        to_lmdb=to_lmdb,
        lmdb_max_gb=lmdb_max_gb,
        save_attrs=save_attrs,
        overwrite=overwrite,
        workers=workers,
        split_strategy=split_strategy,
        ratios=ratios,
    )


def _build_shapenet_profile(
    *,
    points_n: int,
    file_ext: str,
    basename: str,
    normalize: str,
    center: bool,
    dedup: bool,
    fps: bool,
    voxel: float,
    to_lmdb: bool,
    lmdb_max_gb: int,
    save_meta: bool,
    save_attrs: bool,
    overwrite: bool,
    workers: int,
    taxonomy_out: Optional[Path],
) -> ShapeNetProfile:
    return ShapeNetProfile(
        points_n=points_n,
        file_ext=file_ext,
        basename=basename,
        normalize=normalize,
        center=center,
        dedup=dedup,
        fps=fps,
        voxel=voxel,
        to_lmdb=to_lmdb,
        lmdb_max_gb=lmdb_max_gb,
        save_meta=save_meta,
        save_attrs=save_attrs,
        overwrite=overwrite,
        workers=workers,
        taxonomy_out=taxonomy_out,
    )


def register(app: typer.Typer) -> None:
    """Register the command with *app*."""

    @app.command()
    def convert(
        profile: str = typer.Option("pcn"),
        input: Path = typer.Option(..., exists=True, file_okay=False, help="Input directory"),
        out: Path = typer.Option(..., file_okay=False, help="Output directory"),
        manifest: Optional[Path] = typer.Option(None, exists=True, dir_okay=False, help="CSV manifest"),
        split_strategy: str = typer.Option("FILE", show_default=True),
        train_ratio: float = typer.Option(0.9),
        val_ratio: float = typer.Option(0.03),
        test_ratio: float = typer.Option(0.07),
        partial_n: int = typer.Option(2048, help="Points per partial cloud"),
        complete_n: int = typer.Option(16384, help="Points per complete cloud"),
        points_n: int = typer.Option(2048, help="Points per cloud"),
        file_ext: str = typer.Option("ply", help="Output file extension", show_default=True),
        basename: str = typer.Option("points", help="Output filename stem"),
        normalize: str = typer.Option("none", help="Normalization: unit|bbox|none"),
        center: bool = typer.Option(False, help="Center the point cloud"),
        dedup: bool = typer.Option(False, help="Remove duplicate points"),
        fps: bool = typer.Option(False, help="Use farthest point sampling"),
        voxel: float = typer.Option(0.0, help="Voxel down sample size"),
        to_lmdb: bool = typer.Option(False, help="Also export LMDB"),
        lmdb_max_gb: int = typer.Option(64, help="LMDB map size in GB"),
        category_map: Optional[Path] = typer.Option(None, exists=True, dir_okay=False),
        taxonomy_out: Optional[Path] = typer.Option(None, help="Write taxonomy CSV/JSON"),
        save_meta: bool = typer.Option(False, help="Save meta.json per model"),
        save_attrs: bool = typer.Option(False, help="Save extra point attributes"),
        overwrite: bool = typer.Option(False, help="Overwrite existing output"),
        workers: int = typer.Option(8, help="Number of worker threads"),
    ) -> None:
        """Convert raw point clouds into a dataset."""

        ratios: Tuple[float, float, float] = (train_ratio, val_ratio, test_ratio)
        cat_map = load_category_map(category_map) if category_map else None

        if profile == "pcn":
            prof = _build_pcn_profile(
                partial_n=partial_n,
                complete_n=complete_n,
                normalize=normalize,
                center=center,
                dedup=dedup,
                fps=fps,
                voxel=voxel,
                to_lmdb=to_lmdb,
                lmdb_max_gb=lmdb_max_gb,
                save_attrs=save_attrs,
                overwrite=overwrite,
                workers=workers,
                split_strategy=split_strategy,
                ratios=ratios,
            )
            entries = load_entries(
                input,
                manifest,
                split_strategy,
                ratios=ratios,
                category_map=cat_map,
            )
        elif profile == "shapenet":
            prof = _build_shapenet_profile(
                points_n=points_n,
                file_ext=file_ext,
                basename=basename,
                normalize=normalize,
                center=center,
                dedup=dedup,
                fps=fps,
                voxel=voxel,
                to_lmdb=to_lmdb,
                lmdb_max_gb=lmdb_max_gb,
                save_meta=save_meta,
                save_attrs=save_attrs,
                overwrite=overwrite,
                workers=workers,
                taxonomy_out=taxonomy_out,
            )
            entries = load_entries_shapenet(
                input,
                manifest,
                split_strategy,
                ratios=ratios,
                category_map=cat_map,
            )
        else:
            raise typer.BadParameter("Unknown profile")

        prof.convert(entries, out)


__all__ = ["register"]
