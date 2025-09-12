"""Command line interface for :mod:`pcdset`.

This module exposes a :class:`typer.Typer` app with subcommands
implemented in other modules.  The :func:`pcdset.main.main` function
invokes this app.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .profiles.pcn import PCNProfile
from .profiles.shapenet import ShapeNetProfile
from .utils import logging as log_utils
from .utils.manifest import (
    build_example_manifest,
    load_entries,
    load_category_map,
)

app = typer.Typer(add_completion=False, help="Point cloud dataset tools")


@app.callback()
def main_callback(verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging")) -> None:
    """Configure global logging level."""
    log_utils.setup_logging(verbose)


@app.command("list-profiles")
def list_profiles() -> None:
    """List available conversion profiles."""
    pcn = PCNProfile()
    shn = ShapeNetProfile()
    typer.echo(f"pcn - partial_n={pcn.partial_n} complete_n={pcn.complete_n}")
    typer.echo(f"shapenet - points_n={shn.points_n} file_ext={shn.file_ext}")


@app.command("example-manifest")
def example_manifest(profile: str = typer.Option("pcn"),
                     output: Path = typer.Option("manifest.csv", "-o", help="Output CSV path")) -> None:
    """Create an example manifest file for the selected profile."""
    if profile not in ("pcn", "shapenet"):
        raise typer.BadParameter("Unknown profile")
    build_example_manifest(output, profile)
    typer.echo(f"Wrote example manifest to {output}")


@app.command()
def validate(profile: str = typer.Option("pcn"),
            root: Path = typer.Option(..., exists=True, file_okay=False, help="Root dataset directory")) -> None:
    """Validate that a converted dataset appears structurally sound."""
    if profile == "pcn":
        prof = PCNProfile()
    elif profile == "shapenet":
        prof = ShapeNetProfile()
    else:
        raise typer.BadParameter("Unknown profile")
    prof.validate_structure(root)


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
    points_n: int = typer.Option(2048, help="Points per object"),
    file_ext: str = typer.Option("ply", help="Output file extension"),
    basename: str = typer.Option("points", help="Output file stem"),
    normalize: str = typer.Option("none", help="Normalization: unit|bbox|none"),
    center: bool = typer.Option(False, help="Center the point cloud"),
    dedup: bool = typer.Option(False, help="Remove duplicate points"),
    fps: bool = typer.Option(False, help="Use farthest point sampling"),
    voxel: float = typer.Option(0.0, help="Voxel down sample size"),
    to_lmdb: bool = typer.Option(False, help="Also export LMDB"),
    lmdb_max_gb: int = typer.Option(64, help="LMDB map size in GB"),
    category_map: Optional[Path] = typer.Option(None, exists=True, dir_okay=False),
    taxonomy_out: Optional[Path] = typer.Option(None, help="Write taxonomy file"),
    save_meta: bool = typer.Option(False, help="Save meta.json per model"),
    save_attrs: bool = typer.Option(False, help="Save extra point attributes"),
    overwrite: bool = typer.Option(False, help="Overwrite existing output"),
    workers: int = typer.Option(8, help="Number of worker threads"),
) -> None:
    """Convert raw point clouds into a dataset."""
    if profile == "pcn":
        prof = PCNProfile(
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
            ratios=(train_ratio, val_ratio, test_ratio),
        )
    elif profile == "shapenet":
        prof = ShapeNetProfile(
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
            taxonomy_out=taxonomy_out,
            save_meta=save_meta,
            save_attrs=save_attrs,
            overwrite=overwrite,
            workers=workers,
            split_strategy=split_strategy,
            ratios=(train_ratio, val_ratio, test_ratio),
        )
    else:
        raise typer.BadParameter("Unknown profile")

    cat_map = load_category_map(category_map) if category_map else None
    entries = load_entries(
        input,
        manifest,
        split_strategy,
        ratios=(train_ratio, val_ratio, test_ratio),
        category_map=cat_map,
        profile=profile,
    )
    prof.convert(entries, out)
