"""Command line interface for :mod:`pcdset`.

This module exposes a :class:`typer.Typer` application that dispatches to
profile specific conversion logic.  Two profiles are bundled: ``pcn`` and
``shapenet``.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import typer

from .profiles.pcn import PCNProfile
from .profiles.shapenet import ShapeNetProfile
from .utils import logging as log_utils
from .utils.manifest import (
    assign_splits,
    build_example_manifest,
    build_example_manifest_shapenet,
    build_simple_entries,
    load_entries,
    load_entries_shapenet,
    write_manifest,
)
from .utils.taxonomy import load_category_map


PROFILES = {
    "pcn": PCNProfile,
    "shapenet": ShapeNetProfile,
}


app = typer.Typer(add_completion=False, help="Point cloud dataset tools")


@app.callback()
def main_callback(verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging")) -> None:
    """Configure global logging level."""
    log_utils.setup_logging(verbose)


@app.command("list-profiles")
def list_profiles() -> None:
    """List available conversion profiles."""
    typer.echo("pcn - PCN dataset structure (partial_n=2048, complete_n=16384)")
    typer.echo("shapenet - ShapeNet dataset structure (points_n=2048)")


@app.command("example-manifest")
def example_manifest(
    profile: str = typer.Option("pcn"),
    output: Path = typer.Option("manifest.csv", "-o", help="Output CSV path"),
) -> None:
    """Create an example manifest file for the selected profile."""
    if profile == "pcn":
        build_example_manifest(output)
    elif profile == "shapenet":
        build_example_manifest_shapenet(output)
    else:
        raise typer.BadParameter("Unknown profile")
    typer.echo(f"Wrote example manifest to {output}")


@app.command()
def validate(
    profile: str = typer.Option("pcn"),
    root: Path = typer.Option(..., exists=True, file_okay=False, help="Root dataset directory"),
) -> None:
    """Validate that a converted dataset appears structurally sound."""
    if profile not in PROFILES:
        raise typer.BadParameter("Unknown profile")
    prof = PROFILES[profile]()
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
        cat_map = load_category_map(category_map) if category_map else None
        entries = load_entries(
            input,
            manifest,
            split_strategy,
            ratios=(train_ratio, val_ratio, test_ratio),
            category_map=cat_map,
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
            save_meta=save_meta,
            save_attrs=save_attrs,
            overwrite=overwrite,
            workers=workers,
            taxonomy_out=taxonomy_out,
        )
        cat_map = load_category_map(category_map) if category_map else None
        entries = load_entries_shapenet(
            input,
            manifest,
            split_strategy,
            ratios=(train_ratio, val_ratio, test_ratio),
            category_map=cat_map,
        )
    else:
        raise typer.BadParameter("Unknown profile")

    prof.convert(entries, out)



@app.command("auto")
def auto_convert(
    input: Path = typer.Option(..., exists=True, file_okay=False, help="Directory containing point clouds"),
    out: Path = typer.Option(..., file_okay=False, help="Output dataset directory"),
    train_ratio: float = typer.Option(0.8, help="Fraction of samples used for training"),
    val_ratio: float = typer.Option(0.1, help="Fraction of samples used for validation"),
    test_ratio: float = typer.Option(0.1, help="Fraction of samples used for testing"),
    points_n: int = typer.Option(2048, help="Points per output cloud"),
    file_ext: str = typer.Option("ply", help="Output file extension", show_default=True),
    basename: str = typer.Option("points", help="Output filename stem"),
    normalize: str = typer.Option("none", help="Normalization: unit|bbox|none"),
    center: bool = typer.Option(False, help="Center the point cloud"),
    dedup: bool = typer.Option(False, help="Remove duplicate points"),
    fps: bool = typer.Option(False, help="Use farthest point sampling"),
    voxel: float = typer.Option(0.0, help="Voxel down sample size"),
    to_lmdb: bool = typer.Option(False, help="Also export LMDB"),
    lmdb_max_gb: int = typer.Option(64, help="LMDB map size in GB"),
    save_meta: bool = typer.Option(False, help="Save meta.json per model"),
    save_attrs: bool = typer.Option(False, help="Save extra point attributes"),
    overwrite: bool = typer.Option(False, help="Overwrite existing output"),
    workers: int = typer.Option(8, help="Number of worker threads"),
    taxonomy_out: Optional[Path] = typer.Option(None, help="Write taxonomy CSV/JSON"),
    allowed_ext: Optional[str] = typer.Option(None, help="Comma separated list of file extensions to include"),
    default_category: str = typer.Option("default", help="Fallback category name"),
    use_folder_category: bool = typer.Option(True, help="Use top-level folder names as categories when present"),
    manifest_out: Optional[Path] = typer.Option(None, help="Optional path to write the generated manifest"),
    seed: Optional[int] = typer.Option(None, help="Random seed for split shuffling"),
) -> None:
    """Automatically organise a folder of point clouds into a dataset."""

    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise typer.BadParameter("train, val and test ratios must sum to 1.0")

    exts = None
    if allowed_ext:
        exts = [part.strip() for part in allowed_ext.split(',') if part.strip()]

    entries = build_simple_entries(
        input,
        allowed_ext=exts,
        default_category=default_category,
        use_folder_category=use_folder_category,
    )
    if not entries:
        raise typer.BadParameter("No point cloud files found under the input directory")

    assign_splits(entries, (train_ratio, val_ratio, test_ratio), seed=seed)

    profile = ShapeNetProfile(
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

    profile.convert(entries, out)

    if manifest_out:
        write_manifest(entries, manifest_out, base=input)

    summary = {"train": 0, "val": 0, "test": 0}
    extras: dict[str, int] = {}
    for entry in entries:
        if entry.split in summary:
            summary[entry.split] += 1
        else:
            extras[entry.split] = extras.get(entry.split, 0) + 1
    parts = [f"{split}={summary[split]}" for split in ("train", "val", "test")]
    parts.extend(f"{split}={count}" for split, count in sorted(extras.items()))
    typer.echo("Conversion finished. Samples per split: " + ", ".join(parts))


__all__ = ["app"]

