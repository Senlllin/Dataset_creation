"""Implementation of the ``auto`` command."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import typer

from ...manifest import assign_splits, build_simple_entries, write_manifest
from ...profiles import ShapeNetProfile
from ...utils.taxonomy import load_category_map


def register(app: typer.Typer) -> None:
    """Register the command with *app*."""

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
        category_map: Optional[Path] = typer.Option(None, exists=True, dir_okay=False),
    ) -> None:
        """Automatically organise a folder of point clouds into a dataset."""

        total = train_ratio + val_ratio + test_ratio
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise typer.BadParameter("train, val and test ratios must sum to 1.0")

        exts = None
        if allowed_ext:
            exts = [part.strip() for part in allowed_ext.split(",") if part.strip()]

        entries = build_simple_entries(
            input,
            allowed_ext=exts,
            default_category=default_category,
            use_folder_category=use_folder_category,
        )
        if not entries:
            raise typer.BadParameter("No point cloud files found under the input directory")

        assign_splits(entries, (train_ratio, val_ratio, test_ratio), seed=seed)

        category_mapping = load_category_map(category_map) if category_map else None
        if category_mapping:
            for entry in entries:
                entry.category = category_mapping.get(entry.category, entry.category)

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


__all__ = ["register"]
