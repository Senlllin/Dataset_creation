"""Tools for generating PCN style datasets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from ..profiles.pcn import PCNProfile
from ..utils.manifest import load_entries
from ..utils.taxonomy import load_category_map
from ._common import validate_ratios


@dataclass
class PCNConversionConfig:
    """Configuration for converting partial/complete clouds to the PCN format."""

    input: Path
    out: Path
    manifest: Optional[Path] = None
    split_strategy: str = "FILE"
    ratios: Sequence[float] = (0.9, 0.03, 0.07)
    partial_n: int = 2048
    complete_n: int = 16384
    normalize: str = "none"
    center: bool = False
    dedup: bool = False
    fps: bool = False
    voxel: float = 0.0
    to_lmdb: bool = False
    lmdb_max_gb: int = 64
    save_attrs: bool = False
    overwrite: bool = False
    workers: int = 8
    category_map: Optional[Path] = None

    def run(self) -> None:
        """Execute the conversion using :class:`~pcdset.profiles.pcn.PCNProfile`."""

        validate_ratios(self.ratios)

        profile = PCNProfile(
            partial_n=self.partial_n,
            complete_n=self.complete_n,
            normalize=self.normalize,
            center=self.center,
            dedup=self.dedup,
            fps=self.fps,
            voxel=self.voxel,
            to_lmdb=self.to_lmdb,
            lmdb_max_gb=self.lmdb_max_gb,
            save_attrs=self.save_attrs,
            overwrite=self.overwrite,
            workers=self.workers,
            split_strategy=self.split_strategy,
            ratios=self.ratios,
        )

        cat_map = load_category_map(self.category_map) if self.category_map else None
        entries = load_entries(
            self.input,
            self.manifest,
            self.split_strategy,
            ratios=self.ratios,
            category_map=cat_map,
        )

        profile.convert(entries, self.out)


def main() -> None:
    """Example entry point for generating a PCN dataset."""

    config = PCNConversionConfig(
        input=Path("data/raw_pcn"),
        out=Path("data/pcn_dataset"),
        manifest=Path("data/pcn_manifest.csv"),
    )
    config.run()


__all__ = ["PCNConversionConfig", "main"]
