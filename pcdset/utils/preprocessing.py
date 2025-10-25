"""Utility functions for offline point cloud preprocessing.

This module implements three data preparation steps that are frequently
requested when curating datasets before running the training script:

1. Randomly sample a fixed proportion of the input points.
2. Augment the sampled cloud with a configurable amount of noise points and
   persist the result as a new file.
3. Rename all generated files so that their basenames follow a sequential
   number pattern starting from a desired index (``1001`` by default).

All behaviour is controlled via :class:`PreprocessConfig`.  Adjust the
configuration values in the ``__main__`` block at the bottom of the file and
execute the module directly to perform preprocessing before training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from ..io.reader import read_points
from ..io.writer_ply import write_ply
from ..utils.logging import logger


# File formats supported by :mod:`pcdset` readers.  The ``write_points`` helper
# reuses the same extensions to determine how to persist the processed point
# clouds.
SUPPORTED_EXTENSIONS: Sequence[str] = (".ply", ".pcd", ".npz", ".txt", ".csv")


@dataclass
class PreprocessConfig:
    """Configuration for dataset preprocessing.

    Attributes
    ----------
    input_dir:
        Directory containing the source point cloud files.
    output_dir:
        Directory where processed files will be written.  The directory is
        created automatically if it does not exist.
    sample_ratio:
        Fraction of points to keep during random sampling.  Must be in the
        ``(0, 1]`` range.
    noise_ratio:
        Fraction of additional noise points to add relative to the sampled
        point count.  For example ``0.1`` inserts 10% more points.
    noise_scale:
        Amount of bounding-box expansion when generating noise points.  ``0``
        restricts noise to the original bounding box, whereas higher values
        allow outliers to appear slightly outside the cloud extent.
    rename_start:
        Starting index for renamed files.  Each output file increments the
        counter by one, producing names such as ``1001.ply`` and ``1002.ply``.
    seed:
        Seed for ``numpy``'s default random generator to ensure deterministic
        preprocessing runs when desired.
    save_sampled:
        Persist the sampled point cloud before adding noise.  Disable if only
        the noisy variant is required.
    save_noisy:
        Persist the point cloud after noise augmentation.
    """

    input_dir: Path
    output_dir: Path
    sample_ratio: float = 0.8
    noise_ratio: float = 0.1
    noise_scale: float = 0.05
    rename_start: int = 1001
    seed: int | None = 42
    save_sampled: bool = True
    save_noisy: bool = True

    def validate(self) -> None:
        """Validate configuration values."""

        if not 0 < self.sample_ratio <= 1:
            raise ValueError("sample_ratio must be within (0, 1].")
        if self.noise_ratio < 0:
            raise ValueError("noise_ratio cannot be negative.")
        if self.noise_scale < 0:
            raise ValueError("noise_scale cannot be negative.")
        if self.rename_start < 0:
            raise ValueError("rename_start must be non-negative.")


def list_point_clouds(directory: Path) -> List[Path]:
    """Return point cloud files within ``directory`` sorted by name."""

    if not directory.is_dir():
        raise FileNotFoundError(f"Input directory {directory!s} does not exist or is not a directory.")
    files = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS]
    files.sort()
    if not files:
        logger.warning("No supported point cloud files found in %s", directory)
    return files


def random_sample_points(points: np.ndarray, ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Return a random subset of ``points`` using ``ratio`` proportion."""

    if len(points) == 0:
        raise ValueError("Point cloud is empty; cannot sample.")
    sample_size = max(1, int(round(len(points) * ratio)))
    if sample_size > len(points):
        sample_size = len(points)
    indices = rng.choice(len(points), size=sample_size, replace=False)
    sampled = points[indices]
    logger.debug("Sampled %s/%s points (ratio %.3f)", sample_size, len(points), ratio)
    return sampled


def add_noise_points(points: np.ndarray, noise_ratio: float, noise_scale: float, rng: np.random.Generator) -> np.ndarray:
    """Augment ``points`` with uniformly distributed noise points."""

    if noise_ratio <= 0:
        return points
    noise_count = int(round(len(points) * noise_ratio))
    if noise_count == 0:
        return points

    min_corner = points.min(axis=0)
    max_corner = points.max(axis=0)
    span = np.maximum(max_corner - min_corner, 1e-6)
    lower = min_corner - span * noise_scale
    upper = max_corner + span * noise_scale
    noise_points = rng.uniform(lower, upper, size=(noise_count, 3))
    augmented = np.concatenate([points, noise_points], axis=0)
    logger.debug("Added %s noise points (ratio %.3f, scale %.3f)", noise_count, noise_ratio, noise_scale)
    return augmented


def write_points(path: Path, points: np.ndarray) -> None:
    """Persist ``points`` to ``path`` using the appropriate format."""

    ext = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if ext in {".ply", ".pcd", ".npz"}:
        write_ply(path, points)
    elif ext == ".csv":
        np.savetxt(path, points, fmt="%.6f", delimiter=",")
    elif ext == ".txt":
        np.savetxt(path, points, fmt="%.6f")
    else:
        raise ValueError(f"Unsupported output file type: {ext}")


def process_file(path: Path, config: PreprocessConfig, rng: np.random.Generator, next_index: int) -> int:
    """Process a single point cloud file and return the next available index."""

    logger.info("Processing %s", path.name)
    points, _attrs = read_points(path)
    sampled = random_sample_points(points, config.sample_ratio, rng)

    if config.save_sampled:
        output_path = build_output_path(config.output_dir, next_index, path.suffix)
        write_points(output_path, sampled)
        logger.debug("Wrote sampled point cloud to %s", output_path)
        next_index += 1

    if config.save_noisy:
        noisy = add_noise_points(sampled, config.noise_ratio, config.noise_scale, rng)
        output_path = build_output_path(config.output_dir, next_index, path.suffix)
        write_points(output_path, noisy)
        logger.debug("Wrote noisy point cloud to %s", output_path)
        next_index += 1

    return next_index


def build_output_path(output_dir: Path, index: int, extension: str) -> Path:
    """Generate a sequential file name using ``index`` and ``extension``."""

    filename = f"{index:04d}{extension.lower()}"
    return output_dir / filename


def preprocess_directory(config: PreprocessConfig) -> None:
    """Execute preprocessing according to ``config``."""

    config.validate()
    rng = np.random.default_rng(config.seed)
    files = list_point_clouds(config.input_dir)
    if not files:
        return

    next_index = config.rename_start
    for path in files:
        next_index = process_file(path, config, rng, next_index)

    logger.info(
        "Completed preprocessing: %d input files -> %d output files.",
        len(files),
        next_index - config.rename_start,
    )


def _expand_paths(config: PreprocessConfig) -> None:
    """Expand user paths in the configuration in-place."""

    config.input_dir = config.input_dir.expanduser().resolve()
    config.output_dir = config.output_dir.expanduser().resolve()


if __name__ == "__main__":
    # Adjust these parameters as needed before running the training pipeline.
    CONFIG = PreprocessConfig(
        input_dir=Path("./raw_point_clouds"),
        output_dir=Path("./processed_point_clouds"),
        sample_ratio=0.8,
        noise_ratio=0.1,
        noise_scale=0.05,
        rename_start=1001,
        seed=42,
        save_sampled=True,
        save_noisy=True,
    )

    _expand_paths(CONFIG)
    logger.info("Starting preprocessing with configuration: %s", CONFIG)
    preprocess_directory(CONFIG)
