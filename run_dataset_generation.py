"""Example script showing how to run dataset generation tasks programmatically."""
from __future__ import annotations

from pcdset.datasets import auto_shapenet_main, pcn_main, shapenet_main


def main() -> None:
    """Configure and execute dataset generation tasks."""
    # Update the parameters inside the individual main functions in
    # ``pcdset/datasets`` to suit your dataset before running them.

    # Example 1: Convert an existing manifest into the PCN format.
    # pcn_main()

    # Example 2: Convert ShapeNet style dataset using an explicit manifest.
    # shapenet_main()

    # Example 3: Automatically discover point clouds and convert to ShapeNet format.
    auto_shapenet_main()


if __name__ == "__main__":
    main()
