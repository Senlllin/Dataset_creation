"""Compatibility wrapper for :mod:`pcdset.manifest`.

Historically manifest helpers lived in :mod:`pcdset.utils.manifest`.  The
functions were moved into the dedicated :mod:`pcdset.manifest` package to make
module responsibilities clearer.  Importing from this module continues to work
but will eventually be deprecated.
"""
from __future__ import annotations

from warnings import warn

from ..manifest import (
    Entry,
    assign_splits,
    build_example_manifest,
    build_example_manifest_shapenet,
    build_simple_entries,
    load_entries,
    load_entries_shapenet,
    write_manifest,
)

warn(
    "pcdset.utils.manifest is deprecated; import from pcdset.manifest instead",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "Entry",
    "assign_splits",
    "build_example_manifest",
    "build_example_manifest_shapenet",
    "build_simple_entries",
    "load_entries",
    "load_entries_shapenet",
    "write_manifest",
]
