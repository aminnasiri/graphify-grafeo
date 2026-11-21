"""Grafeo-backed storage. JSON is supported only at the interchange boundary."""
from pathlib import Path

from .base import EdgeRecord, GraphSnapshot, GraphStore, NodeRecord
from .grafeo import GrafeoStore, StoreBusyError

__all__ = [
    "EdgeRecord", "GraphSnapshot", "GraphStore", "NodeRecord", "GrafeoStore",
    "StoreBusyError", "open_store",
]


def open_store(path: str | Path | None = None, *, root: str | Path = ".",
               timeout: float = 5.0, read_only: bool = False) -> GrafeoStore:
    """Open the sole graph backend, honoring the configured output directory.

    An explicit relative database path is relative to the project root.
    ``GRAPHIFY_OUT`` may itself be absolute, as for the existing output helpers.
    """
    from graphify.paths import database_path, GRAPHIFY_OUT

    target = Path(path) if path is not None else database_path(Path(root) / GRAPHIFY_OUT, root=root)
    if path is not None and not target.is_absolute():
        target = Path(root) / target
    return GrafeoStore(target, timeout=timeout, read_only=read_only)
