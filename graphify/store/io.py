"""Graph/state boundaries shared by CLI, MCP, watchers and assistant runbooks.

No database handle is cached: detached snapshots let hooks, watch and MCP take
turns owning the embedded database. JSON is read/written only when explicitly
named as an interchange path, or for extraction caches and non-graph artifacts.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator
import json

from . import GrafeoStore
from .pipeline import sync_graph

_ACTIVE: ContextVar[dict[str, GrafeoStore]] = ContextVar("graphify_active_stores", default={})
STATE_NAMES = frozenset({
    "manifest.json", ".graphify_analysis.json", ".graphify_labels.json",
    ".graphify_labels.json.sig", ".graphify_labels.pending.json", "global-manifest.json", ".graphify_learning.json",
})


def is_database(path: str | Path) -> bool:
    path = Path(path)
    if path.suffix.lower() == ".grafeo":
        return True
    if path.suffix.lower() == ".json":
        return False
    # Git's merge driver receives extensionless temporary blob paths.
    try:
        with path.open("rb") as stream:
            return stream.read(4) == b"GRAF"
    except OSError:
        return False


def state_database(path: str | Path) -> Path | None:
    path = Path(path)
    if path.name not in STATE_NAMES:
        return None
    from graphify.paths import database_path
    db = path.parent / "global-graph.grafeo" if path.name == "global-manifest.json" else database_path(path.parent)
    if db.exists():
        return db
    if path.parent == Path("."):
        from graphify.paths import out_path
        fallback = database_path(out_path())
        if fallback.exists():
            return fallback
    return None


@contextmanager
def connection(path: str | Path, *, write: bool = False) -> Iterator[GrafeoStore]:
    import grafeo
    path = Path(path).resolve()
    active = _ACTIVE.get().get(str(path))
    if active is not None:
        yield active
        return
    try:
        with GrafeoStore(path, timeout=30, read_only=not write) as store:
            yield store
    except grafeo.GrafeoError as exc:
        raise ValueError(f"Cannot access graph database {path}: {exc}") from exc


@contextmanager
def transaction(path: str | Path) -> Iterator[GrafeoStore]:
    path = Path(path).resolve()
    with connection(path, write=True) as store:
        token = _ACTIVE.set({**_ACTIVE.get(), str(path): store})
        try:
            with store.transaction():
                yield store
        finally:
            _ACTIVE.reset(token)


def read_graph(path: str | Path) -> dict:
    path = Path(path)
    if is_database(path):
        with connection(path) as store:
            snapshot = store.snapshot()
        data = snapshot.to_node_link()
        data.setdefault("graph", {})["_store_revision"] = snapshot.revision
        return data
    from graphify.security import check_graph_file_size_cap
    check_graph_file_size_cap(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_graph(path: str | Path, data: dict, *, manifest: dict | None = None,
                metadata: dict | None = None, expected_revision: int | None = None) -> None:
    path = Path(path)
    if is_database(path):
        attrs = data.get("graph") or {}
        revision = attrs.get("_store_revision", expected_revision)
        with connection(path, write=True) as store:
            with store.transaction():
                sync_graph(store, data, manifest=manifest, metadata=metadata, expected_revision=revision)
                if store.get_metadata("graph_content_mtime") is not None:
                    store.delete_metadata(["graph_content_mtime"])
        return
    from graphify.paths import write_json_atomic
    write_json_atomic(path, data, compact=True)


def read_json(path: str | Path) -> Any:
    """Read JSON interchange or database-backed graph metadata."""
    path = Path(path)
    if is_database(path):
        return read_graph(path)
    db = state_database(path)
    if db is not None:
        with connection(db) as store:
            if path.name == "manifest.json":
                if store.get_metadata("manifest_present") is False:
                    raise FileNotFoundError(f"manifest metadata not found in {db}")
                return {k.removeprefix("file_hash:"): v for k, v in store.snapshot().metadata.items()
                        if k.startswith("file_hash:")}
            marker = object()
            value = store.get_metadata("state:" + path.name, marker)
            if value is marker:
                raise FileNotFoundError(f"metadata {path.name} not found in {db}")
            return value
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_state(path: str | Path, value: Any) -> bool:
    path = Path(path)
    db = state_database(path)
    if db is None:
        return False
    content_mtime = db.stat().st_mtime
    with connection(db, write=True) as store:
        if store.get_metadata("graph_content_mtime") is None:
            store.set_metadata("graph_content_mtime", content_mtime)
        if path.name == "manifest.json":
            with store.transaction():
                store.set_metadata("manifest_present", True)
                old = store.snapshot().metadata
                wanted = {"file_hash:" + k: v for k, v in value.items()}
                store.delete_metadata([k for k in old if k.startswith("file_hash:") and k not in wanted])
                for key, row in wanted.items():
                    if old.get(key) != row:
                        store.set_metadata(key, row)
        else:
            key = "state:" + path.name
            if store.get_metadata(key, object()) != value:
                store.set_metadata(key, value)
    return True


def state_exists(path: str | Path) -> bool:
    path = Path(path)
    db = state_database(path)
    if db is None:
        return path.exists()
    if path.name == "manifest.json":
        with connection(db) as store:
            return store.get_metadata("manifest_present", True)
    try:
        read_json(path)
        return True
    except FileNotFoundError:
        return False


def state_is_file(path: str | Path) -> bool:
    return state_exists(path) if state_database(path) is not None else Path(path).is_file()


def remove_state(path: str | Path) -> None:
    path = Path(path)
    db = state_database(path)
    if db is None:
        path.unlink(missing_ok=True)
    else:
        with connection(db, write=True) as store:
            keys = [k for k in store.snapshot().metadata if k.startswith("file_hash:")] if path.name == "manifest.json" else ["state:" + path.name]
            with store.transaction():
                store.delete_metadata(keys)
                if path.name == "manifest.json":
                    store.set_metadata("manifest_present", False)


def write_state_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
    if not write_state(path, json.loads(text)):
        from graphify.paths import write_text_atomic
        write_text_atomic(path, text)


def revision(path: str | Path) -> int | None:
    if not is_database(path) or not Path(path).exists():
        return None
    with connection(path) as store:
        return store.get_revision()


def graph_content_mtime(path: str | Path) -> float:
    """Content freshness ignores later metadata-only checkpoints."""
    path = Path(path)
    try:
        with connection(path) as store:
            return store.get_metadata("graph_content_mtime", path.stat().st_mtime)
    except (ValueError, OSError):
        return path.stat().st_mtime
