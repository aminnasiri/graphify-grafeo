"""Fixture I/O for tests ported from graph.json to the Grafeo default.

Assertions inspect node-link data through GraphStore. Physical-format tests
continue to use Path.read_bytes(), and explicit JSON interchange stays textual.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from graphify.store.io import (STATE_NAMES, connection, is_database, read_json as _read_json,
                              state_database, state_exists, write_graph, write_state, remove_state)


def read_json(path):
    data = _read_json(path)
    if is_database(path) and isinstance(data, dict):
        data.get("graph", {}).pop("_store_revision", None)
    return data


def read_text(path, *args, **kwargs):
    if Path(path).suffix == ".grafeo" and Path(path).is_file():
        with Path(path).open("rb") as stream:
            if stream.read(4) != b"GRAF":
                return Path(path).read_text(*args, **kwargs)
    if is_database(path) or state_database(path) is not None:
        return json.dumps(read_json(path), ensure_ascii=False, separators=(",", ":")) + "\n"
    return Path(path).read_text(*args, **kwargs)


def write_text(path, text, *args, **kwargs):
    path = Path(path)
    if path.suffix == ".grafeo":
        try:
            data = json.loads(text)
        except (TypeError, ValueError):
            return path.write_text(text, *args, **kwargs)  # Intentional corruption fixture.
        if isinstance(data, dict):
            data.setdefault("nodes", [])
            if "edges" not in data:
                data.setdefault("links", [])
            data.get("graph", {}).pop("_store_revision", None)
            try:
                write_graph(path, data)
            except (ValueError, TypeError, AttributeError):
                return path.write_text(text, *args, **kwargs)
            # Some fixtures intentionally arrange the old sidecars before the graph.
            for name in STATE_NAMES:
                sidecar = path.parent / name
                if sidecar.is_file():
                    value = json.loads(sidecar.read_text(encoding="utf-8"))
                    if write_state(sidecar, value):
                        sidecar.unlink()
            with connection(path, write=True) as store:
                store.delete_metadata(["graph_content_mtime"])
            return len(text)
    if state_database(path) is not None:
        try:
            value = json.loads(text)
        except ValueError:
            value = text  # Malformed-state fixture, represented as an invalid value.
        if path.name == "manifest.json" and not isinstance(value, dict):
            db = state_database(path)
            assert db is not None
            with connection(db, write=True) as store:
                keys = [key for key in store.snapshot().metadata if key.startswith("file_hash:")]
                for key in keys:
                    store.set_metadata(key, value)
        else:
            write_state(path, value)
        return len(text)
    return path.write_text(text, *args, **kwargs)


def is_file(path):
    return state_exists(path) if state_database(path) is not None else Path(path).is_file()


def read_bytes(path):
    if state_database(path) is not None:
        return read_text(path).encode()
    return Path(path).read_bytes()


def unlink(path, *, missing_ok=False):
    if state_database(path) is not None:
        if not missing_ok and not state_exists(path):
            raise FileNotFoundError(str(path))
        remove_state(path)
    else:
        Path(path).unlink(missing_ok=missing_ok)


def canonical_expected(data):
    if not isinstance(data, dict) or "nodes" not in data:
        return data
    data = deepcopy(data)
    data.setdefault("directed", False)
    from graphify.store.base import infer_multigraph
    data.setdefault("multigraph", infer_multigraph(data))
    data.setdefault("graph", {}).setdefault("schema_version", 1)
    if "links" not in data and "edges" in data:
        data["links"] = data.pop("edges")
    return data


def utime(path, times):
    import os
    return os.utime(state_database(path) or path, times)
