"""Explicit JSON interchange and verified legacy migration commands."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import tempfile

from . import GrafeoStore
from .io import STATE_NAMES, connection
from .pipeline import sync_graph


def _counts(data: dict) -> tuple[int, int, int]:
    return (len(data.get("nodes", [])), len(data.get("links", data.get("edges", []))),
            len({n.get("community") for n in data.get("nodes", []) if n.get("community") is not None}))


def _canonical(data: dict) -> tuple:
    def rows(items):
        return Counter(json.dumps(x, sort_keys=True, ensure_ascii=True) for x in items)
    attrs = {k: v for k, v in data.items() if k not in {"nodes", "links", "edges"}}
    attrs.setdefault("graph", {}).setdefault("schema_version", 1)
    attrs.setdefault("directed", False)
    from .base import infer_multigraph
    attrs.setdefault("multigraph", infer_multigraph(data))
    return (rows(data.get("nodes", [])), rows(data.get("links", data.get("edges", []))), attrs)


def import_file(source: Path, target: Path, *, replace: bool = False) -> tuple[int, int, int]:
    from graphify.paths import read_graph_json
    from graphify.security import check_graph_file_size_cap

    check_graph_file_size_cap(source)
    original = read_graph_json(source)
    with connection(target, write=True) as store:
        with store.transaction():
            store.import_node_link(original, replace=replace)
            from graphify.paths import graph_root_for
            metadata = {"scan_root": str(graph_root_for(source).resolve())}
            manifest = None
            for name in STATE_NAMES:
                sidecar = source.parent / name
                if sidecar.is_file():
                    value = json.loads(sidecar.read_text(encoding="utf-8-sig"))
                    if name == "manifest.json":
                        manifest = value
                    else:
                        metadata["state:" + name] = value
            sync_graph(store, store.snapshot().to_node_link(), metadata=metadata, manifest=manifest)
            actual = store.snapshot().to_node_link()
            if _counts(original) != _counts(actual) or _canonical(original) != _canonical(actual):
                raise ValueError("migration verification failed: graph records or metadata differ")
    return _counts(original)


def migrate(source: Path, target: Path) -> tuple[int, int, int]:
    """Activate a fully verified, closed database; never replace an existing one."""
    if target.exists():
        raise FileExistsError(f"database already exists: {target}; use import json --replace explicitly")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=".graphify-migrate-", suffix=".grafeo", dir=target.parent)
    os.close(fd)
    temporary = Path(raw)
    temporary.unlink()
    try:
        counts = import_file(source, temporary)
        # Grafeo has closed/checkpointed and removed this database's WAL.
        os.link(temporary, target)  # Atomic no-clobber activation.
        return counts
    finally:
        temporary.unlink(missing_ok=True)


def dispatch(command: str, argv: list[str]) -> None:
    from graphify.paths import out_path, database_path

    parser = argparse.ArgumentParser(prog="graphify " + command)
    if command == "migrate":
        parser.add_argument("root", nargs="?", default=".")
        parser.add_argument("--to", choices=["grafeo"], default="grafeo")
        parser.add_argument("--graph", type=Path, help="legacy JSON source")
        parser.add_argument("--database", type=Path)
        args = parser.parse_args(argv)
        out = Path(args.root) / out_path()
        source = args.graph or out / "graph.json"
        target = args.database or database_path(out, root=Path(args.root).resolve())
        counts = migrate(source, target)
        print(f"Migrated to {target}: {counts[0]} nodes, {counts[1]} edges, {counts[2]} communities verified")
    elif command == "import":
        parser.add_argument("format", choices=["json"])
        parser.add_argument("source", type=Path)
        parser.add_argument("--database", "--graph", type=Path, default=database_path(out_path()))
        parser.add_argument("--replace", action="store_true")
        args = parser.parse_args(argv)
        counts = import_file(args.source, args.database, replace=args.replace)
        print(f"Imported {counts[0]} nodes, {counts[1]} edges, {counts[2]} communities into {args.database}")
    elif command == "export json":
        parser.add_argument("--graph", type=Path, default=database_path(out_path()))
        parser.add_argument("--out", "--output", type=Path)
        args = parser.parse_args(argv)
        output = args.out or args.graph.parent / "graph.json"
        if output.resolve() == args.graph.resolve():
            parser.error("JSON output must differ from the database path")
        with connection(args.graph) as store:
            store.export_json(output)
        print(f"Exported JSON to {output}")
