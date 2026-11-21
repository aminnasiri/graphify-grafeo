"""Compare JSON interchange persistence with Grafeo on one canonical corpus.

Run: uv run python scripts/benchmark_stores.py graphify-out/graph.json --query build_merge
Extraction/clustering are identical inputs and are excluded from persistence timings.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import statistics
import tempfile
import time

from graphify.paths import write_json_atomic
from graphify.serve import _read_graph, _query_graph_text
from graphify.store import GrafeoStore
from graphify.store.pipeline import sync_graph


def timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def run(source: Path, query: str) -> dict:
    data = json.loads(source.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="graphify-store-benchmark-") as temp:
        root = Path(temp)
        json_path, db_path = root / "graph.json", root / "graph.grafeo"
        _, json_build = timed(lambda: write_json_atomic(json_path, data, compact=True))
        def build_db():
            with GrafeoStore(db_path) as store:
                store.import_node_link(data)
        _, db_build = timed(build_db)
        files = {n.get("source_file") for n in data["nodes"] if n.get("source_file")}
        source_file = next(n["source_file"] for n in data["nodes"] if n.get("source_file"))
        json_updates, db_updates = [], []
        for index in range(3):
            changed = deepcopy(data)
            for node in changed["nodes"]:
                if node.get("source_file") == source_file:
                    node["benchmark_revision"] = index
            _, elapsed = timed(lambda: write_json_atomic(json_path, changed, compact=True))
            json_updates.append(elapsed)
            def update_db():
                with GrafeoStore(db_path) as store:
                    return sync_graph(store, changed)
            delta, elapsed = timed(update_db)
            db_updates.append(elapsed)
            assert delta.edges_upserted == 0
        cold, warm = {}, {}
        for name, path in (("json", json_path), ("grafeo", db_path)):
            times = []
            for _ in range(3):
                gc.collect()
                _, elapsed = timed(lambda: _query_graph_text(_read_graph(str(path)), query))
                times.append(elapsed)
            cold[name] = statistics.median(times)
            graph = _read_graph(str(path))
            _query_graph_text(graph, query)
            times = [timed(lambda: _query_graph_text(graph, query))[1] for _ in range(7)]
            warm[name] = statistics.median(times)
        return {"nodes": len(data["nodes"]), "edges": len(data.get("links", data.get("edges", []))),
                "source_files": len(files), "updated_file": source_file,
                "updated_nodes": sum(n.get("source_file") == source_file for n in data["nodes"]),
                "query": query,
                "json": {"build_seconds": json_build, "update_seconds": statistics.median(json_updates),
                         "cold_query_seconds": cold["json"], "warm_query_seconds": warm["json"],
                         "bytes": json_path.stat().st_size},
                "grafeo": {"build_seconds": db_build, "update_seconds": statistics.median(db_updates),
                           "cold_query_seconds": cold["grafeo"], "warm_query_seconds": warm["grafeo"],
                           "bytes": db_path.stat().st_size}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--query", default="build_merge")
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.query), indent=2))
