"""End-to-end contracts for the Grafeo-only command pipeline."""
from __future__ import annotations

from collections import Counter
import json
import os
import re
from pathlib import Path
import subprocess
import sys

import pytest

from graphify.store import GrafeoStore
from graphify.store.commands import migrate
from graphify.store.io import read_json
from graphify.store.pipeline import sync_graph


@pytest.fixture(autouse=True)
def isolated_store_env(monkeypatch):
    monkeypatch.delenv("GRAPHIFY_DATABASE", raising=False)
    monkeypatch.delenv("GRAPHIFY_MAX_GRAPH_BYTES", raising=False)
    monkeypatch.setenv("UV_OFFLINE", "1")


def cli(root, *args):
    return subprocess.run([sys.executable, "-m", "graphify", *map(str, args)],
                          cwd=root, capture_output=True, text=True, timeout=90)


def require(result):
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def create_project(root):
    (root / "a.py").write_text("from b import beta\n\ndef alpha():\n    return beta()\n")
    (root / "b.py").write_text("def beta():\n    return 1\n")


def comparable(data):
    def rows(values):
        return Counter(json.dumps(v, sort_keys=True) for v in values)
    return rows(data["nodes"]), rows(data.get("links", data.get("edges", [])))


def test_extract_update_and_exports_have_no_graph_or_state_json(tmp_path):
    create_project(tmp_path)
    require(cli(tmp_path, "extract", ".", "--code-only", "--no-cluster"))
    out = tmp_path / "graphify-out"
    db = out / "graph.grafeo"
    assert db.read_bytes()[:4] == b"GRAF"
    assert not (out / "graph.json").exists()
    assert not (out / "manifest.json").exists()
    with GrafeoStore(db, read_only=True) as store:
        before = store.snapshot()
        b_nodes = {n.id: n for n in before.nodes if n.attributes.get("source_file") == "b.py"}
        native_before = {row["key"]: row["native"] for row in store._execute(
            "MATCH (n:_GraphifyNode) RETURN n.id AS key, id(n) AS native")}
        old_hash = store.get_metadata("file_hash:a.py")["ast_hash"]
    (tmp_path / "a.py").write_text("from b import beta\n\ndef alpha():\n    return beta() + 1\n")
    require(cli(tmp_path, "update", ".", "--no-cluster"))
    with GrafeoStore(db, read_only=True) as store:
        after = store.snapshot()
        assert {n.id: n for n in after.nodes if n.id in b_nodes} == b_nodes
        native_after = {row["key"]: row["native"] for row in store._execute(
            "MATCH (n:_GraphifyNode) RETURN n.id AS key, id(n) AS native")}
        assert native_after == native_before
        assert store.get_metadata("file_hash:a.py")["ast_hash"] != old_hash
    require(cli(tmp_path, "cluster-only", ".", "--no-label", "--no-viz"))
    for format_ in ("html", "wiki", "callflow-html", "neo4j", "falkordb", "graphml", "tree"):
        args = ("tree",) if format_ == "tree" else ("export", format_)
        require(cli(tmp_path, *args))
    for name in ("graph.json", "manifest.json", ".graphify_analysis.json", ".graphify_labels.json"):
        assert not (out / name).exists(), name
    exported = tmp_path / "interchange.json"
    require(cli(tmp_path, "export", "json", "--out", exported))
    data = json.loads(exported.read_text())
    assert data["graph"]["schema_version"] == 1
    assert "_store_revision" not in data["graph"]


@pytest.mark.parametrize("source", sorted((Path(__file__).parents[1] / "worked").glob("*/graph.json")),
                         ids=lambda p: p.parent.name)
def test_worked_migration_and_retrieval_parity(tmp_path, source):
    original = json.loads(source.read_text())
    target = tmp_path / "graph.grafeo"
    counts = migrate(source, target)
    with GrafeoStore(target, read_only=True) as store:
        actual = store.snapshot().to_node_link()
    assert comparable(actual) == comparable(original)
    assert counts[:2] == (len(original["nodes"]), len(original.get("links", original.get("edges", []))))
    label = original["nodes"][0].get("label") or original["nodes"][0]["id"]
    for command in (("query", label), ("explain", label), ("path", label, label)):
        legacy = cli(tmp_path, *command, "--graph", source.resolve())
        native = cli(tmp_path, *command, "--graph", target)
        assert native.returncode == legacy.returncode
        normalize = lambda text: re.sub(r"(?m)^Graph: .*? \(", "Graph: GRAPH (", text)
        assert normalize(native.stdout) == normalize(legacy.stdout)


def test_configured_database_and_explicit_cli_override(tmp_path):
    create_project(tmp_path)
    (tmp_path / ".graphifyrc").write_text("database_path=state/knowledge.grafeo\n")
    require(cli(tmp_path, "extract", ".", "--code-only", "--no-cluster"))
    configured = tmp_path / "state" / "knowledge.grafeo"
    assert configured.exists()
    assert not (tmp_path / "graphify-out" / "graph.grafeo").exists()
    assert "alpha" in require(cli(tmp_path, "query", "alpha"))
    override = tmp_path / "other.grafeo"
    require(cli(tmp_path, "extract", ".", "--code-only", "--no-cluster", "--database", override))
    assert override.exists()
    assert "alpha" in require(cli(tmp_path, "query", "alpha", "--database", override))


def test_import_cli_and_migration_refuse_accidental_overwrite(tmp_path):
    out = tmp_path / "graphify-out"
    out.mkdir()
    source = out / "graph.json"
    source.write_text(json.dumps({"nodes": [{"id": "a", "label": "Alpha", "community": 7}], "links": []}))
    (out / "manifest.json").write_text(json.dumps({"a.py": {"ast_hash": "abc"}}))
    require(cli(tmp_path, "migrate", "--to", "grafeo"))
    assert read_json(out / "manifest.json") == {"a.py": {"ast_hash": "abc"}}
    assert cli(tmp_path, "migrate", "--to", "grafeo").returncode == 1
    assert cli(tmp_path, "import", "json", source).returncode == 1
    require(cli(tmp_path, "import", "json", source, "--replace"))


def test_hook_watch_and_mcp_readers_share_database_without_persistent_owner(tmp_path):
    create_project(tmp_path)
    require(cli(tmp_path, "extract", ".", "--code-only", "--no-cluster"))
    (tmp_path / "a.py").write_text("from b import beta\ndef alpha():\n    return beta() + 2\n")
    db = tmp_path / "graphify-out" / "graph.grafeo"
    watch_script = """
from pathlib import Path
import sys
from graphify.watch import _rebuild_code
assert _rebuild_code(Path(sys.argv[1]), changed_paths=[Path(sys.argv[1])/'a.py'],
                     no_cluster=True, block_on_lock=True)
"""
    mcp_script = """
import sys
from graphify.serve import _GraphContextCache, _query_graph_text
cache = _GraphContextCache(2)
for _ in range(8):
    graph, _ = cache.load(sys.argv[1])
    assert 'alpha' in _query_graph_text(graph, 'alpha')
"""
    processes = [
        subprocess.Popen([sys.executable, "-m", "graphify", "update", str(tmp_path), "--no-cluster"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True),
        subprocess.Popen([sys.executable, "-c", watch_script, str(tmp_path)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True),
        subprocess.Popen([sys.executable, "-c", mcp_script, str(db)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True),
    ]
    for process in processes:
        stdout, stderr = process.communicate(timeout=90)
        assert process.returncode == 0, stdout + stderr
    with GrafeoStore(db, read_only=True) as store:
        assert any(n.attributes.get("label") == "alpha()" for n in store.snapshot().nodes)


def test_revision_cache_reloads_after_metadata_only_commit(tmp_path):
    from graphify.serve import _GraphContextCache

    db = tmp_path / "graph.grafeo"
    with GrafeoStore(db) as store:
        sync_graph(store, {"nodes": [{"id": "a", "label": "Alpha"}], "links": []})
    cache = _GraphContextCache(2)
    first, _ = cache.load(str(db))
    with GrafeoStore(db) as store:
        store.set_metadata("state:.graphify_labels.json", {"0": "New label"})
    second, _ = cache.load(str(db))
    assert second is not first


def test_git_driver_merges_native_extensionless_blobs_with_independent_revisions(tmp_path):
    from graphify.store.io import connection, write_graph
    db = tmp_path / "original.grafeo"
    write_graph(db, {"nodes": [{"id": "a"}, {"id": "b"}], "links": [
        {"source": "b", "target": "a", "relation": "calls", "confidence": "EXTRACTED"}]})
    import shutil
    blobs = [tmp_path / name for name in (".merge_base", ".merge_current", ".merge_other")]
    for blob in blobs:
        shutil.copy2(db, blob)
    with connection(blobs[1], write=True) as store:
        store.set_metadata("local-only", True)
        store.set_metadata("local-only", False)
    with connection(blobs[2], write=True) as store:
        from graphify.store import NodeRecord
        store.upsert_nodes([NodeRecord("c", {"label": "Other branch"})])
    require(cli(tmp_path, "merge-driver", *blobs))
    with connection(blobs[1]) as store:
        data = store.snapshot().to_node_link()
    assert {node["id"] for node in data["nodes"]} == {"a", "b", "c"}
    assert [(e["source"], e["target"]) for e in data["links"]] == [("b", "a")]


def test_environment_cannot_select_a_json_persistence_backend(tmp_path, monkeypatch):
    from graphify.paths import database_path
    monkeypatch.setenv("GRAPHIFY_DATABASE", "accidental.json")
    with pytest.raises(ValueError, match=".grafeo database"):
        database_path(tmp_path / "graphify-out", root=tmp_path)
    assert not (tmp_path / "accidental.json").exists()


def test_native_backups_keep_distinct_graph_revisions(tmp_path):
    from graphify.export import backup_if_protected
    from graphify.store.io import connection, write_graph
    from graphify.store import NodeRecord
    db = tmp_path / "graph.grafeo"
    write_graph(db, {"nodes": [{"id": "a"}], "links": []})
    (tmp_path / ".graphify_semantic_marker").touch()
    first = backup_if_protected(tmp_path)
    assert first is not None
    with connection(db, write=True) as store:
        store.upsert_nodes([NodeRecord("b", {})])
    second = backup_if_protected(tmp_path)
    assert second is not None and first != second
    with connection(first / "graph.grafeo") as store:
        assert [node.id for node in store.snapshot().nodes] == ["a"]
    with connection(second / "graph.grafeo") as store:
        assert {node.id for node in store.snapshot().nodes} == {"a", "b"}
