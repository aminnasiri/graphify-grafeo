"""Storage invariants against the real embedded database, without network calls."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from graphify.store import EdgeRecord, GrafeoStore, NodeRecord, StoreBusyError, open_store


@pytest.fixture
def store(tmp_path):
    with GrafeoStore(tmp_path / "graph.grafeo") as value:
        yield value


def _nodes():
    return [
        NodeRecord("a", {"label": "Alpha", "type": "function", "source_file": "a.py",
                         "community": 0, "_origin": "ast"}),
        NodeRecord("b", {"label": "Beta", "type": "class", "source_file": "b.py",
                         "community": 1, "_origin": "ast"}),
        NodeRecord("c", {"label": "Gamma", "source_file": "b.py", "community": 1}),
    ]


def _edges():
    return [
        EdgeRecord("calls", "a", "b", {"relation": "calls", "confidence": "EXTRACTED",
                                          "source_file": "a.py", "_origin": "ast"}),
        EdgeRecord("imports", "a", "b", {"relation": "imports", "confidence": "INFERRED"}),
        EdgeRecord("uses", "b", "c", {"relation": "uses", "confidence": "AMBIGUOUS"}),
    ]


def test_atomic_write_reopen_and_internal_records_excluded(tmp_path):
    path = tmp_path / "graph.grafeo"
    with GrafeoStore(path) as store:
        with store.transaction():
            store.upsert_nodes(_nodes())
            store.upsert_edges(_edges())
            store.set_metadata("label_signature", {"0": "abc"})
            store.set_metadata("file_hashes", {"a.py": {"ast_hash": "hash", "semantic_hash": ""}})
        assert store.snapshot().revision == 1
    assert path.is_file()
    assert not (tmp_path / "graph.json").exists()
    with GrafeoStore(path, read_only=True) as store:
        snapshot = store.snapshot()
        assert len(snapshot.nodes) == 3
        assert len(snapshot.edges) == 3
        assert store.get_metadata("label_signature") == {"0": "abc"}
        assert store.get_metadata("file_hashes")["a.py"]["ast_hash"] == "hash"
        assert snapshot.revision == 1


def test_exception_rolls_back_records_metadata_and_revision(store):
    store.upsert_nodes(_nodes())
    before = store.snapshot()
    with pytest.raises(RuntimeError, match="failure"):
        with store.transaction():
            store.delete_by_source_file("a.py")
            store.set_metadata("file_hashes", {"a.py": "new"})
            store.upsert_nodes([NodeRecord("new", {"label": "New"})])
            raise RuntimeError("failure")
    assert store.snapshot() == before


def test_caught_nested_error_still_aborts_outer_transaction(store):
    with pytest.raises(RuntimeError, match="nested operation"):
        with store.transaction():
            store.upsert_nodes(_nodes())
            try:
                store.upsert_edges([EdgeRecord("bad", "a", "missing", {"relation": "calls"})])
            except ValueError:
                pass
    assert store.snapshot().nodes == []
    assert store.snapshot().revision == 0


def test_upserts_replace_attributes_and_change_kind_and_edge_endpoints(store):
    store.upsert_nodes(_nodes())
    store.upsert_edges(_edges())
    store.upsert_nodes([NodeRecord("a", {"type": "class", "label": "New"})])
    assert store.get_node("a") == NodeRecord("a", {"type": "class", "label": "New"})
    store.upsert_edges([EdgeRecord("calls", "c", "a", {"relation": "inherits"})])
    edges = {e.id: e for e in store.snapshot().edges}
    assert len(edges) == 3
    assert edges["calls"].source == "c"
    assert edges["calls"].attributes == {"relation": "inherits"}
    # Read the actual labels to ensure a kind change removed the old label.
    assert store._db.get_nodes_by_label("function") == []


def test_lossless_null_float_lists_large_integers_and_reserved_keys(store):
    attrs = {
        "null": None, "empty": [], "float_list": [0.123456789012345, 0.98765432109876],
        "nested": {"null": None, "floats": [0.123456789012345]}, "wide": 10**60,
        "_graphify_parts": "user property", "_graphify_part_0": "also user data",
        "unicode": "سلام 世界", "flag": False,
    }
    store.upsert_nodes([NodeRecord("a", attrs), NodeRecord("b", {})])
    store.upsert_edges([EdgeRecord("a-b", "a", "b", {**attrs, "relation": "calls"})])
    store.set_metadata("arbitrary", attrs)
    assert store.get_node("a").attributes == attrs
    assert store.snapshot().edges[0].attributes == {**attrs, "relation": "calls"}
    assert store.get_metadata("arbitrary") == attrs


def test_document_larger_than_property_cap_stays_in_database(tmp_path):
    path = tmp_path / "graph.grafeo"
    text = "文" * (6 * 1024 * 1024)  # 18 MiB of UTF-8; exceeds Grafeo's 16 MiB cap.
    with GrafeoStore(path) as store:
        store.upsert_nodes([NodeRecord("doc", {"text": text})])
    with GrafeoStore(path, read_only=True) as store:
        node = store.get_node("doc")
        assert node is not None
        assert node.attributes["text"] == text


def test_source_tier_purge_preserves_unrelated_records(store):
    store.upsert_nodes([*_nodes(), NodeRecord("semantic", {"source_file": "a.py", "_origin": "semantic"})])
    store.upsert_edges(_edges())
    before = store.snapshot()
    store.delete_by_source_file("a.py", origin="ast")
    after = store.snapshot()
    assert after.nodes == [n for n in before.nodes if n.id != "a"]
    assert after.edges == [e for e in before.edges if e.id == "uses"]
    assert after.revision == before.revision + 1


def test_read_primitives_keep_parallel_edges_direction_and_communities(store):
    store.upsert_nodes(_nodes())
    store.upsert_edges(_edges())
    assert [n.id for n in store.neighbors("b", direction="in")] == ["a"]
    assert [n.id for n in store.neighbors("b", direction="out")] == ["c"]
    assert store.shortest_path("a", "c") == ["a", "b", "c"]
    assert store.list_communities() == {0: ["a"], 1: ["b", "c"]}
    graph = store.query_subgraph(["b"], depth=1)
    assert len(graph) == 3
    assert graph.number_of_edges() == 3
    assert {d["confidence"] for *_, d in graph.edges(data=True)} == {
        "EXTRACTED", "INFERRED", "AMBIGUOUS",
    }
    assert set(store.query_subgraph(["a"], depth=0)) == {"a"}
    exported = store.snapshot().to_node_link()
    assert exported["graph"]["schema_version"] == 1
    assert exported["hyperedges"] == []
    with pytest.raises(ValueError):
        store.query_subgraph(["a"], depth=-1)
    with pytest.raises(ValueError):
        store.neighbors("a", direction="invalid")


def _canonical(data):
    data = json.loads(json.dumps(data))
    data["nodes"] = sorted(data["nodes"], key=lambda x: json.dumps(x, sort_keys=True))
    data["links"] = sorted(data.pop("edges", data.get("links", [])),
                           key=lambda x: json.dumps(x, sort_keys=True))
    data.setdefault("graph", {}).setdefault("schema_version", 1)
    return data


@pytest.mark.parametrize("source", sorted((Path(__file__).parents[1] / "worked").glob("*/graph.json")),
                         ids=lambda p: p.parent.name)
def test_worked_examples_round_trip(store, tmp_path, source):
    original = json.loads(source.read_text(encoding="utf-8"))
    store.import_json(source)
    output = tmp_path / "export.json"
    store.export_json(output)
    assert _canonical(json.loads(output.read_text(encoding="utf-8"))) == _canonical(original)


def test_interchange_preserves_duplicate_edges_hyperedges_and_unknown_metadata(store):
    data = {
        "nodes": [{"id": "a", "community": None}, {"id": "b"}],
        "links": [
            {"source": "a", "target": "b", "relation": "calls", "key": 7},
            {"source": "a", "target": "b", "relation": "calls", "key": 7},
            {"source": "b", "target": "a", "relation": "imports"},
        ],
        "directed": False, "multigraph": True, "built_at_commit": "abc",
        "graph": {"schema_version": 1, "graphify_version": "test", "custom": {"x": None}},
        "hyperedges": [{"id": "group", "nodes": ["a", "b"], "extra": [0.123456789012345]}],
        "custom_top": {"x": None},
    }
    store.import_node_link(data)
    assert _canonical(store.snapshot().to_node_link()) == _canonical(data)
    graph = store.export_networkx(directed=True)
    assert graph.number_of_edges() == 3
    assert graph.graph["_logical_directed"] is False
    assert graph.graph["_built_at_commit"] == "abc"
    assert graph.graph["hyperedges"] == data["hyperedges"]
    assert [d["key"] for _, _, d in graph.edges(data=True) if "key" in d] == [7, 7]


def test_import_refuses_overwrite_and_bad_endpoints_without_changing_store(store):
    store.upsert_nodes(_nodes())
    before = store.snapshot()
    with pytest.raises(ValueError, match="not empty"):
        store.import_node_link({"nodes": [{"id": "replacement"}]})
    assert store.snapshot() == before
    with pytest.raises(ValueError, match="endpoints"):
        store.import_node_link({"nodes": [], "links": [{"source": "absent", "target": "absent"}]},
                               replace=True)
    assert store.snapshot() == before
    store.import_node_link({"nodes": [{"id": "replacement"}], "edges": []}, replace=True)
    assert [n.id for n in store.snapshot().nodes] == ["replacement"]


def test_labels_and_relations_are_inert_identifiers(store):
    kind = "kind` MATCH (n) DETACH DELETE n //"
    relation = "uses`value"
    store.upsert_nodes([NodeRecord("a", {"type": kind}), NodeRecord("b", {})])
    store.upsert_edges([EdgeRecord("e", "a", "b", {"relation": relation})])
    assert len(store.snapshot().nodes) == 2
    assert store.snapshot().edges[0].attributes["relation"] == relation


def test_file_lock_contention_across_processes_and_reopen(tmp_path):
    path = tmp_path / "graph.grafeo"
    with GrafeoStore(path):
        script = """
import sys
from graphify.store import GrafeoStore, StoreBusyError
try:
    with GrafeoStore(sys.argv[1], timeout=0.1):
        pass
except StoreBusyError:
    sys.exit(23)
"""
        result = subprocess.run([sys.executable, "-c", script, str(path)],
                                capture_output=True, text=True, timeout=20)
        assert result.returncode == 23, result.stderr
    with GrafeoStore(path, timeout=0):
        pass


def test_read_only_rejects_writes_and_closed_store_rejects_reads(tmp_path):
    path = tmp_path / "graph.grafeo"
    with GrafeoStore(path) as store:
        store.upsert_nodes(_nodes())
    with GrafeoStore(path, read_only=True) as reader, GrafeoStore(path, read_only=True) as other:
        assert reader.snapshot() == other.snapshot()
        with pytest.raises(Exception):
            reader.upsert_nodes([NodeRecord("bad", {})])
        assert reader.get_node("bad") is None
    with pytest.raises(RuntimeError, match="closed"):
        reader.get_node("a")
    reader.close()  # Idempotent.


def test_factory_honors_output_directory_and_explicit_path(tmp_path, monkeypatch):
    import graphify.paths as paths

    monkeypatch.setattr(paths, "GRAPHIFY_OUT", "custom-output")
    with open_store(root=tmp_path) as store:
        assert store.path == tmp_path / "custom-output" / "graph.grafeo"
    with open_store("other.grafeo", root=tmp_path) as store:
        assert store.path == tmp_path / "other.grafeo"
    with pytest.raises(ValueError, match="import/export"):
        open_store("graph.json", root=tmp_path)


def test_snapshot_does_not_expose_mutable_store_state(store):
    store.upsert_nodes(_nodes())
    snapshot = store.snapshot()
    snapshot.nodes[0].attributes["label"] = "changed"
    assert store.get_node("a").attributes["label"] == "Alpha"


@pytest.mark.parametrize("timeout", [-1, float("inf"), float("nan")])
def test_invalid_lock_deadlines_are_rejected(tmp_path, timeout):
    with pytest.raises(ValueError, match="finite and non-negative"):
        GrafeoStore(tmp_path / "graph.grafeo", timeout=timeout)
