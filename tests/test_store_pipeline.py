from copy import deepcopy

import pytest

from graphify.store import GrafeoStore
from graphify.store.pipeline import ConcurrentUpdateError, sync_graph, update_communities, update_extraction


def corpus():
    return {"directed": True, "multigraph": True, "graph": {"schema_version": 1},
            "nodes": [
                {"id": "a", "label": "a", "file_type": "code", "type": "function",
                 "source_file": "a.py", "source_location": "L1", "_origin": "ast"},
                {"id": "b", "label": "b", "file_type": "code", "type": "function",
                 "source_file": "b.py", "source_location": "L1", "_origin": "ast"},
                {"id": "doc", "label": "Evidence", "file_type": "document",
                 "source_file": "a.py", "rationale": "Independently recorded evidence"},
            ], "links": [
                {"source": "a", "target": "b", "relation": "calls", "source_file": "a.py",
                 "confidence": "EXTRACTED", "_origin": "ast"},
                {"source": "b", "target": "a", "relation": "uses", "source_file": "b.py",
                 "confidence": "INFERRED", "_origin": "ast"},
            ], "hyperedges": [{"id": "h", "nodes": ["a", "b", "doc"], "source_file": "notes.md"}]}


def native_ids(store):
    nodes = {r["key"]: r["native"] for r in store._execute(
        "MATCH (n:_GraphifyNode) RETURN n.id AS key, id(n) AS native")}
    edges = {r["key"]: r["native"] for r in store._execute(
        "MATCH ()-[e]->() RETURN e._graphify_edge_id AS key, id(e) AS native")}
    return nodes, edges


def test_delta_does_not_clear_or_rewrite_unchanged_records(tmp_path, monkeypatch):
    with GrafeoStore(tmp_path / "graph.grafeo") as store:
        sync_graph(store, corpus(), manifest={"a.py": {"ast_hash": "old"}, "b.py": {"ast_hash": "same"}})
        before = store.snapshot()
        ids = native_ids(store)
        data = before.to_node_link()
        next(n for n in data["nodes"] if n["id"] == "a")["source_location"] = "L2"
        next(e for e in data["links"] if e["source"] == "a")["source_location"] = "L2"
        monkeypatch.setattr(store, "clear", lambda: pytest.fail("full graph replacement"))
        result = sync_graph(store, data, expected_revision=before.revision,
                            manifest={"a.py": {"ast_hash": "new"}, "b.py": {"ast_hash": "same"}})
        assert (result.nodes_upserted, result.edges_upserted, result.nodes_deleted, result.edges_deleted) == (1, 1, 0, 0)
        assert native_ids(store) == ids
        assert store.get_metadata("file_hash:b.py") == {"ast_hash": "same"}
        assert result.revision == before.revision + 1
        again = sync_graph(store, data)
        assert again.revision == result.revision


@pytest.mark.parametrize("cluster", [True, False])
def test_incremental_ast_keeps_semantic_and_incoming_unchanged_evidence(tmp_path, cluster):
    with GrafeoStore(tmp_path / "graph.grafeo") as store:
        initial = corpus()
        sync_graph(store, initial)
        fresh = deepcopy(initial["nodes"][0])
        fresh["source_location"] = "L2"
        update_extraction(store, {"nodes": [fresh], "edges": [], "extracted_sources": ["a.py"]},
                          root=tmp_path, cluster=cluster)
        snapshot = store.snapshot()
        assert {n.id for n in snapshot.nodes} == {"a", "b", "doc"}
        node = store.get_node("a")
        assert node is not None and node.attributes["source_location"] == "L2"
        assert any(e.source == "b" and e.target == "a" for e in snapshot.edges)
        assert snapshot.to_node_link()["hyperedges"][0]["id"] == "h"


def test_failed_extraction_retains_records_and_clears_hash_atomically(tmp_path):
    with GrafeoStore(tmp_path / "graph.grafeo") as store:
        sync_graph(store, corpus())
        ids = native_ids(store)
        update_extraction(store, {"nodes": [], "edges": [], "extracted_sources": ["a.py"],
                                  "failed_sources": ["a.py"]}, root=tmp_path, cluster=False,
                          manifest={"a.py": {"ast_hash": "bad", "semantic_hash": "bad"}})
        assert native_ids(store) == ids
        assert store.get_metadata("file_hash:a.py") == {"ast_hash": "", "semantic_hash": ""}


def test_cluster_updates_keep_native_ids_and_write_labels_with_signatures(tmp_path):
    with GrafeoStore(tmp_path / "graph.grafeo") as store:
        sync_graph(store, corpus())
        ids = native_ids(store)
        result = update_communities(store, {7: ["a", "b"], 9: ["doc"]}, {7: "Code", 9: "Evidence"})
        assert result.edges_upserted == 0
        assert native_ids(store) == ids
        assert store.list_communities() == {7: ["a", "b"], 9: ["doc"]}
        assert store.get_metadata("community_labels")["7"] == "Code"
        assert set(store.get_metadata("label_signature")) == {"7", "9"}


def test_concurrent_baseline_rejected_and_record_metadata_delta_rolls_back(tmp_path, monkeypatch):
    with GrafeoStore(tmp_path / "graph.grafeo") as store:
        sync_graph(store, corpus())
        old = store.snapshot()
        store.set_metadata("another_writer", True)
        with pytest.raises(ConcurrentUpdateError):
            sync_graph(store, old.to_node_link(), expected_revision=old.revision)
        before = store.snapshot()
        data = before.to_node_link()
        data["nodes"][0]["changed"] = True
        def fail(*args):
            raise RuntimeError("metadata write failed")
        monkeypatch.setattr(store, "set_metadata", fail)
        with pytest.raises(RuntimeError):
            sync_graph(store, data, manifest={"a.py": {"ast_hash": "new"}})
        assert store.snapshot() == before


def test_parallel_evidence_reordering_is_a_noop_and_edit_keeps_ids(tmp_path):
    with GrafeoStore(tmp_path / "graph.grafeo") as store:
        data = corpus()
        data["links"].append({**data["links"][0], "source_location": "L7"})
        sync_graph(store, data)
        before = store.snapshot()
        data["links"].reverse()
        result = sync_graph(store, data)
        assert result.revision == before.revision
        assert store.snapshot() == before
