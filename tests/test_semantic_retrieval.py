"""Real-engine optional retrieval, with no model download or remote embedding call."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from graphify.semantic_retrieval import heading_chunks, validated_vectors, embedding_backend
from graphify.store.io import connection, write_graph
from graphify.store.retrieval import semantic_seeds
from graphify.store.pipeline import ConcurrentUpdateError


def corpus(root: Path) -> Path:
    (root / "guide.md").write_text("# Transactions\nAtomic changes use rollback and isolation.\n\n# Locking\nConcurrent writers acquire an exclusive lock.\n")
    db = root / "graph.grafeo"
    write_graph(db, {"nodes": [
        {"id": "guide", "label": "Operations guide", "source_file": "guide.md", "file_type": "document"},
        {"id": "writer", "label": "acquire_writer", "source_file": "store.py", "file_type": "code"},
    ], "links": [{"source": "guide", "target": "writer", "relation": "documents", "confidence": "EXTRACTED"}]})
    return db


def test_chunk_boundaries_share_extractor_rules():
    text = "---\n# Hidden\n---\nIntro\n# One\n```md\n# Example\n```\nBody\n## Two\nNext\n"
    chunks = heading_chunks(text)
    assert [(line, title) for line, title, _ in chunks] == [(4, ""), (5, "One"), (10, "Two")]
    assert "# Example" in chunks[1][2]
    assert "# Hidden" not in "".join(body for _, _, body in chunks)
    assert len(heading_chunks("x" * 40_000)) == 3


def test_indexes_reopen_without_changing_structural_records(tmp_path):
    db = corpus(tmp_path)
    with connection(db) as store:
        before = store.snapshot().to_node_link()
    assert semantic_seeds(db, "exclusive writer lock", root=tmp_path) == ["guide"]
    with connection(db) as store:
        assert store.snapshot().to_node_link() == before
        assert store.get_metadata("retrieval:index")["chunks"] == 2
        assert len(store._db.vector_search("_GraphifyChunk", "embedding", [1.0] + [0.0] * 255, 10)) == 2
        assert store._db.text_search("_GraphifyChunk", "text", "rollback", 10)
        output = tmp_path / "export.json"
        store.export_json(output)
        assert "embedding" not in output.read_text()
        revision = store.get_revision()
    assert semantic_seeds(db, "atomic rollback", root=tmp_path) == ["guide"]
    with connection(db) as store:
        assert store.get_revision() == revision  # Cached index: no structural/meta write.
        assert store.snapshot().to_node_link() == before


def test_source_changes_rebuild_and_deletions_do_not_retrieve_stale_text(tmp_path):
    db = corpus(tmp_path)
    semantic_seeds(db, "rollback", root=tmp_path)
    (tmp_path / "guide.md").write_text("# Changed\nReplication quorum recovery.\n")
    assert semantic_seeds(db, "replication", root=tmp_path) == ["guide"]
    with connection(db) as store:
        assert store.get_metadata("retrieval:index")["chunks"] == 1
        assert not store._db.text_search("_GraphifyChunk", "text", "rollback", 10)
    (tmp_path / "guide.md").unlink()
    assert semantic_seeds(db, "rollback", root=tmp_path) == []


def test_no_chunks_never_imports_embedding_backend(tmp_path):
    db = tmp_path / "graph.grafeo"
    write_graph(db, {"nodes": [{"id": "code", "source_file": "code.py"}], "links": []})
    assert semantic_seeds(db, "anything", root=tmp_path, backend="python:missing:embed") == []
    with connection(db) as store:
        assert not store._execute("MATCH (n:_GraphifyChunk) RETURN n")


@pytest.mark.parametrize("value", [[], [[math.nan]], [[math.inf]], [[0.0]], [[True]], [[1.0], [2.0]]])
def test_invalid_embeddings_fail_before_native_writes(tmp_path, monkeypatch, value):
    import graphify.store.retrieval as retrieval
    db = corpus(tmp_path)
    with connection(db) as store:
        before = store.snapshot()
    monkeypatch.setattr(retrieval, "embedding_backend", lambda _: lambda texts: value)
    with pytest.raises(ValueError):
        semantic_seeds(db, "locks", root=tmp_path)
    with connection(db) as store:
        assert store.snapshot() == before


def test_local_callable_backend_and_revision_check(tmp_path, monkeypatch):
    import graphify.semantic_retrieval as semantic
    import graphify.store.retrieval as retrieval
    db = corpus(tmp_path)
    function = embedding_backend("python:graphify.semantic_retrieval:hash_embeddings")
    assert len(validated_vectors(function, ["hello"])[0]) == 256
    real = semantic.hash_embeddings
    changed = False

    def racing(texts):
        nonlocal changed
        if not changed:
            with connection(db, write=True) as store:
                store.set_metadata("other-writer", True)
            changed = True
        return real(texts)

    monkeypatch.setattr(retrieval, "embedding_backend", lambda _: racing)
    with pytest.raises(ConcurrentUpdateError):
        semantic_seeds(db, "hello", root=tmp_path)
    with connection(db) as store:
        assert not store._execute("MATCH (n:_GraphifyChunk) RETURN n")


def test_outside_sources_and_secrets_are_not_indexed(tmp_path):
    db = corpus(tmp_path)
    with connection(db, write=True) as store:
        from graphify.store import NodeRecord
        store.upsert_nodes([NodeRecord("outside", {"source_file": "../outside.md"})])
    (tmp_path / "guide.md").write_text("# Config\nOPENAI_API_KEY=sk-" + "a1" * 24 + "\n")
    semantic_seeds(db, "config", root=tmp_path)
    with connection(db) as store:
        rows = store._execute("MATCH (n:_GraphifyChunk) RETURN n.text AS text, n.source_file AS source")
    assert all(row["source"] == "guide.md" for row in rows)
    assert "sk-" + "a1" * 24 not in json.dumps(rows)


def test_json_requires_explicit_migration(tmp_path):
    with pytest.raises(ValueError, match="migrate"):
        semantic_seeds(tmp_path / "graph.json", "q", root=tmp_path)


def test_cli_semantic_expands_graph_and_default_never_embeds(tmp_path, monkeypatch, capsys):
    from graphify.cli import dispatch_command
    import graphify.semantic_retrieval as semantic
    import sys
    db = corpus(tmp_path)
    monkeypatch.setenv("GRAPHIFY_OUT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    real = semantic.hash_embeddings
    calls = []

    def tracked(texts):
        calls.append(texts)
        return real(texts)

    monkeypatch.setattr(semantic, "hash_embeddings", tracked)
    monkeypatch.setattr(sys, "argv", ["graphify", "query", "Operations guide", "--graph", str(db)])
    dispatch_command("query")
    assert not calls
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["graphify", "query", "exclusive writer lock", "--graph", str(db),
                                      "--semantic", "--depth", "1", "--budget", "1000"])
    dispatch_command("query")
    output = capsys.readouterr().out
    assert calls
    assert "Operations guide" in output and "acquire_writer" in output
    assert "depth=1" in output
    monkeypatch.setattr(sys, "argv", ["graphify", "query", "locks", "--graph", str(db),
                                      "--semantic", "--embedding-backend", "invalid"])
    with pytest.raises(SystemExit) as error:
        dispatch_command("query")
    assert error.value.code == 1
    assert "semantic retrieval failed" in capsys.readouterr().err


def test_backend_dimension_change_and_import_replace_clear_derived_nodes(tmp_path, monkeypatch):
    import graphify.store.retrieval as retrieval
    db = corpus(tmp_path)
    semantic_seeds(db, "locks", root=tmp_path)
    monkeypatch.setattr(retrieval, "embedding_backend", lambda _: lambda texts: [[1.0, 0.0] for _ in texts])
    assert semantic_seeds(db, "locks", root=tmp_path, backend="custom-v2") == ["guide"]
    with connection(db, write=True) as store:
        assert store.get_metadata("retrieval:index")["dimensions"] == 2
        store.import_node_link({"nodes": [{"id": "new"}], "links": []}, replace=True)
        assert not store._execute("MATCH (n:_GraphifyChunk) RETURN n")
        assert store.get_metadata("retrieval:index") is None


def test_vector_normalization_handles_tiny_finite_values():
    assert validated_vectors(lambda _: [[1e-320, 0.0]], ["tiny"]) == [[1.0, 0.0]]
    with pytest.raises(ValueError):
        validated_vectors(lambda _: [[10 ** 10000]], ["huge"])


def test_heading_match_seeds_its_own_structural_section(tmp_path):
    (tmp_path / "guide.md").write_text("# Transactions\nAtomic rollback.\n\n# Locking\nExclusive writer locks.\n")
    db = tmp_path / "graph.grafeo"
    write_graph(db, {"nodes": [
        {"id": "transactions", "source_file": "guide.md", "source_location": "L1"},
        {"id": "locking", "source_file": "guide.md", "source_location": "L4"},
    ], "links": []})
    assert semantic_seeds(db, "exclusive writer locks", root=tmp_path, limit=1) == ["locking"]


def test_local_model_adapter_refuses_missing_paths_and_disables_fetching(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import numpy as np
    calls = []

    def constructor(path, **kwargs):
        calls.append((path, kwargs))
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
        import os
        assert os.environ["HF_HUB_OFFLINE"] == os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
        return SimpleNamespace(encode=lambda texts, **options: np.array([[1.0, 0.0] for _ in texts]))

    for name in ("HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "TRANSFORMERS_OFFLINE", "DO_NOT_TRACK"):
        monkeypatch.setenv(name, "0")
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=constructor))
    with pytest.raises(ValueError, match="existing local"):
        embedding_backend("sentence-transformers:" + str(tmp_path / "missing"))
    assert not calls
    embed = embedding_backend("sentence-transformers:" + str(tmp_path))
    assert validated_vectors(embed, ["a", "b"]) == [[1.0, 0.0], [1.0, 0.0]]
    assert calls[0][0] == str(tmp_path.resolve())


def test_derived_index_does_not_refresh_structural_freshness(tmp_path):
    import os
    from graphify.store.io import graph_content_mtime
    db = corpus(tmp_path)
    os.utime(db, (100, 100))
    assert graph_content_mtime(db) == 100
    semantic_seeds(db, "locking", root=tmp_path)
    assert db.stat().st_mtime > 100
    assert graph_content_mtime(db) == 100
    semantic_seeds(db, "rollback", root=tmp_path)
    assert graph_content_mtime(db) == 100


def test_cli_does_not_widen_project_reads_using_untrusted_scan_metadata(tmp_path, monkeypatch, capsys):
    import sys
    from graphify.cli import dispatch_command
    project = tmp_path / "project"
    out = project / "graphify-out"
    out.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("# Private\nA stolenneedle outside the project boundary.\n")
    db = out / "graph.grafeo"
    write_graph(db, {"nodes": [{"id": "outside", "label": "External", "source_file": str(outside)}],
                     "links": []}, metadata={"scan_root": str(tmp_path)})
    monkeypatch.chdir(project)
    monkeypatch.setattr(sys, "argv", ["graphify", "query", "stolenneedle", "--graph", str(db), "--semantic"])
    with pytest.raises(SystemExit) as error:
        dispatch_command("query")
    assert error.value.code == 4
    with connection(db) as store:
        assert store.get_metadata("retrieval:index") is None
        assert not store._execute("MATCH (n:_GraphifyChunk) RETURN n")
