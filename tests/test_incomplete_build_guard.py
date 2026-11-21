"""Tests for the incomplete-build shrink-guard on `graphify extract`.

A full build writes the graph with `to_json(..., force=True)`, which bypasses the
#479 shrink guard. When this run's extraction was incomplete (an AST pass crashed
or some semantic chunks failed), forcing the write can silently overwrite a good
complete graph with a smaller partial one. The build now drops back to the shrink
guard (force=False) on an incomplete run — unless `--allow-partial` is passed —
and exits non-zero (before writing the manifest) if the guard refuses.
"""
from __future__ import annotations
from tests.graph_io import unlink as _graph_unlink, read_bytes as _graph_read_bytes, canonical_expected as _canonical_expected
from tests.graph_io import state_exists as _graph_exists, read_text as _graph_read_text, write_text as _graph_write_text

import pytest

import graphify.__main__ as mainmod


def _make_docs_corpus(tmp_path):
    # Docs-only corpus: no code files, so AST extraction is skipped and the only
    # driver of incompleteness is the (stubbed) semantic chunk run.
    _graph_write_text(tmp_path / "README.md", "# Notes\nThe entry point overview.\n")
    _graph_write_text(tmp_path / "GUIDE.md", "# Guide\nHow to use the thing.\n")
    return tmp_path


def _seed_to_json_recorder(monkeypatch, *, returns=True):
    """Patch export.to_json to record the ``force`` it was called with and return
    a fixed bool (True = wrote, False = shrink guard refused)."""
    rec = {"called": False, "force": None}

    def _stub(G, communities, output_path, *, force=False, **kwargs):
        rec["called"] = True
        rec["force"] = force
        return returns

    monkeypatch.setattr("graphify.export.to_json", _stub)
    return rec


def _arm_extract(monkeypatch, tmp_path, *, chunk_total, chunk_succeeded, extra_argv=()):
    corpus = _make_docs_corpus(tmp_path)
    out_dir = tmp_path / "out"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-key")

    def _stub_corpus(paths, **kwargs):
        on_chunk = kwargs.get("on_chunk_done")
        if on_chunk:
            for i in range(chunk_succeeded):
                on_chunk(i, chunk_total, {"nodes": [], "edges": [], "hyperedges": []})
        return {
            "nodes": [{"id": "s1", "source_file": str(corpus / "README.md"),
                       "file_type": "document", "label": "Notes"}],
            "edges": [], "hyperedges": [], "input_tokens": 10, "output_tokens": 5,
        }

    monkeypatch.setattr("graphify.llm.extract_corpus_parallel", _stub_corpus)
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(
        mainmod.sys, "argv",
        ["graphify", "extract", str(corpus), "--backend", "claude",
         "--out", str(out_dir), *extra_argv],
    )
    return out_dir


def test_partial_extraction_refuses_to_shrink_existing_graph(monkeypatch, tmp_path, capsys):
    # 1 of 3 chunks succeeded -> incomplete; the shrink guard refuses (returns False).
    rec = _seed_to_json_recorder(monkeypatch, returns=False)
    out_dir = _arm_extract(monkeypatch, tmp_path, chunk_total=3, chunk_succeeded=1)

    with pytest.raises(SystemExit) as exc:
        mainmod.main()

    assert exc.value.code == 1
    assert rec["called"] and rec["force"] is False, "incomplete build must not force the write"
    err = capsys.readouterr().err
    assert "Refusing to overwrite" in err
    # The manifest must not be stamped for a graph we declined to write.
    assert not _graph_exists(out_dir / "graphify-out" / "manifest.json")


def test_partial_extraction_writes_when_not_shrinking(monkeypatch, tmp_path):
    # Incomplete run, but the new graph is not smaller -> the guard permits the
    # write. force is still False (guard active), and the CLI does not exit 1.
    rec = _seed_to_json_recorder(monkeypatch, returns=True)
    _arm_extract(monkeypatch, tmp_path, chunk_total=3, chunk_succeeded=1)

    mainmod.main()  # no SystemExit

    assert rec["called"] and rec["force"] is False


def test_allow_partial_forces_write_despite_incomplete(monkeypatch, tmp_path):
    rec = _seed_to_json_recorder(monkeypatch, returns=True)
    _arm_extract(monkeypatch, tmp_path, chunk_total=3, chunk_succeeded=1,
                 extra_argv=["--allow-partial"])

    mainmod.main()

    assert rec["called"] and rec["force"] is True, "--allow-partial must restore force=True"


def test_complete_extraction_keeps_force_write(monkeypatch, tmp_path):
    # All chunks succeeded and dedup did not shrink the saved graph. A complete
    # run still force-writes in that case. A dedup shrink is refused unless
    # --allow-dedup-shrink (#3774); see test_dedup_shrink_refuses_force_write.
    rec = _seed_to_json_recorder(monkeypatch, returns=True)
    _arm_extract(monkeypatch, tmp_path, chunk_total=1, chunk_succeeded=1)

    mainmod.main()

    assert rec["called"] and rec["force"] is True


def _seed_existing_graph(gout, n):
    import json
    gout.mkdir(parents=True, exist_ok=True)
    _graph_write_text(gout / "graph.grafeo", json.dumps({"nodes": [{"id": f"keep{i}", "label": f"k{i}"} for i in range(n)],
                    "links": []}), encoding="utf-8")


def _arm_no_cluster(monkeypatch, tmp_path, *, extra_argv=()):
    corpus = _make_docs_corpus(tmp_path)
    out_dir = tmp_path / "out"
    gout = out_dir / "graphify-out"
    _seed_existing_graph(gout, 5)  # existing complete graph
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-key")

    def _stub_corpus(paths, **kwargs):
        on_chunk = kwargs.get("on_chunk_done")
        if on_chunk:
            on_chunk(0, 3, {"nodes": [], "edges": [], "hyperedges": []})  # 1 of 3 -> partial
        return {"nodes": [{"id": "s1", "source_file": str(corpus / "README.md"),
                           "file_type": "document", "label": "Notes"}],
                "edges": [], "hyperedges": [], "input_tokens": 1, "output_tokens": 1}

    monkeypatch.setattr("graphify.llm.extract_corpus_parallel", _stub_corpus)
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(
        mainmod.sys, "argv",
        ["graphify", "extract", str(corpus), "--backend", "claude", "--no-cluster",
         "--out", str(out_dir), *extra_argv],
    )
    return gout / "graph.grafeo"


def test_no_cluster_incomplete_build_refuses_to_shrink(tmp_path, monkeypatch, capsys):
    # --force: the non-incremental raw-dump path, where the shrink guard is the
    # only thing standing between a partial 1-node extraction and the existing
    # complete 5-node graph. (Incremental runs merge the existing graph forward
    # first — #2169 — so a partial run no longer shrinks there; see below.)
    import json
    graph = _arm_no_cluster(monkeypatch, tmp_path, extra_argv=["--force"])

    with pytest.raises(SystemExit) as exc:
        mainmod.main()

    assert exc.value.code == 1
    assert "Refusing to overwrite" in capsys.readouterr().err
    # The existing 5-node graph is untouched — the partial 1-node graph was refused.
    assert len(json.loads(_graph_read_text(graph))["nodes"]) == 5


def test_no_cluster_incremental_incomplete_build_carries_existing_nodes(
    tmp_path, monkeypatch
):
    """#2169: an INCREMENTAL --no-cluster run merges the existing graph forward,
    so even an incomplete extraction does not shrink the graph — the existing
    nodes are carried and this run's partial chunk is added, no guard refusal."""
    import json
    graph = _arm_no_cluster(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as exc:
        mainmod.main()

    assert exc.value.code == 0  # the raw --no-cluster path exits 0 on success
    ids = {n["id"] for n in json.loads(_graph_read_text(graph))["nodes"]}
    assert {f"keep{i}" for i in range(5)} <= ids, ids
    assert "s1" in ids, ids


def test_no_cluster_allow_partial_overwrites(tmp_path, monkeypatch):
    import json
    graph = _arm_no_cluster(
        monkeypatch, tmp_path, extra_argv=["--force", "--allow-partial"]
    )

    with pytest.raises(SystemExit) as exc:
        mainmod.main()

    assert exc.value.code == 0  # the raw --no-cluster path exits 0 on success
    assert len(json.loads(_graph_read_text(graph))["nodes"]) == 1


def test_no_cluster_incomplete_build_fails_closed_on_malformed_existing_graph(
    tmp_path, monkeypatch, capsys
):
    """A present-but-unparseable existing graph.grafeo (corrupt or mid-write) could
    be hiding a complete graph, so an incomplete --no-cluster build must refuse
    to overwrite it — matching to_json's #479 fail-closed handling, not the
    fail-open 'proceed when we can't count' path. --force: the non-incremental
    raw-dump path (the incremental path fails even earlier, at the forward
    merge — see the test below)."""
    graph = _arm_no_cluster(monkeypatch, tmp_path, extra_argv=["--force"])
    _graph_write_text(graph, "{corrupt json", encoding="utf-8")  # non-empty, unparseable

    with pytest.raises(SystemExit) as exc:
        mainmod.main()

    assert exc.value.code == 1
    assert "unparseable" in capsys.readouterr().err
    # The corrupt file is left untouched rather than clobbered by the partial build.
    assert _graph_read_text(graph) == "{corrupt json"


def test_no_cluster_incremental_malformed_existing_graph_refuses_merge(
    tmp_path, monkeypatch, capsys
):
    """#2169: an incremental --no-cluster run must hard-fail on an unparseable
    existing graph.grafeo (build_merge's message) instead of raw-dumping this
    run's chunks over it."""
    graph = _arm_no_cluster(monkeypatch, tmp_path)
    _graph_write_text(graph, "{corrupt json", encoding="utf-8")  # non-empty, unparseable

    with pytest.raises(SystemExit) as exc:
        mainmod.main()

    assert exc.value.code == 1
    assert "Cannot read" in capsys.readouterr().err
    # The corrupt file is left untouched rather than clobbered.
    assert _graph_read_text(graph) == "{corrupt json"
