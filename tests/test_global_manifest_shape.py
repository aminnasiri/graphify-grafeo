from tests.graph_io import unlink as _graph_unlink, read_bytes as _graph_read_bytes, canonical_expected as _canonical_expected
from tests.graph_io import state_exists as _graph_exists, is_file as _graph_is_file, read_text as _graph_read_text, write_text as _graph_write_text
"""Global graph commands recover from JSON with the wrong manifest shape."""

import json

import pytest

from graphify import global_graph


@pytest.mark.parametrize("payload", [None, [], {}, {"repos": []}])
def test_global_list_backs_up_wrong_manifest_shape(tmp_path, monkeypatch, capsys, payload):
    # A non-dict root, or a repos value that is not a map at all, is too broken
    # to recover per-entry: back it up and start fresh.
    manifest = tmp_path / "global-manifest.json"
    original = json.dumps(payload)
    _graph_write_text(manifest, original, encoding="utf-8")
    monkeypatch.setattr(global_graph, "_GLOBAL_MANIFEST", manifest)

    assert global_graph.global_list() == {}

    assert not _graph_exists(manifest)
    backup, = tmp_path.glob("global-manifest.json.corrupt.*")
    assert _graph_read_text(backup, encoding="utf-8") == original
    assert "moved to" in capsys.readouterr().err


def test_global_list_isolates_malformed_repo_entries(tmp_path, monkeypatch, capsys):
    # A single malformed entry must not discard the user's other repos (#5020):
    # drop only the bad one, keep the good ones, and do NOT back up the file.
    manifest = tmp_path / "global-manifest.json"
    good = {"source_path": "graph.grafeo", "node_count": 2}
    _graph_write_text(manifest, json.dumps({"version": 1, "repos": {"goodA": good, "bad": "oops", "goodB": good}}), encoding="utf-8")
    monkeypatch.setattr(global_graph, "_GLOBAL_MANIFEST", manifest)

    assert global_graph.global_list() == {"goodA": good, "goodB": good}

    assert _graph_is_file(manifest)  # not backed up
    assert not list(tmp_path.glob("global-manifest.json.corrupt.*"))
    err = capsys.readouterr().err
    assert "dropped 1 malformed repo" in err and "bad" in err


def test_global_list_drops_all_malformed_entries_without_backup(tmp_path, monkeypatch):
    # An all-malformed repos map is still a valid shape: it collapses to empty
    # rather than being backed up.
    manifest = tmp_path / "global-manifest.json"
    _graph_write_text(manifest, json.dumps({"repos": {"legacy": "invalid"}}), encoding="utf-8")
    monkeypatch.setattr(global_graph, "_GLOBAL_MANIFEST", manifest)

    assert global_graph.global_list() == {}
    assert _graph_is_file(manifest)
    assert not list(tmp_path.glob("global-manifest.json.corrupt.*"))


def test_global_list_preserves_valid_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "global-manifest.json"
    repos = {"project": {"source_path": "graph.grafeo", "node_count": 2}}
    _graph_write_text(manifest, json.dumps({"version": 1, "repos": repos}), encoding="utf-8")
    monkeypatch.setattr(global_graph, "_GLOBAL_MANIFEST", manifest)

    assert global_graph.global_list() == repos
    assert _graph_is_file(manifest)
