from tests.graph_io import unlink as _graph_unlink, read_bytes as _graph_read_bytes, canonical_expected as _canonical_expected
from tests.graph_io import is_file as _graph_is_file, read_text as _graph_read_text, write_text as _graph_write_text
"""A manifest cannot stand in for the graph an incremental update merges into."""

import json

import pytest

from graphify.watch import _rebuild_code


@pytest.mark.parametrize("change_one_source", [False, True])
def test_update_recreates_missing_graph_with_all_sources(tmp_path, change_one_source):
    first = tmp_path / "first.py"
    _graph_write_text(first, "def first(): return 1\n", encoding="utf-8")
    _graph_write_text(tmp_path / "second.py", "def second(): return 2\n", encoding="utf-8")
    out = tmp_path / "graphify-out"
    graph = out / "graph.grafeo"
    assert _rebuild_code(tmp_path, no_cluster=True)
    assert _graph_is_file(out / "manifest.json")
    _graph_unlink(graph)
    if change_one_source:
        _graph_write_text(first, "def renamed(): return 123\n", encoding="utf-8")

    assert _rebuild_code(tmp_path, no_cluster=True, incremental=True)

    labels = {node["label"] for node in json.loads(_graph_read_text(graph, encoding="utf-8"))["nodes"]}
    assert "second()" in labels
    assert ("renamed()" if change_one_source else "first()") in labels
