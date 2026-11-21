from tests.graph_io import unlink as _graph_unlink, read_bytes as _graph_read_bytes, canonical_expected as _canonical_expected
from tests.graph_io import write_text as _graph_write_text
"""Malformed optional learning data cannot hide healthy entries."""

import json

import pytest

from graphify.reflect import LEARNING_SIDECAR_NAME, load_learning_overlay


@pytest.mark.parametrize("payload", [None, [], 7, "unexpected"])
def test_learning_overlay_ignores_non_object_root(tmp_path, payload):
    _graph_write_text(tmp_path / LEARNING_SIDECAR_NAME, json.dumps(payload), encoding="utf-8")

    assert load_learning_overlay(tmp_path / "graph.grafeo") == {}


def test_learning_overlay_keeps_valid_entries_beside_malformed_source(tmp_path):
    payload = {"nodes": {
        "bad": {"source_file": ["app.py"], "status": "preferred"},
        "good": {"status": "preferred", "score": 2},
        "legacy": {"source_file": None, "status": "preferred"},
        "gone": {"source_file": "deleted.py", "status": "preferred"},
    }}
    _graph_write_text(tmp_path / LEARNING_SIDECAR_NAME, json.dumps(payload), encoding="utf-8")

    assert load_learning_overlay(tmp_path / "graph.grafeo") == {
        "good": {"status": "preferred", "score": 2, "stale": False},
        "legacy": {"source_file": None, "status": "preferred", "stale": False},
        "gone": {"source_file": "deleted.py", "status": "preferred", "stale": True},
    }
