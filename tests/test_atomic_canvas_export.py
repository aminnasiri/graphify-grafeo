"""Regression: to_canvas / Obsidian vault writes must be atomic (#3282)."""
from __future__ import annotations
from tests.graph_io import unlink as _graph_unlink, read_bytes as _graph_read_bytes, canonical_expected as _canonical_expected
from tests.graph_io import read_text as _graph_read_text, write_text as _graph_write_text

import json
import os
from pathlib import Path

import networkx as nx
import pytest

from graphify.export import to_canvas, to_obsidian


def _tiny_graph():
    G = nx.Graph()
    G.add_node("a", label="Alpha", file_type="code", source_file="a.py")
    G.add_node("b", label="Beta", file_type="code", source_file="b.py")
    G.add_edge("a", "b", relation="calls", confidence="EXTRACTED", weight=1.0)
    return G, {0: ["a", "b"]}


def test_to_canvas_uses_atomic_replace(tmp_path, monkeypatch):
    G, communities = _tiny_graph()
    out = tmp_path / "graph.canvas"
    _graph_write_text(out, '{"nodes":[],"edges":[]}', encoding="utf-8")

    real_replace = os.replace
    calls: list[tuple[str, str]] = []

    def tracking_replace(src, dst):
        calls.append((str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", tracking_replace)
    to_canvas(G, communities, str(out))

    assert any(Path(dst).resolve() == out.resolve() for _, dst in calls), calls
    data = json.loads(_graph_read_text(out, encoding="utf-8"))
    assert "nodes" in data and "edges" in data
    assert len(data["nodes"]) >= 2


def test_to_canvas_preserves_existing_when_replace_fails(tmp_path, monkeypatch):
    G, communities = _tiny_graph()
    out = tmp_path / "graph.canvas"
    original = '{"nodes":[],"edges":[],"preserved":true}'
    _graph_write_text(out, original, encoding="utf-8")

    def boom(src, dst):
        raise OSError("simulated failure")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        to_canvas(G, communities, str(out))

    assert _graph_read_text(out, encoding="utf-8") == original


def test_to_obsidian_owned_writes_are_atomic(tmp_path, monkeypatch):
    G, communities = _tiny_graph()
    real_replace = os.replace
    calls: list[str] = []

    def tracking_replace(src, dst):
        calls.append(str(dst))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", tracking_replace)
    to_obsidian(G, communities, str(tmp_path), community_labels={0: "Core"})

    # At least one vault artifact should land via os.replace (notes / graph.grafeo).
    assert calls, "expected atomic replaces for Obsidian vault writes"
    assert any(Path(p).name.endswith(".md") or p.endswith("graph.grafeo") for p in calls)
