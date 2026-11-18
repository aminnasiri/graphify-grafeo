from tests.graph_io import unlink as _graph_unlink, read_bytes as _graph_read_bytes, canonical_expected as _canonical_expected
from tests.graph_io import read_text as _graph_read_text, write_text as _graph_write_text
"""A resolver input can change imports in a file whose bytes did not change."""

import json

import pytest

from graphify import extract as extract_mod
from graphify.watch import _rebuild_code


@pytest.mark.parametrize("change", ["edit", "remove", "add"])
def test_update_refreshes_unchanged_typescript_importers(tmp_path, change):
    source = tmp_path / "main.ts"
    _graph_write_text(source, 'import { target } from "@app/target";\nexport function run() { return target(); }\n', encoding="utf-8")
    for name in ("old", "new"):
        directory = tmp_path / name
        directory.mkdir()
        _graph_write_text(directory / "target.ts", f'export function target() {{ return "{name}"; }}\n', encoding="utf-8")
    config = tmp_path / "tsconfig.json"

    def write_config(target):
        _graph_write_text(config, json.dumps({"compilerOptions": {"baseUrl": ".", "paths": {
            "@app/*": [f"./{target}/*"],
        }}}), encoding="utf-8")

    graph = tmp_path / "graphify-out" / "graph.grafeo"

    def local_imports():
        data = json.loads(_graph_read_text(graph, encoding="utf-8"))
        nodes = {node["id"]: node for node in data["nodes"]}
        return {
            nodes[link["target"]].get("source_file", "").replace("\\", "/")
            for link in data["links"]
            if link["relation"] == "imports_from"
            and nodes[link["source"]].get("source_file") == "main.ts"
        } & {"old/target.ts", "new/target.ts"}

    if change != "add":
        write_config("old")
    assert _rebuild_code(tmp_path, no_cluster=True)
    assert local_imports() == (set() if change == "add" else {"old/target.ts"})

    if change == "remove":
        _graph_unlink(config)
    else:
        write_config("new")
    assert _rebuild_code(tmp_path, incremental=True, no_cluster=True)

    assert local_imports() == (set() if change == "remove" else {"new/target.ts"})


def test_update_rechecks_importers_after_lockfile_only_change(tmp_path, monkeypatch):
    source = tmp_path / "main.ts"
    _graph_write_text(source, 'import "external";\nexport function run() {}\n', encoding="utf-8")
    lockfile = tmp_path / "pnpm-lock.yaml"
    _graph_write_text(lockfile, "lockfileVersion: '9.0'\n", encoding="utf-8")
    assert _rebuild_code(tmp_path, no_cluster=True)

    actual_extract = extract_mod.extract
    batches = []

    def record_extract(paths, **kwargs):
        batches.append(list(paths))
        return actual_extract(paths, **kwargs)

    monkeypatch.setattr(extract_mod, "extract", record_extract)
    _graph_write_text(lockfile, "lockfileVersion: '9.0'\nimporters: {}\n", encoding="utf-8")
    assert _rebuild_code(tmp_path, incremental=True, no_cluster=True)
    assert batches == [[source]]

    assert _rebuild_code(tmp_path, incremental=True, no_cluster=True)
    assert batches == [[source]]
