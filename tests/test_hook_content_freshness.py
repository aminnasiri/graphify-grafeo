from tests.graph_io import unlink as _graph_unlink, read_bytes as _graph_read_bytes, canonical_expected as _canonical_expected
from tests.graph_io import read_text as _graph_read_text, write_text as _graph_write_text
"""A checkout's newer timestamps are not evidence of changed extraction content."""
import json
import os
from pathlib import Path

import pytest

import graphify.paths as paths
from graphify.detect import save_manifest
from tests.test_hook_strict import _fixture, _invoke, _is_deny, _read


def _project(tmp_path, *, kind="both"):
    f = _fixture(tmp_path)
    manifest = tmp_path / "graphify-out" / "manifest.json"
    # Use the extraction writer, not a guessed cache/hash schema.
    save_manifest({"code": [str(f)]}, str(manifest), kind=kind, root=tmp_path)
    graph = manifest.with_name("graph.grafeo")
    os.utime(graph, (100, 100))
    from graphify.store.io import connection
    with connection(graph, write=True) as store:
        store.set_metadata("graph_content_mtime", 100)
    os.utime(f, (200, 200))
    return f, manifest


def _assert_stale(f, tmp_path, monkeypatch):
    out = _invoke("read", _read(f), tmp_path, monkeypatch, strict=True)
    assert not _is_deny(out)
    assert "stale" in out.lower()


@pytest.mark.parametrize("kind", ["ast", "semantic", "both"])
@pytest.mark.parametrize("relative", [False, True])
def test_newer_unchanged_file_uses_real_extraction_hash(tmp_path, monkeypatch, kind, relative):
    f, _ = _project(tmp_path, kind=kind)
    target = f.relative_to(tmp_path) if relative else f
    assert _is_deny(_invoke("read", _read(target), tmp_path, monkeypatch, strict=True))


def test_decomposed_filename_matches_normalized_manifest_key(tmp_path, monkeypatch):
    f, manifest = _project(tmp_path)
    decomposed = f.with_name("cafe\u0301.py")
    f.rename(decomposed)
    save_manifest({"code": [str(decomposed)]}, str(manifest), root=tmp_path)
    assert "src/caf\u00e9.py" in json.loads(_graph_read_text(manifest, encoding="utf-8"))
    os.utime(decomposed, (200, 200))

    out = _invoke("read", _read(decomposed), tmp_path, monkeypatch, strict=True)

    assert "stale" not in out.lower()
    assert "knowledge graph at" in out


def test_same_length_content_change_stays_stale(tmp_path, monkeypatch):
    f, _ = _project(tmp_path)
    _graph_write_text(f, "def x():\n    return 2\n", encoding="utf-8")
    _assert_stale(f, tmp_path, monkeypatch)


def test_needs_update_wins_over_matching_hash(tmp_path, monkeypatch):
    f, manifest = _project(tmp_path)
    _graph_write_text(manifest.with_name("needs_update"), "1", encoding="utf-8")
    _assert_stale(f, tmp_path, monkeypatch)


@pytest.mark.parametrize("bad", [
    "missing", "corrupt", "unknown", "non-dict", "non-dict-row",
    "missing-hash", "invalid-hash", "non-string-hash", "empty-hashes", "invalid-tier",
])
def test_uncertain_manifest_stays_stale(tmp_path, monkeypatch, bad):
    f, manifest = _project(tmp_path)
    data = json.loads(_graph_read_text(manifest, encoding="utf-8"))
    if bad == "missing":
        _graph_unlink(manifest)
    elif bad == "corrupt":
        _graph_write_text(manifest, "{", encoding="utf-8")
    elif bad == "oversized":
        # Valid JSON with a matching row must still be rejected over the cap.
        _graph_write_text(manifest, json.dumps(data) + " " * 2_000_001, encoding="utf-8")
    else:
        if bad == "unknown":
            data = {"other.py": data["src/mod.py"]}
        elif bad == "non-dict":
            data = []
        elif bad == "non-dict-row":
            data["src/mod.py"] = 100
        elif bad == "missing-hash":
            data["src/mod.py"] = {"mtime": 100}
        elif bad == "invalid-hash":
            data["src/mod.py"]["ast_hash"] = "not-an-extraction-hash"
        elif bad == "non-string-hash":
            data["src/mod.py"]["ast_hash"] = ["hash"]
        elif bad == "empty-hashes":
            data["src/mod.py"] = {"ast_hash": "", "semantic_hash": ""}
        elif bad == "invalid-tier":
            data["src/mod.py"]["semantic_hash"] = None
        _graph_write_text(manifest, json.dumps(data), encoding="utf-8")
    _assert_stale(f, tmp_path, monkeypatch)


@pytest.mark.parametrize("stale_tier", ["ast_hash", "semantic_hash"])
def test_one_matching_tier_cannot_hide_other_stale_tier(tmp_path, monkeypatch, stale_tier):
    f, manifest = _project(tmp_path)
    data = json.loads(_graph_read_text(manifest, encoding="utf-8"))
    data["src/mod.py"][stale_tier] = "0" * 32
    _graph_write_text(manifest, json.dumps(data), encoding="utf-8")
    _assert_stale(f, tmp_path, monkeypatch)


@pytest.mark.parametrize("absolute_key", [False, True])
def test_legacy_hash_schema(tmp_path, monkeypatch, absolute_key):
    f, manifest = _project(tmp_path)
    row = json.loads(_graph_read_text(manifest, encoding="utf-8"))["src/mod.py"]
    _graph_write_text(manifest, json.dumps({str(f) if absolute_key else "src/mod.py": {
        "mtime": row["mtime"], "hash": row["ast_hash"],
    }}), encoding="utf-8")
    assert _is_deny(_invoke("read", _read(f), tmp_path, monkeypatch, strict=True))


@pytest.mark.parametrize("collision", ["mod.py", "other/src/mod.py", "other/mod.py"])
def test_basename_or_suffix_collision_is_not_extraction_evidence(tmp_path, monkeypatch, collision):
    f, manifest = _project(tmp_path)
    row = json.loads(_graph_read_text(manifest, encoding="utf-8"))["src/mod.py"]
    _graph_write_text(manifest, json.dumps({collision: row}), encoding="utf-8")
    _assert_stale(f, tmp_path, monkeypatch)


@pytest.mark.parametrize("absolute_out", [False, True])
def test_custom_output_manifest_is_used(tmp_path, monkeypatch, absolute_out):
    f, manifest = _project(tmp_path)
    custom = tmp_path / "artifacts" / "graph"
    custom.parent.mkdir()
    manifest.parent.rename(custom)
    monkeypatch.setattr(paths, "GRAPHIFY_OUT", str(custom) if absolute_out else "artifacts/graph")
    assert _is_deny(_invoke("read", _read(f), tmp_path, monkeypatch, strict=True))


def test_older_file_does_not_hash(tmp_path, monkeypatch):
    import graphify.detect as detect

    f, _ = _project(tmp_path)
    os.utime(f, (50, 50))
    calls = []
    monkeypatch.setattr(detect, "_md5_file", lambda p: calls.append(p) or "")
    assert _is_deny(_invoke("read", _read(f), tmp_path, monkeypatch, strict=True))
    assert calls == []


@pytest.mark.parametrize("failure", ["manifest", "source"])
def test_read_errors_stay_stale(tmp_path, monkeypatch, failure):
    f, manifest = _project(tmp_path)
    if failure == "source":
        import builtins
        original_open = builtins.open

        def denied_source(file, *args, **kwargs):
            # The hashing code opens the source through paths.os_long_path,
            # which on win32 prepends the extended-length \\?\ marker so long
            # paths survive open(). Comparing that prefixed string straight
            # against f as plain Path objects never matches on Windows, so
            # the mock never raised and this branch silently tested the
            # readable-file case instead. Normalize both sides the same way
            # the source itself does before comparing.
            if paths.os_long_path(file) == paths.os_long_path(f):
                raise PermissionError("unreadable fixture")
            return original_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", denied_source)
    else:
        import graphify.cli as cli
        original = cli._read_graph_data

        def denied(path):
            if Path(path).name == "manifest.json":
                raise PermissionError("unreadable manifest metadata")
            return original(path)

        monkeypatch.setattr(cli, "_read_graph_data", denied)
    _assert_stale(f, tmp_path, monkeypatch)


def test_stat_index_digest_cannot_bless_changed_extraction(tmp_path, monkeypatch):
    from graphify.cache import file_hash

    f, _ = _project(tmp_path)
    _graph_write_text(f, "def x():\n    return 2\n", encoding="utf-8")
    file_hash(f, root=tmp_path)  # Memoizes current content, not extracted content.
    _assert_stale(f, tmp_path, monkeypatch)
