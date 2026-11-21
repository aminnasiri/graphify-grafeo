"""#4740: `graphify update` must not extract when nothing relevant changed.

_rebuild_code() used to run detect -> extract -> build -> cluster and only then
compare topology, so a zero-change run still cost minutes and gigabytes. These
tests pin the early exit, and — more importantly — the inputs that must NOT take
it. A false "unchanged" ships a stale graph, so every uncertain input has to fall
back to the full path.
"""
from __future__ import annotations
from tests.graph_io import unlink as _graph_unlink, read_bytes as _graph_read_bytes, canonical_expected as _canonical_expected
from tests.graph_io import state_exists as _graph_exists, is_file as _graph_is_file, write_text as _graph_write_text

import importlib
import json
import os
from pathlib import Path

import pytest

import graphify.extract as extract_mod
import graphify.watch as watch_mod
from graphify.watch import _rebuild_code

SKIP_MESSAGE = "No relevant changes"


def _corpus(root: Path) -> Path:
    """A corpus whose graph depends on more than its own code files."""
    (root / "src").mkdir(parents=True, exist_ok=True)
    _graph_write_text(root / "src" / "a.ts", "export const A = 1;\nexport function fa() { return A; }\n")
    _graph_write_text(root / "src" / "b.ts", 'import { fa } from "./a";\nexport const B = fa();\n')
    _graph_write_text(root / "src" / "m.py", "def f():\n    return 1\n")
    _graph_write_text(root / "tsconfig.json", '{"compilerOptions": {"baseUrl": "."}}\n')
    _graph_write_text(root / "package.json", '{"name": "x", "version": "1.0.0"}\n')
    _graph_write_text(root / "package-lock.json", '{"lockfileVersion": 3}\n')
    return root


@pytest.fixture()
def built(tmp_path: Path) -> Path:
    """A corpus with one completed rebuild behind it (manifest stamped)."""
    root = _corpus(tmp_path / "corpus")
    assert _rebuild_code(root, acquire_lock=False) is True
    assert _graph_is_file(root / "graphify-out" / "manifest.json")
    return root


def _spy_extract(monkeypatch: pytest.MonkeyPatch) -> list:
    """Count extract() calls. _rebuild_code imports it at call time."""
    calls: list = []
    real = extract_mod.extract

    def _counting(*args, **kwargs):
        calls.append(args[0] if args else None)
        return real(*args, **kwargs)

    monkeypatch.setattr(extract_mod, "extract", _counting)
    return calls


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        p.name: _graph_read_bytes(p)
        for p in sorted((root / "graphify-out").iterdir())
        if _graph_is_file(p)
    }


def test_unchanged_tree_skips_extraction_and_leaves_outputs_untouched(
    built: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    before = _snapshot(built)
    calls = _spy_extract(monkeypatch)

    assert _rebuild_code(built, acquire_lock=False) is True

    assert SKIP_MESSAGE in capsys.readouterr().out
    assert calls == [], "extract() must not run when nothing changed"
    assert _snapshot(built) == before, "outputs must be byte-identical"


def test_new_file_takes_the_full_path(
    built: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    calls = _spy_extract(monkeypatch)
    _graph_write_text(built / "src" / "c.ts", "export const C = 3;\n")

    assert _rebuild_code(built, acquire_lock=False) is True
    assert SKIP_MESSAGE not in capsys.readouterr().out
    assert calls, "a new file must re-extract"


def test_deleted_file_takes_the_full_path(
    built: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    calls = _spy_extract(monkeypatch)
    _graph_unlink(built / "src" / "m.py")

    assert _rebuild_code(built, acquire_lock=False) is True
    assert SKIP_MESSAGE not in capsys.readouterr().out
    assert calls, "a deletion must re-extract"


def test_edited_tsconfig_takes_the_full_path(
    built: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    calls = _spy_extract(monkeypatch)
    _graph_write_text(built / "tsconfig.json", '{"compilerOptions": {"baseUrl": ".", "paths": {"@app/*": ["src/*"]}}}\n')

    assert _rebuild_code(built, acquire_lock=False) is True
    assert SKIP_MESSAGE not in capsys.readouterr().out
    assert calls, "a tsconfig edit changes JS/TS resolution, not just the graph"


def test_edited_package_json_takes_the_full_path(
    built: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    calls = _spy_extract(monkeypatch)
    _graph_write_text(built / "package.json", '{"name": "x", "version": "2.0.0"}\n')

    assert _rebuild_code(built, acquire_lock=False) is True
    assert SKIP_MESSAGE not in capsys.readouterr().out
    assert calls, "a package.json edit changes workspace resolution"


def test_edited_lock_file_takes_the_full_path(
    built: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    # The lock files are resolution inputs detect() skips by name, so they are
    # the ones the manifest has to carry under the resolution bucket.
    calls = _spy_extract(monkeypatch)
    _graph_write_text(built / "package-lock.json", '{"lockfileVersion": 3, "packages": {"node_modules/x": {}}}\n')

    assert _rebuild_code(built, acquire_lock=False) is True
    assert SKIP_MESSAGE not in capsys.readouterr().out
    assert calls, "a lock-file edit is a resolution-input change"


def test_touch_with_identical_content_still_skips(
    built: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    # Never skip on mtime alone: a bumped mtime with the same bytes is unchanged.
    # Bumping the clock is what `git checkout`/`rsync --times` do, so the gate has
    # to compare content rather than trust the timestamp.
    target = built / "src" / "a.ts"
    stat = target.stat()
    os.utime(target, (stat.st_mtime + 5, stat.st_mtime + 5))
    assert target.stat().st_mtime != stat.st_mtime

    calls = _spy_extract(monkeypatch)
    assert _rebuild_code(built, acquire_lock=False) is True

    assert SKIP_MESSAGE in capsys.readouterr().out
    assert calls == [], "identical content must not re-extract"


def test_changed_paths_supplied_does_not_use_the_gate(
    built: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    calls = _spy_extract(monkeypatch)

    assert _rebuild_code(
        built,
        changed_paths=[built / "src" / "a.ts"],
        acquire_lock=False,
    ) is True

    assert SKIP_MESSAGE not in capsys.readouterr().out
    assert calls, "an explicit change set must still reach extract()"


def test_force_does_not_use_the_gate(
    built: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    calls = _spy_extract(monkeypatch)

    assert _rebuild_code(built, force=True, acquire_lock=False) is True

    assert SKIP_MESSAGE not in capsys.readouterr().out
    assert calls, "--force must run the full path"


def test_gate_fails_closed_without_a_manifest(
    built: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _graph_unlink(built / "graphify-out" / "manifest.json")
    calls = _spy_extract(monkeypatch)

    assert _rebuild_code(built, acquire_lock=False) is True

    assert SKIP_MESSAGE not in capsys.readouterr().out
    assert calls, "no manifest means the tree cannot be proven unchanged"


def test_inputs_unchanged_rejects_an_empty_manifest(tmp_path: Path) -> None:
    from graphify.detect import detect
    from graphify.watch import _inputs_unchanged

    root = _corpus(tmp_path / "corpus")
    detected = detect(root)

    assert _inputs_unchanged(detected, root / "graphify-out" / "manifest.json", root) is False


def test_inputs_unchanged_treats_a_scanned_file_with_no_manifest_row_as_changed(
    built: Path,
) -> None:
    # A scanned source with no manifest row (e.g. one whose AST extraction
    # previously failed, #2543) must NOT fast-path the gate: if it is later fixed
    # with no other change in the tree, the gate has to take the full rebuild so
    # the graph gains its nodes. Tolerating the missing row would skip forever and
    # leave the graph stale. The cost is that a corpus with a permanently-failing
    # file never fast-paths, which is the safe direction.
    from graphify.detect import detect, load_manifest
    from graphify.watch import _inputs_unchanged

    manifest_path = built / "graphify-out" / "manifest.json"
    manifest = load_manifest(str(manifest_path), root=built)
    assert str((built / "src" / "m.py").resolve()) in manifest

    manifest = {k: v for k, v in manifest.items() if not k.endswith("m.py")}
    _graph_write_text(manifest_path, json.dumps(manifest), encoding="utf-8")

    detected = detect(built)
    assert _inputs_unchanged(detected, manifest_path, built) is False


def test_scan_digest_tracks_the_file_set_not_the_contents(built: Path) -> None:
    # The digest is the only thing that can tell "same file, still failing" from
    # "a file appeared", so it has to move for the latter and not for the former.
    from graphify.detect import detect
    from graphify.watch import _scan_digest

    before = _scan_digest(detect(built))
    _graph_write_text(built / "src" / "new.py", "def added(): return 1\n")
    assert _scan_digest(detect(built)) != before

    _graph_unlink(built / "src" / "new.py")
    assert _scan_digest(detect(built)) == before


# --- #4748: a stamp may only describe outputs that are fully on disk ----------
#
# A stamp is a promise that the next run may skip. Writing one over a graph.grafeo
# that a failed or partial write left behind turns that promise into a stale
# graph nothing repairs. Failing to skip only costs time, so every test here
# expects the conservative direction.

STAMP_NAME = ".graphify_build_state.json"

FAILURE_TARGETS = [
    # Each one aborts the run after extraction, at a different write point, so
    # the next run has to notice that the rebuild never finished.
    "graphify.cluster.cluster",
    "graphify.store.io.write_graph",
    "graphify.watch._reconcile_graph_html",
]


@pytest.mark.parametrize("target", FAILURE_TARGETS)
def test_a_rebuild_that_never_finishes_its_outputs_is_not_skipped_next_run(
    built: Path, monkeypatch: pytest.MonkeyPatch, capsys, target: str
) -> None:
    module_name, attr = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    real = getattr(module, attr)

    def boom(*args, **kwargs):
        raise RuntimeError(f"boom: {target}")

    # A topology change, so this run has real work and cannot take one of the
    # no-rewrite short circuits.
    _graph_write_text(built / "src" / "m.py", "def renamed_marker():\n    return 1\n")

    monkeypatch.setattr(module, attr, boom)
    try:
        _rebuild_code(built, acquire_lock=False)
    finally:
        monkeypatch.setattr(module, attr, real)

    calls = _spy_extract(monkeypatch)
    _rebuild_code(built, acquire_lock=False)
    assert SKIP_MESSAGE not in capsys.readouterr().out
    assert calls, "a rebuild that never finished its outputs must not be skipped"


def test_failed_database_transaction_is_never_stamped(built, monkeypatch, capsys):
    from graphify.store import GrafeoStore
    from graphify.store.io import connection
    graph = built / "graphify-out" / "graph.grafeo"
    stamp = built / "graphify-out" / STAMP_NAME
    before_stamp = _graph_read_bytes(stamp)
    with connection(graph) as store:
        before = store.snapshot()
    real = GrafeoStore.upsert_nodes

    def fail(self, nodes):
        real(self, nodes)
        raise RuntimeError("interrupted native transaction")

    _graph_write_text(built / "src" / "m.py", "def renamed_marker():\n    return 1\n")
    with monkeypatch.context() as patch:
        patch.setattr(GrafeoStore, "upsert_nodes", fail)
        assert _rebuild_code(built, acquire_lock=False) is False
    with connection(graph) as store:
        assert store.snapshot() == before
    assert _graph_read_bytes(stamp) == before_stamp
    assert _rebuild_code(built, acquire_lock=False)
    assert SKIP_MESSAGE not in capsys.readouterr().out


def test_write_build_state_refuses_a_graph_that_is_not_a_whole_json_object(
    built: Path,
) -> None:
    from graphify.watch import _write_build_state

    graph = built / "graphify-out" / "graph.grafeo"
    stamp = built / "graphify-out" / STAMP_NAME
    graph.write_bytes(_graph_read_bytes(graph)[:40])
    _graph_unlink(stamp)

    _write_build_state(built / "graphify-out", {}, full_clean=True)

    assert not _graph_exists(stamp)


def test_write_build_state_refuses_while_a_html_rerender_is_pending(
    built: Path,
) -> None:
    from graphify.exporters.html import _HTML_STALE_MARKER
    from graphify.watch import _write_build_state

    stamp = built / "graphify-out" / STAMP_NAME
    _graph_unlink(stamp)
    # The marker means graph.html does not describe the graph yet, so there is no
    # consistent state to promise the next run.
    (built / "graphify-out" / _HTML_STALE_MARKER).touch()

    _write_build_state(built / "graphify-out", {}, full_clean=True)

    assert not _graph_exists(stamp)
