"""`graphify extract --code-only` indexes code without an LLM key (#1734).

A mixed repo (code + docs) with no API key configured used to hard-fail on the
doc/paper/image files. `--code-only` skips the semantic pass so the code graph
still builds, and the no-key error now points users at the flag.
"""
from __future__ import annotations
from tests.graph_io import unlink as _graph_unlink, read_bytes as _graph_read_bytes, canonical_expected as _canonical_expected
from tests.graph_io import state_exists as _graph_exists, read_text as _graph_read_text, write_text as _graph_write_text

import os
import json
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
_KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL",
             "ANTHROPIC_API_KEY", "MOONSHOT_API_KEY", "DEEPSEEK_API_KEY")


def _mixed_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _graph_write_text(repo / "app.py", "def hello():\n    return 1\n")
    _graph_write_text(repo / "README.md", "# Design\n\nHow it works.\n")
    _graph_write_text(repo / "NOTES.txt", "Architecture notes and rationale.\n")
    return repo


def _run(repo: Path, *extra: str):
    env = {k: v for k, v in os.environ.items() if k not in _KEY_VARS}
    env["GRAPHIFY_OUT"] = str(repo / "graphify-out")
    return subprocess.run(
        [PYTHON, "-m", "graphify", "extract", ".", *extra],
        cwd=repo, capture_output=True, text=True, env=env,
    )


def test_code_only_succeeds_without_key(tmp_path):
    repo = _mixed_repo(tmp_path)
    r = _run(repo, "--code-only")
    assert r.returncode == 0, f"--code-only should succeed with no key: {r.stderr}"
    out = r.stdout + r.stderr
    assert "--code-only: skipping" in out
    graph = repo / "graphify-out" / "graph.grafeo"
    assert _graph_exists(graph), "code graph must still be written"
    import json
    g = json.loads(_graph_read_text(graph))
    labels = [n.get("label") for n in g["nodes"]]
    assert any(str(l).startswith("hello") for l in labels), "code was indexed"


def test_mixed_repo_without_key_errors_and_points_at_code_only(tmp_path):
    repo = _mixed_repo(tmp_path)
    r = _run(repo)  # no --code-only, no key
    assert r.returncode != 0, "mixed repo with no key should still error without the flag"
    assert "--code-only" in r.stderr, "the no-key error must point users at --code-only"


def test_extract_usage_advertises_code_only(tmp_path):
    """#2071: --code-only must be discoverable in the extract usage text, not only
    by triggering the no-key error. `graphify extract` with no path prints usage."""
    r = subprocess.run(
        [PYTHON, "-m", "graphify", "extract"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "--code-only" in r.stdout + r.stderr, (
        "extract usage must advertise --code-only (#2071)"
    )


def _run_relative_out(repo: Path, *extra: str):
    """Like _run but with a RELATIVE GRAPHIFY_OUT so --out/--output controls the
    parent dir (an absolute GRAPHIFY_OUT would override the flag)."""
    env = {k: v for k, v in os.environ.items() if k not in _KEY_VARS}
    env["GRAPHIFY_OUT"] = "graphify-out"
    return subprocess.run(
        [PYTHON, "-m", "graphify", "extract", ".", *extra],
        cwd=repo, capture_output=True, text=True, env=env,
    )


def test_output_flag_is_alias_of_out(tmp_path):
    """#2004 part 3: `--output DIR` was silently ignored on extract (output went
    to the default `<path>/graphify-out/`). It is now an alias of `--out`."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _graph_write_text(repo / "app.py", "def hello():\n    return 1\n")
    custom = tmp_path / "elsewhere"

    r = _run_relative_out(repo, "--code-only", "--no-cluster", "--output", str(custom))
    assert r.returncode == 0, r.stderr
    assert _graph_exists(custom / "graphify-out" / "graph.grafeo"), "--output was ignored (#2004)"
    assert not _graph_exists(repo / "graphify-out"), "output must not go to the default dir"


def test_output_flag_inline_form(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _graph_write_text(repo / "app.py", "def hello():\n    return 1\n")
    custom = tmp_path / "out2"
    r = _run_relative_out(repo, "--code-only", "--no-cluster", f"--output={custom}")
    assert r.returncode == 0, r.stderr
    assert _graph_exists(custom / "graphify-out" / "graph.grafeo")


def test_no_gitignore_indexes_vcs_ignored_code_but_keeps_graphifyignore(tmp_path):
    repo = tmp_path / "repo"
    generated = repo / "proj" / "deep" / "generated"
    generated.mkdir(parents=True)
    (repo / ".git" / "info").mkdir(parents=True)
    _graph_write_text(repo / ".git" / "info" / "exclude", "local/\n")
    _graph_write_text(repo / "proj" / ".gitignore", "generated/\n")
    _graph_write_text(repo / "proj" / ".graphifyignore", "hidden/\n")
    _graph_write_text(generated / "Gen.cs", "namespace N { public class Gen {} }\n")
    local = repo / "local"
    local.mkdir()
    _graph_write_text(local / "Local.cs", "namespace N { public class Local {} }\n")
    hidden = repo / "proj" / "hidden"
    hidden.mkdir()
    _graph_write_text(hidden / "Hidden.cs", "namespace N { public class Hidden {} }\n")

    result = _run(repo, "--no-gitignore", "--no-cluster")

    assert result.returncode == 0, result.stderr
    graph = json.loads(_graph_read_text(repo / "graphify-out" / "graph.grafeo"))
    sources = {Path(str(node.get("source_file", ""))).as_posix() for node in graph["nodes"]}
    assert any(source.endswith("proj/deep/generated/Gen.cs") for source in sources)
    assert any(source.endswith("local/Local.cs") for source in sources)
    assert not any(source.endswith("proj/hidden/Hidden.cs") for source in sources)


def test_no_gitignore_setting_persists_across_flagless_extract(tmp_path):
    """#1971 persistence: once --no-gitignore is set, a later flag-less
    `graphify extract` must NOT clobber it back to honoring .gitignore (which
    would make the git-ignored code silently disappear again)."""
    repo = tmp_path / "repo"
    gen = repo / "generated"
    gen.mkdir(parents=True)
    _graph_write_text(repo / ".gitignore", "generated/\n")
    _graph_write_text(repo / "app.py", "def hello():\n    return 1\n")
    _graph_write_text(gen / "Gen.py", "def gen():\n    return 2\n")

    def _sources():
        g = json.loads(_graph_read_text(repo / "graphify-out" / "graph.grafeo"))
        return {Path(str(n.get("source_file", ""))).as_posix() for n in g["nodes"]}

    r1 = _run(repo, "--no-gitignore", "--code-only", "--no-cluster")
    assert r1.returncode == 0, r1.stderr
    assert any(s.endswith("generated/Gen.py") for s in _sources())

    # A plain flag-less re-extract must keep the git-ignored file (setting persisted).
    r2 = _run(repo, "--code-only", "--no-cluster")
    assert r2.returncode == 0, r2.stderr
    assert any(s.endswith("generated/Gen.py") for s in _sources()), (
        "flag-less re-extract clobbered the persisted --no-gitignore setting (#1971)"
    )


def test_exclude_setting_persists_across_flagless_extract(tmp_path):
    repo = tmp_path / "repo"
    vendor = repo / "vendor"
    vendor.mkdir(parents=True)
    _graph_write_text(repo / "app.py", "def app():\n    return 1\n")
    _graph_write_text(vendor / "lib.py", "def vendor():\n    return 2\n")

    def _sources():
        graph = json.loads(_graph_read_text(repo / "graphify-out" / "graph.grafeo"))
        return {
            Path(str(node.get("source_file", ""))).as_posix()
            for node in graph["nodes"]
        }

    first = _run(
        repo, "--exclude", "vendor", "--code-only", "--no-cluster"
    )
    assert first.returncode == 0, first.stderr
    assert any(source.endswith("app.py") for source in _sources())
    assert not any(source.endswith("vendor/lib.py") for source in _sources())

    second = _run(repo, "--code-only", "--no-cluster")
    assert second.returncode == 0, second.stderr
    assert any(source.endswith("app.py") for source in _sources())
    assert not any(source.endswith("vendor/lib.py") for source in _sources())


def test_explicit_exclude_replaces_persisted_setting_with_custom_out(tmp_path):
    repo = tmp_path / "repo"
    vendor = repo / "vendor"
    generated = repo / "generated"
    vendor.mkdir(parents=True)
    generated.mkdir()
    _graph_write_text(repo / "app.py", "def app():\n    return 1\n")
    _graph_write_text(vendor / "lib.py", "def vendor():\n    return 2\n")
    _graph_write_text(generated / "gen.py", "def generated():\n    return 3\n")
    out_root = tmp_path / "custom-output"

    env = {key: value for key, value in os.environ.items() if key not in _KEY_VARS}
    env["GRAPHIFY_OUT"] = "graphify-out"

    def _run_extract(*extra: str):
        return subprocess.run(
            [
                PYTHON,
                "-m",
                "graphify",
                "extract",
                ".",
                "--out",
                str(out_root),
                "--code-only",
                "--no-cluster",
                *extra,
            ],
            cwd=repo,
            capture_output=True,
            text=True,
            env=env,
        )

    first = _run_extract("--exclude", "vendor")
    assert first.returncode == 0, first.stderr

    graph_out = out_root / "graphify-out"
    def _sources():
        graph = json.loads(_graph_read_text(graph_out / "graph.grafeo"))
        return {
            Path(str(node.get("source_file", ""))).as_posix()
            for node in graph["nodes"]
        }

    persisted = _run_extract("--force")
    assert persisted.returncode == 0, persisted.stderr
    assert any(source.endswith("app.py") for source in _sources())
    assert not any(source.endswith("vendor/lib.py") for source in _sources())
    assert any(source.endswith("generated/gen.py") for source in _sources())

    replacement = _run_extract("--exclude", "generated", "--force")
    assert replacement.returncode == 0, replacement.stderr
    sources = _sources()
    assert any(source.endswith("app.py") for source in sources)
    assert any(source.endswith("vendor/lib.py") for source in sources)
    assert not any(source.endswith("generated/gen.py") for source in sources)
    assert json.loads(_graph_read_text(graph_out / ".graphify_build.json")) == _canonical_expected({
        "excludes": ["generated"],
        "code_only": True,
    })


def test_extract_names_skipped_sensitive_files(tmp_path):
    """#2106 traceability: a file dropped by the sensitive-file filter is reported
    by NAME (not just a count), so a wrongly-flagged file is visible."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _graph_write_text(repo / "app.py", "def hello():\n    return 1\n")
    _graph_write_text(repo / "github_token.txt", "ghp_secretvalue\n")  # real secret -> skipped
    r = _run(repo, "--code-only", "--no-cluster")
    assert r.returncode == 0, r.stderr
    out = r.stdout + r.stderr
    assert "skipped as potentially sensitive" in out
    assert "github_token.txt" in out, "the skipped filename must be surfaced (#2106)"


def test_code_only_force_preserves_existing_semantic_layer(tmp_path):
    """#2923 regression: --code-only --force must not drop the existing semantic
    layer. The AST pass is fully replaced (full re-scan, semantic cache reads
    skipped) but the semantic pass is itself skipped, so doc/paper/image nodes
    from graph.grafeo must be carried forward. Before the fix this combination
    silently rewrote graph.grafeo with only the AST tier, losing every semantic
    node and every hyperedge connected to one.
    """
    repo = _mixed_repo(tmp_path)
    out = repo / "graphify-out"
    out.mkdir()
    graph = out / "graph.grafeo"
    # Seed a graph.grafeo as if a prior full extract with an LLM backend had run:
    # 2 AST nodes from app.py + 4 SEMANTIC nodes from README.md/NOTES.txt.
    _graph_write_text(graph, json.dumps({
        "nodes": [
            {"id": "app_py", "label": "app.py", "type": "file",
             "source_file": "app.py", "origin": "AST"},
            {"id": "app_hello", "label": "hello()", "type": "function",
             "source_file": "app.py", "origin": "AST"},
            {"id": "readme_md", "label": "readme.md", "type": "file",
             "source_file": "README.md", "origin": "SEMANTIC"},
            {"id": "readme_design", "label": "Design", "type": "concept",
             "source_file": "README.md", "origin": "SEMANTIC"},
            {"id": "notes_txt", "label": "NOTES.txt", "type": "file",
             "source_file": "NOTES.txt", "origin": "SEMANTIC"},
            {"id": "notes_architecture", "label": "Architecture", "type": "concept",
             "source_file": "NOTES.txt", "origin": "SEMANTIC"},
        ],
        "edges": [
            {"id": "e1", "source": "app_py", "target": "app_hello",
             "relation": "contains", "source_file": "app.py"},
            {"id": "e2", "source": "readme_md", "target": "readme_design",
             "relation": "concept_about", "source_file": "README.md"},
            {"id": "e3", "source": "notes_txt", "target": "notes_architecture",
             "relation": "concept_about", "source_file": "NOTES.txt"},
        ],
        "hyperedges": [],
        "input_tokens": 0,
        "output_tokens": 0,
    }))

    r = _run(repo, "--code-only", "--force", "--no-cluster")
    assert r.returncode == 0, r.stderr

    out_graph = json.loads(_graph_read_text(graph))
    semantic_labels = {n["label"] for n in out_graph["nodes"]
                       if n.get("origin") == "SEMANTIC"}
    semantic_source_files = {
        Path(str(n["source_file"])).name.lower()
        for n in out_graph["nodes"]
        if n.get("origin") == "SEMANTIC"
    }
    # Every seeded semantic node must survive. The AST pass may add new nodes
    # (or relabel existing ones — e.g. hello vs hello()) but it must not
    # silently drop the semantic tier.
    assert {"readme.md", "notes.txt"}.issubset(semantic_source_files), (
        "code-only --force erased the existing semantic layer (#2923); "
        f"semantic nodes remaining: {semantic_labels}"
    )
    # Hyperedges are also semantic tier; the seeded graph had none but the
    # AST re-extract must not have invented any non-semantic work, and the
    # surviving edges list must not have been wholesale replaced.
    assert "links" in out_graph, "native snapshots expose canonical links"
    # And the user-visible console line must explain why a semantic-layer-
    # preserving branch fired.
    assert "existing semantic layer preserved" in r.stdout + r.stderr, (
        "the --force --code-only print must announce the semantic-preserving branch"
    )


def test_code_only_force_prunes_removed_semantic_files(tmp_path):
    """#2923 follow-up: --code-only --force preserves surviving semantic nodes
    but must still prune semantic nodes for files that have been removed from
    disk (the doc/paper/image tier cannot outlive the corpus it indexes).
    """
    repo = _mixed_repo(tmp_path)
    out = repo / "graphify-out"
    out.mkdir()
    graph = out / "graph.grafeo"
    _graph_write_text(graph, json.dumps({
        "nodes": [
            {"id": "app_py", "label": "app.py", "type": "file",
             "source_file": "app.py", "origin": "AST"},
            {"id": "app_hello", "label": "hello()", "type": "function",
             "source_file": "app.py", "origin": "AST"},
            {"id": "notes_txt", "label": "NOTES.txt", "type": "file",
             "source_file": "NOTES.txt", "origin": "SEMANTIC"},
            {"id": "notes_architecture", "label": "Architecture", "type": "concept",
             "source_file": "NOTES.txt", "origin": "SEMANTIC"},
        ],
        "edges": [],
        "hyperedges": [],
        "input_tokens": 0,
        "output_tokens": 0,
    }))

    # Delete NOTES.txt between seed and re-run. The merge's graph_stale_sources
    # path must drop its semantic nodes because the file no longer exists.
    _graph_unlink(repo / "NOTES.txt")

    r = _run(repo, "--code-only", "--force", "--no-cluster")
    assert r.returncode == 0, r.stderr
    out_graph = json.loads(_graph_read_text(graph))
    remaining_sources = {
        Path(n["source_file"]).name
        for n in out_graph["nodes"]
        if n.get("origin") == "SEMANTIC"
    }
    assert "NOTES.txt" not in remaining_sources, (
        "NOTES.txt was deleted from disk; its semantic nodes must be pruned "
        "(#2923 follow-up)"
    )


def test_code_only_force_rescan_re_resolves_tsconfig_paths_and_preserves_semantics(tmp_path: Path):
    """#3125 regression: `extract --code-only --force` over an existing graph
    must perform a full code/AST re-scan so updated tsconfig paths take effect on
    unchanged .ts files, while preserving the existing semantic layer (#2923).
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    src = repo / "src"
    src.mkdir()
    _graph_write_text(src / "utils.ts", "export const helper = 42;\n")
    _graph_write_text(src / "index.ts", "import { helper } from '@/utils';\nconsole.log(helper);\n")
    _graph_write_text(repo / "tsconfig.json", json.dumps({"compilerOptions": {"target": "es2020"}}))

    # 1. Initial extract --code-only
    r1 = _run(repo, "--code-only", "--no-cluster")
    assert r1.returncode == 0, r1.stderr
    graph_path = repo / "graphify-out" / "graph.grafeo"
    g1 = json.loads(_graph_read_text(graph_path, encoding="utf-8"))
    edges1 = [(e["source"], e["target"]) for e in g1.get("edges", g1.get("links", []))]
    assert not any(src == "src_index" and ("src_utils" in tgt or "helper" in tgt) for src, tgt in edges1), (
        "alias import @/utils must not resolve without tsconfig paths mapping"
    )

    # 2. Seed a semantic file and node into graph.grafeo to verify semantic preservation (#2923)
    _graph_write_text(repo / "ARCH.md", "# Architecture\nDesign notes.\n")
    g1["nodes"].append({
        "id": "doc_arch", "label": "Architecture", "type": "concept",
        "source_file": "ARCH.md", "origin": "SEMANTIC", "_origin": "semantic"
    })
    _graph_write_text(graph_path, json.dumps(g1), encoding="utf-8")

    # 3. Modify only tsconfig.json to add baseUrl and paths alias
    _graph_write_text(repo / "tsconfig.json", json.dumps({
        "compilerOptions": {
            "target": "es2020",
            "baseUrl": ".",
            "paths": {"@/*": ["src/*"]}
        }
    }))

    # 4. Run extract --code-only --force
    r2 = _run(repo, "--code-only", "--force", "--no-cluster")
    assert r2.returncode == 0, r2.stderr

    g2 = json.loads(_graph_read_text(graph_path, encoding="utf-8"))
    edges2 = [(e["source"], e["target"]) for e in g2.get("edges", g2.get("links", []))]
    # Verify the import edge now exists
    assert any(src == "src_index" and ("src_utils" in tgt or "helper" in tgt) for src, tgt in edges2), (
        "extract --code-only --force must re-resolve alias imports after tsconfig paths change (#3125)"
    )

    # Verify the semantic entity was preserved
    nodes2 = {n.get("id") for n in g2.get("nodes", [])}
    assert "doc_arch" in nodes2, (
        "existing semantic nodes must survive --code-only --force (#2923/#3125)"
    )


def test_bare_path_without_backend_builds_code_and_defers_docs(tmp_path):
    repo = _mixed_repo(tmp_path)
    env = {k: v for k, v in os.environ.items() if k not in _KEY_VARS}
    env['GRAPHIFY_OUT'] = str(repo / 'graphify-out')
    result = subprocess.run([PYTHON, '-m', 'graphify', '.', '--no-cluster'],
                            cwd=repo, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert 'doc/paper/image file(s) deferred' in result.stdout
    graph = json.loads(_graph_read_text(repo / 'graphify-out' / 'graph.grafeo'))
    assert any(str(n.get('label', '')).startswith('hello') for n in graph['nodes'])
    # A subsequent strict extraction must still see the unprocessed documents.
    retry = _run(repo, '--no-cluster')
    assert retry.returncode != 0
    assert 'no LLM API key found' in retry.stderr
