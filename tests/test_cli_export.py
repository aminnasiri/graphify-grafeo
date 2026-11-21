"""Integration tests for graphify export subcommands and CLI commands.

Each test builds a minimal graph in a temp dir, runs the CLI command as a subprocess,
and asserts the expected output file exists and is non-empty / valid.
"""
from __future__ import annotations
from tests.graph_io import unlink as _graph_unlink, read_bytes as _graph_read_bytes, canonical_expected as _canonical_expected
from tests.graph_io import state_exists as _graph_exists, is_file as _graph_is_file, read_text as _graph_read_text, write_text as _graph_write_text
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

PYTHON = sys.executable
FIXTURES = Path(__file__).parent / "fixtures"


def _rename(src: Path, dst: Path) -> None:
    """Rename a directory, retrying briefly: on Windows a virus scanner or indexer
    can hold a just-written file open and fail the rename with PermissionError."""
    for attempt in range(20):
        try:
            src.rename(dst)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.25)


def _run(args: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYTHON, "-m", "graphify"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )


def _make_graph(tmp_path: Path) -> Path:
    """Build a minimal graph.grafeo + analysis/labels files in tmp_path/graphify-out/."""
    out = tmp_path / "graphify-out"
    out.mkdir()

    extraction = json.loads(_graph_read_text(FIXTURES / "extraction.json"))
    from graphify.build import build_from_json
    from graphify.cluster import cluster, score_all
    from graphify.analyze import god_nodes, surprising_connections
    from graphify.export import to_json

    G = build_from_json(extraction)
    communities = cluster(G)
    cohesion = score_all(G, communities)
    gods = god_nodes(G)
    surprises = surprising_connections(G, communities)
    labels = {cid: f"Community {cid}" for cid in communities}

    to_json(G, communities, str(out / "graph.grafeo"))

    analysis = {
        "communities": {str(k): v for k, v in communities.items()},
        "cohesion": {str(k): v for k, v in cohesion.items()},
        "gods": gods,
        "surprises": surprises,
    }
    _graph_write_text(out / ".graphify_analysis.json", json.dumps(analysis))
    _graph_write_text(out / ".graphify_labels.json", json.dumps({str(k): v for k, v in labels.items()}))
    return out


# ── graphify export html ─────────────────────────────────────────────────────

def test_export_html_creates_file(tmp_path):
    _make_graph(tmp_path)
    r = _run(["export", "html"], tmp_path)
    assert r.returncode == 0, r.stderr
    html = tmp_path / "graphify-out" / "graph.html"
    assert _graph_exists(html)
    assert html.stat().st_size > 0


def test_export_html_prefers_top_level_hyperedges(tmp_path):
    out = _make_graph(tmp_path)
    graph_path = out / "graph.grafeo"
    data = json.loads(_graph_read_text(graph_path))
    data["hyperedges"] = [
        {
            "id": "top-level",
            "label": "top-level edit",
            "nodes": ["n_transformer", "n_attention", "n_layernorm"],
        }
    ]
    data["graph"]["hyperedges"] = [
        {
            "id": "nested",
            "label": "stale nested copy",
            "nodes": ["n_attention", "n_layernorm", "n_concept_attn"],
        }
    ]
    _graph_write_text(graph_path, json.dumps(data))

    r = _run(["export", "html"], tmp_path)

    assert r.returncode == 0, r.stderr
    html = _graph_read_text(out / "graph.html")
    assert "top-level edit" in html
    assert "stale nested copy" not in html


def test_export_html_no_viz_removes_file(tmp_path):
    out = _make_graph(tmp_path)
    _graph_write_text(out / "graph.html", "<html/>")
    r = _run(["export", "html", "--no-viz"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert not _graph_exists(out / "graph.html")


def test_export_html_error_without_graph(tmp_path):
    r = _run(["export", "html"], tmp_path)
    assert r.returncode != 0


# ── graphify export obsidian ─────────────────────────────────────────────────

def test_export_obsidian_creates_vault(tmp_path):
    _make_graph(tmp_path)
    r = _run(["export", "obsidian"], tmp_path)
    assert r.returncode == 0, r.stderr
    vault = tmp_path / "graphify-out" / "obsidian"
    assert _graph_exists(vault)
    md_files = list(vault.glob("*.md"))
    assert len(md_files) > 0


def test_export_obsidian_custom_dir(tmp_path):
    _make_graph(tmp_path)
    custom = tmp_path / "my-vault"
    r = _run(["export", "obsidian", "--dir", str(custom)], tmp_path)
    assert r.returncode == 0, r.stderr
    assert _graph_exists(custom)
    assert len(list(custom.glob("*.md"))) > 0


# ── graphify export wiki ─────────────────────────────────────────────────────

def test_export_wiki_creates_articles(tmp_path):
    _make_graph(tmp_path)
    r = _run(["export", "wiki"], tmp_path)
    assert r.returncode == 0, r.stderr
    wiki = tmp_path / "graphify-out" / "wiki"
    assert _graph_exists(wiki)
    assert _graph_exists(wiki / "index.md")


def test_export_wiki_accepts_edges_only_graph_json(tmp_path):
    out = _make_graph(tmp_path)
    graph_path = out / "graph.grafeo"
    data = json.loads(_graph_read_text(graph_path))
    data["edges"] = data.pop("links")
    _graph_write_text(graph_path, json.dumps(data))

    r = _run(["export", "wiki"], tmp_path)

    assert r.returncode == 0, r.stderr
    assert _graph_exists(out / "wiki" / "index.md")


# ── graphify export graphml ──────────────────────────────────────────────────

def test_export_graphml_creates_file(tmp_path):
    _make_graph(tmp_path)
    r = _run(["export", "graphml"], tmp_path)
    assert r.returncode == 0, r.stderr
    gml = tmp_path / "graphify-out" / "graph.graphml"
    assert _graph_exists(gml)
    assert gml.stat().st_size > 0
    content = _graph_read_text(gml)
    assert "<graphml" in content


# ── graphify export neo4j (cypher) ───────────────────────────────────────────

def test_export_neo4j_creates_cypher(tmp_path):
    _make_graph(tmp_path)
    r = _run(["export", "neo4j"], tmp_path)
    assert r.returncode == 0, r.stderr
    cypher = tmp_path / "graphify-out" / "cypher.txt"
    assert _graph_exists(cypher)
    assert cypher.stat().st_size > 0
    content = _graph_read_text(cypher)
    assert "MERGE" in content or "CREATE" in content


# ── graphify export falkordb (cypher) ────────────────────────────────────────

def test_export_falkordb_creates_cypher(tmp_path):
    _make_graph(tmp_path)
    r = _run(["export", "falkordb"], tmp_path)
    assert r.returncode == 0, r.stderr
    cypher = tmp_path / "graphify-out" / "cypher.txt"
    assert _graph_exists(cypher)
    assert cypher.stat().st_size > 0
    content = _graph_read_text(cypher)
    assert "MERGE" in content or "CREATE" in content


def _dispatch_graphdb_push(
    monkeypatch,
    tmp_path: Path,
    *,
    subcmd: str,
    uri: str,
    password_env: str | None = None,
    user: str | None = None,
) -> tuple[dict[str, object], str]:
    """Run CLI dispatch with the database exporter replaced by a recorder."""
    project = tmp_path / "project with spaces"
    project.mkdir()
    out = _make_graph(project)
    monkeypatch.chdir(project)

    import graphify.export as export_module
    from graphify.cli import dispatch_command

    calls: dict[str, object] = {}

    def fake_push(G, *, uri, user, password, communities):
        calls.update(uri=uri, user=user, password=password, communities=communities)
        return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()}

    monkeypatch.setattr(export_module, f"push_to_{subcmd}", fake_push)
    secret_env = "FALKORDB_PASSWORD" if subcmd == "falkordb" else "NEO4J_PASSWORD"
    monkeypatch.delenv(secret_env, raising=False)
    if password_env is not None:
        monkeypatch.setenv(secret_env, password_env)

    args = ["graphify", "export", subcmd, "--graph", str(out / "graph.grafeo"), "--push", uri]
    if user is not None:
        args.extend(["--user", user])
    monkeypatch.setattr(sys, "argv", args)
    dispatch_command("export")
    return calls, uri


@pytest.mark.parametrize(
    ("subcmd", "uri", "user", "password_env", "expected_password"),
    [
        (
            "neo4j",
            "neo4j+s://db.example:8765/tenant/prod?mode=read%20write&tag=a%26b",
            "graph-writer",
            "neo4j-secret-not-for-output",
            "neo4j-secret-not-for-output",
        ),
        (
            "falkordb",
            "falkordb://db.example:6391/graphs/team%20alpha?tls=true&tag=a%3Bb",
            None,
            None,
            None,
        ),
        (
            "falkordb",
            "redis://db.example:6392/graphs/main?user=writer%40example.com",
            "redis-writer",
            "falkordb-secret-not-for-output",
            "falkordb-secret-not-for-output",
        ),
    ],
)
def test_graphdb_push_forwards_uri_and_credentials_without_a_database_connection(
    monkeypatch,
    tmp_path: Path,
    capsys,
    subcmd: str,
    uri: str,
    user: str | None,
    password_env: str | None,
    expected_password: str | None,
) -> None:
    calls, expected_uri = _dispatch_graphdb_push(
        monkeypatch,
        tmp_path,
        subcmd=subcmd,
        uri=uri,
        user=user,
        password_env=password_env,
    )

    assert calls["uri"] == expected_uri
    assert calls["user"] == (user if user is not None else "neo4j")
    assert calls["password"] == expected_password
    output = capsys.readouterr().out
    assert "db.example" not in output
    assert "secret-not-for-output" not in output


@pytest.mark.parametrize("subcmd", ["neo4j", "falkordb"])
def test_graphdb_export_without_push_never_calls_database(
    monkeypatch, tmp_path: Path, subcmd: str
) -> None:
    project = tmp_path / "project with spaces"
    project.mkdir()
    out = _make_graph(project)
    monkeypatch.chdir(project)

    import graphify.export as export_module
    from graphify.cli import dispatch_command

    def fail_if_pushed(*args, **kwargs):
        pytest.fail("export-only mode must not push to a database")

    monkeypatch.setattr(export_module, f"push_to_{subcmd}", fail_if_pushed)
    monkeypatch.setattr(
        sys,
        "argv",
        ["graphify", "export", subcmd, "--graph", str(out / "graph.grafeo")],
    )

    dispatch_command("export")

    cypher = out / "cypher.txt"
    assert _graph_is_file(cypher)
    assert "MERGE" in _graph_read_text(cypher, encoding="utf-8")


def test_export_reference_preserves_supplied_push_uris_safely() -> None:
    fragment = _graph_read_text(Path(__file__).resolve().parents[1]
        / "tools"
        / "skillgen"
        / "fragments"
        / "references"
        / "shared"
        / "exports.md", encoding="utf-8")
    neo4j = fragment[fragment.index("### Step 7 - Neo4j export") : fragment.index("### Step 7a - FalkorDB")]
    falkordb = fragment[fragment.index("### Step 7a - FalkorDB export") : fragment.index("### Step 7b - SVG export")]

    for section, push_heading, scheme_host, local_default in (
        (
            neo4j,
            "If `--neo4j-push",
            "bolt://remote-host:7687",
            "bolt://localhost:7687",
        ),
        (
            falkordb,
            "If `--falkordb-push",
            "falkordb://remote-host:6379",
            "falkordb://localhost:6379",
        ),
    ):
        push_instructions = section[section.index(push_heading) :]
        code_start = push_instructions.index("```bash\n") + len("```bash\n")
        code_end = push_instructions.index("\n```", code_start)
        push_example = push_instructions[code_start:code_end]

        assert "exact URI from the original `/graphify` command" in section
        assert "quote the URI for the shell that will execute the command" in section
        assert "`shlex.quote` for POSIX shells" in section
        assert "Never replace a supplied URI with localhost" in section
        assert "explicitly requested a push but supplied no URI" in section
        assert "Only push when this flag was explicitly given" in section
        assert "Do not print the URI or credentials" in section
        assert scheme_host in section
        assert scheme_host in push_example
        assert local_default not in push_example

    assert "Default URI is `bolt://localhost:7687`" in neo4j
    assert "Default URI is `falkordb://localhost:6379`" in falkordb
    assert "NEO4J_PASSWORD" in neo4j
    assert "FALKORDB_PASSWORD" in falkordb
    assert "--neo4j-push" in neo4j
    assert "--falkordb-push" in falkordb


# ── graphify query ───────────────────────────────────────────────────────────

def test_query_returns_output(tmp_path):
    _make_graph(tmp_path)
    r = _run(["query", "Transformer"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert len(r.stdout) > 0


def test_query_dfs_flag(tmp_path):
    _make_graph(tmp_path)
    r = _run(["query", "Transformer", "--dfs"], tmp_path)
    assert r.returncode == 0, r.stderr


def test_query_budget_flag(tmp_path):
    _make_graph(tmp_path)
    r = _run(["query", "Transformer", "--budget", "500"], tmp_path)
    assert r.returncode == 0, r.stderr


def test_query_missing_graph_fails(tmp_path):
    r = _run(["query", "anything"], tmp_path)
    assert r.returncode != 0


def test_query_uses_graphify_out_env(tmp_path):
    out = _make_graph(tmp_path)
    custom_out = tmp_path / "custom-graph"
    _rename(out, custom_out)
    env = os.environ.copy()
    env["GRAPHIFY_OUT"] = custom_out.name

    r = _run(["query", "Transformer"], tmp_path, env=env)

    assert r.returncode == 0, r.stderr
    assert len(r.stdout) > 0


def test_extract_writes_to_graphify_out_env(tmp_path):
    """#1423: `graphify extract` honours GRAPHIFY_OUT for where it WRITES, not only
    where readers look — previously it hardcoded graphify-out/ and ignored the
    override. Code-only corpus, so no LLM backend is needed."""
    _graph_write_text(tmp_path / "m.py", "def a():\n    return b()\n\n\ndef b():\n    return 1\n")
    env = os.environ.copy()
    env["GRAPHIFY_OUT"] = "custom-out"

    r = _run(["extract", "."], tmp_path, env=env)

    assert r.returncode == 0, r.stderr
    assert _graph_exists(tmp_path / "custom-out" / "graph.grafeo"), r.stdout
    assert _graph_exists(tmp_path / "custom-out" / "manifest.json")
    # The default dir must NOT be created when the override is set.
    assert not _graph_exists(tmp_path / "graphify-out"), "extract ignored GRAPHIFY_OUT and wrote graphify-out/"
    # Manifest keys are relative to the scan root (portable) — #1417.
    keys = list(json.loads(_graph_read_text(tmp_path / "custom-out" / "manifest.json")).keys())
    assert keys == ["m.py"], keys


# ── graphify path ────────────────────────────────────────────────────────────

def test_path_runs_without_error(tmp_path):
    _make_graph(tmp_path)
    r = _run(["path", "Transformer", "LayerNorm"], tmp_path)
    # May find or not find a path — either is valid, should not crash
    assert r.returncode == 0, r.stderr


def test_path_missing_graph_fails(tmp_path):
    r = _run(["path", "a", "b"], tmp_path)
    assert r.returncode != 0


def test_path_uses_graphify_out_env(tmp_path):
    out = _make_graph(tmp_path)
    custom_out = tmp_path / "custom-graph"
    _rename(out, custom_out)
    env = os.environ.copy()
    env["GRAPHIFY_OUT"] = custom_out.name

    r = _run(["path", "Transformer", "LayerNorm"], tmp_path, env=env)

    assert r.returncode == 0, r.stderr


# ── graphify path direction (#2487) ─────────────────────────────────────────

def _write_path_graph(tmp_path: Path, nodes: list[str], links: list[dict]) -> Path:
    """Write a minimal hand-rolled directed graph.grafeo for path-direction tests."""
    out = tmp_path / "graphify-out"
    out.mkdir()
    _graph_write_text(out / "graph.grafeo", json.dumps({
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": [{"id": n, "label": n} for n in nodes],
        "links": links,
    }))
    return out


def _calls(src: str, tgt: str) -> dict:
    return {"source": src, "target": tgt, "relation": "calls"}


def test_path_directed_respects_direction(tmp_path):
    _write_path_graph(
        tmp_path, ["alpha", "beta", "gamma"],
        [_calls("alpha", "beta"), _calls("beta", "gamma")],
    )
    r = _run(["path", "alpha", "gamma"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("-->") == 2
    assert "<--" not in r.stdout
    # Explicit --directed is the same as the default.
    r2 = _run(["path", "alpha", "gamma", "--directed"], tmp_path)
    assert r2.returncode == 0, r2.stderr
    assert r2.stdout == r.stdout


def test_path_directed_backwards_is_no_path(tmp_path):
    # Default-change guard (#2487): a plain `path` with no flag is directed,
    # so walking the chain backwards must report no directed path.
    _write_path_graph(
        tmp_path, ["alpha", "beta", "gamma"],
        [_calls("alpha", "beta"), _calls("beta", "gamma")],
    )
    r = _run(["path", "gamma", "alpha"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert "No directed path found" in r.stdout
    assert "--undirected" in r.stdout
    assert "-->" not in r.stdout
    assert "<--" not in r.stdout


def test_path_undirected_flag_opt_in(tmp_path):
    _write_path_graph(
        tmp_path, ["alpha", "beta", "gamma"],
        [_calls("alpha", "beta"), _calls("beta", "gamma")],
    )
    r = _run(["path", "gamma", "alpha", "--undirected"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert "Shortest path (2 hops)" in r.stdout
    assert r.stdout.count("<--calls--") == 2
    assert "-->" not in r.stdout


def test_path_directed_legacy_markers(tmp_path):
    # Legacy canonicalized file: the persisted arc is flipped (beta->alpha) but
    # the _src/_tgt markers carry the true direction alpha->beta. Direction
    # truth must come from the markers, not the raw arc order (#2309/#2487).
    _write_path_graph(
        tmp_path, ["alpha", "beta"],
        [{"source": "beta", "target": "alpha",
          "_src": "alpha", "_tgt": "beta", "relation": "calls"}],
    )
    r = _run(["path", "alpha", "beta"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert "Shortest path (1 hops)" in r.stdout
    assert "-->" in r.stdout
    assert "<--" not in r.stdout
    r2 = _run(["path", "beta", "alpha"], tmp_path)
    assert r2.returncode == 0, r2.stderr
    assert "No directed path found" in r2.stdout


def test_path_directed_deterministic(tmp_path):
    # Diamond with two equal-length directed routes: the chosen route must not
    # depend on the process hash seed (#2074 discipline for the digraph too).
    _write_path_graph(
        tmp_path, ["start", "left", "right", "goal"],
        [_calls("start", "left"), _calls("left", "goal"),
         _calls("start", "right"), _calls("right", "goal")],
    )
    outputs = []
    for seed in ("0", "1"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        r = _run(["path", "start", "goal"], tmp_path, env=env)
        assert r.returncode == 0, r.stderr
        assert "-->" in r.stdout
        outputs.append(r.stdout)
    assert outputs[0] == outputs[1]


def test_path_flags_mutually_exclusive(tmp_path):
    _write_path_graph(
        tmp_path, ["alpha", "beta"], [_calls("alpha", "beta")],
    )
    r = _run(["path", "alpha", "beta", "--directed", "--undirected"], tmp_path)
    assert r.returncode != 0
    assert "mutually exclusive" in r.stderr


# ── graphify explain ─────────────────────────────────────────────────────────

def test_explain_runs_without_error(tmp_path):
    _make_graph(tmp_path)
    r = _run(["explain", "Transformer"], tmp_path)
    assert r.returncode == 0, r.stderr


def test_explain_missing_graph_fails(tmp_path):
    r = _run(["explain", "anything"], tmp_path)
    assert r.returncode != 0


def test_explain_uses_graphify_out_env(tmp_path):
    out = _make_graph(tmp_path)
    custom_out = tmp_path / "custom-graph"
    _rename(out, custom_out)
    env = os.environ.copy()
    env["GRAPHIFY_OUT"] = custom_out.name

    r = _run(["explain", "Transformer"], tmp_path, env=env)

    assert r.returncode == 0, r.stderr


# ── graphify export unknown format ───────────────────────────────────────────

def test_export_unknown_format_fails(tmp_path):
    r = _run(["export", "pdf"], tmp_path)
    assert r.returncode != 0


def test_update_no_cluster_writes_raw_graph(tmp_path):
    src = tmp_path / "sample.py"
    _graph_write_text(src, "def f():\n    return 1\n", encoding="utf-8")

    r = _run(["update", ".", "--no-cluster"], tmp_path)
    assert r.returncode == 0, r.stderr

    graph_path = tmp_path / "graphify-out" / "graph.grafeo"
    assert _graph_exists(graph_path)
    data = json.loads(_graph_read_text(graph_path, encoding="utf-8"))
    assert "nodes" in data and "links" in data
    assert all("community" not in node for node in data["nodes"])


def test_extract_no_cluster_marks_parallel_links_as_multigraph(tmp_path):
    src = tmp_path / "sample.ts"
    _graph_write_text(src, "function owner() { function child() { return 1 } return child() }\n", encoding="utf-8")

    r = _run(["extract", ".", "--code-only", "--no-cluster"], tmp_path)
    assert r.returncode == 0, r.stderr

    data = json.loads(_graph_read_text(tmp_path / "graphify-out" / "graph.grafeo"))
    assert data.get("multigraph") is True


# Regression test for #934 - cluster-only crashes when graphify-out/ doesn't exist

def test_cluster_only_creates_output_dir_when_missing(tmp_path):
    """cluster-only must not crash with FileNotFoundError when graphify-out/ is absent (#934)."""
    # Build graph.grafeo somewhere other than the default graphify-out/ location
    # so we can point --graph at it while graphify-out/ doesn't exist yet.
    graph_src = tmp_path / "backup" / "graph.grafeo"
    graph_src.parent.mkdir()

    out_dir = _make_graph(tmp_path)
    graph_json = out_dir / "graph.grafeo"
    # Simulate user archiving the output dir before re-clustering
    import shutil
    shutil.copy(graph_json, graph_src)
    shutil.rmtree(out_dir)

    assert not _graph_exists(tmp_path / "graphify-out")

    r = _run(["cluster-only", ".", "--graph", str(graph_src), "--no-viz"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert _graph_exists(tmp_path / "graphify-out" / "GRAPH_REPORT.md")


def test_cluster_only_graph_in_graphify_out_writes_beside_it(tmp_path):
    """#1747 Case 2: `cluster-only --graph <elsewhere>/graphify-out/graph.grafeo`
    must write GRAPH_REPORT.md and the re-clustered graph beside that graph, not
    into a stray graphify-out/ in the CWD."""
    project = tmp_path / "project"
    project.mkdir()
    out_dir = _make_graph(project)  # project/graphify-out/graph.grafeo

    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    r = _run(
        ["cluster-only", ".", "--graph", str(out_dir / "graph.grafeo"), "--no-viz", "--no-label"],
        cwd,
    )
    assert r.returncode == 0, r.stderr
    assert _graph_exists(out_dir / "GRAPH_REPORT.md")          # beside --graph
    assert not _graph_exists(cwd / "graphify-out")             # no CWD pollution


def test_extract_out_does_not_pollute_corpus(tmp_path):
    """#1747 Case 1: `extract <corpus> --out <elsewhere>` must not leave a stray
    graphify-out/ (cache, stat-index) inside the scanned corpus."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _graph_write_text(corpus / "a.py", "def main():\n    return 1\n")
    out = tmp_path / "scratch"

    r = _run(
        ["extract", str(corpus), "--out", str(out), "--no-cluster", "--code-only"],
        tmp_path,
    )
    assert r.returncode == 0, r.stderr
    assert _graph_exists(out / "graphify-out" / "graph.grafeo")   # graph in --out
    assert not _graph_exists(corpus / "graphify-out")           # corpus untouched


# Regression test for #1027 - cluster-only must remap labels via node overlap

def test_cluster_only_persists_analysis_sidecar(tmp_path):
    """cluster-only must refresh .graphify_analysis.json alongside graph.grafeo.

    Downstream export commands use the sidecar for community membership and
    should not see stale or missing community analysis after a recluster.
    """
    out = _make_graph(tmp_path)
    analysis_path = out / ".graphify_analysis.json"
    _graph_unlink(analysis_path)

    r = _run(["cluster-only", ".", "--no-viz"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert _graph_exists(analysis_path)

    analysis = json.loads(_graph_read_text(analysis_path, encoding="utf-8"))
    assert analysis["communities"]
    assert analysis["cohesion"]
    assert "gods" in analysis
    assert "surprises" in analysis
    assert "questions" in analysis

    graph = json.loads(_graph_read_text(out / "graph.grafeo", encoding="utf-8"))
    graph_cids = {
        str(node["community"])
        for node in graph.get("nodes", [])
        if node.get("community") is not None
    }
    assert graph_cids == set(analysis["communities"])


def test_cluster_only_remaps_labels_to_previous_cids(tmp_path):
    """cluster-only must invoke remap_communities_to_previous so the existing
    .graphify_labels.json keeps tracking the same conceptual communities after
    re-clustering. Without the remap call, Leiden's size-descending cid order
    re-applies labels by raw index and they silently misalign with cluster
    contents (#1027). Mirror of the watch/update fix from #822.
    """
    out = _make_graph(tmp_path)
    graph_json = out / "graph.grafeo"
    labels_json = out / ".graphify_labels.json"

    # Tag every node with an out-of-band community id and write a labels file
    # keyed on those ids. After cluster-only, at least one of those sentinel
    # ids must survive in the labels file (= remap succeeded by node overlap).
    # If the cluster-only branch skips remap, Leiden returns small ints
    # (0, 1, ...) and the sentinel keys disappear entirely.
    g = json.loads(_graph_read_text(graph_json, encoding="utf-8"))
    nodes = g.get("nodes", [])
    assert len(nodes) >= 4, "fixture must have enough nodes to form 2+ communities"
    sentinel_a, sentinel_b = 4242, 9999
    half = len(nodes) // 2
    for i, n in enumerate(nodes):
        n["community"] = sentinel_a if i < half else sentinel_b
    _graph_write_text(graph_json, json.dumps(g), encoding="utf-8")
    _graph_write_text(labels_json, json.dumps({str(sentinel_a): "First Group", str(sentinel_b): "Second Group"}), encoding="utf-8")

    r = _run(["cluster-only", ".", "--no-viz"], tmp_path)
    assert r.returncode == 0, r.stderr

    # Real signal: labels.json keys must align with the community ids actually
    # written to graph.grafeo's per-node community attribute. Without remap,
    # Leiden returns small cids (0, 1, ...) but labels.json still carries the
    # old sentinel keys, so the intersection is empty and labels are orphaned.
    final_graph = json.loads(_graph_read_text(graph_json, encoding="utf-8"))
    final_labels = json.loads(_graph_read_text(labels_json, encoding="utf-8"))
    actual_cids = {n.get("community") for n in final_graph.get("nodes", [])}
    label_cids = {int(k) for k in final_labels.keys()}
    overlap = actual_cids & label_cids
    assert overlap, (
        f"After cluster-only with prior labels keyed on cids {label_cids}, at "
        f"least one of those cids must still appear in graph.grafeo's community "
        f"attribute ({actual_cids}). Without remap_communities_to_previous "
        f"(#1027) Leiden renumbers communities to 0,1,... and the prior labels "
        f"become orphaned. Final labels: {final_labels}"
    )


# ── communities-fallback when .graphify_analysis.json is absent ──────────────
# The watch / post-commit rebuild path only writes graph.grafeo + GRAPH_REPORT.md;
# it does NOT regenerate .graphify_analysis.json. The full `graphify extract`
# pipeline also removes its temp files at the end of the run on some skill
# workflows. In both cases the per-node `community` attribute is intact on
# every node in graph.grafeo — that's the source of truth `to_json` writes.
# Without these tests, `graphify export html|obsidian|wiki|svg|graphml|neo4j`
# silently bails or generates a degraded artifact whenever the sidecar is
# missing, even though the data is right there.

def test_export_html_falls_back_to_node_community_attribute(tmp_path):
    """When .graphify_analysis.json is absent, export html should reconstruct
    communities from the per-node attribute in graph.grafeo rather than bailing
    out with 'Single community - aggregated view not useful.'.
    """
    out = _make_graph(tmp_path)
    # Simulate the watch-rebuild / cleanup case: graph.grafeo + labels survive,
    # analysis sidecar is gone.
    _graph_unlink(out / ".graphify_analysis.json")

    r = _run(["export", "html"], tmp_path)
    assert r.returncode == 0, r.stderr
    html = out / "graph.html"
    assert _graph_exists(html), "graph.html should be generated from the fallback"
    assert html.stat().st_size > 0
    # The success message comes from to_html — confirm we're not hitting the
    # "Single community" bail-out path.
    assert "Single community" not in r.stdout
    assert "Single community" not in r.stderr


def test_export_html_fallback_recovers_multiple_communities(tmp_path):
    """Stronger assertion: the reconstructed `communities` dict should have the
    SAME community count as the analysis sidecar would, so downstream code
    (aggregation thresholds, member counts) sees identical input.
    """
    out = _make_graph(tmp_path)

    # Read the canonical community count from the analysis sidecar
    analysis = json.loads(_graph_read_text(out / ".graphify_analysis.json", encoding="utf-8"))
    expected_count = len(analysis["communities"])

    # And the count we'd reconstruct from graph.grafeo's node attributes
    graph = json.loads(_graph_read_text(out / "graph.grafeo", encoding="utf-8"))
    reconstructed_cids = {
        n["community"] for n in graph.get("nodes", [])
        if n.get("community") is not None
    }
    assert len(reconstructed_cids) == expected_count, (
        f"reconstruction would lose communities: sidecar={expected_count} vs "
        f"graph.grafeo={len(reconstructed_cids)}"
    )

    # Now remove the sidecar and confirm the CLI still succeeds end-to-end.
    _graph_unlink(out / ".graphify_analysis.json")
    r = _run(["export", "html"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert _graph_exists(out / "graph.html")


def test_export_html_no_community_data_at_all_still_succeeds(tmp_path):
    """If a graph.grafeo was somehow written without any per-node `community`
    attribute (older versions of to_json, hand-built graphs), the fallback
    should produce an empty communities dict and the renderer should still
    not crash. Whether the aggregated view is useful is a separate question.
    """
    out = _make_graph(tmp_path)
    _graph_unlink(out / ".graphify_analysis.json")

    # Strip the community attribute from every node
    graph_path = out / "graph.grafeo"
    graph = json.loads(_graph_read_text(graph_path, encoding="utf-8"))
    for n in graph.get("nodes", []):
        n.pop("community", None)
    _graph_write_text(graph_path, json.dumps(graph), encoding="utf-8")

    r = _run(["export", "html"], tmp_path)
    # Should NOT crash. It may print a warning and skip rendering, but exit
    # code stays clean — same behaviour as the pre-fallback empty-communities
    # path, just no longer silently failing on the common case.
    assert r.returncode == 0, r.stderr


# ── #2386: sidecar exists but is STALE, not just absent ──────────────────────
# update/watch advance graph.grafeo's per-node community attribute but never
# regenerate .graphify_analysis.json, so it can describe an earlier
# clustering pass while still being present. That looked identical to a
# fresh sidecar from the outside and kept winning over the correct data
# sitting in graph.grafeo.

def test_export_html_prefers_fresh_data_when_sidecar_is_stale(tmp_path):
    out = _make_graph(tmp_path)
    analysis_path = out / ".graphify_analysis.json"
    analysis = json.loads(_graph_read_text(analysis_path, encoding="utf-8"))
    # Simulate staleness the way the issue describes: the sidecar's node id
    # set no longer matches graph.grafeo's (a node the sidecar never saw, or
    # one it references that graph.grafeo no longer has).
    analysis["communities"] = {"0": ["a_ghost_node_id_not_in_the_graph"]}
    _graph_write_text(analysis_path, json.dumps(analysis))

    r = _run(["export", "html"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert "is stale" in r.stderr
    assert _graph_exists(out / "graph.html")


def test_export_wiki_recomputes_cohesion_when_sidecar_is_stale(tmp_path):
    out = _make_graph(tmp_path)
    analysis_path = out / ".graphify_analysis.json"
    analysis = json.loads(_graph_read_text(analysis_path, encoding="utf-8"))
    analysis["communities"] = {"0": ["a_ghost_node_id_not_in_the_graph"]}
    # A cohesion value that could never be a real score (score_all returns
    # values in a bounded range), so if it survives into the wiki output
    # unchanged, the stale sidecar won instead of being recomputed.
    analysis["cohesion"] = {"0": 999999.0}
    _graph_write_text(analysis_path, json.dumps(analysis))

    r = _run(["export", "wiki"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert "is stale" in r.stderr
    wiki_dir = out / "wiki"
    assert _graph_exists(wiki_dir)
    combined = "\n".join(_graph_read_text(p, encoding="utf-8") for p in wiki_dir.glob("*.md"))
    assert "999999" not in combined, "stale cohesion value leaked into the wiki export"


def test_export_html_uses_sidecar_when_it_still_matches(tmp_path):
    """Negative control: an up to date sidecar must not trigger the stale
    path or its warning."""
    out = _make_graph(tmp_path)

    r = _run(["export", "html"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert "is stale" not in r.stderr
    assert _graph_exists(out / "graph.html")


def test_export_html_detects_stale_sidecar_with_same_nodes_different_partition(tmp_path):
    """A merge, a split, or a node moving from one community to another can
    leave the overall node id set unchanged while still describing a
    different partition -- comparing only the flat node-id set missed this
    exact shape of staleness."""
    out = _make_graph(tmp_path)
    analysis_path = out / ".graphify_analysis.json"
    analysis = json.loads(_graph_read_text(analysis_path, encoding="utf-8"))
    communities = analysis["communities"]
    assert len(communities) >= 2, "fixture must have at least two communities to prove this"
    all_nodes = [n for nodes in communities.values() for n in nodes]
    analysis["communities"] = {"0": all_nodes}
    _graph_write_text(analysis_path, json.dumps(analysis))

    r = _run(["export", "html"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert "is stale" in r.stderr
    assert _graph_exists(out / "graph.html")


def test_graph_json_node_ids_are_portable_across_checkout_paths(tmp_path):
    """#1789: the committed graph.grafeo's node ids must be relative to the scan
    root — not embed the absolute path — so the same repo yields identical ids
    on any machine/checkout and leaks no local username/home."""
    def _build(root: Path):
        (root / "pkg").mkdir(parents=True)
        _graph_write_text(root / "pkg" / "mod.py", "def f(): return 1\n")
        _graph_write_text(root / "pkg" / "app.py", "from pkg.mod import f\ndef g(): return f()\n")
        r = _run(["extract", ".", "--code-only", "--no-cluster"], root)
        assert r.returncode == 0, r.stderr
        data = json.loads(_graph_read_text(root / "graphify-out" / "graph.grafeo"))
        return sorted(n["id"] for n in data["nodes"])

    a = _build(tmp_path / "alice_home" / "proj")
    b = _build(tmp_path / "bob_elsewhere" / "checkout" / "proj")
    assert a == b, f"node ids differ across checkout paths: {a} vs {b}"
    leak = {"alice_home", "bob_elsewhere", "checkout", "tmp", "private", "users", "home", "var"}
    assert not any(part in leak for ident in a for part in ident.split("_")), \
        f"node id embeds an absolute-path component: {a}"


# ── cluster-only silent failures (#2534) + refused-write exit code (#2522) ───


def test_cluster_only_reports_failure_when_write_is_refused(tmp_path, monkeypatch, capsys):
    """A rejected graph commit must leave the previous report and labels intact."""
    out = _make_graph(tmp_path)
    report = out / "GRAPH_REPORT.md"
    report.write_text("PREVIOUS REPORT\n")
    from graphify.store.io import connection
    with connection(out / "graph.grafeo") as store:
        before = store.snapshot()
    import graphify.cli as cli_module
    monkeypatch.setattr("graphify.export.to_json", lambda *args, **kwargs: False)
    monkeypatch.setattr(sys, "argv", ["graphify", "cluster-only", str(tmp_path), "--no-label", "--no-viz"])
    with pytest.raises(SystemExit) as error:
        cli_module.dispatch_command("cluster-only")
    assert error.value.code == 1
    assert "NOT written" in capsys.readouterr().err
    with connection(out / "graph.grafeo") as store:
        assert store.snapshot() == before
    assert report.read_text() == "PREVIOUS REPORT\n"


def test_cluster_only_happy_path_exits_zero(tmp_path):
    """Guard for the #2522 reordering: the normal re-cluster still succeeds."""
    _make_graph(tmp_path)
    r = _run(["cluster-only", ".", "--no-viz"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert "Done -" in r.stdout, r.stdout
    assert "communities" in r.stdout


def test_cluster_only_preserves_parallel_edges(tmp_path):
    """#3999: cluster-only reloads graph.grafeo into a simple Graph (one edge per
    node pair) to re-cluster, then used to write THAT back out, silently
    dropping a second edge (e.g. imports + calls) between a pair that stayed
    connected by the other. Community is a node attribute; re-clustering must
    never change the edge set."""
    out = tmp_path / "graphify-out"
    out.mkdir()
    _graph_write_text(out / "graph.grafeo", json.dumps({
        "directed": False, "multigraph": False, "graph": {},
        "nodes": [
            {"id": "a", "label": "a", "file_type": "code", "source_file": "a.py"},
            {"id": "b", "label": "b", "file_type": "code", "source_file": "a.py"},
            {"id": "c", "label": "c", "file_type": "code", "source_file": "a.py"},
        ],
        "links": [
            {"source": "a", "target": "b", "relation": "imports", "confidence": "EXTRACTED"},
            {"source": "a", "target": "b", "relation": "calls", "confidence": "EXTRACTED"},
            {"source": "b", "target": "c", "relation": "calls", "confidence": "EXTRACTED"},
        ],
    }))

    r = _run(["cluster-only", ".", "--no-viz", "--no-label"], tmp_path)
    assert r.returncode == 0, r.stderr

    data = json.loads(_graph_read_text(out / "graph.grafeo"))
    pairs = sorted((l["source"], l["target"], l["relation"]) for l in data["links"])
    assert pairs == [("a", "b", "calls"), ("a", "b", "imports"), ("b", "c", "calls")], pairs


def test_to_json_original_links_drops_only_a_dangling_endpoint(tmp_path):
    """#3999: original_links writes back every original link verbatim,
    except one whose endpoint no longer exists in G."""
    from graphify.build import build_from_json
    from graphify.cluster import cluster
    from graphify.export import to_json

    extraction = {
        "directed": False,
        "nodes": [
            {"id": "a", "label": "a", "file_type": "code", "source_file": "a.py"},
            {"id": "b", "label": "b", "file_type": "code", "source_file": "a.py"},
        ],
        "links": [
            {"source": "a", "target": "b", "relation": "imports", "confidence": "EXTRACTED"},
            {"source": "a", "target": "b", "relation": "calls", "confidence": "EXTRACTED"},
        ],
    }
    G = build_from_json(extraction)
    communities = cluster(G)
    out = tmp_path / "graph.grafeo"
    original_links = list(extraction["links"]) + [
        {"source": "a", "target": "gone", "relation": "calls", "confidence": "EXTRACTED"},
    ]
    assert to_json(G, communities, str(out), original_links=original_links)

    data = json.loads(_graph_read_text(out))
    pairs = sorted((l["source"], l["target"], l["relation"]) for l in data["links"])
    assert pairs == [("a", "b", "calls"), ("a", "b", "imports")], pairs


def test_cluster_only_warns_when_labeling_flags_are_ignored(tmp_path):
    """#2534 case 1: with a saved .graphify_labels.json the reuse branch never
    calls the LLM, so --backend/--model/--batch-size used to be silently
    ignored while exiting 0. Reuse stays a success (exit 0), but the ignored
    flags must be named on stderr."""
    _make_graph(tmp_path)  # persists .graphify_labels.json -> reuse branch

    r = _run(["cluster-only", ".", "--backend", "openai", "--no-viz"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert "--backend" in r.stderr and "ignored" in r.stderr, r.stderr
    assert "reusing saved labels" in r.stderr, r.stderr
    # the reuse branch must NOT have gone through the LLM labeling path
    assert "Labeling communities" not in r.stdout, r.stdout


def test_cluster_only_reuse_without_labeling_flags_stays_quiet(tmp_path):
    """Control for the #2534 case-1 warning: plain reuse prints no flag warning."""
    _make_graph(tmp_path)
    r = _run(["cluster-only", ".", "--no-viz"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert "ignored" not in r.stderr, r.stderr
    assert "reusing saved labels" not in r.stderr, r.stderr


# ── cluster-only must not re-stamp built_at_commit from the shell's cwd ──────


def _init_git_repo(path: Path, message: str) -> str:
    """git init + one empty commit; returns the HEAD sha."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "-c", "user.email=t@test", "-c", "user.name=t",
         "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-q", "-m", message],
        check=True, capture_output=True,
    )
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def test_cluster_only_preserves_built_at_commit_from_other_repo_cwd(tmp_path):
    """#2534 case 4: cluster-only re-clusters an EXISTING graph, so the
    extract-time built_at_commit must survive — running it from a DIFFERENT
    repo used to re-stamp graph.grafeo with that repo's HEAD."""
    target = tmp_path / "target"
    commit_x = _init_git_repo(target, "init target")
    out = _make_graph(target)
    graph_json = out / "graph.grafeo"
    data = json.loads(_graph_read_text(graph_json, encoding="utf-8"))
    data["built_at_commit"] = commit_x
    _graph_write_text(graph_json, json.dumps(data), encoding="utf-8")

    other = tmp_path / "other"
    commit_y = _init_git_repo(other, "init other")
    assert commit_x != commit_y

    r = _run(["cluster-only", str(target), "--no-viz"], other)
    assert r.returncode == 0, r.stderr
    final = json.loads(_graph_read_text(graph_json, encoding="utf-8"))
    assert final.get("built_at_commit") == commit_x, (
        f"built_at_commit re-stamped from the shell's cwd: expected {commit_x}, "
        f"got {final.get('built_at_commit')} (other repo HEAD is {commit_y})"
    )


def test_cluster_only_preserves_built_at_commit_from_non_repo_cwd(tmp_path):
    """#2534 case 4: from a cwd that is not a git repo at all, the stamp must
    still be preserved — not dropped."""
    target = tmp_path / "target"
    commit_x = _init_git_repo(target, "init target")
    out = _make_graph(target)
    graph_json = out / "graph.grafeo"
    data = json.loads(_graph_read_text(graph_json, encoding="utf-8"))
    data["built_at_commit"] = commit_x
    _graph_write_text(graph_json, json.dumps(data), encoding="utf-8")

    plain = tmp_path / "plain"
    plain.mkdir()
    r = _run(["cluster-only", str(target), "--no-viz"], plain)
    assert r.returncode == 0, r.stderr
    final = json.loads(_graph_read_text(graph_json, encoding="utf-8"))
    assert final.get("built_at_commit") == commit_x


# ── graphify query names its graph (#2789) ───────────────────────────────────

def test_query_command_header_names_the_graph(tmp_path):
    """End-to-end: the CLI `query` command must actually wire the resolved graph
    path into the header. The header logic is unit-tested in
    test_query_names_its_graph.py, but only a real subprocess run proves the CLI
    call site passes graph_path through (the wiring that was the point of #2789)."""
    _make_graph(tmp_path)
    r = _run(["query", "Transformer"], tmp_path)
    assert r.returncode == 0, r.stderr
    first_line = r.stdout.splitlines()[0]
    assert first_line.startswith("Graph: graphify-out/graph.grafeo ("), first_line
    assert "nodes)" in first_line
    assert "Traversal:" in first_line


def test_query_command_names_a_graph_outside_the_cwd(tmp_path):
    """The #2789 scenario: querying an explicit graph that sits outside the CWD
    must show it in full so the operator can tell the answer came from elsewhere."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    out = _make_graph(elsewhere)
    here = tmp_path / "here"
    here.mkdir()
    r = _run(["query", "Transformer", "--graph", str(out / "graph.grafeo")], here)
    assert r.returncode == 0, r.stderr
    first_line = r.stdout.splitlines()[0]
    assert first_line.startswith("Graph: "), first_line
    assert "elsewhere" in first_line, first_line


def test_cluster_only_preserves_parallel_edges_across_an_id_remap(tmp_path):
    """#3999 completion: cluster-only reloads graph.grafeo through build_from_json,
    which rewrites a legacy/pre-migration node id to its canonical stem
    (_semantic_id_remap). The preserved original_links must track the remapped
    id and still carry BOTH parallel edges — not be silently dropped because
    their endpoints no longer match the post-remap node set. build_from_json
    rewrites the shared link endpoints in place before to_json reads them;
    this test guards that invariant against a future refactor."""
    out = tmp_path / "graphify-out"
    out.mkdir()
    # `claude_alpha` + source_file `.claude/CLAUDE.md` remaps to
    # `claude_claude_alpha` (parent-dir == stem case, #1917).
    _graph_write_text(out / "graph.grafeo", json.dumps({
        "directed": False, "multigraph": False, "graph": {},
        "nodes": [
            {"id": "claude_alpha", "label": "alpha", "file_type": "concept",
             "source_file": ".claude/CLAUDE.md", "_origin": "semantic"},
            {"id": "beta", "label": "beta", "file_type": "code",
             "source_file": "pkg/beta.py", "_origin": "semantic"},
        ],
        "links": [
            {"source": "claude_alpha", "target": "beta", "relation": "references", "confidence": "EXTRACTED"},
            {"source": "claude_alpha", "target": "beta", "relation": "mentions", "confidence": "EXTRACTED"},
        ],
    }))

    r = _run(["cluster-only", ".", "--no-viz", "--no-label"], tmp_path)
    assert r.returncode == 0, r.stderr

    data = json.loads(_graph_read_text(out / "graph.grafeo"))
    node_ids = {n["id"] for n in data["nodes"]}
    # both endpoints remap: claude_alpha -> claude_claude_alpha, beta -> pkg_beta
    assert "claude_claude_alpha" in node_ids, node_ids  # remap fired
    assert "claude_alpha" not in node_ids
    pairs = sorted((l["source"], l["target"], l["relation"]) for l in data["links"])
    # both parallel edges survive AND track the remapped endpoint ids
    assert pairs == [
        ("claude_claude_alpha", "pkg_beta", "mentions"),
        ("claude_claude_alpha", "pkg_beta", "references"),
    ], pairs


def test_to_json_declares_multigraph_when_original_links_have_parallel_edges(tmp_path):
    """#4500: to_json restored original_links (which may hold parallel edges,
    #3999) while node_link_data derived "multigraph": false from the simple
    in-memory graph. The flag must match the links actually written."""
    from graphify.build import build_from_json
    from graphify.cluster import cluster
    from graphify.export import to_json

    extraction = {
        "directed": True,
        "nodes": [
            {"id": "a", "label": "App()", "file_type": "code", "source_file": "src/App.tsx"},
            {"id": "b", "label": "applySeed()", "file_type": "code", "source_file": "src/App.tsx"},
        ],
        "links": [
            {"source": "a", "target": "b", "relation": "contains", "confidence": "EXTRACTED",
             "source_file": "src/App.tsx", "source_location": "80"},
            {"source": "a", "target": "b", "relation": "calls", "confidence": "EXTRACTED",
             "source_file": "src/App.tsx", "source_location": "163"},
        ],
    }
    G = build_from_json(extraction, directed=True)
    communities = cluster(G)
    out = tmp_path / "graph.grafeo"
    assert to_json(G, communities, str(out), original_links=extraction["links"])

    data = json.loads(_graph_read_text(out))
    assert data["multigraph"] is True
    pairs = sorted((l["source"], l["target"], l["relation"]) for l in data["links"])
    assert pairs == [("a", "b", "calls"), ("a", "b", "contains")], pairs


def test_to_json_keeps_multigraph_false_without_parallel_edges(tmp_path):
    """#4500: the flag repair must not flip graphs that have no parallel edges."""
    from graphify.build import build_from_json
    from graphify.cluster import cluster
    from graphify.export import to_json

    extraction = {
        "directed": True,
        "nodes": [
            {"id": "a", "label": "a", "file_type": "code", "source_file": "a.py"},
            {"id": "b", "label": "b", "file_type": "code", "source_file": "a.py"},
        ],
        "links": [
            {"source": "a", "target": "b", "relation": "calls", "confidence": "EXTRACTED"},
        ],
    }
    G = build_from_json(extraction, directed=True)
    communities = cluster(G)
    out = tmp_path / "graph.grafeo"
    assert to_json(G, communities, str(out), original_links=extraction["links"])

    data = json.loads(_graph_read_text(out))
    assert data["multigraph"] is False
    assert len(data["links"]) == 1


def test_to_obsidian_lists_every_parallel_relation(tmp_path):
    """#4500 follow-up: graph.grafeo preserves parallel edges and is reloaded as a
    MultiGraph, but the vault note's Connections list went through
    ``edge_data()``, which returns only the FIRST parallel edge. A pair joined by
    both `imports` and `calls` therefore lost one relation from the exported
    note, exactly as serve.py's neighbor output did before it moved to
    ``edge_datas()``."""
    from networkx.readwrite import json_graph

    from graphify.export import to_obsidian

    raw = {
        "directed": False,
        "multigraph": True,
        "nodes": [
            {"id": "a.py", "label": "a.py", "file_type": "code"},
            {"id": "b.py", "label": "b.py", "file_type": "code"},
        ],
        "links": [
            {"source": "a.py", "target": "b.py", "relation": "imports",
             "confidence": "EXTRACTED"},
            {"source": "a.py", "target": "b.py", "relation": "calls",
             "confidence": "INFERRED"},
        ],
    }
    G = json_graph.node_link_graph(raw, edges="links")
    assert G.is_multigraph() and G.number_of_edges() == 2

    to_obsidian(G, {0: ["a.py", "b.py"]}, str(tmp_path))
    note = _graph_read_text(tmp_path / "a.py.md", encoding="utf-8")
    connections = [line for line in note.splitlines() if line.startswith("- [[")]
    assert connections == [
        "- [[b.py]] - `calls` [INFERRED]",
        "- [[b.py]] - `imports` [EXTRACTED]",
    ], connections


def test_cluster_only_accepts_file_mention_folded_into_ast_twin(tmp_path):
    """#4389 follow-up: a saved document mention of a file folds into that
    file's AST node on reload. That is one node, not a lost one, so cluster-only
    must write instead of refusing."""
    out = tmp_path / "graphify-out"
    out.mkdir()
    _graph_write_text(out / "graph.grafeo", json.dumps({
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": [
            {"id": "src_config", "label": "config.php", "source_file": "src/config.php",
             "file_type": "code", "source_location": "L1", "_origin": "ast"},
            {"id": "readme_config", "label": "src/config.php", "source_file": "README.md",
             "file_type": "code", "_origin": "semantic"},
            {"id": "readme", "label": "README.md", "source_file": "README.md",
             "file_type": "document", "_origin": "semantic"},
        ],
        "links": [{"source": "readme", "target": "readme_config", "relation": "describes",
                   "confidence": "EXTRACTED", "source_file": "README.md"}],
    }), encoding="utf-8")

    r = _run(["cluster-only", ".", "--no-viz", "--no-label"], tmp_path)

    assert r.returncode == 0, r.stderr
    assert "NOT written" not in r.stderr, r.stderr
    assert _graph_exists(out / "GRAPH_REPORT.md")
    ids = {n["id"] for n in json.loads(_graph_read_text(out / "graph.grafeo", encoding="utf-8"))["nodes"]}
    assert ids == {"readme", "src_config"}
