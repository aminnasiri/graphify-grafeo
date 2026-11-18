# Graph storage migration

## Revised scope

Graphify requires Python 3.12+ and Grafeo. Grafeo is the sole persistent
graph backend for every feature. There is no `JsonStore`, optional Grafeo extra,
or `store=json` selection. JSON remains an explicit import/export format, with
the existing node-link schema (version 1).

**Phases 1–3:** database storage, incremental updates, CLI, watch, hooks, MCP,
PR analysis, exporters and generated assistant instructions use Grafeo. Existing
JSON graphs are migrated explicitly, never silently replaced.

```bash
graphify migrate --to grafeo
graphify query "build_merge"
graphify export json --out graphify-out/graph.json
graphify import json legacy.json --database another.grafeo
```

Import refuses an occupied store unless `--replace` is supplied. Migration
verifies node, edge and community counts plus all attributes and the schema
envelope before atomically activating the database. Legacy JSON and sidecars
remain available as interchange/backup inputs; runtime uses database metadata.
The default is `graphify-out/graph.grafeo`. Override it with global
`--database PATH`, `GRAPHIFY_DATABASE`, or `.graphifyrc` `database_path`.
The path must end in `.grafeo`; relative config paths resolve at the project root.

## Storage API

`graphify.store.pipeline.sync_graph()` computes record deltas and commits only
changed records and metadata. `update_extraction()` reuses `build_merge` or the
raw merge path with an in-memory baseline; `update_communities()` changes node
assignments and atomically saves labels and membership signatures. Per-file
manifest rows are separate metadata records, so changing one hash does not
rewrite the other rows. A stale `expected_revision` raises
`ConcurrentUpdateError` instead of overwriting another writer. Surviving native
node and edge IDs remain unchanged, including incoming evidence from unchanged
files. Grafeo may still checkpoint larger storage sections internally.

```python
from graphify.store import EdgeRecord, NodeRecord, open_store

with open_store(root="/path/to/project") as store:
    with store.transaction():
        store.upsert_nodes([
            NodeRecord("a", {"label": "Alpha", "type": "function", "source_file": "a.py"}),
            NodeRecord("b", {"label": "Beta", "type": "class", "source_file": "b.py"}),
        ])
        store.upsert_edges([
            EdgeRecord("a-calls-b", "a", "b", {"relation": "calls", "confidence": "EXTRACTED"}),
        ])
        store.set_metadata("file_hashes", {"a.py": {"ast_hash": "...", "semantic_hash": ""}})

    snapshot = store.snapshot()
    graph = store.export_networkx(directed=True)
    store.export_json("/path/to/interchange.json")
```

`open_store()` defaults to `<root>/<GRAPHIFY_OUT>/graph.grafeo`. Its explicit
`path` argument may be absolute or relative to the root. JSON paths are refused
as database paths. An empty store can ingest a legacy graph with
`store.import_json(path)`; replacing existing records requires `replace=True`.
CLI import/migrate commands are deferred to the integration phase.

The contract lives in `graphify/store/base.py`, the Grafeo adapter in
`grafeo.py`, lossless attribute encoding in `codec.py`, and the factory in
`__init__.py`. Imports defer loading the native Grafeo module until a database
is opened, although Grafeo is a required package dependency.

Node and edge upserts replace complete attribute maps. Edge storage IDs are
caller-supplied and separate from exported `key` attributes. Imported links
receive deterministic content-derived IDs with occurrence suffixes, preserving
even identical parallel links. Native ordering properties preserve ingestion order and existing positions across
updates, so traversal tie-breaking agrees with legacy JSON.

`GraphSnapshot` holds a consistent revision, records and metadata. NetworkX
exports use multigraphs to preserve parallel relationships and retain logical
direction separately from an explicitly requested directed traversal view.
Legacy `_src`/`_tgt` markers remain authoritative in that view. JSON exports
preserve original arc order, attributes and envelope fields, including hyperedges
and `built_at_commit`, and include `graph.schema_version` (legacy files without
it are normalized to version 1). Label signatures and other store-only metadata
do not leak into the interchange envelope.

`neighbors`, `shortest_path`, `query_subgraph` and `list_communities` provide
common read primitives. Initially they use snapshots/NetworkX. The path method
is unweighted; it does not replace the CLI's confidence-aware path selection.
Query ranking, token budgets and ambiguity handling remain retrieval concerns.

## Persistence and correctness

Each user node has a `_GraphifyNode` label, a unique indexed `id`, and a kind
label derived from `type`, `kind`, `metadata.kind`, or `file_type`, in that order.
Edges use the original relation as their type. Labels and types are quoted GQL
identifiers; values are bound parameters. All three confidence categories,
`EXTRACTED`, `INFERRED` and `AMBIGUOUS`, survive unchanged.

Grafeo 0.6.3 drops top-level null properties and converts all-float lists to
float32 vectors, including nested lists. `codec.py` therefore keeps a lossless
JSON encoding per record in database properties, with native scalar projections
for lookup and source ownership. User properties using the internal prefix are
also retained in the encoded record. The encoding is split into bounded
properties, so documents larger than Grafeo's default 16 MiB property cap stay
inside the database. No external document file or JSON graph is required by the
store. Non-finite JSON numbers are rejected before writing.

Internal nodes hold metadata and a monotonically increasing revision; they are
excluded from graph snapshots and exports. Transactions cover graph records,
metadata and one revision increment. They use `tx.execute()` exclusively:
database-level direct/batch writes do not join that transaction. An error rolls
back everything. Nested operations join the outer transaction; an inner failure
marks the outer transaction for rollback even if a caller catches the error.

`delete_by_source_file(source, origin=...)` is an explicit purge primitive.
It deletes matching owned nodes and edges, plus edges incident on removed nodes.
It is **not** a safe incremental replacement algorithm on its own. Incremental reconciliation reuses Graphify's existing successful-source authorization, AST/semantic tier
rules, shared/folded identities and cross-file reconciliation, retaining incoming
evidence for surviving nodes. Community assignments can change outside an edited
file after clustering. Shrink/provenance guards must run before a transaction
commits; derived reports must be written after it.

## Process access and durability

Grafeo permits one read-write owner, excluding all other opens. Several
read-only owners may coexist while no writer is open. `GrafeoStore` retries only
Grafeo's database-busy error, then raises `StoreBusyError` after `timeout`
seconds (default 5). Other errors propagate immediately. The Grafeo OS lock is
authoritative; Python threads sharing a store are serialized by an `RLock`.
Always use short context-managed lifetimes. MCP caches detached,
revision-keyed snapshots rather than a permanently open database handle.

The installed Python package includes WAL support. `close()` checkpoints and
releases the database lock. Default WAL durability batches disk synchronization;
power-loss durability is not equivalent to a sync on every commit. Grafeo
checkpoints can rewrite substantial data even when Graphify applies a small
transaction. Incremental application writes do not promise proportional disk
I/O. Active-file copies are not backups; use Grafeo's supported backup/snapshot
APIs. See [persistence](https://docs.grafeo.dev/operate/persistence/persistent/)
and [Python database operations](https://docs.grafeo.dev/languages/python/database/).

## Integration and measured performance

Every feature reads the selected store through shared I/O. Query/path/explain
retain output, budgets, depth and exit codes (0/3/4). Labels, signatures, analysis,
learning overlay and per-file hashes are metadata. Extraction caches and normal
configuration/artifact JSON remain files. A source revision check rejects stale
writers; update/watch can be retried. CLI connections retry busy opens for up to
30 seconds. MCP holds detached, revision-keyed NetworkX caches and releases DB
handles between requests. A three-process integration test runs update, the
watch callback, and MCP reads against one database. Windows execution has not
been verified locally; file access uses Grafeo OS locking and portable paths.

A local benchmark on this repository's graph (26,897 nodes, 57,590 edges,
1,244 source files; one-file update changes 18 nodes) measured the following.
Run `uv run python scripts/benchmark_stores.py graphify-out/graph.json` to repeat.
Build means persistence of the same extracted graph, excluding shared extraction
and clustering. Updates and query latency are medians; filesystem caches are warm
and cold query means opening/loading a fresh graph view.

| Measurement | JSON interchange baseline | Grafeo |
| --- | ---: | ---: |
| Initial persistence | 0.295 s | 2.373 s |
| One-file update | 0.371 s | 1.316 s |
| Cold query | 0.909 s | 1.576 s |
| Cached query | 43.7 ms | 48.5 ms |
| On-disk graph | 27.04 MB | 40.00 MB |

Grafeo currently costs more time and space here. Reconciliation and analytical
reads still materialize full snapshots; only changed graph records are written.
Native checkpoint costs and lossless attribute encoding add overhead. These
numbers establish a baseline, not a speedup claim. Native query pushdown and
longer-lived ownership would require further measured work.

Real-engine tests cover `worked/` migration/attribute round-trips, CLI retrieval
parity, long documents, float/null fidelity, transaction failure, unchanged native
IDs, custom paths and concurrent consumers. Phase 4 adds optional retrieval in a
separate commit.

The supported Grafeo range is `>=0.6.3,<0.7`. Its public API is stable by default
within a minor release, while experimental APIs may change in patches. This
adapter avoids experimental streaming and native CSV loaders. Revalidate the
adapter and database format before accepting a new minor release; see
[Grafeo stability](https://docs.grafeo.dev/project/stability/).
