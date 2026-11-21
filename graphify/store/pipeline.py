"""Reconcile with Graphify's provenance rules, then persist only record deltas."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .base import EdgeRecord, GraphStore, NodeRecord


class ConcurrentUpdateError(RuntimeError):
    """The baseline changed while extraction or clustering was running."""


@dataclass(frozen=True)
class SyncResult:
    nodes_upserted: int
    nodes_deleted: int
    edges_upserted: int
    edges_deleted: int
    revision: int


def _edge_group(source: str, target: str, attrs: dict) -> str:
    return json.dumps([source, target, attrs.get("relation"), "key" in attrs, attrs.get("key")],
                      ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sync_graph(store: GraphStore, data: dict, *, metadata: dict[str, Any] | None = None,
               manifest: dict[str, Any] | None = None,
               expected_revision: int | None = None) -> SyncResult:
    """Apply an already-reconciled graph, without clearing or rewriting the store.

    Match unchanged parallel evidence first, then reuse remaining identities in
    each endpoint/relation/key group. This preserves native IDs on attribute edits.
    Hash rows individually; never serialize the entire graph into one DB value.
    """
    nodes: dict[str, NodeRecord] = {}
    for raw in data.get("nodes", []):
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
            raise ValueError("graph nodes require string IDs")
        if raw["id"] in nodes:
            raise ValueError("duplicate node IDs")
        nodes[raw["id"]] = NodeRecord(raw["id"], {k: v for k, v in raw.items() if k != "id"})
    rows = data.get("links", data.get("edges", []))
    if not isinstance(rows, list):
        raise ValueError("graph links must be a list")
    groups: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("graph links must be objects")
        src, tgt = raw.get("source"), raw.get("target")
        if not isinstance(src, str) or not isinstance(tgt, str) or src not in nodes or tgt not in nodes:
            raise ValueError("edge endpoints must reference graph nodes")
        attrs = {k: v for k, v in raw.items() if k not in {"source", "target"}}
        groups[_edge_group(src, tgt, attrs)].append((src, tgt, attrs))
    envelope = deepcopy({k: v for k, v in data.items() if k not in {"nodes", "links", "edges"}})
    graph_attrs = envelope.setdefault("graph", {})
    if not isinstance(graph_attrs, dict):
        raise ValueError("graph metadata must be an object")
    graph_attrs.pop("_store_revision", None)
    graph_attrs.setdefault("schema_version", 1)
    with store.transaction():
        old = store.snapshot()
        if expected_revision is not None and old.revision != expected_revision:
            raise ConcurrentUpdateError("graph changed during this operation; retry against the current revision")
        old_nodes = {n.id: n for n in old.nodes}
        old_groups: dict[str, list[EdgeRecord]] = defaultdict(list)
        for edge in old.edges:
            old_groups[_edge_group(edge.source, edge.target, edge.attributes)].append(edge)
        desired: dict[str, EdgeRecord] = {}
        occupied = {e.id for e in old.edges}
        for key, group in groups.items():
            available = sorted(old_groups.get(key, []), key=lambda e: e.id)
            pending = []
            for src, tgt, attrs in group:
                match = next((e for e in available if e.attributes == attrs), None)
                if match is not None:
                    desired[match.id] = match
                    available.remove(match)
                else:
                    pending.append((src, tgt, attrs))
            for src, tgt, attrs in pending:
                if available:
                    edge_id = available.pop(0).id
                else:
                    prefix = "edge:" + hashlib.sha256(key.encode()).hexdigest() + ":"
                    index = 0
                    while prefix + str(index) in occupied:
                        index += 1
                    edge_id = prefix + str(index)
                    occupied.add(edge_id)
                desired[edge_id] = EdgeRecord(edge_id, src, tgt, attrs)
        old_edges = {e.id: e for e in old.edges}
        removed_edges = old_edges.keys() - desired.keys()
        removed_nodes = old_nodes.keys() - nodes.keys()
        changed_nodes = [n for key, n in nodes.items() if old_nodes.get(key) != n]
        changed_edges = [e for key, e in desired.items() if old_edges.get(key) != e]
        store.delete_edges(sorted(removed_edges))
        store.delete_nodes(sorted(removed_nodes))
        store.upsert_nodes(changed_nodes)
        store.upsert_edges(changed_edges)
        updates = {**(metadata or {}), "json_envelope": envelope}
        for key in ("schema_version", "graphify_version"):
            if key in graph_attrs:
                updates[key] = graph_attrs[key]
        if manifest is not None:
            updates.setdefault("manifest_present", True)
            file_updates = {"file_hash:" + key: value for key, value in manifest.items()}
            store.delete_metadata([key for key in old.metadata
                                   if key.startswith("file_hash:") and key not in file_updates])
            updates.update(file_updates)
        for key, value in updates.items():
            if key not in old.metadata or old.metadata[key] != value:
                store.set_metadata(key, value)
    return SyncResult(len(changed_nodes), len(removed_nodes), len(changed_edges), len(removed_edges),
                      store.snapshot().revision)


def update_communities(store: GraphStore, communities: dict[int, list[str]],
                       labels: dict[int, str] | None = None) -> SyncResult:
    from graphify.cluster import community_member_sigs

    snapshot = store.snapshot()
    data = snapshot.to_node_link()
    assignments = {nid: cid for cid, members in communities.items() for nid in members}
    for node in data["nodes"]:
        cid = assignments.get(node["id"])
        node["community"] = cid
        if labels and cid is not None:
            node["community_name"] = labels.get(cid, f"Community {cid}")
    return sync_graph(store, data, expected_revision=snapshot.revision, metadata={
        "community_labels": {str(k): v for k, v in (labels or {}).items()},
        "label_signature": {str(k): v for k, v in community_member_sigs(communities).items()},
    })


def update_extraction(store: GraphStore, extraction: dict, *, root: str | Path,
                      prune_sources: list[str] | None = None,
                      manifest: dict | None = None, cluster: bool = True) -> SyncResult:
    """Reuse both existing reconciliation paths; failures retain last good state."""
    from graphify.build import build_merge, merge_raw_extraction, dedupe_nodes, dedupe_edges
    from graphify.cluster import cluster as cluster_graph
    from graphify.export import to_data

    snapshot = store.snapshot()
    baseline = snapshot.to_node_link()
    target = Path(root) / "graphify-out" / "graph.grafeo"
    if cluster:
        graph = build_merge([deepcopy(extraction)], graph_path=target, root=root,
                            prune_sources=prune_sources, existing_data=baseline)
        if graph.graph.get("_unverified_semantic_shrink"):
            raise ValueError("semantic extraction is smaller than saved evidence; refusing replacement")
        data = to_data(graph, cluster_graph(graph), output_path=target)
    else:
        merged = merge_raw_extraction(deepcopy(extraction), target, root=root,
                                      prune_sources=prune_sources, existing_data=baseline)
        if merged.get("_unverified_semantic_shrink"):
            raise ValueError("semantic extraction is smaller than saved evidence; refusing replacement")
        data = {**{k: v for k, v in baseline.items() if k not in {"nodes", "links"}},
                "nodes": dedupe_nodes(merged["nodes"]), "links": dedupe_edges(merged["edges"]),
                "hyperedges": merged.get("hyperedges", [])}
    if extraction.get("failed_sources") and manifest is not None:
        from graphify.build import _norm_source_file
        failures = {_norm_source_file(f, str(root)) for f in extraction["failed_sources"]}
        manifest = deepcopy(manifest)
        for key, value in manifest.items():
            if _norm_source_file(key, str(root)) in failures and isinstance(value, dict):
                value.update(ast_hash="", semantic_hash="")
    return sync_graph(store, data, manifest=manifest, expected_revision=snapshot.revision)
