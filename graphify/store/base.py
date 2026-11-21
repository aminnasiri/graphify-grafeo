"""Storage contract; JSON is an interchange format, never a store backend."""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Self

import networkx as nx


def infer_multigraph(data: dict) -> bool:
    """Missing legacy flags must never collapse parallel or reverse arcs."""
    directed = data.get("directed", False)
    rows = data.get("links", data.get("edges", []))
    pairs = [(row.get("source"), row.get("target")) for row in rows if isinstance(row, dict)]
    keys = pairs if directed else [frozenset(pair) for pair in pairs]
    return len(set(keys)) != len(pairs)


@dataclass(frozen=True)
class NodeRecord:
    id: str
    attributes: dict[str, Any]


@dataclass(frozen=True)
class EdgeRecord:
    """The storage ID is distinct from any exported NetworkX ``key`` attribute."""

    id: str
    source: str
    target: str
    attributes: dict[str, Any]


@dataclass(frozen=True)
class GraphSnapshot:
    nodes: list[NodeRecord]
    edges: list[EdgeRecord]
    metadata: dict[str, Any]
    revision: int

    def to_node_link(self) -> dict[str, Any]:
        """Preserve the interchange envelope and every original record attribute."""
        data = deepcopy(self.metadata.get("json_envelope", {}))
        data.setdefault("directed", False)
        if "multigraph" not in data:
            data["multigraph"] = infer_multigraph({**data, "links": [
                {"source": e.source, "target": e.target} for e in self.edges]})
        graph = data.setdefault("graph", {})
        graph.setdefault("schema_version", 1)
        for key in ("schema_version", "graphify_version"):
            if key in self.metadata:
                graph[key] = deepcopy(self.metadata[key])
        if "json_envelope" not in self.metadata:
            data.setdefault("hyperedges", deepcopy(graph.get("hyperedges", [])))
        data["nodes"] = [{**deepcopy(n.attributes), "id": n.id} for n in self.nodes]
        data["links"] = [
            {**deepcopy(e.attributes), "source": e.source, "target": e.target}
            for e in self.edges
        ]
        return data

    def to_networkx(self, *, directed: bool | None = None) -> nx.Graph:
        """A keyed analytical view, retaining parallel arcs and their direction.

        Use a directed view for traversal; the logical direction flag remains in
        graph metadata. Storage IDs are used as NetworkX keys only in this view.
        """
        data = self.to_node_link()
        logical_directed = bool(data["directed"])
        is_directed = logical_directed if directed is None else directed
        graph: nx.Graph = nx.MultiDiGraph() if is_directed else nx.MultiGraph()
        graph.graph.update(deepcopy(data["graph"]))
        graph.graph["_logical_directed"] = logical_directed
        graph.graph["_store_revision"] = self.revision
        if "built_at_commit" in data:
            graph.graph["_built_at_commit"] = data["built_at_commit"]
        if "hyperedges" in data:
            graph.graph["hyperedges"] = deepcopy(data["hyperedges"])
        for node in self.nodes:
            graph.add_node(node.id, **deepcopy(node.attributes))
        for edge in self.edges:
            attrs = deepcopy(edge.attributes)
            source = attrs.get("_src", edge.source)
            target = attrs.get("_tgt", edge.target)
            attrs.setdefault("_src", source)
            attrs.setdefault("_tgt", target)
            # ``key`` is an original data attribute, not the storage identity.
            original_key = attrs.pop("key", None)
            graph.add_edge(source, target, key=edge.id, **attrs)
            if "key" in edge.attributes:
                graph.edges[source, target, edge.id]["key"] = original_key
        return graph


class GraphStore(ABC):
    """Record operations replace complete attribute maps, without graph rebuilding.

    Callers authorize replacement using Graphify's existing source/tier rules.
    A transaction includes all records, metadata and its monotonically increasing
    revision. Nested transactions join the outer transaction, without savepoints.
    """

    @abstractmethod
    def transaction(self) -> AbstractContextManager[GraphStore]: ...

    @abstractmethod
    def upsert_nodes(self, nodes: Iterable[NodeRecord]) -> None: ...

    @abstractmethod
    def upsert_edges(self, edges: Iterable[EdgeRecord]) -> None: ...

    @abstractmethod
    def delete_nodes(self, ids: Iterable[str]) -> None: ...

    @abstractmethod
    def delete_edges(self, ids: Iterable[str]) -> None: ...

    @abstractmethod
    def delete_metadata(self, keys: Iterable[str]) -> None: ...

    @abstractmethod
    def delete_by_source_file(self, source_file: str, *, origin: str | None = None) -> None:
        """Purge owned nodes/edges and incident arcs, optionally scoped by _origin.

        This is a deletion primitive, not the incremental replacement algorithm.
        Replacement must retain incoming evidence for surviving node identities.
        """
        ...

    @abstractmethod
    def get_node(self, node_id: str) -> NodeRecord | None: ...

    @abstractmethod
    def get_metadata(self, key: str, default: Any = None) -> Any: ...

    @abstractmethod
    def set_metadata(self, key: str, value: Any) -> None: ...

    @abstractmethod
    def snapshot(self) -> GraphSnapshot: ...

    @abstractmethod
    def clear(self) -> None:
        """Explicitly remove the graph and its metadata, inside a transaction."""
        ...

    @abstractmethod
    def backup(self, path: str | Path) -> None:
        """Write an engine-consistent backup while owning the database."""
        ...

    @abstractmethod
    def get_revision(self) -> int: ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def neighbors(self, node_id: str, *, direction: str = "both") -> list[NodeRecord]:
        if direction not in {"in", "out", "both"}:
            raise ValueError("direction must be 'in', 'out' or 'both'")
        snapshot = self.snapshot()
        ids: set[str] = set()
        for edge in snapshot.edges:
            src = edge.attributes.get("_src", edge.source)
            tgt = edge.attributes.get("_tgt", edge.target)
            if direction in {"out", "both"} and src == node_id:
                ids.add(tgt)
            if direction in {"in", "both"} and tgt == node_id:
                ids.add(src)
        return [n for n in snapshot.nodes if n.id in ids]

    def shortest_path(self, source: str, target: str, *, directed: bool = True) -> list[str]:
        """Unweighted path primitive; CLI confidence/tie rules remain in serve.py."""
        graph = self.export_networkx(directed=directed)
        return nx.shortest_path(graph, source, target)

    def query_subgraph(self, seeds: Iterable[str], *, depth: int = 2) -> nx.Graph:
        """Induced weak neighborhood. Ranking and token budgets belong to retrieval."""
        if depth < 0:
            raise ValueError("depth must be non-negative")
        graph = self.export_networkx(directed=True)
        view = graph.to_undirected(as_view=True)
        visited = set(seeds) & set(graph)
        frontier = set(visited)
        for _ in range(depth):
            frontier = {v for u in frontier for v in view.neighbors(u)} - visited
            visited.update(frontier)
        return graph.subgraph(sorted(visited)).copy()

    def list_communities(self) -> dict[int, list[str]]:
        communities: dict[int, list[str]] = {}
        for node in self.snapshot().nodes:
            cid = node.attributes.get("community")
            if cid is not None:
                communities.setdefault(int(cid), []).append(node.id)
        return communities

    def export_networkx(self, *, directed: bool | None = None) -> nx.Graph:
        return self.snapshot().to_networkx(directed=directed)

    def export_json(self, path: str | Path) -> None:
        from graphify.paths import write_json_atomic

        if Path(path).suffix.lower() == ".grafeo":
            raise ValueError("JSON export cannot overwrite a Grafeo database")
        write_json_atomic(path, self.snapshot().to_node_link(), compact=True)

    def import_json(self, path: str | Path, *, replace: bool = False) -> None:
        from graphify.paths import read_graph_json
        from graphify.security import check_graph_file_size_cap

        check_graph_file_size_cap(Path(path))
        self.import_node_link(read_graph_json(path), replace=replace)

    def import_node_link(self, data: dict[str, Any], *, replace: bool = False) -> None:
        """Load interchange data atomically; refuse accidental replacement."""
        envelope = deepcopy({k: v for k, v in data.items() if k not in {"nodes", "links", "edges"}})
        if not isinstance(envelope.get("graph", {}), dict):
            raise ValueError("graph metadata must be an object")
        raw_nodes = data.get("nodes", [])
        raw_edges = data.get("links", data.get("edges", []))
        if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
            raise ValueError("nodes and links must be lists")
        nodes = []
        for row in raw_nodes:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise ValueError("nodes require string IDs")
            nodes.append(NodeRecord(row["id"], {k: v for k, v in row.items() if k != "id"}))
        ids = {n.id for n in nodes}
        if len(ids) != len(nodes):
            raise ValueError("duplicate node IDs in interchange data")
        edges = []
        occurrences: dict[str, int] = {}
        for row in raw_edges:
            if not isinstance(row, dict):
                raise ValueError("links must be objects")
            src, tgt = row.get("source"), row.get("target")
            if not isinstance(src, str) or not isinstance(tgt, str) or src not in ids or tgt not in ids:
                raise ValueError("edge endpoints must reference imported nodes")
            digest = hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
            occurrence = occurrences.get(digest, 0)
            occurrences[digest] = occurrence + 1
            edges.append(EdgeRecord(f"{digest}:{occurrence}", src, tgt,
                                    {k: v for k, v in row.items() if k not in {"source", "target"}}))
        with self.transaction():
            current = self.snapshot()
            if not replace and (current.nodes or current.edges):
                raise ValueError("store is not empty; use replace=True to replace it")
            self.clear()
            self.upsert_nodes(nodes)
            self.upsert_edges(edges)
            self.set_metadata("json_envelope", envelope)
            for key in ("schema_version", "graphify_version"):
                if key in envelope.get("graph", {}):
                    self.set_metadata(key, envelope["graph"][key])
