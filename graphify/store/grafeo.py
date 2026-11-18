"""Embedded Grafeo implementation of the Graphify storage contract."""
from __future__ import annotations

from contextlib import contextmanager
from collections import defaultdict
import math
from pathlib import Path
import threading
import time
from typing import Any, Iterable, Iterator

from .base import EdgeRecord, GraphSnapshot, GraphStore, NodeRecord
from .codec import decode_record, encode_record


def _identifier(value: str) -> str:
    """Quote a GQL identifier; values always travel as query parameters."""
    if not value or "\x00" in value:
        raise ValueError("graph labels and relations must be nonempty and contain no NUL")
    return "`" + value.replace("`", "``") + "`"


def _kind(attributes: dict[str, Any]) -> str:
    metadata = attributes.get("metadata")
    for candidate in (attributes.get("type"), attributes.get("kind"),
                      metadata.get("kind") if isinstance(metadata, dict) else None,
                      attributes.get("file_type")):
        if isinstance(candidate, str) and candidate and not candidate.startswith("_Graphify"):
            return candidate
    return "concept"


class StoreBusyError(TimeoutError):
    """Another database owner did not release the file within the deadline."""


class GrafeoStore(GraphStore):
    """One short-lived database owner; close before yielding to another process.

    The database's OS lock is authoritative. Opens retry only the explicit busy
    error, with a finite deadline. Threads sharing this object are serialized;
    every mutation uses transaction-scoped GQL, never an auto-committing direct
    write. Callers must not keep an MCP context open for the server's lifetime.
    """

    def __init__(self, path: str | Path, *, timeout: float = 5.0, read_only: bool = False):
        import grafeo

        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and non-negative")
        self.path = Path(path).expanduser().resolve()
        if self.path.suffix.lower() == ".json":
            raise ValueError("JSON is an import/export format; use a .grafeo database path")
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._tx: Any = None
        self._dirty = False
        self._rollback_only = False
        self._closed = False
        deadline = time.monotonic() + timeout
        while True:
            try:
                self._db = (grafeo.GrafeoDB.open_read_only(str(self.path)) if read_only
                            else grafeo.GrafeoDB(str(self.path)))
                break
            except grafeo.GrafeoLockedError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise StoreBusyError(f"Graphify database is busy: {self.path}") from exc
                time.sleep(min(0.05, remaining))
        if not read_only:
            try:
                for label, key in (("_GraphifyNode", "id"), ("_GraphifyMetadata", "id"),
                                   ("_GraphifyRevision", "id")):
                    self._db.execute(
                        f"CREATE CONSTRAINT IF NOT EXISTS {label}_key "
                        f"FOR (n:{label}) ON (n.{key}) UNIQUE"
                    )
                self._db.create_property_index("source_file")
            except BaseException:
                self._db.close()
                raise

    def _execute(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Graphify store is closed")
        return list((self._tx if self._tx is not None else self._db).execute(query, params))

    @contextmanager
    def transaction(self) -> Iterator[GrafeoStore]:
        with self._lock:
            if self._closed:
                raise RuntimeError("Graphify store is closed")
            if self._tx is not None:
                try:
                    yield self
                except BaseException:
                    self._rollback_only = True
                    raise
                return
            self._tx = self._db.begin_transaction()
            self._dirty = self._rollback_only = False
            try:
                yield self
                if self._rollback_only:
                    raise RuntimeError("transaction was aborted by a nested operation")
                if self._dirty:
                    self._execute(
                        "MERGE (n:_GraphifyRevision {id:'revision'}) "
                        "SET n.value = coalesce(n.value, 0) + 1"
                    )
                self._tx.commit()
            except BaseException:
                self._tx.rollback()
                raise
            finally:
                self._tx = None
                self._dirty = self._rollback_only = False

    def upsert_nodes(self, nodes: Iterable[NodeRecord]) -> None:
        with self.transaction():
            nodes = list(nodes)
            if not nodes:
                return
            nodes = list({node.id: node for node in nodes}.values())
            prior = {row['id']: row for row in self._execute(
                "MATCH (n:_GraphifyNode) WHERE n.id IN $ids "
                "RETURN n.id AS id, n._graphify_kind AS kind, n._graphify_order AS position",
                {"ids": [n.id for n in nodes]})}
            next_order = self.get_metadata("_next_node_order", 0)
            groups: dict[str, list[dict]] = defaultdict(list)
            for node in nodes:
                if not isinstance(node.id, str):
                    raise ValueError("nodes require string IDs")
                kind = _kind(node.attributes)
                _identifier(kind)
                props = encode_record(node.attributes)
                props.update(id=node.id, _graphify_kind=kind)
                previous = prior.get(node.id, {})
                old = previous.get("kind")
                position = previous.get("position")
                if position is None:
                    position, next_order = next_order, next_order + 1
                props["_graphify_order"] = position
                if old and old != kind:
                    self._execute(
                        f"MATCH (n:_GraphifyNode {{id:$id}}) REMOVE n:{_identifier(old)}",
                        {"id": node.id},
                    )
                groups[kind].append({"id": node.id, "props": props})
            for kind, rows in groups.items():
                self._execute(
                    f"FOR row IN $rows MERGE (n:_GraphifyNode {{id:row.id}}) "
                    f"SET n = row.props SET n:{_identifier(kind)}", {"rows": rows},
                )
                self._dirty = True
            self.set_metadata("_next_node_order", next_order)

    def upsert_edges(self, edges: Iterable[EdgeRecord]) -> None:
        with self.transaction():
            edges = list(edges)
            if not edges:
                return
            # A batch may repeat an ID; its last complete record wins.
            edges = list({edge.id: edge for edge in edges}.values())
            endpoints = {v for edge in edges for v in (edge.source, edge.target)}
            present = {row['id'] for row in self._execute(
                "MATCH (n:_GraphifyNode) WHERE n.id IN $ids RETURN n.id AS id",
                {"ids": sorted(endpoints)})}
            if endpoints - present:
                raise ValueError(f"edge endpoint does not exist: {sorted(endpoints - present)[0]!r}")
            prior = {row['id']: row for row in self._execute(
                "MATCH (a:_GraphifyNode)-[e]->(b:_GraphifyNode) "
                "WHERE e._graphify_edge_id IN $ids RETURN e._graphify_edge_id AS id, "
                "a.id AS source, b.id AS target, type(e) AS relation, e._graphify_order AS position",
                {"ids": [edge.id for edge in edges]})}
            next_order = self.get_metadata("_next_edge_order", 0)
            inserts: dict[str, list[dict]] = defaultdict(list)
            updates: dict[str, list[dict]] = defaultdict(list)
            for edge in edges:
                if not isinstance(edge.id, str) or not edge.id:
                    raise ValueError("edges require nonempty string storage IDs")
                relation = edge.attributes.get("relation", "related_to")
                if not isinstance(relation, str):
                    raise ValueError("edge relation must be a string")
                _identifier(relation)
                props = encode_record(edge.attributes)
                props["_graphify_edge_id"] = edge.id
                old = prior.get(edge.id)
                position = old.get("position") if old else None
                if position is None:
                    position, next_order = next_order, next_order + 1
                props["_graphify_order"] = position
                same = old and (old['source'], old['target'], old['relation']) == (
                    edge.source, edge.target, relation)
                if old and not same:
                    self.delete_edges([edge.id])
                row = {"id": edge.id, "source": edge.source, "target": edge.target, "props": props}
                (updates if same else inserts)[relation].append(row)
            for relation, rows in inserts.items():
                self._execute(
                    "FOR row IN $rows MATCH (a:_GraphifyNode {id:row.source}), "
                    "(b:_GraphifyNode {id:row.target}) "
                    f"INSERT (a)-[e:{_identifier(relation)}]->(b) SET e = row.props", {"rows": rows},
                )
            for relation, rows in updates.items():
                self._execute(
                    f"FOR row IN $rows MATCH (a:_GraphifyNode {{id:row.source}})-"
                    f"[e:{_identifier(relation)}]->(b:_GraphifyNode {{id:row.target}}) "
                    "WHERE e._graphify_edge_id = row.id SET e = row.props", {"rows": rows},
                )
            self._dirty = True
            self.set_metadata("_next_edge_order", next_order)

    def delete_edges(self, ids: Iterable[str]) -> None:
        with self.transaction():
            ids = list(ids)
            if ids:
                self._execute("MATCH (:_GraphifyNode)-[e]->(:_GraphifyNode) "
                              "WHERE e._graphify_edge_id IN $ids DELETE e", {"ids": ids})
                self._dirty = True

    def delete_nodes(self, ids: Iterable[str]) -> None:
        with self.transaction():
            ids = list(ids)
            if ids:
                self._execute("MATCH (n:_GraphifyNode) WHERE n.id IN $ids DETACH DELETE n",
                              {"ids": ids})
                self._dirty = True

    def delete_metadata(self, keys: Iterable[str]) -> None:
        with self.transaction():
            keys = list(keys)
            if keys:
                self._execute("MATCH (n:_GraphifyMetadata) WHERE n.id IN $ids DELETE n", {"ids": keys})
                self._dirty = True

    def delete_by_source_file(self, source_file: str, *, origin: str | None = None) -> None:
        with self.transaction():
            predicate = "n.source_file = $source"
            edge_predicate = "e.source_file = $source"
            params: dict[str, Any] = {"source": source_file}
            if origin is not None:
                predicate += " AND n._origin = $origin"
                edge_predicate += " AND e._origin = $origin"
                params["origin"] = origin
            self._execute(
                "MATCH (:_GraphifyNode)-[e]->(:_GraphifyNode) "
                f"WHERE {edge_predicate} DELETE e", params,
            )
            self._execute(f"MATCH (n:_GraphifyNode) WHERE {predicate} DETACH DELETE n", params)
            self._dirty = True

    def get_node(self, node_id: str) -> NodeRecord | None:
        with self._lock:
            rows = self._execute(
                "MATCH (n:_GraphifyNode {id:$id}) RETURN properties(n) AS props", {"id": node_id},
            )
            return NodeRecord(node_id, decode_record(rows[0]["props"])) if rows else None

    def get_metadata(self, key: str, default: Any = None) -> Any:
        with self._lock:
            rows = self._execute(
                "MATCH (n:_GraphifyMetadata {id:$id}) RETURN properties(n) AS props", {"id": key},
            )
            return decode_record(rows[0]["props"])["value"] if rows else default

    def set_metadata(self, key: str, value: Any) -> None:
        with self.transaction():
            props = encode_record({"value": value})
            props["id"] = key
            self._execute("MERGE (n:_GraphifyMetadata {id:$id}) SET n = $props",
                          {"id": key, "props": props})
            self._dirty = True

    def snapshot(self) -> GraphSnapshot:
        with self.transaction():
            nodes = [
                NodeRecord(row["id"], decode_record(row["props"]))
                for row in self._execute(
                    "MATCH (n:_GraphifyNode) RETURN n.id AS id, properties(n) AS props "
                    "ORDER BY n._graphify_order, n.id"
                )
            ]
            edges = [
                EdgeRecord(row["id"], row["source"], row["target"], decode_record(row["props"]))
                for row in self._execute(
                    "MATCH (a:_GraphifyNode)-[e]->(b:_GraphifyNode) "
                    "RETURN e._graphify_edge_id AS id, a.id AS source, b.id AS target, "
                    "properties(e) AS props ORDER BY e._graphify_order, e._graphify_edge_id"
                )
            ]
            metadata = {
                row["id"]: decode_record(row["props"])["value"]
                for row in self._execute(
                    "MATCH (n:_GraphifyMetadata) RETURN n.id AS id, properties(n) AS props"
                )
            }
            revisions = self._execute("MATCH (n:_GraphifyRevision) RETURN n.value AS value")
            revision = int(revisions[0]["value"]) if revisions else 0
            return GraphSnapshot(nodes, edges, metadata, revision)

    def clear(self) -> None:
        with self.transaction():
            self._execute("MATCH (n:_GraphifyNode) DETACH DELETE n")
            self._execute("MATCH (n:_GraphifyMetadata) DELETE n")
            self._dirty = True

    def get_revision(self) -> int:
        with self._lock:
            rows = self._execute("MATCH (n:_GraphifyRevision) RETURN n.value AS value")
            return int(rows[0]["value"]) if rows else 0

    def backup(self, path: str | Path) -> None:
        with self._lock:
            if self._tx is not None:
                raise RuntimeError("cannot back up an uncommitted transaction")
            self._db.save(str(path))

    def close(self) -> None:
        with self._lock:
            if self._tx is not None:
                raise RuntimeError("cannot close a store inside a transaction")
            if not self._closed:
                self._db.close()
                self._closed = True
