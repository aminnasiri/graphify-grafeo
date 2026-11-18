"""Embedded Grafeo implementation of the Graphify storage contract."""
from __future__ import annotations

from contextlib import contextmanager
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
            for node in nodes:
                if not isinstance(node.id, str) or not node.id:
                    raise ValueError("nodes require nonempty string IDs")
                kind = _kind(node.attributes)
                label = _identifier(kind)
                props = encode_record(node.attributes)
                props.update(id=node.id, _graphify_kind=kind)
                old = self._execute(
                    "MATCH (n:_GraphifyNode {id:$id}) RETURN n._graphify_kind AS kind",
                    {"id": node.id},
                )
                if old and old[0]["kind"] != kind:
                    self._execute(
                        f"MATCH (n:_GraphifyNode {{id:$id}}) REMOVE n:{_identifier(old[0]['kind'])}",
                        {"id": node.id},
                    )
                self._execute(
                    f"MERGE (n:_GraphifyNode {{id:$id}}) SET n = $props SET n:{label}",
                    {"id": node.id, "props": props},
                )
                self._dirty = True

    def upsert_edges(self, edges: Iterable[EdgeRecord]) -> None:
        with self.transaction():
            for edge in edges:
                if not isinstance(edge.id, str) or not edge.id:
                    raise ValueError("edges require nonempty string storage IDs")
                relation = edge.attributes.get("relation", "related_to")
                if not isinstance(relation, str):
                    raise ValueError("edge relation must be a string")
                edge_type = _identifier(relation)
                for endpoint in (edge.source, edge.target):
                    if self.get_node(endpoint) is None:
                        raise ValueError(f"edge endpoint does not exist: {endpoint!r}")
                props = encode_record(edge.attributes)
                props["_graphify_edge_id"] = edge.id
                self._execute(
                    "MATCH (:_GraphifyNode)-[e]->(:_GraphifyNode) "
                    "WHERE e._graphify_edge_id = $id DELETE e", {"id": edge.id},
                )
                self._execute(
                    f"MATCH (a:_GraphifyNode {{id:$source}}), (b:_GraphifyNode {{id:$target}}) "
                    f"INSERT (a)-[e:{edge_type}]->(b) SET e = $props",
                    {"source": edge.source, "target": edge.target, "props": props},
                )
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
                    "MATCH (n:_GraphifyNode) RETURN n.id AS id, properties(n) AS props ORDER BY id"
                )
            ]
            edges = [
                EdgeRecord(row["id"], row["source"], row["target"], decode_record(row["props"]))
                for row in self._execute(
                    "MATCH (a:_GraphifyNode)-[e]->(b:_GraphifyNode) "
                    "RETURN e._graphify_edge_id AS id, a.id AS source, b.id AS target, "
                    "properties(e) AS props ORDER BY id"
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

    def close(self) -> None:
        with self._lock:
            if self._tx is not None:
                raise RuntimeError("cannot close a store inside a transaction")
            if not self._closed:
                self._db.close()
                self._closed = True
