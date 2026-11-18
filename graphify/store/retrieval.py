"""Derived HNSW/BM25 indexes, deliberately separate from structural graph rows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from graphify.semantic_retrieval import document_chunks, embedding_backend, validated_vectors
from .io import connection, is_database
from .pipeline import ConcurrentUpdateError

_LABEL = "_GraphifyChunk"
_INDEX_VERSION = 1


def semantic_seeds(database: Path, question: str, *, root: Path,
                   backend: str = "hash", limit: int = 8, reindex: bool = False) -> list[str]:
    """Fuse native vector/BM25 chunk matches, then return structural node seeds.

    Embedding work happens with no database handle open. A revision check keeps a
    concurrent graph writer from committing an index against stale ownership.
    Chunk nodes never carry _GraphifyNode, so exports/counts/schema stay intact.
    """
    if not is_database(database):
        raise ValueError("--semantic requires a Grafeo database; migrate the JSON graph first")
    if not 1 <= limit <= 100:
        raise ValueError("semantic retrieval limit must be between 1 and 100")
    with connection(database) as store:
        snapshot = store.snapshot()
        content_mtime = store.get_metadata("graph_content_mtime", database.stat().st_mtime)
    chunks, hashes = document_chunks(snapshot.nodes, root)
    fingerprint = hashlib.sha256(json.dumps([
        _INDEX_VERSION, backend, str(root.resolve()), hashes,
        [(c.id, c.parents) for c in chunks],
    ], sort_keys=True, ensure_ascii=True).encode()).hexdigest()
    state = snapshot.metadata.get("retrieval:index", {})
    if not isinstance(state, dict):
        state = {}
    if not chunks:
        # Remove derived copies of deleted documents without importing a model.
        if state:
            with connection(database, write=True) as store:
                if store.get_revision() != snapshot.revision:
                    raise ConcurrentUpdateError("graph changed during retrieval; retry")
                with store.transaction():
                    if store.get_metadata("graph_content_mtime") is None:
                        store.set_metadata("graph_content_mtime", content_mtime)
                    store._execute(f"MATCH (n:{_LABEL}) DELETE n")
                    store.delete_metadata(["retrieval:index"])
        return []
    function = embedding_backend(backend)
    fresh = not reindex and state.get("fingerprint") == fingerprint
    dimensions = state.get("dimensions") if fresh else None
    query_vector = validated_vectors(function, [question], dimensions)[0]
    rows = None
    if not fresh:
        vectors = []
        for start in range(0, len(chunks), 64):
            batch = validated_vectors(function, [c.text for c in chunks[start:start + 64]], len(query_vector))
            vectors.extend(batch)
        rows = [{"id": chunk.id, "text": chunk.text, "heading": chunk.heading,
                 "source_file": chunk.source_file, "source_location": chunk.source_location,
                 "parents": chunk.parents, "embedding": vector}
                for chunk, vector in zip(chunks, vectors)]
    # Index declarations are idempotent, persist across reopen, and are outside
    # GQL transactions in Grafeo's Python API. Invalidate the cache before them;
    # a declaration/write failure forces a safe rebuild on the next query.
    with connection(database, write=not fresh) as store:
        if store.get_revision() != snapshot.revision:
            raise ConcurrentUpdateError("graph changed while building semantic retrieval; retry the query")
        if rows is not None:
            with store.transaction():
                if store.get_metadata("graph_content_mtime") is None:
                    store.set_metadata("graph_content_mtime", content_mtime)
                store.delete_metadata(["retrieval:index"])
                store._execute(f"MATCH (n:{_LABEL}) DELETE n")
            store._db.drop_vector_index(_LABEL, "embedding")
            store._db.create_vector_index(_LABEL, "embedding", dimensions=len(query_vector), metric="cosine")
            store._db.create_text_index(_LABEL, "text")
            with store.transaction():
                for start in range(0, len(rows), 64):
                    store._execute(f"FOR row IN $rows INSERT (n:{_LABEL}) SET n = row", {"rows": rows[start:start + 64]})
                store.set_metadata("retrieval:index", {"fingerprint": fingerprint, "backend": backend,
                                                       "dimensions": len(query_vector), "chunks": len(rows)})
        matches = store._db.hybrid_search(_LABEL, "text", "embedding", question, limit,
                                          query_vector=query_vector, fusion="rrf")
        seeds = []
        # Native search returns internal node IDs; parents use stable graph IDs.
        ids = {node.id for node in snapshot.nodes}
        for native_id, _score in matches:
            node = store._db.get_node(native_id)
            if node is None:
                continue
            for parent in node.properties().get("parents", []):
                if parent in ids and parent not in seeds:
                    seeds.append(parent)
        return seeds[:32]
