"""Opt-in local document retrieval; structural extraction never calls this module."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
from typing import Callable, cast

from graphify.sections import _headings, _resolve_inside
from graphify.extractors.markdown import _split_frontmatter
from graphify.security import redact_secrets

EmbeddingFunction = Callable[[list[str]], list[list[float]]]
_DOCUMENT_SUFFIXES = frozenset({".md", ".mdx", ".qmd", ".skill", ".txt", ".rst"})
_MAX_DOCUMENT_BYTES = 32 * 1024 * 1024
_CHUNK_CHARS = 16_000


@dataclass(frozen=True)
class Chunk:
    id: str
    source_file: str
    source_location: str
    heading: str
    text: str
    parents: list[str]


def heading_chunks(text: str) -> list[tuple[int, str, str]]:
    """Disjoint sections, using the extractor's fence/frontmatter/heading rules."""
    lines = text.splitlines(keepends=True)
    _, body_start = _split_frontmatter(lines)
    headings = _headings(lines)
    starts = [(line - 1, title) for line, _, title in headings]
    if not starts or starts[0][0] > body_start:
        starts.insert(0, (body_start, ""))
    out = []
    for pos, (start, title) in enumerate(starts):
        end = starts[pos + 1][0] if pos + 1 < len(starts) else len(lines)
        body = "".join(lines[start:end])
        # Bound individual properties and backend inputs even for heading-free docs.
        line = start + 1
        for offset in range(0, len(body), _CHUNK_CHARS):
            piece = body[offset:offset + _CHUNK_CHARS]
            if piece.strip():
                out.append((line, title, piece))
            line += piece.count("\n")
    return out


def document_chunks(nodes, root: Path) -> tuple[list[Chunk], list[tuple[str, str]]]:
    """Read only existing indexed sources inside the same project boundary."""
    by_source: dict[str, list] = {}
    for node in nodes:
        source = node.attributes.get("source_file")
        if isinstance(source, str) and Path(source).suffix.lower() in _DOCUMENT_SUFFIXES:
            by_source.setdefault(source, []).append(node)
    chunks, hashes = [], []
    for source, owners in sorted(by_source.items()):
        path = _resolve_inside(source, root)
        if path is None:
            continue
        try:
            with path.open("rb") as stream:
                raw = stream.read(_MAX_DOCUMENT_BYTES + 1)
        except OSError:
            continue
        if len(raw) > _MAX_DOCUMENT_BYTES:
            raise ValueError(f"semantic document exceeds {_MAX_DOCUMENT_BYTES} bytes: {source}")
        hashes.append((source, hashlib.sha256(raw).hexdigest()))
        text = redact_secrets(raw.decode("utf-8-sig", errors="replace"))
        for ordinal, (line, title, body) in enumerate(heading_chunks(text)):
            ident = hashlib.sha256(json.dumps([source, ordinal], ensure_ascii=True).encode()).hexdigest()
            end_line = line + body.count("\n") - int(body.endswith("\n"))
            parents = []
            for owner in owners:
                location = re.match(r"^L(\d+)", str(owner.attributes.get("source_location") or ""))
                if location is None or line <= int(location.group(1)) <= max(line, end_line):
                    parents.append(owner.id)
            # Older document nodes may only describe the whole source file.
            if not parents:
                parents = [owner.id for owner in owners if owner.attributes.get("node_kind") == "page"]
            if not parents:
                parents = [owner.id for owner in owners]
            chunks.append(Chunk(ident, source, f"L{line}", title, body, sorted(parents)))
    return chunks, hashes


def hash_embeddings(texts: list[str]) -> list[list[float]]:
    """Offline lexical reference vectors; use a local model for semantic meaning."""
    out = []
    for text in texts:
        vector = [0.0] * 256
        for word in re.findall(r"\w+", text.casefold()):
            digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
            vector[int.from_bytes(digest[:4], "little") % len(vector)] += 1.0
        # Empty documents/queries still need a nonzero cosine vector.
        if not any(vector):
            vector[0] = 1.0
        out.append(vector)
    return out


def embedding_backend(spec: str) -> EmbeddingFunction:
    """Select explicitly configured, local code. Never fetch a model or service."""
    if spec == "hash":
        return hash_embeddings
    if spec.startswith("python:"):
        parts = spec.split(":")
        if len(parts) != 3 or not parts[1] or not parts[2]:
            raise ValueError("embedding backend must be python:module:function")
        function = getattr(importlib.import_module(parts[1]), parts[2])
        if not callable(function):
            raise ValueError("embedding backend is not callable")
        return cast(EmbeddingFunction, function)
    if spec.startswith("sentence-transformers:"):
        model_path = Path(spec.split(":", 1)[1]).expanduser().resolve()
        if not model_path.is_dir():
            raise ValueError("sentence-transformers backend needs an existing local model directory")
        # Set before optional SDK imports; keep the opt-in local adapter offline.
        for name in ("HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "TRANSFORMERS_OFFLINE", "DO_NOT_TRACK"):
            os.environ[name] = "1"
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        model = SentenceTransformer(str(model_path), local_files_only=True, trust_remote_code=False)
        return lambda texts: model.encode(texts, normalize_embeddings=True).tolist()
    raise ValueError("embedding backend must be hash, python:module:function, or sentence-transformers:/local/model")


def validated_vectors(function: EmbeddingFunction, texts: list[str], dimensions: int | None = None) -> list[list[float]]:
    rows = function(texts)
    if not isinstance(rows, list) or len(rows) != len(texts):
        raise ValueError("embedding backend must return one vector per input text")
    out = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or not row or len(row) > 4096:
            raise ValueError("embedding dimensions must be between 1 and 4096")
        if dimensions is None:
            dimensions = len(row)
        if len(row) != dimensions or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in row):
            raise ValueError("embedding vectors must have consistent dimensions and numeric values")
        if any(abs(x) > 1e30 for x in row):
            raise ValueError("embedding values must fit float32")
        vector = [float(x) for x in row]
        if any(not math.isfinite(x) for x in vector) or not any(vector):
            raise ValueError("embedding vectors must be finite, nonzero and fit float32")
        scale = max(abs(x) for x in vector)
        vector = [x / scale for x in vector]
        norm = math.sqrt(sum(x * x for x in vector))
        out.append([x / norm for x in vector])
    return out
