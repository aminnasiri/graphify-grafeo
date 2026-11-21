"""Lossless JSON attributes, with scalar projections for indexed database reads.

Grafeo removes null properties and converts all-float lists to f32 vectors.
The encoded record preserves JSON values exactly; projections are derived data.
Chunked properties keep even long documents inside the database's property cap.
"""
from __future__ import annotations

import json
from typing import Any

_PREFIX = "_graphify_"
_CHUNK_CHARS = 256 * 1024  # At most 1 MiB of UTF-8 per property.


def encode_record(attributes: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(attributes, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                         allow_nan=False)
    parts = [payload[i:i + _CHUNK_CHARS] for i in range(0, len(payload), _CHUNK_CHARS)]
    properties: dict[str, Any] = {f"{_PREFIX}parts": len(parts)}
    properties.update({f"{_PREFIX}part_{i}": part for i, part in enumerate(parts)})
    # No native nested collections: their float lists would lose precision.
    for key, value in attributes.items():
        if key.startswith(_PREFIX) or value is None:
            continue
        if isinstance(value, (str, bool, float)) or (
            isinstance(value, int) and -(2**63) <= value < 2**63
        ):
            # Long strings remain in the chunked record without a second copy.
            if not isinstance(value, str) or len(value.encode("utf-8")) <= _CHUNK_CHARS:
                properties[key] = value
    return properties


def decode_record(properties: dict[str, Any]) -> dict[str, Any]:
    try:
        count = properties[f"{_PREFIX}parts"]
        if not isinstance(count, int) or count < 1 or count > len(properties):
            raise ValueError("invalid encoded property count")
        payload = "".join(properties[f"{_PREFIX}part_{i}"] for i in range(count))
        result = json.loads(payload)
        if not isinstance(result, dict):
            raise ValueError("encoded record must be an object")
        return result
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("corrupt Graphify database record") from exc
