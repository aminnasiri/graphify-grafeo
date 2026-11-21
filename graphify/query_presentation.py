"""Local, evidence-only descriptions of retrieved graph relationships."""
from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

from graphify.build import edge_datas
from graphify.security import sanitize_label


def describe_subgraph(
    graph: nx.Graph,
    nodes: Iterable[str],
    edges: Iterable[tuple[str, str]],
    seeds: list[str],
    token_budget: int,
    *,
    direct_only: bool = False,
) -> str:
    """Describe stored facts without inferring execution order or intent.

    Seed relationships precede wider context; implementation precedes test
    evidence. Like the retrieval renderer, the budget is approximately chars/4.
    """
    def clean(value: object) -> str:
        return sanitize_label(str(value or ""))

    def label(nid: str) -> str:
        return clean(graph.nodes[nid].get("label", nid))

    def location(data: dict) -> str:
        return ":".join(clean(data.get(k)) for k in ("source_file", "source_location") if data.get(k))

    edges = list(edges)
    selected = set(nodes)
    roots = set(seeds)
    lines = ["Graph evidence (relationships do not establish execution order):"]
    for nid in seeds:
        data = graph.nodes[nid]
        at = location(data)
        lines.append(f"{label(nid)} is defined at {at or 'an unknown source location'}.")
    # Extractors retain docstrings as rationale nodes. Surface their own words
    # before relationship evidence without inventing a summary from symbol names.
    for nid in seeds:
        for u, v in edges:
            if nid not in (u, v):
                continue
            other = v if u == nid else u
            if other not in graph:
                continue
            if any(d.get("relation") == "rationale_for" for d in edge_datas(graph, u, v)):
                text = label(other)
                if text:
                    lines.append(f"Recorded description: {text} [{location(graph.nodes[other]) or 'source unknown'}]")
    facts = []
    seen = set()
    for u, v in edges:
        if u not in selected or v not in selected:
            continue
        for data in edge_datas(graph, u, v):
            src, tgt = data.get("_src", u), data.get("_tgt", v)
            if {src, tgt} != {u, v}:
                src, tgt = u, v
            direct = src in roots or tgt in roots
            if direct_only and not direct:
                continue
            rel = clean(data.get("relation")) or "is connected to"
            key = (src, tgt, rel, location(data))
            if key in seen:
                continue
            seen.add(key)
            source = str(graph.nodes[src].get("source_file", "")).replace("\\", "/")
            is_test = source.startswith("tests/") or "/tests/" in source
            confidence = clean(data.get("confidence")) or "confidence unspecified"
            at = location(data) or location(graph.nodes[src])
            fact = f"- {label(src)} {rel.replace('_', ' ')} {label(tgt)}. [{confidence}; {at or 'source unknown'}]"
            facts.append(((not direct, is_test, rel != "calls"), fact))
    facts.sort(key=lambda pair: pair[0])
    limit = max(0, token_budget * 4)
    omitted = 0
    shown = 0
    for _, fact in facts:
        if (len("\n".join([*lines, fact])) + 160 > limit
                or (direct_only and token_budget < 100_000 and shown >= 12)):
            omitted += 1
        else:
            lines.append(fact)
            shown += 1
    if not facts:
        lines.append("No matching relationships were found within this traversal.")
    if omitted:
        hint = "use --all" if direct_only else "increase --budget or narrow --depth/--context"
        lines.append(f"{omitted} additional relationships; {hint}.")
    lines.append("Use --raw for graph records.")
    return "\n".join(lines)
