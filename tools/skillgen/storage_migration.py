"""Exact, reviewable instruction edits for the Grafeo storage migration.

Apply the same transformation to frozen instruction baselines before their
round-trip comparison. Unrelated text still fails the existing multiset guards.
"""
from __future__ import annotations

import re


def migrate_storage_instructions(text: str) -> str:
    text = text.replace("graph.json", "graph.grafeo")
    text = text.replace(
        "A committed graphify-out/graph.grafeo is rewritten whole on every build, so a large diff to it in a pull request is expected. Do not flag it as churn or suggest skipping the graph rebuild or its hooks.",
        "Grafeo commits record deltas to graphify-out/graph.grafeo. Treat this database as binary; use graphify export json for a reviewable interchange snapshot. Keep rebuild hooks enabled.",
    )
    pattern = r'json\.loads\((Path\(([\"\']).*?\2\))\.read_text\([^)]*\)\)'
    def read(match):
        if any(key in match[1] for key in ("graph.grafeo", ".graphify_analysis.json", ".graphify_labels.json", "manifest.json")):
            return "__import__('graphify.store.io', fromlist=['read_json']).read_json(" + match[1] + ")"
        return match[0]
    text = re.sub(pattern, read, text)
    pattern = r'(Path\(([\"\']).*?\2\))\.write_text\('
    def write(match):
        if any(key in match[1] for key in (".graphify_analysis.json", ".graphify_labels.json", "manifest.json")):
            return "__import__('graphify.store.io', fromlist=['write_state_text']).write_state_text(" + match[1] + ", "
        return match[0]
    text = re.sub(pattern, write, text)
    pattern = r'(Path\(([\"\']).*?\2\))\.exists\(\)'
    def exists(match):
        if any(key in match[1] for key in (".graphify_analysis.json", ".graphify_labels.json", "manifest.json")):
            return "__import__('graphify.store.io', fromlist=['state_exists']).state_exists(" + match[1] + ")"
        return match[0]
    text = re.sub(pattern, exists, text)
    text = text.replace("cp graphify-out/graph.grafeo graphify-out/.graphify_old.json", "graphify export json --out graphify-out/.graphify_old.json")
    text = text.replace("cp graphify-out/graph.grafeo .graphify_old.json", "graphify export json --out .graphify_old.json")
    return text
