"""Assemble ancestor outputs into NodeSpec.context. Ancestors only, never siblings."""

import json

from kurogami.contracts import NodeSpec
from kurogami.engine.store import TreeStore

# Seen live: per-ancestor caps of 2000 chars minus 300 per level handed the final
# decision node 9,300 of the investigation's 49,000 chars -- 9 of its 10 inputs cut
# mid-JSON -- so it could only restate headlines and couldn't cross-check anything.
# A whole investigation is ~12k tokens; this budget is about that, and when it's
# exceeded the most distant ancestors are dropped whole rather than every input sliced.
CONTEXT_BUDGET_CHARS = 60_000


def assemble(store: TreeStore, node: NodeSpec) -> dict[str, str]:
    """Walk parent_ids transitively; nothing from siblings or the rest of the tree.

    Direct parents pass their full output -- that is what this node builds on.
    More distant ancestors pass their structured payload, whole, preferring it
    over raw output (ARCHITECTURE.md section 5a). Nothing is cut mid-string:
    nearest ancestors are added first, and any that don't fit the budget are
    left out entirely.
    """
    candidates: list[tuple[int, str, str]] = []
    visited: set[str] = set()
    frontier: list[tuple[str, int]] = [(pid, 1) for pid in node.parent_ids]
    while frontier:
        ancestor_id, distance = frontier.pop(0)
        if ancestor_id in visited:
            continue
        visited.add(ancestor_id)
        ancestor = store.get(ancestor_id)
        result = store.result(ancestor_id)
        if result is not None:
            if distance > 1 and result.structured:
                text = json.dumps(result.structured, sort_keys=True)
            else:
                text = result.output
            candidates.append((distance, ancestor_id, text))
        frontier.extend((pid, distance + 1) for pid in ancestor.parent_ids)

    context: dict[str, str] = {}
    used = 0
    for _, ancestor_id, text in sorted(candidates, key=lambda c: c[0]):
        if used + len(text) > CONTEXT_BUDGET_CHARS:
            continue
        context[ancestor_id] = text
        used += len(text)
    return context
