"""Assemble ancestor outputs into NodeSpec.context. Ancestors only, never siblings."""

import json

from kurogami.contracts import NodeSpec
from kurogami.engine.store import TreeStore

_BASE_CHARS = 2000
_MIN_CHARS = 200
_DECAY_PER_DEPTH = 300


def _budget_for_distance(distance: int) -> int:
    """Nearer ancestors get more room; distant ones are truncated harder."""
    return max(_MIN_CHARS, _BASE_CHARS - _DECAY_PER_DEPTH * distance)


def assemble(store: TreeStore, node: NodeSpec) -> dict[str, str]:
    """Walk parent_ids transitively; nothing from siblings or the rest of the tree.

    Prefers each ancestor's structured payload over its raw output text, and
    truncates more aggressively the further back the ancestor sits, so a deep
    node doesn't blow its context window pulling in the whole tree
    (ARCHITECTURE.md section 5a).
    """
    context: dict[str, str] = {}
    visited: set[str] = set()
    frontier: list[tuple[str, int]] = [(pid, 1) for pid in node.parent_ids]
    while frontier:
        ancestor_id, distance = frontier.pop(0)
        if ancestor_id in visited:
            continue
        visited.add(ancestor_id)

        ancestor = store.get(ancestor_id)
        result = store.result(ancestor_id)
        budget = _budget_for_distance(distance)

        if result is not None and result.structured:
            text = json.dumps(result.structured, sort_keys=True)
        elif result is not None:
            text = result.output
        else:
            text = ""
        context[ancestor_id] = text[:budget]

        frontier.extend((pid, distance + 1) for pid in ancestor.parent_ids)
    return context
