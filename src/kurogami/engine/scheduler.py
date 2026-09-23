"""Execution order: DFS, parents-before-children, skipping anything not PENDING."""

from kurogami.contracts import NodeSpec, NodeStatus
from kurogami.engine.store import TreeStore


def _parents_passed(store: TreeStore, node: NodeSpec) -> bool:
    return all(store.status(pid) == NodeStatus.PASSED for pid in node.parent_ids)


def next(store: TreeStore) -> NodeSpec | None:
    """The next runnable node: DFS from the roots, descending only through PASSED nodes.

    Returns the first PENDING node whose parents have all PASSED, or None if
    nothing is currently runnable.
    """
    visited: set[str] = set()

    def visit(node_id: str) -> NodeSpec | None:
        if node_id in visited:
            return None
        visited.add(node_id)

        node = store.get(node_id)
        status = store.status(node_id)
        if status == NodeStatus.PENDING and _parents_passed(store, node):
            return node
        if status == NodeStatus.PASSED:
            for child in store.children(node_id):
                found = visit(child.node_id)
                if found is not None:
                    return found
        return None

    for root_id in store.root_ids():
        found = visit(root_id)
        if found is not None:
            return found
    return None
