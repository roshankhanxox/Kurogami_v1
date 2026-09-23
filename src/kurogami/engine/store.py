"""TreeStore: the node graph, its status, and invalidation. No LLM calls happen here."""

from kurogami.contracts import BacktrackEvent, NodeResult, NodeSpec, NodeStatus, TreeSnapshot


class UnknownNodeError(KeyError):
    """Raised when a TreeStore operation references a node_id the store has never seen."""


class DuplicateNodeError(ValueError):
    """Raised when seed()/attach() is given a node_id the store already has.

    Seen live: a real planner call generated a root node and, later, an
    unrelated expand()-produced child with the same node_id. Without this
    check, _add() silently overwrote the root's spec and status with the
    child's -- corrupting the tree (the root vanished from its own subtree,
    while root_ids still listed it, producing an inconsistent render).
    """


class TreeStore:
    """Owns the node graph. Nodes are never deleted, only re-flagged.

    Kept even after invalidation so the trace can show what was thrown away
    (ARCHITECTURE.md section 5c).
    """

    def __init__(self) -> None:
        self._specs: dict[str, NodeSpec] = {}
        self._statuses: dict[str, NodeStatus] = {}
        self._results: dict[str, NodeResult] = {}
        self._root_ids: list[str] = []
        self._requeue_reasons: dict[str, str] = {}
        self._backtrack_events: list[BacktrackEvent] = []

    def seed(self, nodes: list[NodeSpec]) -> None:
        """Attach root nodes. Every node passed here must have no parents."""
        for node in nodes:
            if node.parent_ids:
                raise ValueError(f"seed() expects root nodes; {node.node_id} has parents")
            self._add(node)
            self._root_ids.append(node.node_id)

    def attach(self, parent: NodeSpec, children: list[NodeSpec]) -> None:
        """Attach children produced by planner.expand() beneath an already-known parent."""
        if parent.node_id not in self._specs:
            raise UnknownNodeError(parent.node_id)
        for child in children:
            if parent.node_id not in child.parent_ids:
                raise ValueError(f"{child.node_id} does not list {parent.node_id} in parent_ids")
            self._add(child)

    def _add(self, node: NodeSpec) -> None:
        if node.node_id in self._specs:
            raise DuplicateNodeError(
                f"node_id {node.node_id!r} already exists in the store; ids must be unique"
            )
        self._specs[node.node_id] = node
        self._statuses[node.node_id] = NodeStatus.PENDING

    def update_spec(self, node: NodeSpec) -> None:
        """Replace the stored spec for an already-known node, e.g. after context assembly."""
        self.get(node.node_id)
        self._specs[node.node_id] = node

    def get(self, node_id: str) -> NodeSpec:
        try:
            return self._specs[node_id]
        except KeyError:
            raise UnknownNodeError(node_id) from None

    def status(self, node_id: str) -> NodeStatus:
        try:
            return self._statuses[node_id]
        except KeyError:
            raise UnknownNodeError(node_id) from None

    def result(self, node_id: str) -> NodeResult | None:
        self.get(node_id)
        return self._results.get(node_id)

    def all_ids(self) -> list[str]:
        return list(self._specs.keys())

    def root_ids(self) -> list[str]:
        return list(self._root_ids)

    def children(self, node_id: str) -> list[NodeSpec]:
        self.get(node_id)
        return [n for n in self._specs.values() if node_id in n.parent_ids]

    def ancestors(self, node_id: str) -> list[NodeSpec]:
        """All transitive ancestors, each appearing once."""
        seen: dict[str, NodeSpec] = {}
        frontier = list(self.get(node_id).parent_ids)
        while frontier:
            pid = frontier.pop()
            if pid in seen:
                continue
            node = self.get(pid)
            seen[pid] = node
            frontier.extend(node.parent_ids)
        return list(seen.values())

    def descendants(self, node_id: str) -> list[NodeSpec]:
        """All transitive descendants, each appearing once. A DAG node counts once."""
        self.get(node_id)
        seen: dict[str, NodeSpec] = {}
        frontier = self.children(node_id)
        while frontier:
            node = frontier.pop()
            if node.node_id in seen:
                continue
            seen[node.node_id] = node
            frontier.extend(self.children(node.node_id))
        return list(seen.values())

    def mark_running(self, node_id: str) -> None:
        self.get(node_id)
        self._statuses[node_id] = NodeStatus.RUNNING

    def mark_passed(self, node_id: str, result: NodeResult) -> None:
        self.get(node_id)
        self._statuses[node_id] = NodeStatus.PASSED
        self._results[node_id] = result

    def mark_failed(self, node_id: str) -> None:
        self.get(node_id)
        self._statuses[node_id] = NodeStatus.FAILED

    def mark_skipped(self, node_id: str) -> None:
        """Budget exhausted. Only a still-PENDING node can be skipped."""
        self.get(node_id)
        if self._statuses[node_id] == NodeStatus.PENDING:
            self._statuses[node_id] = NodeStatus.SKIPPED

    def mark_dirty_descendants(self, node_id: str) -> list[str]:
        """Invalidate node_id's existing descendants, e.g. after a human interrupt."""
        ids = [n.node_id for n in self.descendants(node_id)]
        for nid in ids:
            self._statuses[nid] = NodeStatus.INVALIDATED
        return ids

    def invalidate_subtree(self, target_id: str) -> list[str]:
        """Mark target PENDING, every transitive descendant INVALIDATED. Nothing is deleted."""
        self.get(target_id)
        invalidated = [n.node_id for n in self.descendants(target_id)]
        for nid in invalidated:
            self._statuses[nid] = NodeStatus.INVALIDATED
        self._statuses[target_id] = NodeStatus.PENDING
        self._results.pop(target_id, None)
        return invalidated

    def requeue(self, node_id: str, *, with_failure_context: str | None = None) -> None:
        self.get(node_id)
        self._statuses[node_id] = NodeStatus.PENDING
        if with_failure_context is not None:
            self._requeue_reasons[node_id] = with_failure_context
        else:
            self._requeue_reasons.pop(node_id, None)

    def pop_requeue_reason(self, node_id: str) -> str | None:
        """Consume and clear the failure context attached by the last requeue, if any."""
        return self._requeue_reasons.pop(node_id, None)

    def record_backtrack(self, event: BacktrackEvent) -> None:
        self._backtrack_events.append(event)

    def has_pending(self) -> bool:
        return any(status == NodeStatus.PENDING for status in self._statuses.values())

    def pending_ids(self) -> list[str]:
        return [nid for nid, status in self._statuses.items() if status == NodeStatus.PENDING]

    def snapshot(self) -> TreeSnapshot:
        return TreeSnapshot(
            root_ids=self.root_ids(),
            specs=dict(self._specs),
            statuses=dict(self._statuses),
            results=dict(self._results),
            backtrack_events=list(self._backtrack_events),
        )
