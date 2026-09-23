"""TreeSnapshot / BacktrackEvent: the point-in-time view of the node store."""

from pydantic import BaseModel

from kurogami.contracts.node import NodeResult, NodeSpec, NodeStatus


class BacktrackEvent(BaseModel):
    """Record of one localisation + invalidation, for the trace and the renderer."""

    failing_node_id: str
    target_node_id: str
    invalidated_node_ids: list[str]
    reason_summary: str


class TreeSnapshot(BaseModel):
    """Immutable, serialisable view of TreeStore at a point in time."""

    root_ids: list[str]
    specs: dict[str, NodeSpec]
    statuses: dict[str, NodeStatus]
    results: dict[str, NodeResult] = {}
    backtrack_events: list[BacktrackEvent] = []
