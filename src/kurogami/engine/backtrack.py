"""Fault localisation and invalidation policy. The failing node is usually not the wrong node."""

from kurogami.contracts import BacktrackEvent, FailureReason
from kurogami.engine.store import TreeStore


def locate(reason: FailureReason, store: TreeStore, failing_node_id: str) -> str:
    """The shallowest ancestor the verifier actually suspects, or the failing node itself.

    Intersects the verifier's suspect_node_ids with the failing node's real
    ancestor set (a hallucinated or irrelevant suspect id is ignored), then
    picks the shallowest — repairing the shallowest wrong premise repairs
    everything beneath it (ARCHITECTURE.md section 5b).
    """
    ancestor_ids = {a.node_id for a in store.ancestors(failing_node_id)}
    candidates = [nid for nid in reason.suspect_node_ids if nid in ancestor_ids]
    if not candidates:
        return failing_node_id
    return min(candidates, key=lambda nid: store.get(nid).depth)


def apply(store: TreeStore, failing_node_id: str, reason: FailureReason) -> BacktrackEvent:
    """Locate the target, invalidate its subtree, and requeue it with the failure context."""
    target_id = locate(reason, store, failing_node_id)
    invalidated = store.invalidate_subtree(target_id)
    store.requeue(target_id, with_failure_context=reason.summary)

    event = BacktrackEvent(
        failing_node_id=failing_node_id,
        target_node_id=target_id,
        invalidated_node_ids=invalidated,
        reason_summary=reason.summary,
    )
    store.record_backtrack(event)
    return event
