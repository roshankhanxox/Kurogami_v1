"""Fault localisation: shallowest ancestor suspect, not the failing node itself."""

from kurogami.contracts import FailureReason, NodeStatus
from kurogami.engine import backtrack


def test_localiser_picks_shallowest_ancestor(fixture_store):
    reason = FailureReason(
        summary="Pricing exceeds the WTP ceiling established by market sizing.",
        violated="semantic",
        evidence="Tier 2 priced above the ceiling.",
        suspect_node_ids=["n_003", "n_002"],  # n_002 (depth 1) is shallower than n_003 (depth 2)
    )
    target = backtrack.locate(reason, fixture_store, failing_node_id="n_004")
    assert target == "n_002"


def test_localiser_falls_back_to_failing_node_when_no_ancestor_suspects(fixture_store):
    reason = FailureReason(
        summary="Output is malformed.",
        violated="schema",
        evidence="not JSON",
        suspect_node_ids=["n_006"],  # a real node, but not an ancestor of n_004
    )
    target = backtrack.locate(reason, fixture_store, failing_node_id="n_004")
    assert target == "n_004"


def test_localiser_ignores_suspects_that_are_not_ancestors(fixture_store):
    reason = FailureReason(
        summary="x",
        violated="semantic",
        evidence="x",
        suspect_node_ids=["n_006", "n_002"],  # n_006 is on a sibling branch, not an ancestor
    )
    target = backtrack.locate(reason, fixture_store, failing_node_id="n_004")
    assert target == "n_002"


def test_apply_invalidates_and_requeues_the_target(fixture_store):
    reason = FailureReason(
        summary="Pricing exceeds the WTP ceiling.",
        violated="semantic",
        evidence="x",
        suspect_node_ids=["n_002"],
    )
    event = backtrack.apply(fixture_store, "n_004", reason)

    assert event.target_node_id == "n_002"
    assert event.failing_node_id == "n_004"
    assert set(event.invalidated_node_ids) == {"n_003", "n_004", "n_009", "n_010", "n_011", "n_012"}
    assert fixture_store.status("n_002") == NodeStatus.PENDING
    assert fixture_store.pop_requeue_reason("n_002") == f"{reason.summary} Evidence: x"


def test_apply_falls_back_to_self_when_no_ancestor_is_suspected(fixture_store):
    reason = FailureReason(
        summary="x",
        violated="schema",
        evidence="x",
        suspect_node_ids=[],
    )
    event = backtrack.apply(fixture_store, "n_012", reason)
    assert event.target_node_id == "n_012"
    assert event.invalidated_node_ids == []  # n_012 is a leaf
    assert fixture_store.status("n_012") == NodeStatus.PENDING
