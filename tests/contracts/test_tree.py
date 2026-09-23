"""Round-trip tests for TreeSnapshot and BacktrackEvent."""

from kurogami.contracts import (
    BacktrackEvent,
    NodeKind,
    NodeSpec,
    NodeStatus,
    PassCondition,
    TreeSnapshot,
)


def _spec(node_id: str, parent_ids: list[str], depth: int) -> NodeSpec:
    return NodeSpec(
        node_id=node_id,
        parent_ids=parent_ids,
        depth=depth,
        kind=NodeKind.ANALYSIS,
        title=node_id,
        node_goal=f"goal for {node_id}",
        generated_prompt=f"prompt for {node_id}",
        pass_condition=PassCondition(assertions=[], semantic_check="ok?"),
    )


def test_backtrack_event_round_trip():
    event = BacktrackEvent(
        failing_node_id="n_004_pricing",
        target_node_id="n_002_differentiation",
        invalidated_node_ids=["n_003_positioning", "n_004_pricing"],
        reason_summary="Pricing tiers exceed WTP ceiling.",
    )
    restored = BacktrackEvent.model_validate_json(event.model_dump_json())
    assert restored == event


def test_tree_snapshot_round_trip():
    root = _spec("n_001", [], 0)
    child = _spec("n_002", ["n_001"], 1)
    snapshot = TreeSnapshot(
        root_ids=["n_001"],
        specs={"n_001": root, "n_002": child},
        statuses={"n_001": NodeStatus.PASSED, "n_002": NodeStatus.PENDING},
    )
    restored = TreeSnapshot.model_validate_json(snapshot.model_dump_json())
    assert restored == snapshot


def test_tree_snapshot_defaults_are_empty():
    snapshot = TreeSnapshot(root_ids=["n_001"], specs={}, statuses={})
    assert snapshot.results == {}
    assert snapshot.backtrack_events == []
