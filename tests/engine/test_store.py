"""TreeStore: invalidation is the core claim -- kill descendants, nothing else."""

import pytest

from kurogami.contracts import NodeKind, NodeResult, NodeSpec, NodeStatus, PassCondition
from kurogami.engine.demo_tree import FIXTURE_ORDER
from kurogami.engine.store import DuplicateNodeError, TreeStore, UnknownNodeError


def _dummy_result(node_id: str) -> NodeResult:
    return NodeResult(
        node_id=node_id, output="x", tokens_in=1, tokens_out=1, latency_ms=1,
        model_id="fake", prompt_version="v1",
    )

_DIFFERENTIATION_DESCENDANTS = {"n_003", "n_004", "n_009", "n_010", "n_011", "n_012"}
_UNTOUCHED_BY_DIFFERENTIATION_INVALIDATION = {"n_001", "n_005", "n_006", "n_007", "n_008"}


def test_invalidate_kills_only_descendants(fixture_store):
    store = fixture_store

    invalidated = store.invalidate_subtree("n_002")

    assert set(invalidated) == _DIFFERENTIATION_DESCENDANTS
    for node_id in _DIFFERENTIATION_DESCENDANTS:
        assert store.status(node_id) == NodeStatus.INVALIDATED
    assert store.status("n_002") == NodeStatus.PENDING
    for node_id in _UNTOUCHED_BY_DIFFERENTIATION_INVALIDATION:
        assert store.status(node_id) == NodeStatus.PASSED


def test_invalidate_subtree_of_leaf_has_no_descendants(fixture_store):
    store = fixture_store
    invalidated = store.invalidate_subtree("n_012")
    assert invalidated == []
    assert store.status("n_012") == NodeStatus.PENDING


def test_invalidate_subtree_clears_the_target_result(fixture_store):
    store = fixture_store
    assert store.result("n_002") is not None
    store.invalidate_subtree("n_002")
    assert store.result("n_002") is None


def test_ancestors_are_transitive(fixture_store):
    ancestor_ids = {n.node_id for n in fixture_store.ancestors("n_012")}
    assert ancestor_ids == {
        "n_001",
        "n_002",
        "n_005",
        "n_006",
        "n_007",
        "n_008",
        "n_009",
        "n_010",
        "n_011",
    }


def test_descendants_are_transitive(fixture_store):
    descendant_ids = {n.node_id for n in fixture_store.descendants("n_001")}
    assert descendant_ids == set(FIXTURE_ORDER) - {"n_001"}


def test_seed_rejects_nodes_with_parents():
    store = TreeStore()
    bad = NodeSpec(
        node_id="n_x",
        parent_ids=["n_y"],
        depth=1,
        kind=NodeKind.ANALYSIS,
        title="x",
        node_goal="x",
        generated_prompt="x",
        pass_condition=PassCondition(assertions=[], semantic_check="ok?"),
    )
    with pytest.raises(ValueError, match="root nodes"):
        store.seed([bad])


def test_unknown_node_id_raises():
    store = TreeStore()
    with pytest.raises(UnknownNodeError):
        store.get("does_not_exist")


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


def test_seed_rejects_a_duplicate_node_id():
    store = TreeStore()
    store.seed([_spec("n_a", [], 0), _spec("n_b", [], 0)])
    with pytest.raises(DuplicateNodeError):
        store.seed([_spec("n_a", [], 0)])


def test_attach_rejects_a_child_id_colliding_with_an_existing_root():
    """Regression: seen live -- a planner call produced a root node and,
    later, an unrelated expand()-produced child with the same node_id. This
    must be rejected, not silently overwrite the root's spec and status.
    """
    store = TreeStore()
    store.seed([_spec("n_a", [], 0), _spec("marketing_strategy", [], 0)])
    store.mark_passed("n_a", _dummy_result("n_a"))

    with pytest.raises(DuplicateNodeError):
        store.attach(store.get("n_a"), [_spec("marketing_strategy", ["n_a"], 1)])

    # the original root must be untouched
    assert store.get("marketing_strategy").parent_ids == []
    assert store.status("marketing_strategy") == NodeStatus.PENDING


def test_snapshot_round_trips_through_the_store(fixture_store):
    snapshot = fixture_store.snapshot()
    assert set(snapshot.specs.keys()) == set(FIXTURE_ORDER)
    assert all(status == NodeStatus.PASSED for status in snapshot.statuses.values())
    assert snapshot.root_ids == ["n_001"]


# --- scoped planning: blueprint seeding, regeneration, gap-fill wiring ------------------


def _chain() -> TreeStore:
    """a -> b -> c, with c's check and assertion naming b."""
    store = TreeStore()
    c = _spec("c", ["b"], 2).model_copy(
        update={
            "generated_prompt": "use the findings of b",
            "pass_condition": PassCondition(
                assertions=["context['b'] != ''"], semantic_check="Consistent with b?"
            ),
        }
    )
    store.seed_blueprint([_spec("a", [], 0), _spec("b", ["a"], 1), c])
    return store


def test_seed_blueprint_rejects_a_node_before_its_parent():
    store = TreeStore()
    with pytest.raises(ValueError, match="before its parents"):
        store.seed_blueprint([_spec("b", ["a"], 1), _spec("a", [], 0)])


def test_regeneration_clones_the_subtree_and_remaps_ancestor_references():
    store = _chain()
    for n in ("a", "b", "c"):
        store.mark_passed(n, _dummy_result(n))
    store.invalidate_subtree("a")
    store.mark_passed("a", _dummy_result("a"))

    assert store.awaiting_regeneration("a")
    clones = {c.node_id: c for c in store.regenerate_subtree("a")}

    assert set(clones) == {"b~r1", "c~r1"}
    c = clones["c~r1"]
    assert c.parent_ids == ["b~r1"]
    assert c.generated_prompt == "use the findings of b~r1"
    assert c.pass_condition.assertions == ["context['b~r1'] != ''"]
    assert c.pass_condition.semantic_check == "Consistent with b~r1?"
    assert store.status("b") == NodeStatus.INVALIDATED  # originals kept, flagged
    assert store.status("b~r1") == NodeStatus.PENDING
    assert not store.awaiting_regeneration("a")


def test_a_higher_backtrack_before_regeneration_still_rebuilds_the_whole_subtree():
    """b is backtracked (c invalidated) but, before b re-passes, a is backtracked too.
    Regenerating a must rebuild b AND c -- not just what the second backtrack touched.
    """
    store = _chain()
    for n in ("a", "b", "c"):
        store.mark_passed(n, _dummy_result(n))
    store.invalidate_subtree("b")  # c invalidated, b pending
    store.invalidate_subtree("a")  # b invalidated, a pending
    store.mark_passed("a", _dummy_result("a"))

    clones = {c.node_id: c for c in store.regenerate_subtree("a")}

    assert set(clones) == {"b~r1", "c~r1"}
    assert clones["c~r1"].parent_ids == ["b~r1"]
    assert not store.awaiting_regeneration("b")  # stale entry cleared with its node


def test_second_regeneration_gets_the_next_version():
    store = _chain()
    for _ in range(2):
        for n in [x for x in store.all_ids() if store.status(x) == NodeStatus.PENDING]:
            store.mark_passed(n, _dummy_result(n))
        store.invalidate_subtree("a")
        store.mark_passed("a", _dummy_result("a"))
        store.regenerate_subtree("a")
    assert {"b~r1", "c~r1", "b~r2", "c~r2"} <= set(store.all_ids())
    assert store.get("c~r2").parent_ids == ["b~r2"]


def test_gap_node_becomes_an_extra_parent_of_pending_children():
    store = _chain()
    store.mark_passed("a", _dummy_result("a"))

    added = store.add_gap_node(_spec("gap", ["a"], 1), "a", max_depth=5)

    assert added is True
    assert store.get("b").parent_ids == ["a", "gap"]
    assert store.get("b").depth == 2
    assert store.get("c").depth == 3


def test_gap_node_that_would_exceed_the_depth_limit_changes_nothing():
    store = _chain()
    store.mark_passed("a", _dummy_result("a"))

    added = store.add_gap_node(_spec("gap", ["a"], 1), "a", max_depth=2)

    assert added is False
    assert "gap" not in store.all_ids()
    assert store.get("b").parent_ids == ["a"]
    assert store.get("c").depth == 2


def test_gap_node_with_a_colliding_id_is_rejected():
    store = _chain()
    with pytest.raises(DuplicateNodeError):
        store.add_gap_node(_spec("b", ["a"], 1), "a", max_depth=5)
