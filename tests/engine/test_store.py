"""TreeStore: invalidation is the core claim -- kill descendants, nothing else."""

import pytest

from kurogami.contracts import NodeKind, NodeSpec, NodeStatus, PassCondition
from kurogami.engine.demo_tree import FIXTURE_ORDER
from kurogami.engine.store import TreeStore, UnknownNodeError

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


def test_snapshot_round_trips_through_the_store(fixture_store):
    snapshot = fixture_store.snapshot()
    assert set(snapshot.specs.keys()) == set(FIXTURE_ORDER)
    assert all(status == NodeStatus.PASSED for status in snapshot.statuses.values())
    assert snapshot.root_ids == ["n_001"]
