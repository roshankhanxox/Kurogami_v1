"""Scheduler: DFS, parents-before-children, skip anything not PENDING and ready."""

from kurogami.engine import scheduler
from kurogami.engine.demo_tree import dummy_result


def test_scheduler_respects_dependencies(empty_fixture_store):
    store = empty_fixture_store
    first = scheduler.next(store)
    assert first is not None
    assert first.node_id == "n_001"  # the only root; nothing else can be ready yet


def test_scheduler_advances_to_children_once_parent_passes(empty_fixture_store):
    store = empty_fixture_store
    store.mark_passed("n_001", dummy_result("n_001"))

    nxt = scheduler.next(store)
    assert nxt is not None
    assert nxt.node_id in {"n_002", "n_005"}  # n_001's two children, both now ready


def test_scheduler_returns_none_when_nothing_is_ready(empty_fixture_store):
    store = empty_fixture_store
    store.mark_running("n_001")  # in flight: not PENDING, and it's the only root
    assert scheduler.next(store) is None


def test_scheduler_waits_for_all_parents_of_a_synthesis_node(fixture_store):
    store = fixture_store
    # n_012's parents are n_008 (still PASSED) and n_011 (now back to PENDING).
    store.invalidate_subtree("n_011")

    nxt = scheduler.next(store)
    assert nxt is not None
    assert nxt.node_id == "n_011"  # n_012 must not be picked; one of its parents isn't PASSED
