"""Shared fixtures. Everything here runs offline, no network, no API key."""

import pytest

from kurogami.engine.demo_tree import FIXTURE_ORDER, build_fixture_tree, dummy_result


@pytest.fixture
def fixture_store():
    """The 12-node, depth-5 fixture tree, every node PASSED with a dummy result."""
    store = build_fixture_tree()
    for node_id in FIXTURE_ORDER:
        store.mark_passed(node_id, dummy_result(node_id))
    return store


@pytest.fixture
def empty_fixture_store():
    """The 12-node, depth-5 fixture tree, every node still PENDING."""
    return build_fixture_tree()
