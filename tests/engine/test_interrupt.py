"""ScriptedInterrupt: pure, offline, pauses at exactly the given node ids."""

from kurogami.engine.interrupt import ScriptedInterrupt


def test_should_pause_only_at_scripted_node_ids(fixture_store):
    node = fixture_store.get("n_002")
    other = fixture_store.get("n_003")
    interrupt = ScriptedInterrupt(["n_002"])

    assert interrupt.should_pause(node) is True
    assert interrupt.should_pause(other) is False


def test_collect_returns_scripted_constraints_for_that_node(fixture_store):
    node = fixture_store.get("n_002")
    interrupt = ScriptedInterrupt(["n_002"], {"n_002": ["Stay under INR 500/month."]})

    assert interrupt.collect(node) == ["Stay under INR 500/month."]


def test_collect_returns_empty_for_an_unscripted_node(fixture_store):
    node = fixture_store.get("n_003")
    interrupt = ScriptedInterrupt(["n_002"], {"n_002": ["x"]})

    assert interrupt.collect(node) == []
