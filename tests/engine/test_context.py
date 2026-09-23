"""Context assembly: ancestors only, never siblings; nothing sliced mid-string."""

from kurogami.contracts import NodeResult
from kurogami.engine import context

_ANCESTORS_OF_PRICING = {"n_001", "n_002", "n_003"}
_NOT_ANCESTORS_OF_PRICING = {"n_005", "n_006", "n_007", "n_008", "n_009", "n_010", "n_011", "n_012"}


def test_context_excludes_siblings(fixture_store):
    node = fixture_store.get("n_004")  # pricing: parent n_003
    assembled = context.assemble(fixture_store, node)

    assert set(assembled.keys()) == _ANCESTORS_OF_PRICING
    assert not (set(assembled.keys()) & _NOT_ANCESTORS_OF_PRICING)


def test_context_is_empty_for_a_root_node(fixture_store):
    node = fixture_store.get("n_001")
    assert context.assemble(fixture_store, node) == {}


def _pass(store, node_id: str, output: str, structured: dict | None = None) -> None:
    store.mark_passed(
        node_id,
        NodeResult(
            node_id=node_id, output=output, structured=structured or {}, tokens_in=1,
            tokens_out=1, latency_ms=1, model_id="fake", prompt_version="v1",
        ),
    )


def test_a_direct_parent_passes_its_full_output():
    """Seen live: the decision node got 1,700 of its parent's chars, cut mid-JSON."""
    from kurogami.engine.demo_tree import build_fixture_tree

    store = build_fixture_tree()
    long_output = "finding " * 2000  # 16,000 chars
    _pass(store, "n_001", long_output, {"wtp_ceiling_inr": 500})

    assembled = context.assemble(store, store.get("n_002"))  # n_001 is its parent

    assert assembled["n_001"] == long_output


def test_a_distant_ancestor_passes_its_structured_payload_whole(fixture_store):
    _pass(fixture_store, "n_001", "raw prose output", {"wtp_ceiling_inr": 500})

    assembled = context.assemble(fixture_store, fixture_store.get("n_003"))  # grandparent

    assert assembled["n_001"] == '{"wtp_ceiling_inr": 500}'


def test_over_budget_the_most_distant_ancestors_are_dropped_whole_never_sliced(fixture_store):
    big = "x" * (context.CONTEXT_BUDGET_CHARS // 2 + 1)
    for node_id in ("n_001", "n_002", "n_003"):
        _pass(fixture_store, node_id, big)

    assembled = context.assemble(fixture_store, fixture_store.get("n_004"))

    assert list(assembled) == ["n_003"]  # nearest kept; farther ones dropped entirely
    assert assembled["n_003"] == big
