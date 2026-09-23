"""Context assembly must include only ancestors, never siblings, and truncate by distance."""

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


def test_context_prefers_structured_over_output(fixture_store):
    fixture_store.mark_passed(
        "n_001",
        NodeResult(
            node_id="n_001",
            output="raw prose output",
            structured={"wtp_ceiling_inr": 500},
            tokens_in=1,
            tokens_out=1,
            latency_ms=1,
            model_id="fake",
            prompt_version="v1",
        ),
    )
    node = fixture_store.get("n_002")
    assembled = context.assemble(fixture_store, node)
    assert "wtp_ceiling_inr" in assembled["n_001"]
    assert "raw prose output" not in assembled["n_001"]


def test_context_truncates_further_ancestors_more(fixture_store):
    long_text = "x" * 5000
    for node_id in ("n_001", "n_002", "n_003"):
        fixture_store.mark_passed(
            node_id,
            NodeResult(
                node_id=node_id,
                output=long_text,
                tokens_in=1,
                tokens_out=1,
                latency_ms=1,
                model_id="fake",
                prompt_version="v1",
            ),
        )
    node = fixture_store.get("n_004")
    assembled = context.assemble(fixture_store, node)
    assert len(assembled["n_003"]) > len(assembled["n_002"]) > len(assembled["n_001"])
