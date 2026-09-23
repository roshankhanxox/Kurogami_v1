"""Round-trip and validation tests for NodeKind, NodeStatus, NodeSpec, NodeResult."""

import pytest
from pydantic import ValidationError

from kurogami.contracts import NodeKind, NodeResult, NodeSpec, NodeStatus, PassCondition


def _pass_condition() -> PassCondition:
    return PassCondition(
        assertions=["len(structured['competitors']) >= 5"],
        semantic_check="Do the pricing tiers respect the WTP ceiling from market sizing?",
    )


def _node_spec(**overrides) -> NodeSpec:
    kwargs = {
        "node_id": "n_001",
        "parent_ids": [],
        "depth": 0,
        "kind": NodeKind.ANALYSIS,
        "title": "Market sizing",
        "node_goal": "Establish the willingness-to-pay ceiling for the segment.",
        "generated_prompt": "Given the target market, estimate willingness to pay...",
        "pass_condition": _pass_condition(),
    }
    kwargs.update(overrides)
    return NodeSpec(**kwargs)


def test_node_spec_round_trip():
    spec = _node_spec()
    restored = NodeSpec.model_validate_json(spec.model_dump_json())
    assert restored == spec


def test_node_spec_parent_ids_is_a_list_not_scalar():
    spec = _node_spec(parent_ids=["n_000", "n_000b"])
    assert spec.parent_ids == ["n_000", "n_000b"]


def test_node_spec_defaults():
    spec = _node_spec()
    assert spec.context == {}
    assert spec.injected_constraints == []


def test_node_kind_rejects_unknown_value():
    kwargs = {
        "node_id": "n_001",
        "parent_ids": [],
        "depth": 0,
        "kind": "not_a_kind",
        "title": "x",
        "node_goal": "x",
        "generated_prompt": "x",
        "pass_condition": _pass_condition(),
    }
    with pytest.raises(ValidationError):
        NodeSpec(**kwargs)


def test_node_status_values():
    assert {s.value for s in NodeStatus} == {
        "pending",
        "running",
        "passed",
        "failed",
        "invalidated",
        "skipped",
    }


def test_node_result_round_trip():
    result = NodeResult(
        node_id="n_001",
        output="The WTP ceiling is roughly INR 500/month.",
        structured={"wtp_ceiling_inr": 500},
        tokens_in=120,
        tokens_out=340,
        latency_ms=850,
        model_id="fake-llm",
        prompt_version="execute.md@abc123",
    )
    restored = NodeResult.model_validate_json(result.model_dump_json())
    assert restored == result


def test_node_result_structured_defaults_to_empty_dict():
    result = NodeResult(
        node_id="n_001",
        output="x",
        tokens_in=1,
        tokens_out=1,
        latency_ms=1,
        model_id="fake-llm",
        prompt_version="execute.md@abc123",
    )
    assert result.structured == {}
