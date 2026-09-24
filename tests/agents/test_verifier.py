"""Verifier: the semantic check, offline via FakeLLM. node_id/checked_by are never trusted."""

import pytest

from kurogami.adapters.llm.fake import FakeLLM
from kurogami.agents._prompt_loader import load_prompt
from kurogami.agents.verifier import Verifier, _VerifyResponse
from kurogami.contracts import FailureReason, NodeKind, NodeResult, NodeSpec, PassCondition


def _node() -> NodeSpec:
    return NodeSpec(
        node_id="n_004",
        parent_ids=["n_003"],
        depth=3,
        kind=NodeKind.DECISION,
        title="Pricing",
        node_goal="Propose pricing tiers.",
        generated_prompt="propose pricing",
        pass_condition=PassCondition(
            assertions=[],
            semantic_check="Do the tiers respect the WTP ceiling from n_001?",
        ),
    )


def _result() -> NodeResult:
    return NodeResult(
        node_id="n_004",
        output="Tier 2: INR 2000/month.",
        tokens_in=1,
        tokens_out=1,
        latency_ms=1,
        model_id="fake",
        prompt_version="v1",
    )


def _prompt_for(node: NodeSpec, result: NodeResult, context: dict[str, str]) -> str:
    import json

    return load_prompt("verify").format(
        node_json=node.model_dump_json(indent=2),
        result_json=result.model_dump_json(indent=2),
        context_json=json.dumps(context, indent=2),
        semantic_check=node.pass_condition.semantic_check,
    )


def test_verifier_returns_pass_and_forces_checked_by_llm():
    node, result, context = _node(), _result(), {"n_001": "WTP ceiling: 500"}
    prompt = _prompt_for(node, result, context)
    llm = FakeLLM(responses={prompt: _VerifyResponse(checked_claims=[], verdict="PASS")})

    verdict = Verifier(llm).check(node, result, context)

    assert verdict.verdict == "PASS"
    assert verdict.checked_by == "llm"
    assert verdict.node_id == "n_004"  # always the node under review, never trusted from the model


def test_verifier_returns_fail_with_suspect_node_ids():
    node, result, context = _node(), _result(), {"n_001": "WTP ceiling: 500"}
    prompt = _prompt_for(node, result, context)
    reason = FailureReason(
        summary="Tier 2 exceeds the WTP ceiling.",
        violated="semantic",
        evidence="Tier 2: INR 2000/month.",
        suspect_node_ids=["n_002"],
    )
    llm = FakeLLM(responses={prompt: _VerifyResponse(checked_claims=[], verdict="FAIL", reason=reason)})

    verdict = Verifier(llm).check(node, result, context)

    assert verdict.verdict == "FAIL"
    assert verdict.reason == reason


def test_verifier_rejects_a_fail_with_no_reason():
    node, result, context = _node(), _result(), {}
    prompt = _prompt_for(node, result, context)
    llm = FakeLLM(responses={prompt: _VerifyResponse(checked_claims=[], verdict="FAIL", reason=None)})

    with pytest.raises(ValueError, match="no reason"):
        Verifier(llm).check(node, result, context)
