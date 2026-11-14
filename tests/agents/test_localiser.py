"""Localiser: an LLM decides who is to blame for a failed cross-node check, and only then."""

from kurogami.agents.localiser import Localiser, _LocaliseResponse
from kurogami.contracts import (
    FailureReason,
    LLMResponse,
    NodeKind,
    NodeResult,
    NodeSpec,
    PassCondition,
)

_CROSS = "structured['price'] <= ancestors['wtp']['ceiling']"
_OWN = "len(structured['tiers']) >= 2"
_CONTEXT = {"wtp": '{"ceiling": 500}', "positioning": "Position it as premium.", "_backtrack_reason": "x"}


class _ScriptedLLM:
    def __init__(self, response: _LocaliseResponse) -> None:
        self.response, self.prompts = response, []

    def complete(self, *, prompt, system=None, schema=None, temperature=0.0):
        self.prompts.append(prompt)
        return LLMResponse(text="", parsed=self.response, tokens_in=1, tokens_out=1, latency_ms=0, model_id="f")


def _node() -> NodeSpec:
    return NodeSpec(
        node_id="pricing", parent_ids=["positioning", "wtp"], depth=2, kind=NodeKind.ANALYSIS,
        title="p", node_goal="set price", generated_prompt="set a price",
        pass_condition=PassCondition(assertions=[_OWN, _CROSS], semantic_check="?"),
    )


def _result() -> NodeResult:
    return NodeResult(node_id="pricing", output="INR 999", structured={"price": 999}, tokens_in=1,
                      tokens_out=1, latency_ms=1, model_id="f", prompt_version="v")


def _reason(evidence: str) -> FailureReason:
    return FailureReason(summary="assertion evaluated to False", violated="assertion", evidence=evidence)


def test_a_failed_check_on_the_nodes_own_output_costs_no_llm_call():
    llm = _ScriptedLLM(_LocaliseResponse(explanation="x", suspect_node_ids=["positioning"]))
    reason = _reason(_OWN)
    assert Localiser(llm).localise(_node(), _result(), reason, _CONTEXT) is reason
    assert llm.prompts == []


def test_a_failed_cross_node_check_is_localised_to_the_named_ancestor():
    llm = _ScriptedLLM(_LocaliseResponse(explanation="premium positioning forced it",
                                         suspect_node_ids=["positioning"]))
    localised = Localiser(llm).localise(_node(), _result(), _reason(_CROSS + "; with ..."), _CONTEXT)
    assert localised.suspect_node_ids == ["positioning"]
    assert "premium positioning forced it" in localised.summary
    assert "Position it as premium." in llm.prompts[0]


def test_invented_ids_and_the_node_itself_are_not_suspects():
    llm = _ScriptedLLM(_LocaliseResponse(explanation="x",
                                         suspect_node_ids=["pricing", "nonexistent", "_backtrack_reason"]))
    localised = Localiser(llm).localise(_node(), _result(), _reason(_CROSS), _CONTEXT)
    assert localised.suspect_node_ids == []
