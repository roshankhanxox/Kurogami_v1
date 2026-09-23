"""Planner: plan() writes roots, expand() writes children -- offline via FakeLLM."""

from kurogami.adapters.llm.fake import FakeLLM
from kurogami.agents._prompt_loader import load_prompt
from kurogami.agents.planner import Planner, _NodeSpecBatch, _PlannerNodeSpec
from kurogami.contracts import (
    GoalSpec,
    LLMResponse,
    NodeKind,
    NodeResult,
    NodeSpec,
    PassCondition,
)


def _goal() -> GoalSpec:
    return GoalSpec(
        raw_text="x",
        product_description="x",
        target_market="x",
        decision_type="market_entry",
        success_definition="x",
    )


def _node_kwargs(node_id: str = "n_a", parent_ids: list[str] | None = None, depth: int = 0) -> dict:
    return {
        "node_id": node_id,
        "parent_ids": parent_ids or [],
        "depth": depth,
        "kind": NodeKind.ANALYSIS,
        "title": "title",
        "node_goal": "goal",
        "generated_prompt": "a sufficiently long generated prompt " * 3,
        "pass_condition": PassCondition(assertions=[], semantic_check="ok?"),
    }


def _node(node_id: str = "n_a", parent_ids: list[str] | None = None, depth: int = 0) -> NodeSpec:
    """What the planner is expected to return: a full NodeSpec, context/injected_constraints defaulted."""
    return NodeSpec(**_node_kwargs(node_id, parent_ids, depth))


def _planner_node(node_id: str = "n_a", parent_ids: list[str] | None = None, depth: int = 0) -> _PlannerNodeSpec:
    """What the LLM is actually asked to produce -- no context/injected_constraints."""
    return _PlannerNodeSpec(**_node_kwargs(node_id, parent_ids, depth))


def _result(node_id: str = "n_a") -> NodeResult:
    return NodeResult(
        node_id=node_id,
        output="x",
        tokens_in=1,
        tokens_out=1,
        latency_ms=1,
        model_id="fake",
        prompt_version="v1",
    )


def test_plan_returns_the_llms_node_specs():
    goal = _goal()
    llm_output = [_planner_node("n_001"), _planner_node("n_002")]
    expected = [_node("n_001"), _node("n_002")]
    prompt = load_prompt("plan").format(goal_json=goal.model_dump_json(indent=2))
    llm = FakeLLM(responses={prompt: _NodeSpecBatch(nodes=llm_output)})

    result = Planner(llm).plan(goal)

    assert result == expected


def test_expand_returns_the_llms_children():
    goal = _goal()
    parent = _node("n_001")
    result = _result("n_001")
    llm_output = [_planner_node("n_002", parent_ids=["n_001"], depth=1)]
    expected = [_node("n_002", parent_ids=["n_001"], depth=1)]
    prompt = load_prompt("expand").format(
        parent_json=parent.model_dump_json(indent=2),
        result_json=result.model_dump_json(indent=2),
        goal_json=goal.model_dump_json(indent=2),
    )
    llm = FakeLLM(responses={prompt: _NodeSpecBatch(nodes=llm_output)})

    assert Planner(llm).expand(parent, result, goal) == expected


def test_planner_node_spec_excludes_engine_managed_fields():
    """Regression guard: context/injected_constraints must never be part of what
    we ask the LLM to produce -- they're engine-managed, and a schema built
    from full NodeSpec (with these free-form/defaulted fields) is rejected
    by OpenAI's Structured Outputs mode. See planner.py's _PlannerNodeSpec
    docstring for the incident.
    """
    assert "context" not in _PlannerNodeSpec.model_fields
    assert "injected_constraints" not in _PlannerNodeSpec.model_fields


def test_to_node_spec_defaults_context_and_injected_constraints_empty():
    node = _planner_node("n_001").to_node_spec()
    assert node.context == {}
    assert node.injected_constraints == []


def test_expand_can_return_zero_children_to_terminate_a_branch():
    goal = _goal()
    parent = _node("n_001")
    result = _result("n_001")
    prompt = load_prompt("expand").format(
        parent_json=parent.model_dump_json(indent=2),
        result_json=result.model_dump_json(indent=2),
        goal_json=goal.model_dump_json(indent=2),
    )
    llm = FakeLLM(responses={prompt: _NodeSpecBatch(nodes=[])})

    assert Planner(llm).expand(parent, result, goal) == []


def _planner_node_with(node_id: str, assertions: list[str]) -> _PlannerNodeSpec:
    kwargs = _node_kwargs(node_id)
    kwargs["pass_condition"] = PassCondition(assertions=assertions, semantic_check="ok?")
    return _PlannerNodeSpec(**kwargs)


def test_plan_reasks_once_when_an_assertion_cannot_be_evaluated():
    """Regression: seen live -- an assertion calling a non-whitelisted function
    failed identically on every retry of its node until the budget broke,
    because re-executing a node can't change its own assertion. The planner
    now catches this at authoring time and asks the model to fix it.
    """
    goal = _goal()
    bad = _planner_node_with("n_001", ["open('x')"])
    good = _planner_node_with("n_001", ["len(structured['items']) >= 1"])
    llm = _ScriptedLLM([_NodeSpecBatch(nodes=[bad]), _NodeSpecBatch(nodes=[good])])

    [node] = Planner(llm).plan(goal)

    assert node.pass_condition.assertions == ["len(structured['items']) >= 1"]
    assert len(llm.prompts) == 2
    retry_prompt = llm.prompts[1]
    assert retry_prompt.startswith(llm.prompts[0])  # original prompt plus a correction
    assert "open('x')" in retry_prompt
    assert "cannot be evaluated" in retry_prompt


def test_plan_drops_assertions_still_invalid_after_the_reask():
    bad = _NodeSpecBatch(nodes=[_planner_node_with("n_001", ["open('x')", "len(structured['a']) > 0"])])
    llm = _ScriptedLLM([bad, bad])

    [node] = Planner(llm).plan(_goal())

    assert node.pass_condition.assertions == ["len(structured['a']) > 0"]
    assert len(llm.prompts) == 2  # original + exactly one corrective re-ask


def test_plan_does_not_reask_when_all_assertions_are_valid():
    good = _NodeSpecBatch(nodes=[_planner_node_with("n_001", ["len(structured['a']) > 0"])])
    llm = _ScriptedLLM([good])
    Planner(llm).plan(_goal())
    assert len(llm.prompts) == 1


def test_expand_also_validates_assertions():
    bad = _NodeSpecBatch(nodes=[_planner_node_with("n_002", ["structured.keys()"])])
    llm = _ScriptedLLM([bad, bad])

    [child] = Planner(llm).expand(_node("n_001"), _result("n_001"), _goal())

    assert child.pass_condition.assertions == []
    assert len(llm.prompts) == 2


class _ScriptedLLM:
    """Returns the given parsed batches in order, one per call; records every prompt."""

    def __init__(self, batches: list[_NodeSpecBatch]) -> None:
        self._batches = list(batches)
        self.prompts: list[str] = []

    def complete(self, *, prompt, system=None, schema=None, temperature=0.0):
        self.prompts.append(prompt)
        batch = self._batches.pop(0)
        return LLMResponse(
            text=batch.model_dump_json(), parsed=batch,
            tokens_in=1, tokens_out=1, latency_ms=0, model_id="fake",
        )
