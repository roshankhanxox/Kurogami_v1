"""Planner: plan() writes roots, expand() writes children -- offline via FakeLLM."""

from kurogami.adapters.llm.fake import FakeLLM
from kurogami.agents._prompt_loader import load_prompt
from kurogami.agents.planner import Planner, _NodeSpecBatch, _PlannerNodeSpec
from kurogami.contracts import GoalSpec, NodeKind, NodeResult, NodeSpec, PassCondition


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
