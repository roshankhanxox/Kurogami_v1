"""Planner: plan() writes roots, expand() writes children -- offline via FakeLLM."""

from kurogami.adapters.llm.fake import FakeLLM
from kurogami.agents._prompt_loader import load_prompt
from kurogami.agents.planner import Planner, _NodeSpecBatch
from kurogami.contracts import GoalSpec, NodeKind, NodeResult, NodeSpec, PassCondition


def _goal() -> GoalSpec:
    return GoalSpec(
        raw_text="x",
        product_description="x",
        target_market="x",
        decision_type="market_entry",
        success_definition="x",
    )


def _node(node_id: str = "n_a", parent_ids: list[str] | None = None, depth: int = 0) -> NodeSpec:
    return NodeSpec(
        node_id=node_id,
        parent_ids=parent_ids or [],
        depth=depth,
        kind=NodeKind.ANALYSIS,
        title="title",
        node_goal="goal",
        generated_prompt="a sufficiently long generated prompt " * 3,
        pass_condition=PassCondition(assertions=[], semantic_check="ok?"),
    )


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
    roots = [_node("n_001"), _node("n_002")]
    prompt = load_prompt("plan").format(goal_json=goal.model_dump_json(indent=2))
    llm = FakeLLM(responses={prompt: _NodeSpecBatch(nodes=roots)})

    result = Planner(llm).plan(goal)

    assert result == roots


def test_expand_returns_the_llms_children():
    goal = _goal()
    parent = _node("n_001")
    result = _result("n_001")
    children = [_node("n_002", parent_ids=["n_001"], depth=1)]
    prompt = load_prompt("expand").format(
        parent_json=parent.model_dump_json(indent=2),
        result_json=result.model_dump_json(indent=2),
        goal_json=goal.model_dump_json(indent=2),
    )
    llm = FakeLLM(responses={prompt: _NodeSpecBatch(nodes=children)})

    assert Planner(llm).expand(parent, result, goal) == children


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
