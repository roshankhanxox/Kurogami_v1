"""Level 2 and the expansion step: an LLM writes its own child agents' prompts."""

from pydantic import BaseModel

from kurogami.agents._prompt_loader import load_prompt
from kurogami.contracts import GoalSpec, LLMPort, NodeResult, NodeSpec


class _NodeSpecBatch(BaseModel):
    """Decoding envelope only: the planner's LLM call returns a list, not one model."""

    nodes: list[NodeSpec]


class Planner:
    """Writes root nodes (plan) and, once a node passes, its children (expand).

    Two failure modes this is designed against (IMPLEMENTATION_PLAN.md Track B):
    bushy shallow trees, and vacuous pass conditions. Both are pushed onto the
    prompt (prompts/plan.md, prompts/expand.md), not enforced in code here --
    it is a prompting problem, not a coding one.
    """

    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    def plan(self, goal: GoalSpec) -> list[NodeSpec]:
        template = load_prompt("plan")
        prompt = template.format(goal_json=goal.model_dump_json(indent=2))
        response = self._llm.complete(prompt=prompt, schema=_NodeSpecBatch)
        return self._unpack(response.parsed)

    def expand(self, node: NodeSpec, result: NodeResult, goal: GoalSpec) -> list[NodeSpec]:
        template = load_prompt("expand")
        prompt = template.format(
            parent_json=node.model_dump_json(indent=2),
            result_json=result.model_dump_json(indent=2),
            goal_json=goal.model_dump_json(indent=2),
        )
        response = self._llm.complete(prompt=prompt, schema=_NodeSpecBatch)
        return self._unpack(response.parsed)

    @staticmethod
    def _unpack(parsed: BaseModel | None) -> list[NodeSpec]:
        if not isinstance(parsed, _NodeSpecBatch):
            raise TypeError(f"Planner expected a parsed _NodeSpecBatch, got {type(parsed).__name__}")
        return parsed.nodes
