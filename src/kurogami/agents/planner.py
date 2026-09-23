"""Level 2 and the expansion step: an LLM writes its own child agents' prompts."""

import logging

from pydantic import BaseModel

from kurogami.agents._prompt_loader import load_prompt
from kurogami.agents.rules import ALLOWED_FUNCTIONS, UnsafeAssertionError, validate_assertion
from kurogami.contracts import (
    GoalSpec,
    LLMPort,
    NodeKind,
    NodeResult,
    NodeSpec,
    PassCondition,
)

_log = logging.getLogger(__name__)


class _PlannerNodeSpec(BaseModel):
    """Everything in NodeSpec except context and injected_constraints.

    Those two fields are engine-managed (assembled ancestor context, and
    interrupt-injected constraints) -- the planner never legitimately fills
    them; they're always empty on a freshly created node. Excluding them
    also sidesteps a real incompatibility: NodeSpec gives them defaults, but
    OpenAI's Structured Outputs mode requires every schema property to
    appear in "required" with no implicit defaults, so a schema built
    directly from NodeSpec is rejected by the API.
    """

    node_id: str
    parent_ids: list[str]
    depth: int
    kind: NodeKind
    title: str
    node_goal: str
    generated_prompt: str
    pass_condition: PassCondition

    def to_node_spec(self) -> NodeSpec:
        return NodeSpec(**self.model_dump())


class _NodeSpecBatch(BaseModel):
    """Decoding envelope only: the planner's LLM call returns a list, not one model."""

    nodes: list[_PlannerNodeSpec]


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
        return self._complete_validated(prompt)

    def expand(self, node: NodeSpec, result: NodeResult, goal: GoalSpec) -> list[NodeSpec]:
        template = load_prompt("expand")
        prompt = template.format(
            parent_json=node.model_dump_json(indent=2),
            result_json=result.model_dump_json(indent=2),
            goal_json=goal.model_dump_json(indent=2),
        )
        return self._complete_validated(prompt)

    def _complete_validated(self, prompt: str) -> list[NodeSpec]:
        """Reject un-evaluable assertions at authoring time, not at run time.

        A node whose assertion can never be evaluated fails on every retry --
        the assertion is baked into the NodeSpec, so re-executing the node
        can't fix it (seen live: four identical FAILs until the budget broke).
        So: one corrective re-ask naming the bad assertions; anything still
        invalid after that is dropped and logged -- the semantic verifier
        still checks the node.
        """
        nodes = self._unpack(self._llm.complete(prompt=prompt, schema=_NodeSpecBatch).parsed)
        problems = _invalid_assertions(nodes)
        if not problems:
            return nodes

        retry_prompt = prompt + _correction_note(problems)
        nodes = self._unpack(
            self._llm.complete(prompt=retry_prompt, schema=_NodeSpecBatch).parsed
        )
        return [_drop_invalid_assertions(node) for node in nodes]

    @staticmethod
    def _unpack(parsed: BaseModel | None) -> list[NodeSpec]:
        if not isinstance(parsed, _NodeSpecBatch):
            raise TypeError(f"Planner expected a parsed _NodeSpecBatch, got {type(parsed).__name__}")
        return [node.to_node_spec() for node in parsed.nodes]


def _assertion_error(assertion: str) -> str | None:
    try:
        validate_assertion(assertion)
    except (UnsafeAssertionError, SyntaxError) as exc:
        return str(exc)
    return None


def _invalid_assertions(nodes: list[NodeSpec]) -> list[tuple[str, str, str]]:
    problems = []
    for node in nodes:
        for assertion in node.pass_condition.assertions:
            error = _assertion_error(assertion)
            if error is not None:
                problems.append((node.node_id, assertion, error))
    return problems


def _correction_note(problems: list[tuple[str, str, str]]) -> str:
    lines = "\n".join(f"- {node_id}: `{a}` -> {err}" for node_id, a, err in problems)
    return load_prompt("correct_assertions").format(
        problems=lines, allowed_functions=", ".join(ALLOWED_FUNCTIONS)
    )


def _drop_invalid_assertions(node: NodeSpec) -> NodeSpec:
    kept = []
    for assertion in node.pass_condition.assertions:
        error = _assertion_error(assertion)
        if error is None:
            kept.append(assertion)
        else:
            _log.warning("dropping un-evaluable assertion on %s: %r (%s)", node.node_id, assertion, error)
    if len(kept) == len(node.pass_condition.assertions):
        return node
    return node.model_copy(
        update={"pass_condition": node.pass_condition.model_copy(update={"assertions": kept})}
    )
