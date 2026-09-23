"""The Master: scope the whole investigation, then write every child agent's prompt.

Seen live, open-ended expansion (decide children one node at a time) never knew
when the job was done: it drilled one topic to the depth cap while sibling
branches never ran. So the Master now works out up front exactly which
questions the answer needs, writes the complete bounded tree, and only grows
it at runtime when a node reports a prerequisite nobody planned for.
"""

import json
import logging
from typing import TypeVar

from pydantic import BaseModel

from kurogami.agents._prompt_loader import load_prompt
from kurogami.agents.rules import ALLOWED_FUNCTIONS, UnsafeAssertionError, validate_assertion
from kurogami.contracts import (
    BudgetLimits,
    GoalSpec,
    LLMPort,
    NodeKind,
    NodeResult,
    NodeSpec,
    PassCondition,
)

_log = logging.getLogger(__name__)

MIN_SCOPE_ITEMS = 10
MAX_SCOPE_ITEMS = 14
MIN_TREE_DEPTH = 4
MISSING_PREREQUISITES_KEY = "missing_prerequisites"

_Model = TypeVar("_Model", bound=BaseModel)


class BlueprintError(RuntimeError):
    """The Master could not produce a valid plan, even after one correction."""


class _ScopeItem(BaseModel):
    id: str
    question: str
    kind: NodeKind
    depends_on: list[str]


class _Scope(BaseModel):
    items: list[_ScopeItem]


class _BlueprintNode(BaseModel):
    """What the LLM writes per node; structure (parents, depth) comes from the scope.

    Excludes NodeSpec's engine-managed context/injected_constraints: a free-form
    dict field is also unrepresentable in OpenAI's strict Structured Outputs.
    """

    node_id: str
    title: str
    node_goal: str
    generated_prompt: str
    pass_condition: PassCondition


class _Blueprint(BaseModel):
    nodes: list[_BlueprintNode]


class _GapFill(BaseModel):
    node: _BlueprintNode | None


class Planner:
    """Plans the complete tree up front; expand() is bounded gap-filling only."""

    def __init__(self, llm: LLMPort, *, max_depth: int = BudgetLimits().max_depth) -> None:
        self._llm = llm
        self._max_depth = max_depth
        self._scope: list[_ScopeItem] = []
        self._issued_ids: set[str] = set()

    def plan(self, goal: GoalSpec) -> list[NodeSpec]:
        """The whole tree, in topological order. Raises BlueprintError if unfixable."""
        scope = self._scope_goal(goal)
        nodes = self._write_blueprint(goal, scope)
        self._scope = scope
        self._issued_ids = {n.node_id for n in nodes}
        return nodes

    def expand(self, node: NodeSpec, result: NodeResult, goal: GoalSpec) -> list[NodeSpec]:
        """At most one node, and only if this node reported an unplanned prerequisite."""
        reported = result.structured.get(MISSING_PREREQUISITES_KEY)
        gaps = [g for g in reported if isinstance(g, str) and g.strip()] if isinstance(
            reported, list
        ) else []
        if not gaps:
            return []

        prompt = load_prompt("gap_fill").format(
            goal_json=goal.model_dump_json(indent=2),
            scope_json=_scope_json(self._scope),
            node_json=node.model_dump_json(indent=2, exclude={"context"}),
            output=result.output,
            gaps="\n".join(f"- {g}" for g in gaps),
            reporter_id=node.node_id,
            allowed_functions=", ".join(ALLOWED_FUNCTIONS),
        )
        filled = self._ask(prompt, _GapFill).node
        if filled is None:
            return []
        if filled.node_id in self._issued_ids:
            _log.warning("gap-fill reused existing id %s; dropped", filled.node_id)
            return []
        if node.node_id not in filled.pass_condition.semantic_check:
            _log.warning("gap-fill %s does not name its reporter; dropped", filled.node_id)
            return []

        self._issued_ids.add(filled.node_id)
        return [
            _to_node_spec(
                filled,
                parent_ids=[node.node_id],
                depth=node.depth + 1,
                kind=NodeKind.ANALYSIS,
            )
        ]

    def _scope_goal(self, goal: GoalSpec) -> list[_ScopeItem]:
        prompt = load_prompt("scope").format(
            goal_json=goal.model_dump_json(indent=2),
            min_items=MIN_SCOPE_ITEMS,
            max_items=MAX_SCOPE_ITEMS,
            min_levels=MIN_TREE_DEPTH + 1,
            max_levels=self._max_depth,
        )
        scope = self._ask(prompt, _Scope).items
        problems = _scope_problems(scope, self._max_depth)
        if problems:
            scope = self._ask(prompt + _correction(problems), _Scope).items
            problems = _scope_problems(scope, self._max_depth)
            if problems:
                raise BlueprintError("scope still invalid after correction: " + "; ".join(problems))
        return scope

    def _write_blueprint(self, goal: GoalSpec, scope: list[_ScopeItem]) -> list[NodeSpec]:
        prompt = load_prompt("plan").format(
            goal_json=goal.model_dump_json(indent=2),
            scope_json=_scope_json(scope),
            allowed_functions=", ".join(ALLOWED_FUNCTIONS),
        )
        blueprint = self._ask(prompt, _Blueprint).nodes
        problems = _blueprint_problems(blueprint, scope)
        if problems:
            blueprint = self._ask(prompt + _correction(problems), _Blueprint).nodes
            # Unevaluable assertions can be dropped; structural problems cannot.
            problems = _blueprint_problems(blueprint, scope, check_assertions=False)
            if problems:
                raise BlueprintError(
                    "blueprint still invalid after correction: " + "; ".join(problems)
                )
        return _assemble(scope, blueprint)

    def _ask(self, prompt: str, schema: type[_Model]) -> _Model:
        parsed = self._llm.complete(prompt=prompt, schema=schema).parsed
        if not isinstance(parsed, schema):
            raise TypeError(f"Planner expected a parsed {schema.__name__}, got {type(parsed).__name__}")
        return parsed


def _scope_json(scope: list[_ScopeItem]) -> str:
    return json.dumps([item.model_dump(mode="json") for item in scope], indent=2)


def _correction(problems: list[str]) -> str:
    return load_prompt("correct_blueprint").format(
        problems="\n".join(f"- {p}" for p in problems),
        allowed_functions=", ".join(ALLOWED_FUNCTIONS),
    )


def _topological_order(deps: dict[str, list[str]]) -> list[str] | None:
    """Kahn's algorithm; None if there is a cycle."""
    remaining = {node: set(parents) for node, parents in deps.items()}
    order: list[str] = []
    while remaining:
        ready = sorted(n for n, parents in remaining.items() if not parents)
        if not ready:
            return None
        for n in ready:
            order.append(n)
            del remaining[n]
        for parents in remaining.values():
            parents.difference_update(ready)
    return order


def _depths(order: list[str], deps: dict[str, list[str]]) -> dict[str, int]:
    depth: dict[str, int] = {}
    for n in order:
        depth[n] = 1 + max((depth[p] for p in deps[n]), default=-1)
    return depth


def _ancestors(node_id: str, deps: dict[str, list[str]]) -> set[str]:
    seen: set[str] = set()
    frontier = list(deps.get(node_id, []))
    while frontier:
        p = frontier.pop()
        if p not in seen:
            seen.add(p)
            frontier.extend(deps.get(p, []))
    return seen


def _scope_problems(scope: list[_ScopeItem], max_depth: int) -> list[str]:
    problems: list[str] = []
    ids = [item.id for item in scope]
    if not MIN_SCOPE_ITEMS <= len(ids) <= MAX_SCOPE_ITEMS:
        problems.append(
            f"has {len(ids)} items; it must have {MIN_SCOPE_ITEMS}-{MAX_SCOPE_ITEMS}"
        )
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        problems.append(f"duplicate ids: {duplicates}")
    deps = {item.id: item.depends_on for item in scope}
    for item in scope:
        unknown = [d for d in item.depends_on if d not in deps]
        if unknown:
            problems.append(f"{item.id} depends on unknown ids {unknown}")
        if item.id in item.depends_on:
            problems.append(f"{item.id} depends on itself")
    if problems:
        return problems

    order = _topological_order(deps)
    if order is None:
        return ["the dependencies contain a cycle"]

    depended_on = {d for item in scope for d in item.depends_on}
    sinks = [item for item in scope if item.id not in depended_on]
    decisions = [item.id for item in scope if item.kind == NodeKind.DECISION]
    if len(sinks) != 1:
        problems.append(
            f"exactly one item must be the final one nothing depends on; found "
            f"{[s.id for s in sinks]} -- make them all feed the final decision"
        )
    elif sinks[0].kind != NodeKind.DECISION:
        problems.append(f"the final item {sinks[0].id} must have kind 'decision'")
    if len(decisions) != 1:
        problems.append(f"exactly one item may have kind 'decision'; found {decisions}")

    deepest = max(_depths(order, deps).values())
    if not MIN_TREE_DEPTH <= deepest <= max_depth - 1:
        problems.append(
            f"the longest dependency chain has {deepest + 1} items; it must have "
            f"{MIN_TREE_DEPTH + 1}-{max_depth}"
        )
    return problems


def _blueprint_problems(
    blueprint: list[_BlueprintNode], scope: list[_ScopeItem], *, check_assertions: bool = True
) -> list[str]:
    problems: list[str] = []
    scope_ids = {item.id for item in scope}
    written = [n.node_id for n in blueprint]
    missing = sorted(scope_ids - set(written))
    extra = sorted(set(written) - scope_ids)
    if missing:
        problems.append(f"no node written for scope items {missing}")
    if extra:
        problems.append(f"nodes not in the scope: {extra}")
    duplicates = sorted({i for i in written if written.count(i) > 1})
    if duplicates:
        problems.append(f"more than one node for {duplicates}")

    deps = {item.id: item.depends_on for item in scope}
    for node in blueprint:
        if node.node_id not in deps:
            continue
        ancestors = _ancestors(node.node_id, deps)
        if ancestors and not any(a in node.pass_condition.semantic_check for a in ancestors):
            problems.append(
                f"{node.node_id}: semantic_check must name one of its ancestor ids "
                f"verbatim ({sorted(ancestors)})"
            )
        if check_assertions:
            for assertion in node.pass_condition.assertions:
                error = _assertion_error(assertion)
                if error is not None:
                    problems.append(f"{node.node_id}: assertion `{assertion}` -> {error}")
    return problems


def _assemble(scope: list[_ScopeItem], blueprint: list[_BlueprintNode]) -> list[NodeSpec]:
    deps = {item.id: item.depends_on for item in scope}
    kinds = {item.id: item.kind for item in scope}
    order = _topological_order(deps)
    assert order is not None  # _scope_problems already rejected cycles
    depths = _depths(order, deps)
    written = {n.node_id: n for n in blueprint}
    return [
        _to_node_spec(written[n], parent_ids=deps[n], depth=depths[n], kind=kinds[n])
        for n in order
    ]


def _to_node_spec(
    node: _BlueprintNode, *, parent_ids: list[str], depth: int, kind: NodeKind
) -> NodeSpec:
    kept = []
    for assertion in node.pass_condition.assertions:
        error = _assertion_error(assertion)
        if error is None:
            kept.append(assertion)
        else:
            _log.warning(
                "dropping un-evaluable assertion on %s: %r (%s)", node.node_id, assertion, error
            )
    return NodeSpec(
        node_id=node.node_id,
        parent_ids=parent_ids,
        depth=depth,
        kind=kind,
        title=node.title,
        node_goal=node.node_goal,
        generated_prompt=node.generated_prompt,
        pass_condition=PassCondition(
            assertions=kept, semantic_check=node.pass_condition.semantic_check
        ),
    )


def _assertion_error(assertion: str) -> str | None:
    try:
        validate_assertion(assertion)
    except (UnsafeAssertionError, SyntaxError) as exc:
        return str(exc)
    return None
