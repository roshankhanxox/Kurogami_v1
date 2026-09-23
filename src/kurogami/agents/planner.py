"""The Master: scope the whole investigation, then write every child agent's prompt.

Seen live, open-ended expansion (decide children one node at a time) never knew
when the job was done: it drilled one topic to the depth cap while sibling
branches never ran. So the Master now works out up front exactly which
questions the answer needs, writes the complete bounded tree, and only grows
it at runtime when a node reports a prerequisite nobody planned for.
"""

import json
import logging
import re
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
# Seen live: gpt-4o-mini handled ranges ("5-7 stages") poorly -- it swung between one
# long chain and a flat plan -- but follows a concrete shape. 6 stages at two items per
# middle stage gives 11-12 nodes at depth 5: inside G3 (depth >= 4, >= 10 nodes) and
# inside the max_depth=6 budget.
SCOPE_STAGES = 6
MAX_CORRECTIONS = 2
MIN_PROMPT_WORDS = 40
MIN_PROMPT_FLOOR = 30
MISSING_PREREQUISITES_KEY = "missing_prerequisites"
_UNBOUNDED = re.compile(
    r"\b(all|every|each and every|comprehensive(?:ly)?|complete(?:ly)?|exhaustive(?:ly)?|"
    r"entire(?:ly)?|fully)\b",
    re.IGNORECASE,
)

_Model = TypeVar("_Model", bound=BaseModel)


class BlueprintError(RuntimeError):
    """The Master could not produce a valid plan, even after its bounded corrections."""


class _ScopeItem(BaseModel):
    """stage: seen live, models can't count a dependency chain's length but reason
    well in stages, so they're asked to label one. It's a planning aid only --
    validation runs on the dependency graph, never on the labels.
    """

    id: str
    question: str
    kind: NodeKind
    stage: int
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
        if not _names(filled.pass_condition.semantic_check, node.node_id):
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
            stages=SCOPE_STAGES,
            penultimate=SCOPE_STAGES - 1,
        )
        scope = self._ask(prompt, _Scope).items
        problems = _scope_problems(scope, self._max_depth)
        for _ in range(MAX_CORRECTIONS):
            if not problems:
                return scope
            scope = self._ask(prompt + _correction(problems), _Scope).items
            problems = _scope_problems(scope, self._max_depth)
        if problems:
            raise BlueprintError(
                f"scope still invalid after {MAX_CORRECTIONS} corrections: " + "; ".join(problems)
            )
        return scope

    def _write_blueprint(self, goal: GoalSpec, scope: list[_ScopeItem]) -> list[NodeSpec]:
        prompt = load_prompt("plan").format(
            goal_json=goal.model_dump_json(indent=2),
            scope_json=_scope_json(scope),
            allowed_functions=", ".join(ALLOWED_FUNCTIONS),
        )
        blueprint = self._ask(prompt, _Blueprint).nodes
        for _ in range(MAX_CORRECTIONS):
            problems = _blueprint_problems(blueprint, scope)
            if not problems:
                break
            # Seen live: re-asking for the whole blueprint made the model rewrite the
            # nodes it had just fixed, and they regressed. Keep what passed; re-ask
            # only for the failing nodes and merge them back in.
            failing = sorted({node_id for node_id, _ in problems})
            note = load_prompt("correct_nodes").format(
                problems="\n".join(f"- {message}" for _, message in problems),
                node_ids=", ".join(failing),
                allowed_functions=", ".join(ALLOWED_FUNCTIONS),
            )
            fixed = {n.node_id: n for n in self._ask(prompt + note, _Blueprint).nodes}
            blueprint = _merge_corrections(blueprint, fixed, problems, _ids(scope))
        # Unevaluable assertions left after the corrections are dropped (and logged);
        # structural problems cannot be.
        structural = _blueprint_problems(blueprint, scope, strict=False)
        if structural:
            raise BlueprintError(
                f"blueprint still invalid after {MAX_CORRECTIONS} corrections: "
                + "; ".join(message for _, message in structural)
            )
        return _assemble(scope, blueprint)

    def _ask(self, prompt: str, schema: type[_Model]) -> _Model:
        parsed = self._llm.complete(prompt=prompt, schema=schema).parsed
        if not isinstance(parsed, schema):
            raise TypeError(f"Planner expected a parsed {schema.__name__}, got {type(parsed).__name__}")
        return parsed


def _ids(scope: list[_ScopeItem]) -> set[str]:
    return {item.id for item in scope}


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

    # Stage labels are scaffolding that help the model shape the plan; the plan itself
    # is the dependency graph. Seen live: gpt-4.1-mini kept a correct dependency
    # (adoption <- product-market fit) but labelled both stage 2, and was rejected
    # twice over the label. So only the graph is validated.
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

    # Depth is computed from the graph and the exact chain is named, so a correction
    # is actionable (seen live: "the chain has 8 items" alone couldn't be fixed).
    chain = _longest_chain(order, deps)
    if len(chain) > max_depth + 1:
        problems.append(
            f"the longest dependency chain is {' -> '.join(chain)} ({len(chain)} items); no "
            f"chain may exceed {max_depth + 1} -- let a later item depend on an earlier one "
            f"directly instead of on the one just before it, or merge two links"
        )
    elif len(chain) < MIN_TREE_DEPTH + 1:
        problems.append(
            f"the longest dependency chain is {' -> '.join(chain)} ({len(chain)} items); at "
            f"least one chain must have {MIN_TREE_DEPTH + 1} -- make items depend on an "
            f"answer from the stage just before theirs"
        )
    return problems


def _longest_chain(order: list[str], deps: dict[str, list[str]]) -> list[str]:
    depth = _depths(order, deps)
    node = max(order, key=lambda n: depth[n])
    chain = [node]
    while deps[node]:
        node = max(deps[node], key=lambda p: depth[p])
        chain.append(node)
    return list(reversed(chain))


def _blueprint_problems(
    blueprint: list[_BlueprintNode], scope: list[_ScopeItem], *, strict: bool = True
) -> list[tuple[str, str]]:
    """(node_id, message) pairs, so a correction can target just the failing nodes."""
    problems: list[tuple[str, str]] = []
    scope_ids = {item.id for item in scope}
    written = [n.node_id for n in blueprint]
    for missing in sorted(scope_ids - set(written)):
        problems.append((missing, f"{missing}: no node was written for this scope item"))
    for extra in sorted(set(written) - scope_ids):
        problems.append((extra, f"{extra}: not a scope item -- remove it"))
    for dup in sorted({i for i in written if written.count(i) > 1}):
        problems.append((dup, f"{dup}: more than one node was written"))

    deps = {item.id: item.depends_on for item in scope}
    for node in blueprint:
        if node.node_id not in deps:
            continue
        ancestors = _ancestors(node.node_id, deps)
        check = node.pass_condition.semantic_check
        words = len(node.generated_prompt.split())
        # Asked for 40 (G3: "over ~40"); only a genuinely thin prompt blocks the plan.
        # Seen live: a hard 40 aborted a whole plan over prompts of 37-39 words.
        if words < (MIN_PROMPT_WORDS if strict else MIN_PROMPT_FLOOR):
            message = (
                f"{node.node_id}: generated_prompt has {words} words; write at least "
                f"{MIN_PROMPT_WORDS}, specific to this goal"
            )
            problems.append((node.node_id, message))
        unbounded = _UNBOUNDED.search(check)
        if strict and unbounded:
            # Seen live: "Does the analysis cover all major competitors?" -- the
            # verifier kept moving the goalposts and no retry could ever pass it.
            message = (
                f"{node.node_id}: semantic_check asks for unbounded completeness "
                f"('{unbounded.group(0)}'); state a bar the output can verifiably meet, "
                f"e.g. 'names at least 4 competitors with their pricing'"
            )
            problems.append((node.node_id, message))
        if ancestors and not any(_names(check, a) for a in ancestors):
            message = (
                f"{node.node_id}: semantic_check must name one of its ancestor ids, "
                f"e.g. {min(ancestors)} (any of {sorted(ancestors)})"
            )
            problems.append((node.node_id, message))
        if strict:
            for assertion in node.pass_condition.assertions:
                error = _assertion_error(assertion)
                if error is not None:
                    problems.append(
                        (node.node_id, f"{node.node_id}: assertion `{assertion}` -> {error}")
                    )
    return problems


def _merge_corrections(
    blueprint: list[_BlueprintNode],
    fixed: dict[str, _BlueprintNode],
    problems: list[tuple[str, str]],
    scope_ids: set[str],
) -> list[_BlueprintNode]:
    """Take from a correction only the fields that had problems.

    Seen live: asked to fix assertions, the model rewrote whole nodes and good
    prompts shrank from 32-58 words to 24-41. A correction may now only change
    what it was asked to change.
    """
    current = {n.node_id: n for n in blueprint}
    merged: dict[str, _BlueprintNode] = {i: n for i, n in current.items() if i in scope_ids}
    for node_id in {i for i, _ in problems} & scope_ids:
        new = fixed.get(node_id)
        old = current.get(node_id)
        if new is None:
            continue
        if old is None:
            merged[node_id] = new
            continue
        messages = " ".join(m for i, m in problems if i == node_id)
        condition = old.pass_condition
        merged[node_id] = old.model_copy(
            update={
                "generated_prompt": new.generated_prompt
                if "generated_prompt" in messages
                else old.generated_prompt,
                "pass_condition": condition.model_copy(
                    update={
                        "semantic_check": new.pass_condition.semantic_check
                        if "semantic_check" in messages
                        else condition.semantic_check,
                        "assertions": new.pass_condition.assertions
                        if "assertion" in messages
                        else condition.assertions,
                    }
                ),
            }
        )
    return list(merged.values())


def _names(text: str, node_id: str) -> bool:
    """Whether a check names this node: its id, or the id read as words.

    Seen live, the model writes "the identified marketing channels" for
    marketing_channels -- the verifier sees context keyed by that id, so the
    reference is unambiguous. Vaguer paraphrases ("market size" for
    target_market_size) still don't count.
    """
    lowered = text.lower()
    return node_id.lower() in lowered or node_id.replace("_", " ").lower() in lowered


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
