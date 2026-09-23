"""Planner: scope the whole investigation, write the full tree, gap-fill only when asked.

Offline via a scripted LLM that returns parsed models in order and records every
prompt it was sent.
"""

import pytest

from kurogami.agents.planner import (
    BlueprintError,
    Planner,
    _Blueprint,
    _blueprint_problems,
    _BlueprintNode,
    _GapFill,
    _names,
    _Scope,
    _scope_problems,
    _ScopeItem,
)
from kurogami.contracts import (
    GoalSpec,
    LLMResponse,
    NodeKind,
    NodeResult,
    NodeSpec,
    PassCondition,
)

MAX_DEPTH = 6

# A valid 10-item scope: one decision sink, longest chain a0-a1-c0-c1-c2-final (depth 5).
_SCOPE_SHAPE = {
    "a0": (NodeKind.RESEARCH, []),
    "a1": (NodeKind.ANALYSIS, ["a0"]),
    "a2": (NodeKind.ANALYSIS, ["a1"]),
    "a3": (NodeKind.ANALYSIS, ["a2"]),
    "b0": (NodeKind.RESEARCH, []),
    "b1": (NodeKind.ANALYSIS, ["b0"]),
    "c0": (NodeKind.ANALYSIS, ["a1"]),
    "c1": (NodeKind.SYNTHESIS, ["c0", "b1"]),
    "c2": (NodeKind.ANALYSIS, ["c1"]),
    "final": (NodeKind.DECISION, ["a3", "c2"]),
}


def _goal() -> GoalSpec:
    return GoalSpec(
        raw_text="Should I launch my invoicing tool?",
        product_description="x",
        target_market="x",
        decision_type="launch",
        success_definition="x",
    )


def _stages(shape: dict) -> dict[str, int]:
    """Stage = longest-path depth + 1 (a well-formed scope); unknown deps count as stage 0."""
    stage: dict[str, int] = {}

    def of(i: str, seen: frozenset = frozenset()) -> int:
        if i not in shape or i in seen:
            return 0
        if i not in stage:
            stage[i] = 1 + max((of(d, seen | {i}) for d in shape[i][1]), default=0)
        return stage[i]

    for i in shape:
        of(i)
    return stage


def _scope(
    shape: dict[str, tuple[NodeKind, list[str]]] = _SCOPE_SHAPE, stages: dict[str, int] | None = None
) -> _Scope:
    stage = {**_stages(shape), **(stages or {})}
    return _Scope(
        items=[
            _ScopeItem(id=i, question=f"question {i}?", kind=kind, stage=stage[i], depends_on=deps)
            for i, (kind, deps) in shape.items()
        ]
    )


def _bp_node(node_id: str, deps: list[str], assertions: list[str] | None = None) -> _BlueprintNode:
    check = f"Is this consistent with {deps[0]}?" if deps else "Does it answer the question?"
    return _BlueprintNode(
        node_id=node_id,
        title=node_id,
        node_goal=f"goal {node_id}",
        generated_prompt=f"prompt for {node_id} " * 15,
        pass_condition=PassCondition(assertions=assertions or [], semantic_check=check),
    )


def _blueprint(**overrides: list[str]) -> _Blueprint:
    return _Blueprint(
        nodes=[_bp_node(i, deps, overrides.get(i)) for i, (_, deps) in _SCOPE_SHAPE.items()]
    )


class _ScriptedLLM:
    """Returns the given parsed models in order, one per call; records every prompt."""

    def __init__(self, parsed: list) -> None:
        self._parsed = list(parsed)
        self.prompts: list[str] = []

    def complete(self, *, prompt, system=None, schema=None, temperature=0.0):
        self.prompts.append(prompt)
        model = self._parsed.pop(0)
        assert isinstance(model, schema), f"scripted {type(model).__name__}, asked {schema}"
        return LLMResponse(
            text=model.model_dump_json(), parsed=model, tokens_in=1, tokens_out=1,
            latency_ms=0, model_id="fake",
        )


# --- plan(): the whole tree up front -------------------------------------------------


def test_plan_returns_the_whole_tree_in_topological_order():
    nodes = Planner(_ScriptedLLM([_scope(), _blueprint()])).plan(_goal())

    assert [n.node_id for n in nodes] and len(nodes) == len(_SCOPE_SHAPE)
    position = {n.node_id: i for i, n in enumerate(nodes)}
    for node in nodes:
        assert all(position[p] < position[node.node_id] for p in node.parent_ids)


def test_plan_takes_structure_from_the_scope_and_computes_depth():
    nodes = {n.node_id: n for n in Planner(_ScriptedLLM([_scope(), _blueprint()])).plan(_goal())}

    assert nodes["c1"].parent_ids == ["c0", "b1"]
    assert nodes["a0"].depth == 0
    assert nodes["c1"].depth == 3  # longest path a0-a1-c0-c1, not the shortest via b0-b1
    assert nodes["final"].depth == 5
    assert nodes["final"].kind == NodeKind.DECISION


def test_plan_reasks_once_when_the_scope_is_invalid():
    cyclic = _scope({**_SCOPE_SHAPE, "a0": (NodeKind.RESEARCH, ["a3"])})
    llm = _ScriptedLLM([cyclic, _scope(), _blueprint()])

    Planner(llm).plan(_goal())

    assert len(llm.prompts) == 3
    assert llm.prompts[1].startswith(llm.prompts[0])
    assert "cycle" in llm.prompts[1]


def test_plan_raises_blueprint_error_when_the_scope_is_still_invalid():
    cyclic = _scope({**_SCOPE_SHAPE, "a0": (NodeKind.RESEARCH, ["a3"])})
    with pytest.raises(BlueprintError, match="cycle"):
        Planner(_ScriptedLLM([cyclic, cyclic, cyclic])).plan(_goal())


def test_plan_reasks_then_drops_an_assertion_that_is_still_unevaluable():
    bad = _blueprint(a1=["structured.keys()", "len(structured['items']) > 0"])
    still_bad_a1 = _Blueprint(nodes=[n for n in bad.nodes if n.node_id == "a1"])
    llm = _ScriptedLLM([_scope(), bad, still_bad_a1, still_bad_a1])

    nodes = {n.node_id: n for n in Planner(llm).plan(_goal())}

    assert len(llm.prompts) == 4  # scope + blueprint + two corrections
    assert "structured.keys()" in llm.prompts[3]
    assert nodes["a1"].pass_condition.assertions == ["len(structured['items']) > 0"]


def test_plan_raises_when_the_blueprint_is_still_structurally_wrong():
    missing = _Blueprint(nodes=_blueprint().nodes[:-1])  # no node for "final"
    with pytest.raises(BlueprintError, match="final"):
        Planner(_ScriptedLLM([_scope(), missing, missing, missing])).plan(_goal())


# --- validation rules ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ({**_SCOPE_SHAPE, "a0": (NodeKind.RESEARCH, ["a3"])}, "cycle"),
        ({**_SCOPE_SHAPE, "a1": (NodeKind.ANALYSIS, ["nope"])}, "unknown ids"),
        ({**_SCOPE_SHAPE, "extra_leaf": (NodeKind.ANALYSIS, ["a0"])}, "exactly one item must be the final"),
        ({**_SCOPE_SHAPE, "final": (NodeKind.SYNTHESIS, ["a3", "c2"])}, "must have kind 'decision'"),
        (dict(list(_SCOPE_SHAPE.items())[:9]), "items; it must have"),
    ],
)
def test_scope_problems_are_detected(shape, expected):
    problems = _scope_problems(_scope(shape).items, MAX_DEPTH)
    assert any(expected in p for p in problems), problems


def test_scope_too_shallow_is_rejected():
    flat = {f"s{i}": (NodeKind.RESEARCH, []) for i in range(9)}
    flat["final"] = (NodeKind.DECISION, list(flat))
    problems = _scope_problems(_scope(flat).items, MAX_DEPTH)
    assert any("at least one chain must have 5" in p for p in problems), problems


def test_a_mislabelled_stage_does_not_reject_a_valid_plan():
    """Live incident: gpt-4.1-mini kept a correct dependency (adoption <- product-market
    fit) but labelled both stage 2, and was rejected twice over the label. Stage labels
    are a planning aid; only the dependency graph is validated.
    """
    assert _scope_problems(_scope(stages={"a2": 2, "final": 99}).items, MAX_DEPTH) == []


def test_too_deep_a_plan_names_the_exact_chain_to_shorten():
    """Live incident: 'the chain has 8 items' alone couldn't be fixed by the model."""
    chain = {f"d{i}": (NodeKind.ANALYSIS, [f"d{i-1}"] if i else []) for i in range(8)}
    chain["d0"] = (NodeKind.RESEARCH, [])
    chain["side"] = (NodeKind.RESEARCH, [])
    chain["final"] = (NodeKind.DECISION, ["d7", "side"])
    problems = _scope_problems(_scope(chain).items, MAX_DEPTH)
    assert any("d0 -> d1 -> d2" in p and "-> final (9 items)" in p for p in problems), problems


def test_a_chain_of_exactly_max_depth_plus_one_items_is_allowed():
    chain = {f"d{i}": (NodeKind.ANALYSIS, [f"d{i-1}"] if i else []) for i in range(6)}
    chain.update({"s0": (NodeKind.RESEARCH, []), "s1": (NodeKind.ANALYSIS, ["s0"]),
                  "s2": (NodeKind.ANALYSIS, ["s1"])})
    chain["final"] = (NodeKind.DECISION, ["d5", "s2"])  # 7-item chain d0..d5-final
    assert _scope_problems(_scope(chain).items, MAX_DEPTH) == []


def test_valid_scope_has_no_problems():
    assert _scope_problems(_scope().items, MAX_DEPTH) == []


def test_semantic_check_must_name_an_ancestor():
    nodes = _blueprint().nodes
    nodes[1] = nodes[1].model_copy(
        update={"pass_condition": PassCondition(assertions=[], semantic_check="Is it good?")}
    )
    problems = _blueprint_problems(nodes, _scope().items)
    assert ("a1", "a1: semantic_check must name one of its ancestor ids, e.g. a0 (any of ['a0'])") in problems


def test_an_ancestor_id_written_as_words_counts_as_named():
    """Seen live: "aligned with the identified marketing channels" for marketing_channels."""
    assert _names("Aligned with the identified A0 findings?", "a0")
    assert _names("Consistent with the marketing channels found?", "marketing_channels")
    assert not _names("Realistic given the market size?", "target_market_size")


def test_corrections_keep_accepted_nodes_and_reask_only_the_failing_ones():
    """Seen live: re-asking for the whole blueprint made the model rewrite nodes it had
    just fixed, and they regressed. Only the failing node is re-asked and merged back.
    """
    bad_check = PassCondition(assertions=[], semantic_check="Is it good?")
    first = _blueprint()
    first.nodes[1] = first.nodes[1].model_copy(update={"pass_condition": bad_check})  # a1
    fixed_a1 = _Blueprint(nodes=[_bp_node("a1", ["a0"])])
    llm = _ScriptedLLM([_scope(), first, fixed_a1])

    nodes = {n.node_id: n for n in Planner(llm).plan(_goal())}

    assert len(llm.prompts) == 3
    assert "Return ONLY the corrected nodes for these ids: a1." in llm.prompts[2]
    assert nodes["a1"].pass_condition.semantic_check == "Is this consistent with a0?"
    assert nodes["final"].generated_prompt == first.nodes[-1].generated_prompt  # untouched


@pytest.mark.parametrize(
    "check",
    [
        "Does the analysis cover all major competitors, per a0?",
        "Is it a comprehensive view of a0?",
        "Is every risk from a0 addressed?",
    ],
)
def test_an_unbounded_semantic_check_is_a_problem(check):
    """Live incident: 'cover all major competitors' could never pass -- the verifier
    kept naming more after each retry."""
    nodes = _blueprint().nodes
    nodes[1] = nodes[1].model_copy(
        update={"pass_condition": PassCondition(assertions=[], semantic_check=check)}
    )
    problems = _blueprint_problems(nodes, _scope().items)
    assert any(n == "a1" and "unbounded completeness" in m for n, m in problems), problems


def test_an_unbounded_check_is_not_structural_so_it_never_sinks_the_plan():
    nodes = _blueprint().nodes
    nodes[1] = nodes[1].model_copy(
        update={"pass_condition": PassCondition(assertions=[], semantic_check="All of a0?")}
    )
    assert _blueprint_problems(nodes, _scope().items, strict=False) == []


def test_a_short_generated_prompt_is_a_problem():
    nodes = _blueprint().nodes
    nodes[0] = nodes[0].model_copy(update={"generated_prompt": "Research the market."})
    problems = _blueprint_problems(nodes, _scope().items)
    assert any(node_id == "a0" and "3 words" in m for node_id, m in problems), problems


def test_a_distant_ancestor_counts_as_named():
    nodes = _blueprint().nodes
    nodes[3] = nodes[3].model_copy(  # a3's ancestors are a2, a1, a0
        update={"pass_condition": PassCondition(assertions=[], semantic_check="Uses a0?")}
    )
    assert _blueprint_problems(nodes, _scope().items) == []


# --- expand(): bounded gap-filling ------------------------------------------------------


def _reporter() -> NodeSpec:
    return NodeSpec(
        node_id="c1", parent_ids=["c0", "b1"], depth=3, kind=NodeKind.SYNTHESIS, title="c1",
        node_goal="g", generated_prompt="p", pass_condition=PassCondition(assertions=[], semantic_check="?"),
    )


def _result(structured: dict) -> NodeResult:
    return NodeResult(
        node_id="c1", output="out", structured=structured, tokens_in=1, tokens_out=1,
        latency_ms=1, model_id="fake", prompt_version="v1",
    )


def _planned() -> tuple[Planner, _ScriptedLLM]:
    llm = _ScriptedLLM([_scope(), _blueprint()])
    planner = Planner(llm)
    planner.plan(_goal())
    return planner, llm


def test_expand_makes_no_llm_call_without_a_reported_gap():
    planner, llm = _planned()
    assert planner.expand(_reporter(), _result({"x": 1}), _goal()) == []
    assert len(llm.prompts) == 2  # scope + blueprint only


def test_expand_returns_one_node_under_the_reporter_for_a_real_gap():
    planner, llm = _planned()
    gap = _bp_node("regulatory_limits", ["c1"])
    llm._parsed.append(_GapFill(node=gap))

    [node] = planner.expand(
        _reporter(), _result({"missing_prerequisites": ["What GST rules apply?"]}), _goal()
    )

    assert node.node_id == "regulatory_limits"
    assert node.parent_ids == ["c1"]
    assert node.depth == 4
    assert "What GST rules apply?" in llm.prompts[-1]


def test_expand_returns_nothing_when_the_gap_is_already_covered():
    planner, llm = _planned()
    llm._parsed.append(_GapFill(node=None))
    assert planner.expand(_reporter(), _result({"missing_prerequisites": ["q?"]}), _goal()) == []


def test_expand_drops_a_gap_node_reusing_a_planned_id():
    planner, llm = _planned()
    llm._parsed.append(_GapFill(node=_bp_node("a0", ["c1"])))
    assert planner.expand(_reporter(), _result({"missing_prerequisites": ["q?"]}), _goal()) == []


def test_expand_drops_a_gap_node_that_does_not_name_its_reporter():
    planner, llm = _planned()
    llm._parsed.append(_GapFill(node=_bp_node("new_gap", [])))  # check doesn't mention c1
    assert planner.expand(_reporter(), _result({"missing_prerequisites": ["q?"]}), _goal()) == []
