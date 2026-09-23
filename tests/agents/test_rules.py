"""RuleChecker: cheap deterministic assertions, evaluated in a restricted namespace."""

from kurogami.agents.rules import RuleChecker
from kurogami.contracts import NodeKind, NodeResult, NodeSpec, PassCondition


def _node(assertions: list[str], context: dict[str, str] | None = None) -> NodeSpec:
    return NodeSpec(
        node_id="n_004",
        parent_ids=["n_003"],
        depth=1,
        kind=NodeKind.DECISION,
        title="Pricing",
        node_goal="Propose pricing tiers.",
        generated_prompt="propose pricing",
        pass_condition=PassCondition(assertions=assertions, semantic_check="ok?"),
        context=context or {},
    )


def _result(structured: dict) -> NodeResult:
    return NodeResult(
        node_id="n_004",
        output="x",
        structured=structured,
        tokens_in=1,
        tokens_out=1,
        latency_ms=1,
        model_id="fake",
        prompt_version="v1",
    )


def test_passes_when_all_assertions_hold():
    node = _node(["len(structured['tiers']) >= 2", "structured['tiers'][0] < 1000"])
    result = _result({"tiers": [500, 900]})

    verdict = RuleChecker().check(node, result)

    assert verdict.verdict == "PASS"
    assert verdict.checked_by == "rules"
    assert verdict.reason is None


def test_fails_when_an_assertion_is_false():
    node = _node(["len(structured['tiers']) >= 5"])
    result = _result({"tiers": [500]})

    verdict = RuleChecker().check(node, result)

    assert verdict.verdict == "FAIL"
    assert verdict.reason is not None
    assert verdict.reason.violated == "assertion"


def test_can_reference_ancestor_context():
    node = _node(["structured['price'] <= 500"], context={"n_001": "500"})
    result = _result({"price": 400})
    verdict = RuleChecker().check(node, result)
    assert verdict.verdict == "PASS"


def test_context_lookup_itself_is_permitted_syntax():
    node = _node(["context['n_001'] == '500'"], context={"n_001": "500"})
    result = _result({})
    verdict = RuleChecker().check(node, result)
    assert verdict.verdict == "PASS"


def test_generator_expression_inside_any_is_permitted_syntax():
    """Regression: seen live -- `all(x > 0 for x in structured['tiers'])`
    was rejected (GeneratorExp/comprehension weren't in the whitelist).
    """
    node = _node(["all(x > 0 for x in structured['tiers'])"])
    result = _result({"tiers": [100, 200, 300]})
    verdict = RuleChecker().check(node, result)
    assert verdict.verdict == "PASS"


def test_generator_expression_can_still_fail_on_real_data():
    node = _node(["all(x > 0 for x in structured['tiers'])"])
    result = _result({"tiers": [100, -1, 300]})
    verdict = RuleChecker().check(node, result)
    assert verdict.verdict == "FAIL"


def test_list_comprehension_is_permitted_syntax():
    node = _node(["len([x for x in structured['tiers'] if x > 100]) >= 1"])
    result = _result({"tiers": [50, 150, 300]})
    verdict = RuleChecker().check(node, result)
    assert verdict.verdict == "PASS"


def test_comprehension_loop_variable_does_not_leak_as_an_allowed_free_name():
    """The loop variable `x` is only safe *inside* the comprehension that binds
    it -- it must not be usable as a free-standing name elsewhere.
    """
    node = _node(["x"])
    result = _result({})
    verdict = RuleChecker().check(node, result)
    assert verdict.verdict == "FAIL"
    assert verdict.reason.violated == "schema"


def test_no_assertions_always_passes():
    node = _node([])
    result = _result({})
    verdict = RuleChecker().check(node, result)
    assert verdict.verdict == "PASS"


def test_key_error_in_assertion_is_a_typed_fail_not_a_crash():
    node = _node(["structured['does_not_exist'] > 0"])
    result = _result({})

    verdict = RuleChecker().check(node, result)

    assert verdict.verdict == "FAIL"
    assert verdict.reason.violated == "assertion"


def test_disallowed_syntax_is_rejected_as_a_schema_violation():
    node = _node(["__import__('os').system('echo hi')"])
    result = _result({})

    verdict = RuleChecker().check(node, result)

    assert verdict.verdict == "FAIL"
    assert verdict.reason.violated == "schema"


def test_disallowed_name_is_rejected():
    node = _node(["open('/etc/passwd').read()"])
    result = _result({})

    verdict = RuleChecker().check(node, result)

    assert verdict.verdict == "FAIL"
    assert verdict.reason.violated == "schema"


def test_is_not_none_is_permitted_syntax():
    """Regression: seen live -- a real assertion used `is not None`, which the
    whitelist originally rejected (Is/IsNot weren't in _ALLOWED_NODE_TYPES).
    """
    node = _node(["structured['budget'] is not None"])
    result = _result({"budget": 500})
    verdict = RuleChecker().check(node, result)
    assert verdict.verdict == "PASS"


def test_is_none_is_permitted_syntax():
    node = _node(["structured['x'] is None"])
    result = _result({"x": None})
    verdict = RuleChecker().check(node, result)
    assert verdict.verdict == "PASS"


def test_natural_language_assertion_is_a_typed_fail_not_a_crash():
    """Regression: seen live against a real LLM -- the planner wrote
    "The output includes a summary of current tools used by freelance
    designers." as an assertion. ast.parse() raises a bare SyntaxError on
    that, which must never propagate out of RuleChecker.check().
    """
    node = _node(["The output includes a summary of current tools used."])
    result = _result({})

    verdict = RuleChecker().check(node, result)

    assert verdict.verdict == "FAIL"
    assert verdict.reason.violated == "schema"


def test_attribute_access_is_rejected():
    node = _node(["structured.__class__"])
    result = _result({})

    verdict = RuleChecker().check(node, result)

    assert verdict.verdict == "FAIL"
    assert verdict.reason.violated == "schema"
