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


def test_attribute_access_is_rejected():
    node = _node(["structured.__class__"])
    result = _result({})

    verdict = RuleChecker().check(node, result)

    assert verdict.verdict == "FAIL"
    assert verdict.reason.violated == "schema"
