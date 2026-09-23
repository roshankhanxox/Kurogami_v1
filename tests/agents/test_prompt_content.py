"""Regression guards on prompt content -- load-bearing rules a future edit must not drop.

Each rule here exists because of a design decision or a live incident; the
Python-side validation enforces the same rules, the prompt makes them likely to
be met on the first attempt.
"""

from kurogami.agents._prompt_loader import load_prompt


def _normalized(name: str) -> str:
    return " ".join(load_prompt(name).split())


def test_scope_prompt_bounds_the_investigation_and_forces_a_single_decision():
    content = _normalized("scope")
    assert "{min_items}" in content and "{max_items}" in content
    assert "The last stage has exactly one item: the final recommendation" in content
    assert "Use exactly {stages} stages" in content


def test_scope_prompt_makes_dependencies_point_to_earlier_stages():
    """Live incident: the model couldn't count chain length, so depth is bounded by stages."""
    assert "They must all be in EARLIER stages" in _normalized("scope")


def test_scope_prompt_surfaces_hard_limits_later_items_must_respect():
    """How the pricing-vs-willingness-to-pay contradiction becomes detectable."""
    assert "hard limit that later answers must respect" in _normalized("scope")


def test_plan_prompt_writes_exactly_the_scoped_nodes():
    assert "exactly one node per item, nothing added and nothing removed" in _normalized("plan")


def test_plan_prompt_requires_python_boolean_assertions():
    """Live incident: plain-English assertions crashed ast.parse()."""
    assert "Python boolean expression" in _normalized("plan")


def test_plan_prompt_forbids_vacuous_checks_and_requires_an_ancestor():
    content = _normalized("plan").lower()
    assert "vacuous" in content
    assert "ancestor ids verbatim" in content


def test_gap_fill_prompt_prefers_returning_null():
    """Runtime growth is the exception: most reported gaps are already covered."""
    content = _normalized("gap_fill")
    assert "return node as null" in content
    assert "{reporter_id}" in content


def test_executor_always_offers_the_missing_prerequisites_key():
    assert '"missing_prerequisites"' in _normalized("execute_structured")


def test_interpret_prompt_requires_flagging_ambiguity_not_resolving_it():
    content = load_prompt("interpret")
    assert "ambiguities" in content
    assert "Do NOT resolve ambiguity" in content
