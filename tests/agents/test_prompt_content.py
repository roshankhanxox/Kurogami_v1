"""Regression guards on prompt content -- protects the two named failure modes
from IMPLEMENTATION_PLAN.md Track B (bushy shallow trees, vacuous pass
conditions) from being silently weakened by a future prompt edit.
"""

from kurogami.agents._prompt_loader import load_prompt


def test_plan_prompt_caps_root_nodes_between_two_and_three():
    content = load_prompt("plan")
    assert "two and three root nodes" in content


def test_plan_and_expand_prompts_forbid_vacuous_pass_conditions():
    for name in ("plan", "expand"):
        content = load_prompt(name).lower()
        assert "vacuous" in content
        assert "ancestor" in content


def test_expand_prompt_allows_zero_children():
    content = load_prompt("expand")
    assert "zero children" in content


def test_interpret_prompt_requires_flagging_ambiguity_not_resolving_it():
    content = load_prompt("interpret")
    assert "ambiguities" in content
    assert "Do NOT resolve ambiguity" in content
