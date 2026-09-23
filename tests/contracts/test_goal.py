"""Round-trip and validation tests for GoalSpec."""

import pytest
from pydantic import ValidationError

from kurogami.contracts import GoalSpec


def _valid_kwargs() -> dict:
    return {
        "raw_text": "Should I launch my invoicing tool for freelance designers in India?",
        "product_description": "Invoicing and late-payment chasing tool for freelancers.",
        "target_market": "Freelance designers in India",
        "decision_type": "market_entry",
        "success_definition": "A clear go/no-go with a pricing and positioning rationale.",
    }


def test_round_trip():
    goal = GoalSpec(**_valid_kwargs())
    restored = GoalSpec.model_validate_json(goal.model_dump_json())
    assert restored == goal


def test_defaults_are_empty_lists():
    goal = GoalSpec(**_valid_kwargs())
    assert goal.known_constraints == []
    assert goal.ambiguities == []


def test_decision_type_rejects_unknown_literal():
    kwargs = _valid_kwargs()
    kwargs["decision_type"] = "world_domination"
    with pytest.raises(ValidationError):
        GoalSpec(**kwargs)


def test_missing_required_field_rejected():
    kwargs = _valid_kwargs()
    del kwargs["success_definition"]
    with pytest.raises(ValidationError):
        GoalSpec(**kwargs)
