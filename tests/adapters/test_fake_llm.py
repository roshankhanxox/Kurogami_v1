"""FakeLLM: canned responses, including the dict/list-validated-against-schema path
used by the CLI's --fake-script flag to script a conversation as plain JSON.
"""

import pytest
from pydantic import BaseModel, ValidationError

from kurogami.adapters.llm.fake import FakeLLM, UnregisteredPromptError
from kurogami.contracts import GoalSpec


def test_returns_a_model_instance_registered_directly():
    goal = GoalSpec(
        raw_text="x", product_description="x", target_market="x",
        decision_type="market_entry", success_definition="x",
    )
    llm = FakeLLM(responses={"p": goal})
    response = llm.complete(prompt="p", prompt_name="p", schema=GoalSpec)
    assert response.parsed == goal


def test_validates_a_plain_dict_against_the_requested_schema():
    payload = {
        "raw_text": "x",
        "product_description": "x",
        "target_market": "x",
        "decision_type": "pricing",
        "success_definition": "x",
    }
    llm = FakeLLM(responses={"p": payload})
    response = llm.complete(prompt="p", prompt_name="p", schema=GoalSpec)
    assert isinstance(response.parsed, GoalSpec)
    assert response.parsed.decision_type == "pricing"


def test_validates_a_plain_list_against_a_list_wrapping_schema():
    class _Batch(BaseModel):
        values: list[int]

    llm = FakeLLM(responses={"p": {"values": [1, 2, 3]}})
    response = llm.complete(prompt="p", prompt_name="p", schema=_Batch)
    assert response.parsed == _Batch(values=[1, 2, 3])


def test_mismatched_dict_raises_a_validation_error_not_a_silent_pass():
    llm = FakeLLM(responses={"p": {"not": "a goal spec"}})
    with pytest.raises(ValidationError):
        llm.complete(prompt="p", prompt_name="p", schema=GoalSpec)


def test_unregistered_prompt_raises():
    llm = FakeLLM(responses={})
    with pytest.raises(UnregisteredPromptError):
        llm.complete(prompt="anything")


def test_plain_string_response_without_schema():
    llm = FakeLLM(responses={"p": "hello"})
    response = llm.complete(prompt="p", prompt_name="p")
    assert response.text == "hello"
    assert response.parsed is None
