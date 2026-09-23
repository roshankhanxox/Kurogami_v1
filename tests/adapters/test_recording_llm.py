"""RecordingLLM: every call -- including failures -- lands in the log, then passes through."""

import json

import pytest

from kurogami.adapters.llm.fake import FakeLLM, UnregisteredPromptError
from kurogami.adapters.llm.recording import RecordingLLM
from kurogami.contracts import GoalSpec


def _goal() -> GoalSpec:
    return GoalSpec(
        raw_text="x", product_description="x", target_market="x",
        decision_type="launch", success_definition="x",
    )


def _lines(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_records_the_prompt_and_the_parsed_response(tmp_path):
    log = tmp_path / "calls.jsonl"
    llm = RecordingLLM(FakeLLM(responses={"p": _goal()}), log)

    response = llm.complete(prompt="p", schema=GoalSpec)

    assert response.parsed == _goal()
    [record] = _lines(log)
    assert record["seq"] == 1
    assert record["schema"] == "GoalSpec"
    assert record["prompt"] == "p"
    assert record["parsed"]["decision_type"] == "launch"


def test_records_free_text_calls_in_order(tmp_path):
    log = tmp_path / "calls.jsonl"
    llm = RecordingLLM(FakeLLM(responses={"a": "first", "b": "second"}), log)

    llm.complete(prompt="a")
    llm.complete(prompt="b")

    assert [(r["seq"], r["text"], r["schema"]) for r in _lines(log)] == [
        (1, "first", None),
        (2, "second", None),
    ]


def test_a_failed_call_is_recorded_and_still_raised(tmp_path):
    log = tmp_path / "calls.jsonl"
    llm = RecordingLLM(FakeLLM(responses={}), log)

    with pytest.raises(UnregisteredPromptError):
        llm.complete(prompt="missing")

    [record] = _lines(log)
    assert record["error"].startswith("UnregisteredPromptError")
