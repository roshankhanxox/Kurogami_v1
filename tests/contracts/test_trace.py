"""TraceRecord tests, including an explicit field-for-field schema lock.

This test is deliberately verbose: it exists so a later refactor cannot
silently drift from the schema submitted alongside deck Slide 7. If this
test needs to change, the change is to Slide 7 too, not just the code.
"""

from typing import Literal
from uuid import UUID, uuid4

from kurogami.contracts import TraceRecord

EXPECTED_FIELD_TYPES = {
    "run_id": UUID,
    "seq": int,
    "node_id": str,
    "parent_node_id": str | None,
    "depth": int,
    "generated_prompt": str,
    "node_goal": str,
    "output": str,
    "verifier_verdict": Literal["PASS", "FAIL"],
    "failure_reason": str | None,
    "backtrack_target": str | None,
    "tokens_in": int,
    "tokens_out": int,
    "latency_ms": int,
    "human_interrupt": bool,
}


def _record(**overrides) -> TraceRecord:
    kwargs = {
        "run_id": uuid4(),
        "seq": 1,
        "node_id": "n_004_pricing",
        "parent_node_id": "n_002_differentiation",
        "depth": 2,
        "generated_prompt": "Given the differentiation output, propose pricing tiers...",
        "node_goal": "Propose pricing tiers consistent with the WTP ceiling.",
        "output": "Tier 1: INR 500/mo. Tier 2: INR 2000/mo.",
        "verifier_verdict": "FAIL",
        "failure_reason": "Tier 2 exceeds the WTP ceiling established by market sizing.",
        "backtrack_target": "n_002_differentiation",
        "tokens_in": 310,
        "tokens_out": 140,
        "latency_ms": 1200,
        "human_interrupt": False,
    }
    kwargs.update(overrides)
    return TraceRecord(**kwargs)


def test_trace_record_field_names_match_architecture_section_4_4():
    assert set(TraceRecord.model_fields.keys()) == set(EXPECTED_FIELD_TYPES.keys())


def test_trace_record_field_annotations_match_architecture_section_4_4():
    for name, expected_type in EXPECTED_FIELD_TYPES.items():
        actual_type = TraceRecord.model_fields[name].annotation
        assert actual_type == expected_type, (
            f"TraceRecord.{name} is {actual_type}, expected {expected_type} "
            "per ARCHITECTURE.md section 4.4"
        )


def test_trace_record_round_trip():
    record = _record()
    restored = TraceRecord.model_validate_json(record.model_dump_json())
    assert restored == record


def test_trace_record_pass_has_no_failure_reason_or_backtrack_target():
    record = _record(
        verifier_verdict="PASS",
        failure_reason=None,
        backtrack_target=None,
    )
    assert record.failure_reason is None
    assert record.backtrack_target is None
