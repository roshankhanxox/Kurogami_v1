"""Round-trip and validation tests for PassCondition, FailureReason, Verdict."""

import pytest
from pydantic import ValidationError

from kurogami.contracts import FailureReason, PassCondition, Verdict


def test_pass_condition_round_trip():
    pc = PassCondition(
        assertions=["len(structured['tiers']) >= 2"],
        semantic_check="Do the tiers respect the WTP ceiling?",
    )
    assert PassCondition.model_validate_json(pc.model_dump_json()) == pc


def test_failure_reason_defaults_suspect_node_ids_empty():
    fr = FailureReason(
        summary="Pricing exceeds WTP ceiling.",
        violated="semantic",
        evidence="Tier 2 is priced at INR 2000/month.",
    )
    assert fr.suspect_node_ids == []


def test_failure_reason_rejects_unknown_violated_literal():
    with pytest.raises(ValidationError):
        FailureReason(summary="x", violated="vibes", evidence="x")


def test_verdict_pass_has_no_reason_required():
    v = Verdict(node_id="n_004", verdict="PASS", checked_by="rules")
    assert v.reason is None


def test_verdict_fail_round_trip_with_reason():
    fr = FailureReason(
        summary="Pricing exceeds WTP ceiling.",
        violated="semantic",
        evidence="Tier 2 is priced at INR 2000/month.",
        suspect_node_ids=["n_002_differentiation"],
    )
    v = Verdict(node_id="n_004_pricing", verdict="FAIL", reason=fr, checked_by="both")
    restored = Verdict.model_validate_json(v.model_dump_json())
    assert restored == v


def test_verdict_rejects_unknown_verdict_literal():
    with pytest.raises(ValidationError):
        Verdict(node_id="n_004", verdict="MAYBE", checked_by="rules")
