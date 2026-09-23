"""TraceRecord: the evaluation dataset. Must match deck Slide 7 field-for-field."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel


class TraceRecord(BaseModel):
    run_id: UUID
    seq: int
    node_id: str
    parent_node_id: str | None
    depth: int
    generated_prompt: str
    node_goal: str
    output: str
    verifier_verdict: Literal["PASS", "FAIL"]
    failure_reason: str | None
    backtrack_target: str | None
    tokens_in: int
    tokens_out: int
    latency_ms: int
    human_interrupt: bool
