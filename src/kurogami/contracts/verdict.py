"""PassCondition / Verdict: the two-layer check a NodeResult must clear."""

from typing import Literal

from pydantic import BaseModel


class PassCondition(BaseModel):
    """Two-layer: cheap deterministic assertions + one semantic question."""

    assertions: list[str]
    semantic_check: str


class FailureReason(BaseModel):
    summary: str
    violated: Literal["assertion", "semantic", "schema"]
    evidence: str
    suspect_node_ids: list[str] = []


class Verdict(BaseModel):
    node_id: str
    verdict: Literal["PASS", "FAIL"]
    reason: FailureReason | None = None
    checked_by: Literal["rules", "llm", "both"]
