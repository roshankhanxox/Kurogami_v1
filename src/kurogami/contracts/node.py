"""NodeSpec / NodeResult: the unit the planner writes and the executor fills."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from kurogami.contracts.verdict import PassCondition


class NodeKind(StrEnum):
    RESEARCH = "research"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    DECISION = "decision"


class NodeStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    INVALIDATED = "invalidated"
    SKIPPED = "skipped"


class NodeSpec(BaseModel):
    """Written BY THE PLANNER, not by us. This is the self-prompting claim."""

    node_id: str
    parent_ids: list[str]
    depth: int
    kind: NodeKind
    title: str
    node_goal: str
    generated_prompt: str
    pass_condition: PassCondition
    context: dict[str, str] = {}
    injected_constraints: list[str] = []


class NodeResult(BaseModel):
    node_id: str
    output: str
    structured: dict[str, Any] = {}
    tokens_in: int
    tokens_out: int
    latency_ms: int
    model_id: str
    prompt_version: str
