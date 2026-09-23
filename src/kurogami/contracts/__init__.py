"""Layer 0: typed contracts. Pure data, zero internal imports."""

from kurogami.contracts.budget import BudgetLimits, BudgetState
from kurogami.contracts.goal import GoalSpec
from kurogami.contracts.node import NodeKind, NodeResult, NodeSpec, NodeStatus
from kurogami.contracts.ports import (
    ClockPort,
    ExecutorPort,
    InterruptPort,
    LLMPort,
    LLMResponse,
    PlannerPort,
    RuntimePort,
    SearchHit,
    SearchPort,
    TraceSink,
    VerifierPort,
)
from kurogami.contracts.trace import TraceRecord
from kurogami.contracts.tree import BacktrackEvent, TreeSnapshot
from kurogami.contracts.verdict import FailureReason, PassCondition, Verdict

__all__ = [
    "BacktrackEvent",
    "BudgetLimits",
    "BudgetState",
    "ClockPort",
    "ExecutorPort",
    "FailureReason",
    "GoalSpec",
    "InterruptPort",
    "LLMPort",
    "LLMResponse",
    "NodeKind",
    "NodeResult",
    "NodeSpec",
    "NodeStatus",
    "PassCondition",
    "PlannerPort",
    "RuntimePort",
    "SearchHit",
    "SearchPort",
    "TraceRecord",
    "TraceSink",
    "TreeSnapshot",
    "Verdict",
    "VerifierPort",
]
