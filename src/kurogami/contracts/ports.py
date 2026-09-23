"""The reusability claim, made concrete. Every port has a Fake* implementation."""

from datetime import datetime
from typing import Protocol

from pydantic import BaseModel

from kurogami.contracts.goal import GoalSpec
from kurogami.contracts.node import NodeResult, NodeSpec
from kurogami.contracts.trace import TraceRecord
from kurogami.contracts.tree import TreeSnapshot
from kurogami.contracts.verdict import Verdict


class LLMResponse(BaseModel):
    """What an LLMPort call returns: raw text, optionally schema-parsed."""

    text: str
    parsed: BaseModel | None = None
    tokens_in: int
    tokens_out: int
    latency_ms: int
    model_id: str


class SearchHit(BaseModel):
    title: str
    url: str
    snippet: str


class LLMPort(Protocol):
    def complete(
        self,
        *,
        prompt: str,
        system: str | None = None,
        schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


class SearchPort(Protocol):
    def search(self, query: str, *, max_results: int = 5) -> list[SearchHit]: ...


class TraceSink(Protocol):
    def emit(self, record: TraceRecord) -> None: ...
    def close(self) -> None: ...


class InterruptPort(Protocol):
    def should_pause(self, node: NodeSpec) -> bool: ...
    def collect(self, node: NodeSpec) -> list[str]: ...


class RunnerHooks(Protocol):
    """Callbacks the runtime invokes as it advances the tree, e.g. for rendering."""

    def on_node_started(self, node: NodeSpec) -> None: ...
    def on_node_finished(self, node: NodeSpec, result: NodeResult, verdict: Verdict) -> None: ...
    def on_backtrack(self, target_node_id: str, invalidated_node_ids: list[str]) -> None: ...


class RuntimePort(Protocol):
    def run(self, plan: TreeSnapshot, *, hooks: RunnerHooks) -> TreeSnapshot: ...


class PlannerPort(Protocol):
    def plan(self, goal: GoalSpec) -> list[NodeSpec]: ...
    def expand(self, node: NodeSpec, result: NodeResult, goal: GoalSpec) -> list[NodeSpec]: ...


class ExecutorPort(Protocol):
    def run(self, node: NodeSpec) -> NodeResult: ...


class VerifierPort(Protocol):
    def check(self, node: NodeSpec, result: NodeResult, context: dict[str, str]) -> Verdict: ...


class ClockPort(Protocol):
    def now(self) -> datetime: ...
    def monotonic_ms(self) -> int: ...
