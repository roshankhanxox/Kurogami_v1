"""Integration: the runner loop closes without exception and produces a trace, offline.

The agents that will eventually plug into these ports (Interpreter, Planner,
Executor, RuleChecker, Verifier) belong to other tracks and don't exist yet.
These are minimal, self-contained test doubles that satisfy the same ports,
so engine/runner.py can be proven correct in isolation.
"""

from kurogami.contracts import (
    FailureReason,
    GoalSpec,
    NodeKind,
    NodeResult,
    NodeSpec,
    NodeStatus,
    PassCondition,
    TraceRecord,
    Verdict,
)
from kurogami.engine.runner import Runner


def _pass_condition() -> PassCondition:
    return PassCondition(assertions=[], semantic_check="ok?")


class _FakeInterpreter:
    def run(self, raw_text: str) -> GoalSpec:
        return GoalSpec(
            raw_text=raw_text,
            product_description="Invoicing tool for freelance designers.",
            target_market="Freelance designers in India",
            decision_type="market_entry",
            success_definition="A clear go/no-go.",
        )


class _FakePlanner:
    """Root n_a always passes and expands once into n_b. n_b fails once, then passes."""

    def __init__(self) -> None:
        self.expand_calls = 0

    def plan(self, goal: GoalSpec) -> list[NodeSpec]:
        return [
            NodeSpec(
                node_id="n_a",
                parent_ids=[],
                depth=0,
                kind=NodeKind.ANALYSIS,
                title="root",
                node_goal="root goal",
                generated_prompt="root prompt",
                pass_condition=_pass_condition(),
            )
        ]

    def expand(self, node: NodeSpec, result: NodeResult, goal: GoalSpec) -> list[NodeSpec]:
        self.expand_calls += 1
        if node.node_id != "n_a" or self.expand_calls > 1:
            return []
        return [
            NodeSpec(
                node_id="n_b",
                parent_ids=["n_a"],
                depth=1,
                kind=NodeKind.ANALYSIS,
                title="child",
                node_goal="child goal",
                generated_prompt="child prompt",
                pass_condition=_pass_condition(),
            )
        ]


class _FakeExecutor:
    def __init__(self) -> None:
        self.n_b_calls = 0

    def run(self, node: NodeSpec) -> NodeResult:
        if node.node_id == "n_b":
            self.n_b_calls += 1
        return NodeResult(
            node_id=node.node_id,
            output=f"output for {node.node_id}",
            tokens_in=10,
            tokens_out=10,
            latency_ms=1,
            model_id="fake",
            prompt_version="v1",
        )


class _AlwaysPassRules:
    def check(self, node: NodeSpec, result: NodeResult) -> Verdict:
        return Verdict(node_id=node.node_id, verdict="PASS", checked_by="rules")


class _FailOnceVerifier:
    """Fails n_b's first execution, blaming n_b itself; passes everything else."""

    def __init__(self) -> None:
        self._failed_once = False

    def check(self, node: NodeSpec, result: NodeResult, context: dict[str, str]) -> Verdict:
        if node.node_id == "n_b" and not self._failed_once:
            self._failed_once = True
            return Verdict(
                node_id=node.node_id,
                verdict="FAIL",
                checked_by="llm",
                reason=FailureReason(
                    summary="first attempt is wrong",
                    violated="semantic",
                    evidence="x",
                    suspect_node_ids=["n_b"],
                ),
            )
        return Verdict(node_id=node.node_id, verdict="PASS", checked_by="llm")


class _NeverInterrupt:
    def should_pause(self, node: NodeSpec) -> bool:
        return False

    def collect(self, node: NodeSpec) -> list[str]:
        return []


class _MemoryTraceSink:
    def __init__(self) -> None:
        self.records: list[TraceRecord] = []
        self.closed = False

    def emit(self, record: TraceRecord) -> None:
        self.records.append(record)

    def close(self) -> None:
        self.closed = True


def test_full_run_with_fake_llm():
    executor = _FakeExecutor()
    trace_sink = _MemoryTraceSink()

    runner = Runner(
        interpreter=_FakeInterpreter(),
        planner=_FakePlanner(),
        executor=executor,
        rules_checker=_AlwaysPassRules(),
        verifier=_FailOnceVerifier(),
        interrupt=_NeverInterrupt(),
        trace_sink=trace_sink,
    )

    report = runner.run("Should I launch my invoicing tool?")

    assert report.budget_breached is False
    assert executor.n_b_calls == 2  # failed once, backtracked to itself, ran again

    snapshot = report.snapshot
    assert snapshot.statuses["n_a"] == NodeStatus.PASSED
    assert snapshot.statuses["n_b"] == NodeStatus.PASSED

    assert trace_sink.closed is True
    verdicts = [r.verifier_verdict for r in trace_sink.records]
    assert "FAIL" in verdicts
    assert verdicts[-1] == "PASS"

    fail_record = next(r for r in trace_sink.records if r.verifier_verdict == "FAIL")
    assert fail_record.backtrack_target == "n_b"

    pass_records = [r for r in trace_sink.records if r.node_id == "n_b" and r.verifier_verdict == "PASS"]
    assert pass_records[0].verifier_verdict == "PASS"
