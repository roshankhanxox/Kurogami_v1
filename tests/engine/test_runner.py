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
from kurogami.engine.budget import Budget
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


class _AlwaysPassVerifier:
    def check(self, node: NodeSpec, result: NodeResult, context: dict[str, str]) -> Verdict:
        return Verdict(node_id=node.node_id, verdict="PASS", checked_by="llm")


class _CollidingChildPlanner:
    """Root n_a always passes, but expand() always tries to attach a child
    with node_id="n_a" -- the same id as the node being expanded. Regression
    for a live incident where a colliding node_id silently corrupted the
    tree instead of being rejected.
    """

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
        return [
            NodeSpec(
                node_id="n_a",
                parent_ids=["n_a"],
                depth=1,
                kind=NodeKind.ANALYSIS,
                title="colliding child",
                node_goal="x",
                generated_prompt="x",
                pass_condition=_pass_condition(),
            )
        ]


def test_colliding_gap_fill_is_dropped_and_the_planned_tree_still_finishes():
    """Regression: a node_id collision once silently corrupted the tree. A colliding
    gap-fill is now rejected by the store and dropped -- the planned tree completes.
    """
    runner = Runner(
        interpreter=_FakeInterpreter(),
        planner=_CollidingChildPlanner(),
        executor=_FakeExecutor(),
        rules_checker=_AlwaysPassRules(),
        verifier=_AlwaysPassVerifier(),
        interrupt=_NeverInterrupt(),
        trace_sink=_MemoryTraceSink(),
    )

    report = runner.run("Should I launch my invoicing tool?")  # must not raise

    assert report.budget_breached is False
    assert report.snapshot.statuses == {"n_a": NodeStatus.PASSED}


# --- scoped planning: the whole tree is planned up front ---------------------------------


def _spec(node_id: str, parents: list[str], depth: int, assertions: list[str] | None = None) -> NodeSpec:
    return NodeSpec(
        node_id=node_id,
        parent_ids=parents,
        depth=depth,
        kind=NodeKind.ANALYSIS,
        title=node_id,
        node_goal=f"goal {node_id}",
        generated_prompt=f"prompt for {node_id}",
        pass_condition=PassCondition(assertions=assertions or [], semantic_check="ok?"),
    )


class _BlueprintPlanner:
    """Plans a->b->c up front; expand() returns whatever gaps it was scripted with."""

    def __init__(self, gaps: dict[str, list[NodeSpec]] | None = None) -> None:
        self._gaps = gaps or {}
        self.expand_calls: list[str] = []

    def plan(self, goal: GoalSpec) -> list[NodeSpec]:
        return [_spec("a", [], 0), _spec("b", ["a"], 1), _spec("c", ["b"], 2)]

    def expand(self, node: NodeSpec, result: NodeResult, goal: GoalSpec) -> list[NodeSpec]:
        self.expand_calls.append(node.node_id)
        return self._gaps.pop(node.node_id, [])


class _RecordingExecutor:
    def __init__(self) -> None:
        self.runs: list[NodeSpec] = []

    def run(self, node: NodeSpec) -> NodeResult:
        self.runs.append(node)
        return NodeResult(
            node_id=node.node_id, output=f"output for {node.node_id}", tokens_in=1,
            tokens_out=1, latency_ms=1, model_id="fake", prompt_version="v1",
        )


class _FailFirstVerifier:
    """FAILs `node_id` once, blaming `suspect`; PASSes everything else."""

    def __init__(self, node_id: str, suspect: str) -> None:
        self._node_id, self._suspect, self._done = node_id, suspect, False

    def check(self, node: NodeSpec, result: NodeResult, context: dict[str, str]) -> Verdict:
        if node.node_id == self._node_id and not self._done:
            self._done = True
            return Verdict(
                node_id=node.node_id, verdict="FAIL", checked_by="llm",
                reason=FailureReason(
                    summary="contradicts an ancestor", violated="semantic", evidence="x",
                    suspect_node_ids=[self._suspect],
                ),
            )
        return Verdict(node_id=node.node_id, verdict="PASS", checked_by="llm")


def _runner(planner, executor=None, verifier=None, interrupt=None, trace_sink=None, budget=None):
    return Runner(
        interpreter=_FakeInterpreter(),
        planner=planner,
        executor=executor or _RecordingExecutor(),
        rules_checker=_AlwaysPassRules(),
        verifier=verifier or _AlwaysPassVerifier(),
        interrupt=interrupt or _NeverInterrupt(),
        trace_sink=trace_sink or _MemoryTraceSink(),
        budget=budget,
    )


def test_planned_tree_runs_to_completion_and_nothing_is_added():
    planner = _BlueprintPlanner()
    report = _runner(planner).run("goal")

    assert report.budget_breached is False
    assert report.snapshot.statuses == {
        "a": NodeStatus.PASSED, "b": NodeStatus.PASSED, "c": NodeStatus.PASSED,
    }
    # Offered after every PASS; with no reported gap it adds nothing (and costs no LLM call).
    assert planner.expand_calls == ["a", "b", "c"]


def test_backtrack_regenerates_the_same_slots_as_versioned_clones():
    report = _runner(_BlueprintPlanner(), verifier=_FailFirstVerifier("c", suspect="a")).run("goal")
    statuses = report.snapshot.statuses

    assert report.budget_breached is False
    assert statuses["b"] == NodeStatus.INVALIDATED  # originals kept, flagged (ARCHITECTURE 5c)
    assert statuses["c"] == NodeStatus.INVALIDATED
    assert statuses["b~r1"] == NodeStatus.PASSED
    assert statuses["c~r1"] == NodeStatus.PASSED
    assert report.snapshot.specs["c~r1"].parent_ids == ["b~r1"]
    assert report.snapshot.specs["b~r1"].parent_ids == ["a"]


def test_a_plan_that_cannot_be_made_is_a_clean_outcome_not_a_crash():
    from kurogami.agents.planner import BlueprintError

    class _Unplannable(_BlueprintPlanner):
        def plan(self, goal: GoalSpec) -> list[NodeSpec]:
            raise BlueprintError("scope still invalid after correction: cycle")

    sink = _MemoryTraceSink()
    report = _runner(_Unplannable(), trace_sink=sink).run("goal")

    assert report.aborted_reason is not None and "cycle" in report.aborted_reason
    assert [r.node_id for r in sink.records] == ["__plan__"]
    assert sink.closed is True


def test_interrupt_constraints_reach_the_node_and_every_planned_descendant():
    from kurogami.engine.interrupt import ScriptedInterrupt

    executor = _RecordingExecutor()
    interrupt = ScriptedInterrupt(["a"], {"a": ["Keep the price under INR 500/month."]})
    _runner(_BlueprintPlanner(), executor=executor, interrupt=interrupt).run("goal")

    for ran in executor.runs:
        assert "Keep the price under INR 500/month." in ran.injected_constraints


def test_gap_fill_feeds_the_reporters_pending_children():
    gap = _spec("gap", ["a"], 1)
    executor = _RecordingExecutor()
    report = _runner(_BlueprintPlanner(gaps={"a": [gap]}), executor=executor).run("goal")

    specs = report.snapshot.specs
    assert specs["b"].parent_ids == ["a", "gap"]
    assert specs["b"].depth == 2 and specs["c"].depth == 3  # pushed down by the new level
    order = [n.node_id for n in executor.runs]
    assert order.index("gap") < order.index("b")


def test_gap_fills_stop_at_the_cap():
    gaps = {"a": [_spec("g1", ["a"], 1)], "b": [_spec("g2", ["b"], 2)], "c": [_spec("g3", ["c"], 3)]}
    planner = _BlueprintPlanner(gaps=gaps)
    report = _runner(planner, budget=Budget(max_gap_fills=1)).run("goal")

    assert "g1" in report.snapshot.specs
    assert "g2" not in report.snapshot.specs and "g3" not in report.snapshot.specs
    assert planner.expand_calls == ["a"]


class _AlwaysFailSelfVerifier:
    """FAILs `node_id` every time, blaming only itself."""

    def __init__(self, node_id: str) -> None:
        self._node_id = node_id

    def check(self, node: NodeSpec, result: NodeResult, context: dict[str, str]) -> Verdict:
        if node.node_id == self._node_id:
            return Verdict(
                node_id=node.node_id, verdict="FAIL", checked_by="llm",
                reason=FailureReason(summary="not good enough", violated="semantic", evidence="x"),
            )
        return Verdict(node_id=node.node_id, verdict="PASS", checked_by="llm")


class _TwoBranchPlanner(_BlueprintPlanner):
    def plan(self, goal: GoalSpec) -> list[NodeSpec]:
        return [_spec("a", [], 0), _spec("b", ["a"], 1), _spec("x", [], 0), _spec("y", ["x"], 1)]


def test_a_self_failing_node_gives_up_without_spending_the_backtrack_budget():
    budget = Budget()
    report = _runner(
        _TwoBranchPlanner(), verifier=_AlwaysFailSelfVerifier("a"), budget=budget
    ).run("goal")
    statuses = report.snapshot.statuses

    assert statuses["a"] == NodeStatus.FAILED  # gave up after its own retries
    assert statuses["b"] == NodeStatus.SKIPPED  # could never run
    assert statuses["x"] == NodeStatus.PASSED and statuses["y"] == NodeStatus.PASSED
    assert budget.state.backtracks == 0  # self-retries are not ancestor backtracks
    assert budget.state.node_retries["a"] == 3  # 1 attempt + max_node_retries (2) retries
    assert report.budget_breached is False
    assert report.incomplete_reason is not None and "gave up on a" in report.incomplete_reason
