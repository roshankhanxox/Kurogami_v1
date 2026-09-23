"""THE LOOP. Composes store, scheduler, context, budget, and backtrack behind the ports.

Implements ARCHITECTURE.md section 5 exactly. Every dependency arrives as a
port; this module imports no adapters and no concrete agent classes.
"""

import uuid
from dataclasses import dataclass
from typing import Protocol

from kurogami.contracts import (
    FailureReason,
    GoalSpec,
    InterruptPort,
    NodeResult,
    NodeSpec,
    TraceRecord,
    TraceSink,
    TreeSnapshot,
    Verdict,
)
from kurogami.engine import backtrack, context, scheduler
from kurogami.engine.budget import Budget
from kurogami.engine.store import DuplicateNodeError, TreeStore


class Interpreter(Protocol):
    def run(self, raw_text: str) -> GoalSpec: ...


class Planner(Protocol):
    def plan(self, goal: GoalSpec) -> list[NodeSpec]: ...
    def expand(self, node: NodeSpec, result: NodeResult, goal: GoalSpec) -> list[NodeSpec]: ...


class Executor(Protocol):
    def run(self, node: NodeSpec) -> NodeResult: ...


class RuleChecker(Protocol):
    def check(self, node: NodeSpec, result: NodeResult) -> Verdict: ...


class Verifier(Protocol):
    def check(self, node: NodeSpec, result: NodeResult, context: dict[str, str]) -> Verdict: ...


@dataclass
class RunReport:
    """What a caller (CLI, bench harness) reads after a run. Not a frozen contract."""

    run_id: uuid.UUID
    snapshot: TreeSnapshot
    budget_breached: bool
    breach_reason: str | None


class Runner:
    """The whole system. See ARCHITECTURE.md section 5 for the loop this implements."""

    def __init__(
        self,
        *,
        interpreter: Interpreter,
        planner: Planner,
        executor: Executor,
        rules_checker: RuleChecker,
        verifier: Verifier,
        interrupt: InterruptPort,
        trace_sink: TraceSink,
        budget: Budget | None = None,
    ) -> None:
        self._interpreter = interpreter
        self._planner = planner
        self._executor = executor
        self._rules_checker = rules_checker
        self._verifier = verifier
        self._interrupt = interrupt
        self._trace_sink = trace_sink
        self._budget = budget or Budget()

    def run(self, raw_goal: str) -> RunReport:
        run_id = uuid.uuid4()
        seq = 0
        store = TreeStore()

        goal = self._interpreter.run(raw_goal)
        roots = self._planner.plan(goal)
        for root in roots:
            self._budget.record_node_created(
                root.node_id, depth=root.depth, node_goal=root.node_goal, parent_ids=root.parent_ids
            )
        store.seed(roots)

        while store.has_pending() and self._budget.ok():
            node = scheduler.next(store)
            if node is None:
                break  # nothing runnable; remaining PENDING nodes are blocked, not our problem here

            interrupted = self._interrupt.should_pause(node)
            if interrupted:
                constraints = self._interrupt.collect(node)
                node = node.model_copy(
                    update={"injected_constraints": [*node.injected_constraints, *constraints]}
                )
                store.update_spec(node)
                store.mark_dirty_descendants(node.node_id)

            assembled_context = context.assemble(store, node)
            reason_text = store.pop_requeue_reason(node.node_id)
            if reason_text is not None:
                assembled_context = {**assembled_context, "_backtrack_reason": reason_text}
            node = node.model_copy(update={"context": assembled_context})
            store.update_spec(node)

            store.mark_running(node.node_id)
            result = self._executor.run(node)
            self._budget.record_tokens(result.tokens_in, result.tokens_out)

            verdict = self._rules_checker.check(node, result)
            if verdict.verdict == "PASS":
                verdict = self._verifier.check(node, result, node.context).model_copy(
                    update={"checked_by": "both"}
                )

            seq += 1
            backtrack_target: str | None = None
            if verdict.verdict == "PASS":
                store.mark_passed(node.node_id, result)
                children = self._planner.expand(node, result, goal)
                try:
                    for child in children:
                        self._budget.record_node_created(
                            child.node_id,
                            depth=child.depth,
                            node_goal=child.node_goal,
                            parent_ids=child.parent_ids,
                        )
                    store.attach(node, children)
                except DuplicateNodeError as exc:
                    # The planner produced a child node_id colliding with an
                    # existing node -- structurally invalid, not a real PASS.
                    # invalidate_subtree() below reverts mark_passed() above.
                    verdict = Verdict(
                        node_id=node.node_id,
                        verdict="FAIL",
                        checked_by="rules",
                        reason=FailureReason(
                            summary="planner produced a colliding node_id",
                            violated="schema",
                            evidence=str(exc),
                        ),
                    )
                    assert verdict.reason is not None
                    event = backtrack.apply(store, node.node_id, verdict.reason)
                    backtrack_target = event.target_node_id
                    self._budget.record_backtrack()
            else:
                store.mark_failed(node.node_id)
                assert verdict.reason is not None  # a FAIL verdict always carries a reason
                event = backtrack.apply(store, node.node_id, verdict.reason)
                backtrack_target = event.target_node_id
                self._budget.record_backtrack()

            self._trace_sink.emit(
                TraceRecord(
                    run_id=run_id,
                    seq=seq,
                    node_id=node.node_id,
                    parent_node_id=node.parent_ids[0] if node.parent_ids else None,
                    depth=node.depth,
                    generated_prompt=node.generated_prompt,
                    node_goal=node.node_goal,
                    output=result.output,
                    verifier_verdict=verdict.verdict,
                    failure_reason=verdict.reason.summary if verdict.reason else None,
                    backtrack_target=backtrack_target,
                    tokens_in=result.tokens_in,
                    tokens_out=result.tokens_out,
                    latency_ms=result.latency_ms,
                    human_interrupt=interrupted,
                )
            )

        if not self._budget.ok():
            for pending_id in store.pending_ids():
                store.mark_skipped(pending_id)
            seq += 1
            self._trace_sink.emit(
                TraceRecord(
                    run_id=run_id,
                    seq=seq,
                    node_id="__budget__",
                    parent_node_id=None,
                    depth=0,
                    generated_prompt="",
                    node_goal="budget breach",
                    output="",
                    verifier_verdict="FAIL",
                    failure_reason=self._budget.state.breach_reason,
                    backtrack_target=None,
                    tokens_in=0,
                    tokens_out=0,
                    latency_ms=0,
                    human_interrupt=False,
                )
            )

        self._trace_sink.close()
        return RunReport(
            run_id=run_id,
            snapshot=store.snapshot(),
            budget_breached=not self._budget.ok(),
            breach_reason=self._budget.state.breach_reason,
        )
