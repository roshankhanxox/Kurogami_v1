"""THE LOOP. Composes store, scheduler, context, budget, and backtrack behind the ports.

Implements ARCHITECTURE.md section 5 as amended for scoped planning: the Master
plans the whole tree up front, the run ends when that tree is done, the tree
grows at runtime only through bounded gap-fills, and a backtrack regenerates the
same planned slots. Every dependency arrives as a port; no adapters are imported.
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Protocol

from kurogami.agents.planner import BlueprintError
from kurogami.contracts import (
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

_log = logging.getLogger(__name__)


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
    aborted_reason: str | None = None


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
        try:
            blueprint = self._planner.plan(goal)
        except BlueprintError as exc:
            return self._abort(run_id, store, f"planning failed: {exc}")
        for planned in blueprint:
            self._record_created(planned)
        store.seed_blueprint(blueprint)

        while store.has_pending() and self._budget.ok():
            node = scheduler.next(store)
            if node is None:
                break  # nothing runnable; remaining PENDING nodes are blocked, not our problem here

            interrupted = self._interrupt.should_pause(node)
            if interrupted:
                constraints = self._interrupt.collect(node)
                # The whole subtree is already planned, so the constraint goes to the
                # node and every descendant -- regenerated clones inherit it too.
                for target in [node, *store.descendants(node.node_id)]:
                    added = [c for c in constraints if c not in target.injected_constraints]
                    store.update_spec(
                        target.model_copy(
                            update={"injected_constraints": [*target.injected_constraints, *added]}
                        )
                    )
                node = store.get(node.node_id)

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
                if store.awaiting_regeneration(node.node_id):
                    for clone in store.regenerate_subtree(node.node_id):
                        self._record_created(clone)
                if self._budget.gap_fills_remaining() and self._budget.ok():
                    self._fill_gaps(store, node, result, goal)
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

    def _record_created(self, node: NodeSpec) -> None:
        self._budget.record_node_created(
            node.node_id, depth=node.depth, node_goal=node.node_goal, parent_ids=node.parent_ids
        )

    def _fill_gaps(self, store: TreeStore, node: NodeSpec, result: NodeResult, goal: GoalSpec) -> None:
        """Bounded runtime growth: only when a node reports an unplanned prerequisite."""
        for gap in self._planner.expand(node, result, goal):
            try:
                added = store.add_gap_node(
                    gap, node.node_id, max_depth=self._budget.limits.max_depth - 1
                )
            except DuplicateNodeError:
                added = False
            if not added:
                _log.warning("dropping gap-fill %s under %s", gap.node_id, node.node_id)
                continue
            self._budget.record_gap_fill()
            self._record_created(store.get(gap.node_id))

    def _abort(self, run_id: uuid.UUID, store: TreeStore, reason: str) -> RunReport:
        """A run that cannot start is a reported outcome with a trace record, not a crash."""
        self._trace_sink.emit(
            TraceRecord(
                run_id=run_id,
                seq=1,
                node_id="__plan__",
                parent_node_id=None,
                depth=0,
                generated_prompt="",
                node_goal="plan the investigation",
                output="",
                verifier_verdict="FAIL",
                failure_reason=reason,
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
            budget_breached=False,
            breach_reason=None,
            aborted_reason=reason,
        )
