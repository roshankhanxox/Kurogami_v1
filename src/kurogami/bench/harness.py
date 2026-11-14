"""Archive a paired development pilot with fresh model responses and full graph evidence."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAIError
from pydantic import BaseModel

from kurogami.adapters.llm.openai import DEFAULT_MODEL, OpenAILLM, OpenAIRefusalError
from kurogami.adapters.llm.recording import RecordingLLM
from kurogami.adapters.trace.jsonl import JsonlTraceSink
from kurogami.agents._prompt_loader import load_prompt
from kurogami.agents.executor import Executor
from kurogami.agents.interpreter import Interpreter
from kurogami.agents.planner import Planner
from kurogami.agents.rules import RuleChecker
from kurogami.agents.verifier import Verifier
from kurogami.contracts import GoalSpec, NodeKind, NodeResult, NodeSpec, NodeStatus, TreeSnapshot
from kurogami.engine.budget import Budget
from kurogami.engine.interrupt import ScriptedInterrupt
from kurogami.engine.runner import Runner

DEV_GOALS = Path(__file__).with_name("goals") / "dev"
FAULT_OUTPUT = (
    "Confirmed pricing premise: pilot shops agreed to pay an upfront onboarding "
    "fee of INR 5,000. The launch recommendation should require this upfront fee."
)


class PilotGoal(BaseModel):
    goal_id: str
    raw_text: str
    split: Literal["dev"]
    planted_fault: str | None = None


class FaultPlanner(Planner):
    """Record the chosen upstream target before any node is executed."""

    def __init__(self, llm: RecordingLLM, manifest: Path) -> None:
        super().__init__(llm)
        self.target: str | None = None
        self._manifest = manifest

    def plan(self, goal: GoalSpec) -> list[NodeSpec]:
        nodes = super().plan(goal)
        self.target = next(node.node_id for node in nodes if not node.parent_ids)
        _write_json(self._manifest, {"expected_target": self.target, "injected_output": FAULT_OUTPUT})
        return nodes


class FaultExecutor(Executor):
    """Plant the fault once, before the real rules and verifier see the output."""

    def __init__(self, llm: RecordingLLM, planner: FaultPlanner) -> None:
        super().__init__(llm)
        self._planner = planner
        self._planted = False

    def run(self, node: NodeSpec) -> NodeResult:
        result = super().run(node)
        if node.node_id != self._planner.target or self._planted:
            return result
        self._planted = True
        return result.model_copy(update={"output": FAULT_OUTPUT, "structured": {}})


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sum_tokens(log: Path) -> int | None:
    if not log.exists():
        return None
    with log.open(encoding="utf-8") as stream:
        return sum(
            row.get("tokens_in", 0) + row.get("tokens_out", 0)
            for row in map(json.loads, stream)
        )


def graph_is_valid(snapshot: TreeSnapshot) -> bool:
    """Check the complete active graph, including dependencies elided by replay."""
    active = {
        node_id: spec for node_id, spec in snapshot.specs.items()
        if snapshot.statuses[node_id] != NodeStatus.INVALIDATED
    }
    if not active or max(spec.depth for spec in active.values()) < 4:
        return False
    for spec in active.values():
        if any(parent not in active for parent in spec.parent_ids):
            return False
        if spec.depth != 1 + max((active[parent].depth for parent in spec.parent_ids), default=-1):
            return False
    return True


def _run_kurogami(goal: PilotGoal, model: str, folder: Path) -> dict[str, object]:
    llm = RecordingLLM(OpenAILLM(model=model, max_retries=0, timeout_s=30), folder / "kurogami.calls.jsonl")
    planner = Planner(llm)
    executor = Executor(llm)
    fault_planner = None
    if goal.planted_fault:
        fault_planner = FaultPlanner(llm, folder / "fault.json")
        planner = fault_planner
        executor = FaultExecutor(llm, fault_planner)

    trace = JsonlTraceSink(folder / "kurogami.jsonl")
    try:
        report = Runner(
            interpreter=Interpreter(llm), planner=planner, executor=executor,
            rules_checker=RuleChecker(), verifier=Verifier(llm), interrupt=ScriptedInterrupt([]),
            trace_sink=trace, budget=Budget(),
        ).run(goal.raw_text)
    finally:
        trace.close()

    snapshot = report.snapshot
    _write_json(folder / "graph.json", snapshot.model_dump(mode="json"))
    failure = report.aborted_reason or report.breach_reason or report.incomplete_reason
    decision = next(
        (result for node_id, result in snapshot.results.items()
         if snapshot.statuses[node_id] == NodeStatus.PASSED
         and snapshot.specs[node_id].kind == NodeKind.DECISION),
        None,
    )
    if decision is not None and failure is None:
        (folder / "kurogami.txt").write_text(decision.output, encoding="utf-8")
    target = fault_planner.target if fault_planner is not None else None
    return {
        "complete": failure is None and decision is not None,
        "failure": failure,
        "valid_graph": graph_is_valid(snapshot),
        "correct_ancestor_backtrack": (
            any(event.target_node_id == target and event.failing_node_id != target
                for event in snapshot.backtrack_events) if target is not None else None
        ),
        "recovery": "review repaired output against zero upfront fee" if target is not None else None,
    }


def _run_baseline(goal: PilotGoal, model: str, folder: Path) -> dict[str, object]:
    llm = RecordingLLM(OpenAILLM(model=model, max_retries=0, timeout_s=30), folder / "baseline.calls.jsonl")
    response = llm.complete(prompt=load_prompt("baseline").format(raw_goal=goal.raw_text))
    (folder / "baseline.txt").write_text(response.text, encoding="utf-8")
    return {"complete": bool(response.text.strip())}


def one_pair(goal: PilotGoal, model: str, folder: Path) -> dict[str, object]:
    """Preserve each arm independently, including partial call logs on error."""
    folder.mkdir()
    _write_json(folder / "goal.json", goal.model_dump(mode="json"))
    result: dict[str, object] = {"goal_id": goal.goal_id}
    for name, run in (("kurogami", _run_kurogami), ("baseline", _run_baseline)):
        try:
            arm = run(goal, model, folder)
        except (OpenAIError, OpenAIRefusalError, ValueError) as exc:
            arm = {"complete": False, "failure": f"{type(exc).__name__}: {exc}"}
        arm["tokens"] = _sum_tokens(folder / f"{name}.calls.jsonl")
        result[name] = arm
    _write_json(folder / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--jobs", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    load_dotenv()
    args.out.mkdir(parents=True)
    goals = [PilotGoal.model_validate_json((DEV_GOALS / f"g00{i}.json").read_text(encoding="utf-8"))
             for i in (1, 2, 3)]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = [pool.submit(one_pair, goal, args.model, args.out / goal.goal_id) for goal in goals]
        pairs = [job.result() for job in pending]
    _write_json(args.out / "pilot.json", {
        "model": args.model, "retrieval": "none", "runs_per_goal": 1, "pairs": pairs,
    })


if __name__ == "__main__":
    main()
