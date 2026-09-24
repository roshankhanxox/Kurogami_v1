"""The Textual app, driven headless through Textual's pilot against a real Runner, offline."""

import asyncio

from textual.widgets import TextArea, Tree

from kurogami.cli.tui import ConstraintScreen, KurogamiApp, primary_parent
from kurogami.contracts import (
    GoalSpec,
    NodeKind,
    NodeResult,
    NodeSpec,
    NodeStatus,
    PassCondition,
    TraceRecord,
    TreeSnapshot,
    Verdict,
)
from kurogami.engine.runner import Runner


def _spec(node_id: str, parents: list[str], depth: int, kind=NodeKind.ANALYSIS) -> NodeSpec:
    return NodeSpec(
        node_id=node_id, parent_ids=parents, depth=depth, kind=kind, title=node_id.title(),
        node_goal=f"goal {node_id}", generated_prompt=f"prompt {node_id}",
        pass_condition=PassCondition(assertions=[], semantic_check="ok?"),
    )


# a -> b, a -> c, (b, c) -> d: d has two parents and must still be drawn once.
_PLAN = [_spec("a", [], 0), _spec("b", ["a"], 1), _spec("c", ["a", "b"], 2),
         _spec("d", ["b", "c"], 3, NodeKind.DECISION)]


class _Interpreter:
    def run(self, raw_text: str) -> GoalSpec:
        return GoalSpec(raw_text=raw_text, product_description="x", target_market="x",
                        decision_type="launch", success_definition="x")


class _Planner:
    def plan(self, goal):
        return list(_PLAN)

    def expand(self, node, result, goal):
        return []


class _Executor:
    def __init__(self) -> None:
        self.runs: list[NodeSpec] = []

    def run(self, node: NodeSpec) -> NodeResult:
        self.runs.append(node)
        return NodeResult(node_id=node.node_id, output=f"# answer {node.node_id}", tokens_in=1,
                          tokens_out=1, latency_ms=1, model_id="f", prompt_version="v")


class _Pass:
    def check(self, node, result, context=None) -> Verdict:
        return Verdict(node_id=node.node_id, verdict="PASS", checked_by="rules")


class _Sink:
    def emit(self, record: TraceRecord) -> None:
        pass

    def close(self) -> None:
        pass


def _app(executor: _Executor, **kwargs) -> KurogamiApp:
    def start(bridge):
        return Runner(
            interpreter=_Interpreter(), planner=_Planner(), executor=executor,
            rules_checker=_Pass(), verifier=_Pass(), interrupt=bridge, trace_sink=_Sink(),
            observer=bridge,
        ).run("Should I launch?")

    return KurogamiApp("Should I launch?", start_run=start, **kwargs)


async def _until(pilot, condition, timeout: float = 5.0) -> None:
    for _ in range(int(timeout / 0.05)):
        if condition():
            return
        await pilot.pause(0.05)
    raise AssertionError("condition never became true")


def test_nothing_runs_until_the_plan_is_approved():
    async def scenario():
        executor = _Executor()
        app = _app(executor)
        async with app.run_test() as pilot:
            await _until(pilot, lambda: app.run_screen.state.snapshot is not None)
            await pilot.pause(0.3)
            assert executor.runs == []
            await pilot.press("r")
            await _until(pilot, lambda: app.run_screen.state.phase == "finished")
            assert [n.node_id for n in executor.runs] == ["a", "b", "c", "d"]
            assert app.run_screen.state.outcome == "✓ complete"

    asyncio.run(scenario())


def test_every_node_is_drawn_exactly_once_under_its_deepest_parent():
    async def scenario():
        app = _app(_Executor(), review_plan=False)
        async with app.run_test() as pilot:
            await _until(pilot, lambda: app.run_screen.state.phase == "finished")
            tree = app.run_screen.query_one("#tree", Tree)
            drawn = []

            def walk(node):
                for child in node.children:
                    drawn.append((child.data, node.data))
                    walk(child)

            walk(tree.root)
            assert sorted(drawn) == [("a", None), ("b", "a"), ("c", "b"), ("d", "c")]

    asyncio.run(scenario())


def test_a_constraint_typed_at_a_pause_reaches_the_node_and_its_descendants():
    async def scenario():
        executor = _Executor()
        app = _app(executor)
        async with app.run_test() as pilot:
            await _until(pilot, lambda: "c" in app.run_screen._tree_nodes)
            app.run_screen.pause_ids.add("c")
            await pilot.press("r")
            await _until(pilot, lambda: isinstance(app.screen, ConstraintScreen))
            app.screen.query_one(TextArea).text = "Price must stay under INR 500/month"
            await pilot.click("#apply")
            await _until(pilot, lambda: app.run_screen.state.phase == "finished")
            by_id = {n.node_id: n for n in executor.runs}
            assert by_id["c"].injected_constraints == ["Price must stay under INR 500/month"]
            assert by_id["d"].injected_constraints == ["Price must stay under INR 500/month"]
            assert by_id["b"].injected_constraints == []

    asyncio.run(scenario())


def test_a_finished_run_opens_offline_on_the_final_answer():
    specs = {s.node_id: s for s in _PLAN}
    snapshot = TreeSnapshot(
        root_ids=["a"], specs=specs,
        statuses=dict.fromkeys(specs, NodeStatus.PASSED),
        results={i: NodeResult(node_id=i, output=f"answer {i}", tokens_in=1, tokens_out=1,
                               latency_ms=1, model_id="f", prompt_version="v") for i in specs},
    )

    async def scenario():
        app = KurogamiApp("goal", snapshot=snapshot)
        async with app.run_test() as pilot:
            await pilot.pause(0.2)
            assert app.run_screen._shown == "d"

    asyncio.run(scenario())


def test_primary_parent_is_the_deepest_one():
    snapshot = TreeSnapshot(root_ids=["a"], specs={s.node_id: s for s in _PLAN},
                            statuses={s.node_id: NodeStatus.PENDING for s in _PLAN})
    assert primary_parent(_PLAN[3], snapshot) == "c"
    assert primary_parent(_PLAN[0], snapshot) is None


def test_closing_the_app_mid_pause_releases_the_runner_thread():
    """Live incident: the app closed with a pause dialog open; the runner waited forever."""
    import threading

    async def scenario():
        app = _app(_Executor())
        async with app.run_test() as pilot:
            await _until(pilot, lambda: "c" in app.run_screen._tree_nodes)
            app.run_screen.pause_ids.add("c")
            await pilot.press("r")
            await _until(pilot, lambda: isinstance(app.screen, ConstraintScreen))
            app.exit()  # not via q
        for _ in range(50):
            if threading.active_count() <= baseline:
                break
            await asyncio.sleep(0.05)

    baseline = threading.active_count()
    asyncio.run(scenario())
    assert threading.active_count() <= baseline
