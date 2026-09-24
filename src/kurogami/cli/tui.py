"""Textual app: watch a run live, review the plan before it runs, steer it, browse it after.

Lives in cli/ (it does terminal I/O). The Runner runs in a thread worker; every
engine event arrives through a RunObserver and is handed to the UI thread with
call_from_thread. A paused node or the plan review blocks the runner's thread on
a threading.Event until the human answers -- the engine itself never knows a UI
exists.
"""

import logging
import threading
import time
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Label,
    Markdown,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
    Tree,
)
from textual.widgets.tree import TreeNode

from kurogami.cli.export import export_run
from kurogami.contracts import (
    BacktrackEvent,
    NodeKind,
    NodeResult,
    NodeSpec,
    NodeStatus,
    TreeSnapshot,
    Verdict,
)
from kurogami.engine.runner import RunReport

_log = logging.getLogger(__name__)

_ICONS = {
    NodeStatus.PENDING: ("○", "dim"),
    NodeStatus.RUNNING: ("◐", "bold yellow"),
    NodeStatus.PASSED: ("✓", "green"),
    NodeStatus.FAILED: ("✗", "bold red"),
    NodeStatus.INVALIDATED: ("⤫", "dim strike"),
    NodeStatus.SKIPPED: ("⊘", "yellow"),
}


class RunCancelled(Exception):
    """The human quit mid-run; unwinds the runner's thread."""


@dataclass
class Attempt:
    result: NodeResult
    verdict: Verdict
    backtrack_target: str | None = None


@dataclass
class RunState:
    """Everything the UI knows, owned by the UI thread."""

    goal: str
    snapshot: TreeSnapshot | None = None
    attempts: dict[str, list[Attempt]] = field(default_factory=dict)
    backtracks: list[BacktrackEvent] = field(default_factory=list)
    phase: str = "planning"
    started: float = field(default_factory=time.monotonic)
    finished: float | None = None
    outcome: str = ""


def primary_parent(spec: NodeSpec, snapshot: TreeSnapshot) -> str | None:
    """Draw a node once, under its deepest parent (the one on its longest chain)."""
    parents = [p for p in spec.parent_ids if p in snapshot.specs]
    if not parents:
        return None
    return max(parents, key=lambda p: (snapshot.specs[p].depth, -parents.index(p)))


def node_label(node_id: str, snapshot: TreeSnapshot, *, paused: bool, preset: bool) -> Text:
    spec = snapshot.specs[node_id]
    icon, style = _ICONS[snapshot.statuses[node_id]]
    label = Text()
    label.append(f"{icon} ", style=style.replace(" strike", ""))
    label.append(spec.title, style=style)
    label.append(f"  {node_id}", style="dim")
    if spec.kind == NodeKind.DECISION:
        label.append("  ★", style="bold magenta")
    if paused:
        label.append("  ⏸", style="bold cyan")
    if preset or spec.injected_constraints:
        label.append("  ✎", style="cyan")
    return label


class ConstraintScreen(ModalScreen[list[str] | None]):
    """Ask the human for constraints, one per line."""

    DEFAULT_CSS = """
    ConstraintScreen { align: center middle; }
    #dialog { width: 80; height: auto; max-height: 90%; border: thick $accent;
              background: $surface; padding: 1 2; }
    #dialog TextArea { height: 8; margin: 1 0; }
    #buttons { height: auto; align-horizontal: right; }
    #buttons Button { margin-left: 2; }
    """

    def __init__(self, title: str, detail: str, initial: list[str]) -> None:
        super().__init__()
        self._title, self._detail, self._initial = title, detail, initial

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(f"[b]{self._title}[/b]")
            yield Static(self._detail)
            yield Label("One constraint per line. It applies to this node and everything below it.")
            yield TextArea("\n".join(self._initial), id="constraints")
            with Horizontal(id="buttons"):
                yield Button("Skip", id="skip")
                yield Button("Apply", id="apply", variant="primary")

    def on_mount(self) -> None:
        self.query_one(TextArea).focus()

    @on(Button.Pressed, "#apply")
    def _apply(self) -> None:
        text = self.query_one(TextArea).text
        self.dismiss([line.strip() for line in text.splitlines() if line.strip()])

    @on(Button.Pressed, "#skip")
    def _skip(self) -> None:
        self.dismiss(None)


class _Bridge:
    """Runner-thread side: turns engine callbacks into UI updates, and blocks for answers."""

    def __init__(self, screen: "RunScreen") -> None:
        self._app = screen

    def _check(self) -> None:
        if self._app.cancelled.is_set():
            raise RunCancelled()

    def _ui(self, callback: Callable[..., object], *args: object) -> None:
        """Run on the UI thread; once the human has quit, stop the run instead."""
        self._check()
        try:
            self._app.app.call_from_thread(callback, *args)
        except RuntimeError as exc:  # the app is no longer running
            raise RunCancelled() from exc

    # RunObserver
    def on_planned(self, snapshot: TreeSnapshot) -> None:
        self._ui(self._app.show_plan, snapshot)
        if self._app.review_plan:
            self._app.go.wait()
            self._check()
        self._ui(self._app.set_phase, "running")

    def on_node_started(self, node: NodeSpec, snapshot: TreeSnapshot) -> None:
        self._ui(self._app.apply_snapshot, snapshot)

    def on_node_finished(
        self, node: NodeSpec, result: NodeResult, verdict: Verdict, snapshot: TreeSnapshot
    ) -> None:
        self._ui(self._app.record_attempt, node.node_id, result, verdict, snapshot)

    def on_backtrack(self, event: BacktrackEvent, snapshot: TreeSnapshot) -> None:
        self._ui(self._app.record_backtrack, event, snapshot)

    # InterruptPort
    def should_pause(self, node: NodeSpec) -> bool:
        return node.node_id in self._app.pause_ids or node.node_id in self._app.presets

    def collect(self, node: NodeSpec) -> list[str]:
        preset = self._app.presets.pop(node.node_id, [])
        if node.node_id not in self._app.pause_ids:
            return preset
        answered = threading.Event()
        answer: list[list[str] | None] = [None]

        def done(value: list[str] | None) -> None:
            answer[0] = value
            answered.set()

        def ask() -> None:
            self._app.set_phase(f"paused at {node.node_id}")
            self._app.app.push_screen(
                ConstraintScreen(
                    f"Paused before {node.title}",
                    f"[dim]{node.node_id}[/dim]\n{node.node_goal}",
                    preset,
                ),
                done,
            )

        self._ui(ask)
        while not answered.wait(0.2):
            self._check()
        self._ui(self._app.set_phase, "running")
        return answer[0] or []


class RunScreen(Screen[RunReport | None]):
    """Live view of one run, or a browser over a finished one."""

    DEFAULT_CSS = """
    #body { height: 1fr; }
    #tree-pane { width: 45%; border-right: solid $panel-lighten-2; }
    #tree { height: 1fr; }
    #detail { width: 55%; }
    #node-head { height: auto; padding: 0 1; background: $boost; }
    #status { height: 1; padding: 0 1; background: $panel; }
    TabPane { padding: 0 1; }
    """
    BINDINGS = [  # noqa: RUF012 -- Textual reads this class attribute
        Binding("r", "go", "Run plan"),
        Binding("p", "toggle_pause", "Pause here"),
        Binding("c", "constrain", "Add constraint"),
        Binding("a", "answer", "Final answer"),
        Binding("e", "export", "Open graph in browser"),
        Binding("escape", "back", "Back"),
    ]

    def __init__(
        self,
        goal: str,
        *,
        start_run: Callable[["_Bridge"], RunReport] | None = None,
        snapshot: TreeSnapshot | None = None,
        review_plan: bool = True,
        snapshot_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.state = RunState(goal=goal, snapshot=snapshot)
        self.snapshot_path = snapshot_path
        self.report: RunReport | None = None
        self._start_run = start_run
        self.review_plan = review_plan
        self.go = threading.Event()
        self.cancelled = threading.Event()
        self.pause_ids: set[str] = set()
        self.presets: dict[str, list[str]] = {}
        self._tree_nodes: dict[str, TreeNode[str]] = {}
        self._shown: str | None = None
        if snapshot is not None:
            self.state.phase = "finished run (offline)"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="tree-pane"):
                tree: Tree[str] = Tree(self.state.goal, id="tree")
                tree.root.expand()
                yield tree
            with Vertical(id="detail"):
                yield Static("Select a node.", id="node-head")
                with TabbedContent(id="tabs"):
                    with TabPane("Output", id="tab-output"), VerticalScroll():
                        yield Markdown("", id="output")
                    with TabPane("Prompt", id="tab-prompt"), VerticalScroll():
                        yield Static("", id="prompt")
                    with TabPane("Checks", id="tab-checks"), VerticalScroll():
                        yield Static("", id="checks")
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = self.state.goal
        if self.state.snapshot is not None:
            self._rebuild_tree()
            self.action_answer()
        elif self._start_run is not None:
            self._run()
        self.set_interval(1.0, self._refresh_status)
        self._refresh_status()

    # --- runner thread ------------------------------------------------------------------

    @work(thread=True, exit_on_error=False)
    def _run(self) -> None:
        assert self._start_run is not None
        try:
            report = self._start_run(_Bridge(self))
        except RunCancelled:
            return
        except Exception as exc:  # noqa: BLE001 -- UI boundary: logged and shown, not swallowed
            _log.exception("run crashed")
            if not self.cancelled.is_set():
                self.app.call_from_thread(self._failed, f"{type(exc).__name__}: {exc}")
            return
        if not self.cancelled.is_set():
            self.app.call_from_thread(self._finished, report)

    # --- UI-thread updates from the runner ----------------------------------------------

    def show_plan(self, snapshot: TreeSnapshot) -> None:
        self.state.snapshot = snapshot
        self._rebuild_tree()
        if self.review_plan:
            self.set_phase("plan review — p: pause at node · c: add constraint · r: run")
            self.notify(f"The Master planned {len(snapshot.specs)} nodes. Review, then press r.")
        tree = self.query_one("#tree", Tree)
        tree.focus()

    def set_phase(self, phase: str) -> None:
        self.state.phase = phase
        self._refresh_status()

    def apply_snapshot(self, snapshot: TreeSnapshot) -> None:
        self.state.snapshot = snapshot
        self._rebuild_tree()

    def record_attempt(
        self, node_id: str, result: NodeResult, verdict: Verdict, snapshot: TreeSnapshot
    ) -> None:
        self.state.attempts.setdefault(node_id, []).append(Attempt(result, verdict))
        if verdict.verdict == "FAIL" and verdict.reason is not None:
            self.notify(f"{node_id} failed: {verdict.reason.summary[:160]}", severity="error")
        self.apply_snapshot(snapshot)

    def record_backtrack(self, event: BacktrackEvent, snapshot: TreeSnapshot) -> None:
        self.state.backtracks.append(event)
        attempts = self.state.attempts.get(event.failing_node_id)
        if attempts:
            attempts[-1].backtrack_target = event.target_node_id
        if event.target_node_id != event.failing_node_id:
            self.notify(
                f"Backtrack: {event.failing_node_id} → {event.target_node_id}; "
                f"{len(event.invalidated_node_ids)} node(s) thrown away",
                severity="warning",
                timeout=8,
            )
        self.apply_snapshot(snapshot)

    def _finished(self, report: RunReport) -> None:
        self.state.snapshot = report.snapshot
        self.state.finished = time.monotonic()
        problem = report.aborted_reason or report.breach_reason or report.incomplete_reason
        self.state.outcome = f"✗ {problem}" if problem else "✓ complete"
        self.state.phase = "finished"
        self._rebuild_tree()
        self.action_answer()
        self.notify("Run finished." if not problem else problem, severity="error" if problem else "information")
        self.report = report

    def _failed(self, message: str) -> None:
        self.state.finished = time.monotonic()
        self.state.outcome = f"✗ crashed: {message}"
        self.state.phase = "finished"
        self._refresh_status()
        self.notify(message, severity="error", timeout=30)

    # --- rendering ----------------------------------------------------------------------

    def _rebuild_tree(self) -> None:
        snapshot = self.state.snapshot
        if snapshot is None:
            return
        tree = self.query_one("#tree", Tree)
        cursor_id = self._highlighted_id()
        tree.clear()
        self._tree_nodes = {}
        children: dict[str | None, list[str]] = {}
        for node_id, spec in snapshot.specs.items():
            children.setdefault(primary_parent(spec, snapshot), []).append(node_id)

        def add(parent: TreeNode[str], node_id: str) -> None:
            kids = sorted(children.get(node_id, []), key=lambda i: snapshot.specs[i].depth)
            label = node_label(
                node_id, snapshot,
                paused=node_id in self.pause_ids, preset=node_id in self.presets,
            )
            if kids:
                branch = parent.add(label, data=node_id, expand=True)
            else:
                branch = parent.add_leaf(label, data=node_id)
            self._tree_nodes[node_id] = branch
            for kid in kids:
                add(branch, kid)

        for root in children.get(None, []):
            add(tree.root, root)
        if cursor_id in self._tree_nodes:
            # Line numbers exist only after the tree lays itself out again.
            tree.call_after_refresh(tree.move_cursor, self._tree_nodes[cursor_id])
        self._refresh_status()
        if self._shown is not None:
            self._show(self._shown)

    def _highlighted_id(self) -> str | None:
        node = self.query_one("#tree", Tree).cursor_node
        return node.data if node is not None else None

    @on(Tree.NodeHighlighted)
    def _highlighted(self, event: Tree.NodeHighlighted[str]) -> None:
        if event.node.data is not None:
            self._show(event.node.data)

    def _show(self, node_id: str) -> None:
        snapshot = self.state.snapshot
        if snapshot is None or node_id not in snapshot.specs:
            return
        self._shown = node_id
        spec = snapshot.specs[node_id]
        status = snapshot.statuses[node_id]
        attempts = self.state.attempts.get(node_id, [])
        result = snapshot.results.get(node_id) or (attempts[-1].result if attempts else None)

        head = Text()
        head.append(f"{spec.title}", style="bold")
        head.append(f"   {node_id} · {spec.kind.value} · depth {spec.depth} · {status.value}\n")
        head.append(f"{spec.node_goal}\n", style="italic")
        if spec.parent_ids:
            head.append("builds on: " + ", ".join(spec.parent_ids), style="dim")
        if result is not None:
            head.append(
                f"   ·   {len(result.output.split())} words · "
                f"{result.tokens_in + result.tokens_out:,} tokens", style="dim"
            )
        self.query_one("#node-head", Static).update(head)

        if result is not None:
            output = result.output
        elif status == NodeStatus.RUNNING:
            output = "_Running…_"
        else:
            output = "_Not run yet._"
        self.query_one("#output", Markdown).update(output)

        prompt = Text(spec.generated_prompt)
        if spec.injected_constraints or node_id in self.presets:
            prompt.append("\n\nHuman constraints:\n", style="bold cyan")
            for c in [*spec.injected_constraints, *self.presets.get(node_id, [])]:
                prompt.append(f"  • {c}\n", style="cyan")
        self.query_one("#prompt", Static).update(prompt)

        checks = Text()
        checks.append("Assertions (deterministic)\n", style="bold")
        for a in spec.pass_condition.assertions or ["(none)"]:
            checks.append(f"  • {a}\n", style="cyan" if "ancestors" in a else "")
        checks.append("\nVerifier question\n", style="bold")
        checks.append(f"  {spec.pass_condition.semantic_check}\n")
        for i, attempt in enumerate(attempts, 1):
            v = attempt.verdict
            style = "green" if v.verdict == "PASS" else "bold red"
            checks.append(f"\nAttempt {i}: {v.verdict}", style=style)
            checks.append(f"  (checked by {v.checked_by})\n", style="dim")
            if v.reason is not None:
                checks.append(f"  {v.reason.summary}\n")
                checks.append(f"  evidence: {v.reason.evidence}\n", style="dim")
                if v.reason.suspect_node_ids:
                    checks.append(f"  suspects: {', '.join(v.reason.suspect_node_ids)}\n")
            if attempt.backtrack_target:
                where = (
                    "retry itself" if attempt.backtrack_target == node_id
                    else f"backtrack → {attempt.backtrack_target}"
                )
                checks.append(f"  ↩ {where}\n", style="yellow")
        self.query_one("#checks", Static).update(checks)

    def _refresh_status(self) -> None:
        snapshot = self.state.snapshot
        status = Text()
        status.append(f" {self.state.phase} ", style="reverse")
        if snapshot is not None:
            counts = {s: 0 for s in NodeStatus}
            for s in snapshot.statuses.values():
                counts[s] += 1
            live = len(snapshot.specs) - counts[NodeStatus.INVALIDATED]
            status.append(f"  ✓ {counts[NodeStatus.PASSED]}/{live}", style="green")
            fails = sum(
                1 for a in self.state.attempts.values() for x in a if x.verdict.verdict == "FAIL"
            )
            status.append(f"  ✗ {fails} fails", style="red" if fails else "dim")
            real = [b for b in self.state.backtracks if b.target_node_id != b.failing_node_id]
            status.append(f"  ↩ {len(real)} backtracks", style="yellow" if real else "dim")
            tokens = sum(r.tokens_in + r.tokens_out for a in self.state.attempts.values()
                         for r in (x.result for x in a))
            if tokens:
                status.append(f"  {tokens:,} node tokens", style="dim")
        if self._start_run is not None:
            end = self.state.finished or time.monotonic()
            status.append(f"  {int(end - self.state.started)}s", style="dim")
        if self.state.outcome:
            status.append(f"  {self.state.outcome}", style="bold")
        self.query_one("#status", Static).update(status)

    # --- actions ------------------------------------------------------------------------

    def action_go(self) -> None:
        if not self.go.is_set() and self.state.snapshot is not None and self.review_plan:
            self.go.set()
            self.notify("Running the plan.")

    def action_toggle_pause(self) -> None:
        node_id = self._highlighted_id()
        if node_id is None or self._not_pending(node_id):
            return
        self.pause_ids.symmetric_difference_update({node_id})
        self._rebuild_tree()

    def action_constrain(self) -> None:
        node_id = self._highlighted_id()
        if node_id is None or self._not_pending(node_id):
            return
        spec = self.state.snapshot.specs[node_id]  # type: ignore[union-attr]

        def done(value: list[str] | None) -> None:
            if value:
                self.presets[node_id] = value
            else:
                self.presets.pop(node_id, None)
            self._rebuild_tree()

        self.app.push_screen(
            ConstraintScreen(f"Constraints for {spec.title}", spec.node_goal,
                             self.presets.get(node_id, [])),
            done,
        )

    def _not_pending(self, node_id: str) -> bool:
        snapshot = self.state.snapshot
        if snapshot is None or snapshot.statuses[node_id] != NodeStatus.PENDING:
            self.notify("Only a node that hasn't run yet can be paused or constrained.")
            return True
        return False

    def action_answer(self) -> None:
        snapshot = self.state.snapshot
        if snapshot is None:
            return
        decisions = [
            i for i, s in snapshot.specs.items()
            if s.kind == NodeKind.DECISION and snapshot.statuses[i] != NodeStatus.INVALIDATED
        ]
        if decisions and decisions[-1] in self._tree_nodes:
            tree = self.query_one("#tree", Tree)
            tree.call_after_refresh(tree.move_cursor, self._tree_nodes[decisions[-1]])
            self._show(decisions[-1])
            self.query_one("#tabs", TabbedContent).active = "tab-output"

    def on_unmount(self) -> None:
        """However the screen closes -- Esc, q, Ctrl+C, an error -- release the runner's thread.

        Seen in a live session: the app closed while a pause dialog was open, the
        runner thread waited on it forever, and the process never exited.
        """
        self.cancelled.set()
        self.go.set()

    def action_back(self) -> None:
        if self._start_run is not None and self.state.phase != "finished":
            self.notify("Stopping the run after the current LLM call.", severity="warning")
        self.cancelled.set()
        self.go.set()
        self.dismiss(self.report)

    def action_export(self) -> None:
        if self.snapshot_path is None or not self.snapshot_path.exists():
            self.notify("The graph can be exported once the run has finished.")
            return
        page = export_run(self.snapshot_path)
        webbrowser.open(page.resolve().as_uri())
        self.notify(f"Opened {page.name} in your browser.")


class KurogamiApp(App[RunReport | None]):
    """One run, full screen: `kurogami tui` and the tests."""

    TITLE = "kurogami"
    BINDINGS = [Binding("q", "quit", "Quit")]  # noqa: RUF012 -- Textual reads this class attribute

    def __init__(self, goal: str, **run_screen_args: Any) -> None:
        super().__init__()
        self.run_screen = RunScreen(goal, **run_screen_args)

    def on_mount(self) -> None:
        self.push_screen(self.run_screen, self.exit)

    async def action_quit(self) -> None:
        self.run_screen.cancelled.set()
        self.run_screen.go.set()
        self.exit(self.run_screen.report)
