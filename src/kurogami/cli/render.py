"""Render CLI runs and saved traces with Rich."""

import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from kurogami.contracts import NodeStatus, TraceRecord, TraceSink, TreeSnapshot

_STATUS_STYLE = {
    NodeStatus.INVALIDATED: "yellow",
    NodeStatus.FAILED: "bold red",
    NodeStatus.PASSED: "green",
    NodeStatus.SKIPPED: "yellow",
}


def render_run_intro(console: Console, goal: str, mode: str, model: str, search: str) -> None:
    body = Text(goal, style="bold")
    body.append(f"\n\n{mode}  ·  Model: {model}  ·  Search: {search}", style="dim")
    console.print(Panel(body, title="Kurogami", subtitle="Market-entry analysis", border_style="cyan"))
    console.print("Interpreting your goal and building the plan. The first update follows verification.")


class ConsoleTraceSink:
    """Show attempt outcomes as they happen while preserving the JSONL trace."""

    def __init__(self, sink: TraceSink, console: Console) -> None:
        self._sink = sink
        self._console = console

    def emit(self, record: TraceRecord) -> None:
        self._sink.emit(record)
        if record.node_id in {"__plan__", "__budget__", "__incomplete__"}:
            return
        label = Text(f"{record.verifier_verdict:<4}", style=(
            "green" if record.verifier_verdict == "PASS" else "bold red"
        ))
        label.append(f"  {record.node_id}", style="bold")
        label.append(f"  {record.node_goal}")
        self._console.print(label, markup=False)
        if record.failure_reason:
            self._console.print(Text(f"      {record.failure_reason}", style="yellow"))
        if record.backtrack_target:
            action = (
                "Retrying this task"
                if record.backtrack_target == record.node_id
                else f"{record.node_id} -> backtrack to {record.backtrack_target}"
            )
            self._console.print(Text(f"      {action}", style="bold cyan"))

    def close(self) -> None:
        self._sink.close()


def render_tree(snapshot: TreeSnapshot, console: Console, *, title: str = "goal") -> None:
    """List each task once and show all of its dependencies."""
    console.print(Text(title, style="bold cyan"))
    table = Table(box=None, header_style="bold cyan", expand=True, pad_edge=False)
    table.add_column("Task ID", overflow="fold")
    table.add_column("Task", ratio=1, overflow="fold")
    table.add_column("Depends on", ratio=1, overflow="fold")
    table.add_column("Status", no_wrap=True)
    for spec in sorted(snapshot.specs.values(), key=lambda item: (item.depth, item.node_id)):
        status = snapshot.statuses[spec.node_id]
        table.add_row(
            Text(spec.node_id),
            Text(spec.title if spec.title != spec.node_id else spec.node_goal),
            Text(", ".join(spec.parent_ids) or "—"),
            Text(status.value, style=_STATUS_STYLE.get(status, "dim")),
        )
    console.print(table)


def render_outcome(console: Console, title: str, message: str, *, failure: bool = False) -> None:
    body = Text(message) if failure else Markdown(message, hyperlinks=False)
    console.print(
        Panel(body, title=title, border_style="red" if failure else "green", expand=True)
    )


def render_trace(records: list[TraceRecord], console: Console, *, title: str = "replay") -> None:
    """Render a tree from a sequence of TraceRecords -- offline, no network (Gate G7).

    TraceRecord carries a single parent_node_id, not the full parent_ids DAG
    edge set a NodeSpec has, so this is a best-effort reconstruction for
    display -- accurate for the tree shape, not a full NodeSpec round-trip.
    """
    latest_by_node: dict[str, TraceRecord] = {}
    for record in records:
        latest_by_node[record.node_id] = record  # last write wins, e.g. after a retry

    tree = Tree(Text(title, style="bold cyan"))
    branches: dict[str, Tree] = {}
    for record in sorted(latest_by_node.values(), key=lambda r: r.depth):
        label = Text(f"{record.node_id} ")
        label.append(
            f"({record.verifier_verdict})",
            style="green" if record.verifier_verdict == "PASS" else "red",
        )
        if record.backtrack_target:
            label.append(f" -> backtrack to {record.backtrack_target}", style="cyan")
        if record.human_interrupt:
            label.append(" · User input", style="yellow")
        parent_branch = branches.get(record.parent_node_id, tree) if record.parent_node_id else tree
        branches[record.node_id] = parent_branch.add(label)
    console.print(tree)


def main() -> None:
    """python -m kurogami.cli.render --demo -- renders the engine fixture tree."""
    from kurogami.engine.demo_tree import FIXTURE_ORDER, build_fixture_tree, dummy_result

    if "--demo" not in sys.argv:
        print("Usage: python -m kurogami.cli.render --demo")
        raise SystemExit(1)

    store = build_fixture_tree()
    for node_id in FIXTURE_ORDER:
        store.mark_passed(node_id, dummy_result(node_id))

    render_tree(store.snapshot(), Console(), title="fixture tree")


if __name__ == "__main__":
    main()
