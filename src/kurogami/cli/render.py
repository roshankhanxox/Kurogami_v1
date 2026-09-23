"""rich tree rendering: colour by NodeStatus, invalidated nodes struck through.

The evaluator's whole understanding of whether backtracking happened comes
from what they see here (IMPLEMENTATION_PLAN.md Track C).
"""

import sys

from rich.console import Console
from rich.tree import Tree

from kurogami.contracts import NodeStatus, TraceRecord, TreeSnapshot


def _label(node_id: str, snapshot: TreeSnapshot) -> str:
    spec = snapshot.specs[node_id]
    status = snapshot.statuses[node_id]
    text = f"{node_id} {spec.title} ({status.value})"
    if status == NodeStatus.INVALIDATED:
        return f"[strike dim]{text}[/strike dim]"
    if status == NodeStatus.FAILED:
        return f"[bold red]{text}[/bold red]"
    if status == NodeStatus.PASSED:
        return f"[green]{text}[/green]"
    if status == NodeStatus.SKIPPED:
        return f"[yellow]{text}[/yellow]"
    return text


def render_tree(snapshot: TreeSnapshot, console: Console, *, title: str = "goal") -> None:
    """Render a live TreeSnapshot, e.g. straight out of a Runner.run() report."""
    tree = Tree(title)
    branches: dict[str, Tree] = {}

    def add(node_id: str, parent_branch: Tree) -> None:
        branch = parent_branch.add(_label(node_id, snapshot))
        branches[node_id] = branch
        children = sorted(
            nid for nid, spec in snapshot.specs.items() if node_id in spec.parent_ids
        )
        for child_id in children:
            add(child_id, branch)

    for root_id in snapshot.root_ids:
        add(root_id, tree)
    console.print(tree)


def render_trace(records: list[TraceRecord], console: Console, *, title: str = "replay") -> None:
    """Render a tree from a sequence of TraceRecords -- offline, no network (Gate G7).

    TraceRecord carries a single parent_node_id, not the full parent_ids DAG
    edge set a NodeSpec has, so this is a best-effort reconstruction for
    display -- accurate for the tree shape, not a full NodeSpec round-trip.
    """
    latest_by_node: dict[str, TraceRecord] = {}
    for record in records:
        latest_by_node[record.node_id] = record  # last write wins, e.g. after a retry

    tree = Tree(title)
    branches: dict[str, Tree] = {}
    for record in sorted(latest_by_node.values(), key=lambda r: r.depth):
        label = f"{record.node_id} ({record.verifier_verdict})"
        if record.backtrack_target:
            label += f" -> backtrack to {record.backtrack_target}"
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
