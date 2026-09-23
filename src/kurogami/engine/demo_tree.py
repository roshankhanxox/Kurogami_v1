"""Gate A: the 12-node, depth-5 fixture tree. Forces a FAIL at pricing and backtracks.

Run with `python -m kurogami.engine.demo_tree`. Every test in tests/engine
runs against the same fixture built here, so the demo and the test suite
can never silently drift apart.
"""

from rich.console import Console
from rich.tree import Tree

from kurogami.contracts import (
    FailureReason,
    NodeKind,
    NodeResult,
    NodeSpec,
    NodeStatus,
    PassCondition,
)
from kurogami.engine import backtrack
from kurogami.engine.store import TreeStore

_TITLES: dict[str, str] = {
    "n_001": "Market sizing",
    "n_002": "Differentiation",
    "n_003": "Positioning",
    "n_004": "Pricing",
    "n_005": "Competitor scan",
    "n_006": "Feature scope",
    "n_007": "Tech stack",
    "n_008": "Launch channel",
    "n_009": "Messaging",
    "n_010": "Onboarding flow",
    "n_011": "Support plan",
    "n_012": "Launch plan",
}


def _spec(
    node_id: str, parent_ids: list[str], depth: int, kind: NodeKind = NodeKind.ANALYSIS
) -> NodeSpec:
    title = _TITLES[node_id]
    return NodeSpec(
        node_id=node_id,
        parent_ids=parent_ids,
        depth=depth,
        kind=kind,
        title=title,
        node_goal=f"Goal for {title}.",
        generated_prompt=f"Generated prompt for {title}.",
        pass_condition=PassCondition(assertions=[], semantic_check="Is this consistent?"),
    )


FIXTURE_SPECS: dict[str, NodeSpec] = {
    "n_001": _spec("n_001", [], 0),
    "n_002": _spec("n_002", ["n_001"], 1),
    "n_005": _spec("n_005", ["n_001"], 1, NodeKind.RESEARCH),
    "n_003": _spec("n_003", ["n_002"], 2),
    "n_009": _spec("n_009", ["n_002"], 2, NodeKind.SYNTHESIS),
    "n_006": _spec("n_006", ["n_005"], 2),
    "n_004": _spec("n_004", ["n_003"], 3, NodeKind.DECISION),
    "n_010": _spec("n_010", ["n_009"], 3),
    "n_007": _spec("n_007", ["n_006"], 3),
    "n_008": _spec("n_008", ["n_007"], 4, NodeKind.DECISION),
    "n_011": _spec("n_011", ["n_010"], 4),
    "n_012": _spec("n_012", ["n_008", "n_011"], 5, NodeKind.SYNTHESIS),
}

# Parents before children, safe for sequential store.attach() calls.
FIXTURE_ORDER: list[str] = [
    "n_001",
    "n_002",
    "n_005",
    "n_003",
    "n_009",
    "n_006",
    "n_004",
    "n_010",
    "n_007",
    "n_008",
    "n_011",
    "n_012",
]


def build_fixture_tree() -> TreeStore:
    """The 12-node, depth-5 fixture tree every engine test runs against, offline."""
    store = TreeStore()
    store.seed([FIXTURE_SPECS["n_001"]])
    for node_id in FIXTURE_ORDER[1:]:
        spec = FIXTURE_SPECS[node_id]
        parent = FIXTURE_SPECS[spec.parent_ids[0]]
        store.attach(parent, [spec])
    return store


def dummy_result(node_id: str) -> NodeResult:
    return NodeResult(
        node_id=node_id,
        output=f"Output for {_TITLES[node_id]}.",
        structured={},
        tokens_in=50,
        tokens_out=50,
        latency_ms=10,
        model_id="fake-llm",
        prompt_version="demo_tree@0",
    )


def _status_label(node_id: str, store: TreeStore) -> str:
    status = store.status(node_id)
    title = FIXTURE_SPECS[node_id].title
    label = f"{node_id} {title} ({status.value})"
    if status == NodeStatus.INVALIDATED:
        return f"[strike dim]{label}[/strike dim]"
    if status == NodeStatus.FAILED:
        return f"[bold red]{label}[/bold red]"
    if status == NodeStatus.PASSED:
        return f"[green]{label}[/green]"
    return label


def _render(store: TreeStore, console: Console) -> None:
    tree = Tree("goal: freelance invoicing tool for designers in India")

    def add(node_id: str, parent_branch: Tree) -> None:
        branch = parent_branch.add(_status_label(node_id, store))
        for child in sorted(store.children(node_id), key=lambda c: c.node_id):
            add(child.node_id, branch)

    for root_id in store.root_ids():
        add(root_id, tree)
    console.print(tree)


def main() -> None:
    console = Console()
    store = build_fixture_tree()
    for node_id in FIXTURE_ORDER:
        store.mark_passed(node_id, dummy_result(node_id))

    console.print("[bold]Fixture tree, all 12 nodes PASSED:[/bold]")
    _render(store, console)

    reason = FailureReason(
        summary="Pricing tiers exceed the willingness-to-pay ceiling established by market sizing.",
        violated="semantic",
        evidence="Tier 2 priced at INR 2000/month against a WTP ceiling of INR 500/month.",
        suspect_node_ids=["n_002", "n_003"],
    )
    store.mark_failed("n_004")
    event = backtrack.apply(store, "n_004", reason)

    console.print()
    console.print(f"[bold red]Pricing (n_004) FAILED:[/bold red] {reason.summary}")
    console.print(
        f"[bold]Localised to[/bold] {event.target_node_id} "
        f"({FIXTURE_SPECS[event.target_node_id].title}) — "
        f"invalidated: {sorted(event.invalidated_node_ids)}"
    )
    console.print()
    console.print("[bold]Tree after backtrack — differentiation's subtree struck through, "
                  "siblings untouched:[/bold]")
    _render(store, console)


if __name__ == "__main__":
    main()
