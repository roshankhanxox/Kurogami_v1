"""cli/render.py: colour by NodeStatus, invalidated struck through -- offline."""

from io import StringIO
from uuid import uuid4

from rich.console import Console

from kurogami.cli.render import render_trace, render_tree
from kurogami.contracts import TraceRecord
from kurogami.engine.demo_tree import FIXTURE_ORDER, build_fixture_tree, dummy_result


def _console() -> tuple[Console, StringIO]:
    buffer = StringIO()
    return Console(file=buffer, force_terminal=False, width=200), buffer


def test_render_tree_shows_every_fixture_node():
    store = build_fixture_tree()
    for node_id in FIXTURE_ORDER:
        store.mark_passed(node_id, dummy_result(node_id))

    console, buffer = _console()
    render_tree(store.snapshot(), console)

    output = buffer.getvalue()
    for node_id in FIXTURE_ORDER:
        assert node_id in output


def test_render_tree_shows_invalidated_status_after_backtrack():
    from kurogami.contracts import FailureReason
    from kurogami.engine import backtrack

    store = build_fixture_tree()
    for node_id in FIXTURE_ORDER:
        store.mark_passed(node_id, dummy_result(node_id))
    store.mark_failed("n_004")
    backtrack.apply(
        store,
        "n_004",
        FailureReason(summary="x", violated="semantic", evidence="x", suspect_node_ids=["n_002"]),
    )

    console, buffer = _console()
    render_tree(store.snapshot(), console)

    output = buffer.getvalue()
    assert "invalidated" in output
    assert "pending" in output


def _trace_record(node_id: str, seq: int, **overrides) -> TraceRecord:
    kwargs = {
        "run_id": uuid4(),
        "seq": seq,
        "node_id": node_id,
        "parent_node_id": None,
        "depth": 0,
        "generated_prompt": "p",
        "node_goal": "g",
        "output": "o",
        "verifier_verdict": "PASS",
        "failure_reason": None,
        "backtrack_target": None,
        "tokens_in": 1,
        "tokens_out": 1,
        "latency_ms": 1,
        "human_interrupt": False,
    }
    kwargs.update(overrides)
    return TraceRecord(**kwargs)


def test_render_trace_shows_backtrack_target():
    records = [
        _trace_record("n_a", 1),
        _trace_record(
            "n_b", 2, parent_node_id="n_a", depth=1,
            verifier_verdict="FAIL", backtrack_target="n_a",
        ),
    ]
    console, buffer = _console()
    render_trace(records, console)

    output = buffer.getvalue()
    assert "n_a" in output
    assert "n_b" in output
    assert "backtrack to n_a" in output


def test_render_trace_keeps_only_the_latest_record_per_node():
    records = [
        _trace_record("n_a", 1, verifier_verdict="FAIL"),
        _trace_record("n_a", 2, verifier_verdict="PASS"),
    ]
    console, buffer = _console()
    render_trace(records, console)

    output = buffer.getvalue()
    assert output.count("n_a") == 1
