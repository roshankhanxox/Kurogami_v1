"""Export a finished run as one self-contained HTML page: the real dependency graph.

A terminal tree can draw each node under one parent only; the plan is a DAG, and
the edges it hides are the ones that matter (what a node sees, what a backtrack
redoes). The page embeds the run as JSON and draws every edge.
"""

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from kurogami.contracts import TraceRecord


def trace_path_for(snapshot_path: Path) -> Path:
    return snapshot_path.with_name(snapshot_path.name.replace(".snapshot.json", ".jsonl"))


def attempts_by_node(trace_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Every attempt of every node, in order, from the trace (failed ones included)."""
    attempts: dict[str, list[dict[str, Any]]] = {}
    if not trace_path.exists():
        return attempts
    for line in trace_path.read_text().splitlines():
        if not line.strip():
            continue
        record = TraceRecord.model_validate_json(line)
        if record.node_id.startswith("__"):
            continue
        attempts.setdefault(record.node_id, []).append(
            {
                "verdict": record.verifier_verdict,
                "failure_reason": record.failure_reason,
                "backtrack_target": record.backtrack_target,
                "output": record.output,
                "tokens": record.tokens_in + record.tokens_out,
            }
        )
    return attempts


def render_html(saved: dict[str, Any], attempts: dict[str, list[dict[str, Any]]]) -> str:
    payload = json.dumps(
        {
            "goal": saved["goal"],
            "outcome": saved.get("outcome"),
            "snapshot": saved["snapshot"],
            "attempts": attempts,
        }
    )
    # Inside <script>, "</" could end the element early; LLM output is hostile text.
    payload = payload.replace("</", "<\\/")
    template = files("kurogami.cli").joinpath("templates/run.html").read_text()
    return template.replace("__RUN_DATA__", payload)


def export_run(snapshot_path: Path, out: Path | None = None) -> Path:
    saved = json.loads(snapshot_path.read_text())
    html = render_html(saved, attempts_by_node(trace_path_for(snapshot_path)))
    target = out or snapshot_path.with_name(
        snapshot_path.name.replace(".snapshot.json", ".html")
    )
    target.write_text(html)
    return target
