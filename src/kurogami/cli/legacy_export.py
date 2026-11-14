"""Render execution traces saved before graph snapshots were available."""

import json
from html import escape
from pathlib import Path
from urllib.parse import quote

from kurogami.contracts import TraceRecord, TreeSnapshot

_CALL_NAMES = {
    "GoalSpec": "Interpret goal",
    "_Scope": "Plan dependencies",
    "_Blueprint": "Write task prompts",
    "_VerifyResponse": "Verify task",
    "_GapFill": "Plan missing prerequisite",
    None: "Execute task",
}


def export_run(trace: Path) -> Path:
    records = [
        TraceRecord.model_validate_json(line)
        for line in trace.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    calls_path = trace.with_suffix(".calls.jsonl")
    calls = (
        [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()
         if line.strip()] if calls_path.exists() else []
    )
    snapshot_path = trace.with_suffix(".snapshot.json")
    saved = json.loads(snapshot_path.read_text(encoding="utf-8")) if snapshot_path.exists() else {}
    snapshot = TreeSnapshot.model_validate(saved["snapshot"]) if saved else None
    first_call = calls[0] if calls else {}
    goal = saved.get("goal") or (first_call.get("parsed") or {}).get("raw_text") or trace.stem
    outcome = saved.get("outcome") or next(
        (row.failure_reason for row in reversed(records) if row.node_id.startswith("__")), None
    )
    if snapshot is None and not outcome:
        outcome = "Trace-only view. The final outcome was not saved."

    attempts: dict[str, list[TraceRecord]] = {}
    for record in records:
        if not record.node_id.startswith("__"):
            attempts.setdefault(record.node_id, []).append(record)
    tasks: list[str] = []
    nodes: list[tuple[str, int, list[str], str]] = []
    if snapshot is not None:
        for spec in sorted(snapshot.specs.values(), key=lambda item: (item.depth, item.node_id)):
            rows = attempts.get(spec.node_id, [])
            result = snapshot.results.get(spec.node_id)
            nodes.append((spec.node_id, spec.depth, spec.parent_ids, snapshot.statuses[spec.node_id].value))
            tasks.append(_task(
                spec.node_id, spec.title, spec.depth, spec.parent_ids,
                snapshot.statuses[spec.node_id].value, spec.generated_prompt,
                result.output if result is not None else (rows[-1].output if rows else ""), rows,
            ))
    else:
        for node_id, rows in sorted(attempts.items(), key=lambda item: item[1][-1].depth):
            row = rows[-1]
            parents = [row.parent_node_id] if row.parent_node_id else []
            nodes.append((node_id, row.depth, parents, row.verifier_verdict))
            tasks.append(_task(
                node_id, row.node_goal, row.depth, parents,
                row.verifier_verdict, row.generated_prompt, row.output, rows,
            ))

    stack = []
    for call in calls:
        stage = _CALL_NAMES.get(call.get("schema"), call.get("schema") or "Model call")
        summary = f'{call["seq"]}. {stage} · {call.get("wall_ms", 0):,} ms'
        response = call.get("text") or call.get("error") or ""
        stack.append(
            f"<details><summary>{escape(summary)}</summary>"
            f'<h3>Prompt</h3><pre>{escape(call.get("prompt") or "")}</pre>'
            f"<h3>Response</h3><pre>{escape(response)}</pre></details>"
        )

    target = trace.with_suffix(".viewer.html")
    graph_note = (
        "Expand a task to inspect its output and attempt history."
        if snapshot is not None
        else "Legacy trace. Only the first prerequisite per task was recorded."
    )
    task_list = "".join(tasks) or "<p>No task attempts were recorded.</p>"
    call_list = "".join(stack) or "<p>No model calls were recorded.</p>"
    target.write_text(
        f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Kurogami run</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font: 16px/1.55 system-ui, sans-serif; max-width: 68rem; margin: auto; padding: 2rem 1rem; }}
header {{ border-bottom: 2px solid #2ba6a0; padding-bottom: 1rem; }}
h1 {{ line-height: 1.2; }}
h2 {{ margin-top: 2rem; }}
h3 {{ font-size: 1rem; }}
details {{ padding: .65rem .85rem; margin: .5rem 0; border: 1px solid #7c879580; border-radius: .5rem; }}
summary {{ cursor: pointer; font-weight: 600; }}
.meta {{ opacity: .8; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
nav a {{ margin-right: 1rem; }}
.graph {{ overflow: auto; padding: .5rem; }}
.graph rect {{ fill: Canvas; stroke: #758395; }}
.graph text {{ fill: CanvasText; font: 13px system-ui, sans-serif; }}
.graph .passed rect, .graph .PASS rect {{ stroke: #289d68; }}
.graph .failed rect, .graph .FAIL rect {{ stroke: #cc4d4d; }}
.graph .invalidated rect {{ stroke: #ad7a28; stroke-dasharray: 4; }}
</style>
<header><strong>Kurogami</strong><h1>{escape(goal)}</h1>
<p>{escape(outcome or "Completed run")}</p>
<nav><a href="#tasks">Task dependencies</a><a href="#calls">Model call stack</a></nav></header>
<main><section id="tasks"><h2>Task dependencies</h2><p>{graph_note}</p>{_graph(nodes)}{task_list}</section>
<section id="calls"><h2>Model call stack ({len(stack)})</h2>{call_list}</section></main>
<script>
function showTask() {{
  const task = document.getElementById(decodeURIComponent(location.hash.slice(1)));
  if (task instanceof HTMLDetailsElement) task.open = true;
}}
window.addEventListener("hashchange", showTask);
showTask();
</script>
</html>
""",
        encoding="utf-8",
    )
    return target


def _graph(nodes: list[tuple[str, int, list[str], str]]) -> str:
    if not nodes:
        return ""
    columns = {depth: index for index, depth in enumerate(sorted({node[1] for node in nodes}))}
    rows: dict[int, int] = {}
    positions = {}
    for node_id, depth, _, _ in nodes:
        row = rows.get(depth, 0)
        positions[node_id] = (8 + 220 * columns[depth], 8 + 58 * row)
        rows[depth] = row + 1

    edges = []
    labels = []
    for node_id, _, parents, status in nodes:
        x, y = positions[node_id]
        for parent in parents:
            if parent in positions:
                px, py = positions[parent]
                edges.append(
                    f'<path d="M {px + 176} {py + 19} C {px + 198} {py + 19}, '
                    f'{x - 22} {y + 19}, {x} {y + 19}"/>'
                )
        labels.append(
            f'<a href="#task-{quote(node_id, safe="")}" class="{escape(status)}">'
            f'<title>{escape(node_id)} · {escape(status)}</title>'
            f'<rect x="{x}" y="{y}" width="176" height="38" rx="6"/>'
            f'<text x="{x + 8}" y="{y + 24}">{escape(node_id[:22])}</text></a>'
        )
    return (
        f'<div class="graph"><svg width="{220 * len(columns)}" height="{58 * max(rows.values())}" '
        'aria-label="Task dependency graph">'
        f'<g fill="none" stroke="#758395">{"".join(edges)}</g>{"".join(labels)}</svg></div>'
    )


def _task(
    node_id: str, title: str, depth: int, parents: list[str], status: str,
    prompt: str, output: str, attempts: list[TraceRecord],
) -> str:
    history = []
    for attempt in attempts:
        result = attempt.verifier_verdict
        if attempt.failure_reason:
            result += f" · {attempt.failure_reason}"
        if attempt.backtrack_target:
            result += f" · Backtrack or retry to {attempt.backtrack_target}"
        history.append(f"<li>{escape(result)}</li>")
    label = f"{node_id} · {title}" if title != node_id else node_id
    dependencies = ", ".join(
        f'<a href="#task-{quote(parent, safe="")}">{escape(parent)}</a>' for parent in parents
    )
    return (
        f'<details id="task-{escape(node_id, quote=True)}">'
        f"<summary>{escape(label)} · {escape(status)}</summary>"
        f'<p class="meta">Depth {depth} · Depends on {dependencies or "—"}</p>'
        f'<h3>Output</h3><pre>{escape(output or "No output recorded.")}</pre>'
        f'<h3>Attempts</h3><ol>{"".join(history) or "<li>No attempts recorded.</li>"}</ol>'
        f"<h3>Task prompt</h3><pre>{escape(prompt)}</pre></details>"
    )
