"""Export a finished run as one self-contained HTML page: the real dependency graph.

A terminal tree can draw each node under one parent only; the plan is a DAG, and
the edges it hides are the ones that matter (what a node sees, what a backtrack
redoes). The page embeds the run as JSON and draws every edge.
"""

import json
import os
import shutil
import subprocess
import webbrowser
from base64 import b64encode
from importlib.resources import files
from pathlib import Path, PureWindowsPath
from typing import Any

from markdown_it import MarkdownIt

from kurogami.contracts import TraceRecord


def trace_path_for(snapshot_path: Path) -> Path:
    return snapshot_path.with_name(snapshot_path.name.replace(".snapshot.json", ".jsonl"))


def attempts_by_node(trace_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Every attempt of every node, in order, from the trace (failed ones included)."""
    attempts: dict[str, list[dict[str, Any]]] = {}
    if not trace_path.exists():
        return attempts
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = TraceRecord.model_validate_json(line)
        if record.node_id.startswith("__"):
            continue
        attempts.setdefault(record.node_id, []).append(
            {
                "sequence": record.seq,
                "verdict": record.verifier_verdict,
                "failure_reason": record.failure_reason,
                "backtrack_target": record.backtrack_target,
                "output": record.output,
                "tokens": record.tokens_in + record.tokens_out,
            }
        )
    return attempts


def render_html(
    saved: dict[str, Any], attempts: dict[str, list[dict[str, Any]]],
    calls: list[dict[str, Any]] | None = None,
) -> str:
    markdown = MarkdownIt("commonmark", {"html": False, "breaks": True}).enable("table")
    markdown.disable("image")
    results = saved["snapshot"].get("results", {})
    calls = calls or []
    payload = json.dumps(
        {
            "goal": saved["goal"],
            "outcome": saved.get("outcome"),
            "snapshot": saved["snapshot"],
            "attempts": attempts,
            "calls": calls,
            "formatted": {
                "results": {
                    node: markdown.render(result.get("output") or "")
                    for node, result in results.items()
                },
                "attempts": {
                    node: [markdown.render(attempt.get("output") or "") for attempt in rows]
                    for node, rows in attempts.items()
                },
                "responses": [
                    markdown.render(call["text"])
                    if call.get("text") and not call.get("schema") else None
                    for call in calls
                ],
            },
        }
    )
    # Inside <script>, "</" could end the element early; LLM output is hostile text.
    payload = payload.replace("</", "<\\/")
    template = files("kurogami.cli").joinpath("templates/run.html").read_text()
    return template.replace("__RUN_DATA__", payload)


def export_run(snapshot_path: Path, out: Path | None = None) -> Path:
    saved = json.loads(snapshot_path.read_text(encoding="utf-8"))
    trace = trace_path_for(snapshot_path)
    log = trace.with_suffix(".calls.jsonl")
    calls = (
        [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
        if log.exists() else []
    )
    html = render_html(saved, attempts_by_node(trace), calls)
    target = out or snapshot_path.with_name(
        snapshot_path.name.replace(".snapshot.json", ".html")
    )
    target.write_text(html, encoding="utf-8")
    return target


def open_browser(page: Path) -> bool:
    """Launch a Windows browser directly when running in WSL."""
    path = page.resolve()
    try:
        if os.environ.get("WSL_DISTRO_NAME"):
            windows_path = subprocess.check_output(
                ["wslpath", "-w", str(path)], text=True, timeout=5
            ).strip()
            uri = PureWindowsPath(windows_path).as_uri()
            browsers = (
                "Microsoft/Edge/Application/msedge.exe",
                "Google/Chrome/Application/chrome.exe",
            )
            for program in browsers:
                for folder in ("/mnt/c/Program Files (x86)", "/mnt/c/Program Files"):
                    browser = Path(folder) / program
                    if browser.is_file():
                        process = subprocess.Popen(
                            [str(browser), uri], stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
                        )
                        try:
                            if process.wait(timeout=0.3) == 0:
                                return True
                        except subprocess.TimeoutExpired:
                            return True
            powershell = shutil.which("powershell.exe")
            if powershell:
                literal = windows_path.replace("'", "''")
                command = f"$ErrorActionPreference = 'Stop'; Start-Process -FilePath '{literal}'"
                encoded = b64encode(command.encode("utf-16-le")).decode("ascii")
                return subprocess.run(
                    [powershell, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                    capture_output=True, timeout=10, check=False,
                ).returncode == 0
        return webbrowser.open(path.as_uri())
    except (OSError, subprocess.SubprocessError, webbrowser.Error):
        return False
