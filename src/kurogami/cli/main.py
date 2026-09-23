"""Composition root. Concrete adapters are constructed only here (CLAUDE.md R1).

Only the fake LLM/search adapters exist so far (Track D -- real Anthropic/
OpenAI adapters -- hasn't merged yet), so --llm/--search only support
"fake" today. --llm anthropic fails with a clear error rather than
pretending to work.
"""

import json
import uuid
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from kurogami.adapters.llm.fake import FakeLLM
from kurogami.adapters.search.fake import FakeSearch
from kurogami.adapters.trace.jsonl import JsonlTraceSink
from kurogami.agents.executor import Executor
from kurogami.agents.interpreter import Interpreter
from kurogami.agents.planner import Planner
from kurogami.agents.rules import RuleChecker
from kurogami.agents.verifier import Verifier
from kurogami.cli.interrupt import CliInterrupt
from kurogami.cli.render import render_trace, render_tree
from kurogami.contracts import LLMPort, SearchPort, TraceRecord
from kurogami.engine.budget import Budget
from kurogami.engine.interrupt import ScriptedInterrupt
from kurogami.engine.runner import Runner

app = typer.Typer(help="Kurogami: agent orchestration for market-entry strategy decisions.")
console = Console()


def _build_llm(name: str, fake_script: Path | None) -> LLMPort:
    if name != "fake":
        raise typer.BadParameter(
            f"--llm {name} is not available yet; only 'fake' is implemented "
            "(the real adapters are Track D / feat/bench-harness)."
        )
    responses: dict[str, Any] = {}
    if fake_script is not None:
        responses = json.loads(fake_script.read_text())
    return FakeLLM(responses=responses)


def _build_search(name: str) -> SearchPort | None:
    if name == "none":
        return None
    if name != "fake":
        raise typer.BadParameter(f"--search {name} is not available yet; only 'fake' or 'none'.")
    return FakeSearch()


def _load_raw_text(goal_file: Path) -> str:
    payload = json.loads(goal_file.read_text())
    return str(payload["raw_text"])


@app.command()
def interpret(
    text: Annotated[str, typer.Option("--text", help="A single sentence describing the decision.")],
    llm: Annotated[str, typer.Option("--llm")] = "fake",
    fake_script: Annotated[Path | None, typer.Option("--fake-script")] = None,
) -> None:
    """Interpret a sentence into a GoalSpec and print it."""
    goal = Interpreter(_build_llm(llm, fake_script)).run(text)
    console.print_json(goal.model_dump_json())


@app.command()
def plan(
    goal_file: Annotated[Path, typer.Option("--goal-file", exists=True)],
    llm: Annotated[str, typer.Option("--llm")] = "fake",
    fake_script: Annotated[Path | None, typer.Option("--fake-script")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Interpret a goal file and print the planner's root NodeSpecs."""
    llm_port = _build_llm(llm, fake_script)
    goal_spec = Interpreter(llm_port).run(_load_raw_text(goal_file))
    roots = Planner(llm_port).plan(goal_spec)
    for node in roots:
        console.print_json(node.model_dump_json())
    if dry_run:
        console.print(f"[bold]{len(roots)} root node(s), nothing executed (--dry-run).[/bold]")


@app.command(name="run")
def run_cmd(
    goal_file: Annotated[Path, typer.Option("--goal-file", exists=True)],
    llm: Annotated[str, typer.Option("--llm")] = "fake",
    fake_script: Annotated[Path | None, typer.Option("--fake-script")] = None,
    search: Annotated[str, typer.Option("--search")] = "fake",
    interrupt_at: Annotated[list[str], typer.Option("--interrupt-at")] = [],  # noqa: B006
    trace_dir: Annotated[Path, typer.Option("--trace-dir")] = Path("traces"),
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Run the full loop end to end and write a JSONL trace."""
    llm_port = _build_llm(llm, fake_script)
    search_port = _build_search(search)
    interrupt_port = CliInterrupt(interrupt_at) if interrupt_at else ScriptedInterrupt([])

    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"{uuid.uuid4()}.jsonl"
    trace_sink = JsonlTraceSink(trace_path)

    runner = Runner(
        interpreter=Interpreter(llm_port),
        planner=Planner(llm_port),
        executor=Executor(llm_port, search=search_port),
        rules_checker=RuleChecker(),
        verifier=Verifier(llm_port),
        interrupt=interrupt_port,
        trace_sink=trace_sink,
        budget=Budget(),
    )
    report = runner.run(_load_raw_text(goal_file))

    if verbose:
        render_tree(report.snapshot, console, title=_load_raw_text(goal_file))
    console.print(f"trace written to {trace_path}")

    if report.budget_breached:
        console.print(f"[bold red]budget breached: {report.breach_reason}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def replay(
    trace_file: Annotated[Path, typer.Argument(exists=True)],
) -> None:
    """Re-render a tree from a JSONL trace, offline -- no network (Gate G7)."""
    lines = [line for line in trace_file.read_text().splitlines() if line.strip()]
    records = [TraceRecord.model_validate_json(line) for line in lines]
    render_trace(records, console, title=str(trace_file))


if __name__ == "__main__":
    app()
