"""Composition root. Concrete adapters are constructed only here (CLAUDE.md R1).

Only the fake and OpenAI LLM adapters exist so far (the Anthropic adapter
isn't built), so --llm only supports "fake" and "openai" today. --llm
anthropic fails with a clear error rather than pretending to work.
"""

import json
import uuid
from pathlib import Path
from typing import Annotated, Any

import typer
from dotenv import load_dotenv
from rich.console import Console

from kurogami.adapters.llm.cached import CachedLLM
from kurogami.adapters.llm.fake import FakeLLM
from kurogami.adapters.llm.openai import DEFAULT_MODEL, OpenAILLM
from kurogami.adapters.llm.recording import RecordingLLM
from kurogami.adapters.search.fake import FakeSearch
from kurogami.adapters.trace.jsonl import JsonlTraceSink
from kurogami.agents.executor import Executor
from kurogami.agents.interpreter import Interpreter
from kurogami.agents.planner import BlueprintError, Planner
from kurogami.agents.rules import RuleChecker
from kurogami.agents.verifier import Verifier
from kurogami.cli.interrupt import CliInterrupt
from kurogami.cli.render import render_trace, render_tree
from kurogami.contracts import LLMPort, SearchPort, TraceRecord
from kurogami.engine.budget import Budget
from kurogami.engine.interrupt import ScriptedInterrupt
from kurogami.engine.runner import Runner
from kurogami.engine.store import TreeStore

app = typer.Typer(help="Kurogami: agent orchestration for market-entry strategy decisions.")
console = Console()

load_dotenv()


def _build_llm(
    name: str,
    fake_script: Path | None,
    *,
    model: str = DEFAULT_MODEL,
    no_cache: bool = False,
    cache_dir: Path = Path(".cache/llm"),
) -> LLMPort:
    if name == "fake":
        responses: dict[str, Any] = {}
        if fake_script is not None:
            responses = json.loads(fake_script.read_text())
        return FakeLLM(responses=responses)

    if name == "openai":
        llm: LLMPort = OpenAILLM(model=model)
        if no_cache:
            return llm
        return CachedLLM(llm, model_id=model, cache_dir=cache_dir)

    raise typer.BadParameter(
        f"--llm {name} is not available yet; only 'fake' and 'openai' are implemented "
        "(the Anthropic adapter isn't built)."
    )


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
    model: Annotated[str, typer.Option("--model")] = DEFAULT_MODEL,
    fake_script: Annotated[Path | None, typer.Option("--fake-script")] = None,
    no_cache: Annotated[bool, typer.Option("--no-cache")] = False,
) -> None:
    """Interpret a sentence into a GoalSpec and print it."""
    goal = Interpreter(_build_llm(llm, fake_script, model=model, no_cache=no_cache)).run(text)
    console.print_json(goal.model_dump_json())


@app.command()
def plan(
    goal_file: Annotated[Path, typer.Option("--goal-file", exists=True)],
    llm: Annotated[str, typer.Option("--llm")] = "fake",
    model: Annotated[str, typer.Option("--model")] = DEFAULT_MODEL,
    fake_script: Annotated[Path | None, typer.Option("--fake-script")] = None,
    no_cache: Annotated[bool, typer.Option("--no-cache")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    trace_dir: Annotated[Path, typer.Option("--trace-dir")] = Path("traces"),
) -> None:
    """Interpret a goal file and print the Master's complete planned tree (Gate G3)."""
    calls_log = trace_dir / f"plan-{uuid.uuid4()}.calls.jsonl"
    llm_port = RecordingLLM(_build_llm(llm, fake_script, model=model, no_cache=no_cache), calls_log)
    console.print(f"every LLM call is recorded in {calls_log}")
    goal_spec = Interpreter(llm_port).run(_load_raw_text(goal_file))
    try:
        blueprint = Planner(llm_port).plan(goal_spec)
    except BlueprintError as exc:
        console.print(f"[bold red]planning failed: {exc}[/bold red]")
        raise typer.Exit(code=1) from exc
    for node in blueprint:
        console.print_json(node.model_dump_json())

    store = TreeStore()
    store.seed_blueprint(blueprint)
    render_tree(store.snapshot(), console, title=goal_spec.raw_text)
    deepest = max(node.depth for node in blueprint)
    summary = f"{len(blueprint)} planned node(s), max depth {deepest}"
    if dry_run:
        summary += ", nothing executed (--dry-run)"
    console.print(f"[bold]{summary}.[/bold]")


@app.command(name="run")
def run_cmd(
    goal_file: Annotated[Path, typer.Option("--goal-file", exists=True)],
    llm: Annotated[str, typer.Option("--llm")] = "fake",
    model: Annotated[str, typer.Option("--model")] = DEFAULT_MODEL,
    fake_script: Annotated[Path | None, typer.Option("--fake-script")] = None,
    no_cache: Annotated[bool, typer.Option("--no-cache")] = False,
    search: Annotated[str, typer.Option("--search")] = "fake",
    interrupt_at: Annotated[list[str], typer.Option("--interrupt-at")] = [],  # noqa: B006
    trace_dir: Annotated[Path, typer.Option("--trace-dir")] = Path("traces"),
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Run the full loop end to end and write a JSONL trace."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"{uuid.uuid4()}.jsonl"
    trace_sink = JsonlTraceSink(trace_path)
    calls_log = trace_path.with_suffix(".calls.jsonl")

    llm_port = RecordingLLM(_build_llm(llm, fake_script, model=model, no_cache=no_cache), calls_log)
    search_port = _build_search(search)
    interrupt_port = CliInterrupt(interrupt_at) if interrupt_at else ScriptedInterrupt([])

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
    console.print(f"every LLM call is recorded in {calls_log}")

    if report.aborted_reason:
        console.print(f"[bold red]{report.aborted_reason}[/bold red]")
        raise typer.Exit(code=1)
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
