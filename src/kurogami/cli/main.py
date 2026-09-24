"""Composition root. Concrete adapters are constructed only here (CLAUDE.md R1).

Only the fake and OpenAI LLM adapters exist so far (the Anthropic adapter
isn't built), so --llm only supports "fake" and "openai" today. --llm
anthropic fails with a clear error rather than pretending to work.
"""

import json
import os
import uuid
import webbrowser
from collections.abc import Callable
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
from kurogami.adapters.search.tavily import TavilySearch
from kurogami.adapters.trace.jsonl import JsonlTraceSink
from kurogami.agents.executor import Executor
from kurogami.agents.interpreter import Interpreter
from kurogami.agents.localiser import Localiser
from kurogami.agents.planner import BlueprintError, Planner
from kurogami.agents.rules import RuleChecker
from kurogami.agents.verifier import Verifier
from kurogami.cli.export import export_run
from kurogami.cli.home import HomeApp, Settings
from kurogami.cli.interrupt import CliInterrupt
from kurogami.cli.render import render_trace, render_tree
from kurogami.cli.tui import KurogamiApp
from kurogami.contracts import (
    BudgetLimits,
    InterruptPort,
    LLMPort,
    SearchPort,
    TraceRecord,
    TreeSnapshot,
)
from kurogami.engine.budget import Budget
from kurogami.engine.interrupt import ScriptedInterrupt
from kurogami.engine.runner import Runner, RunObserver, RunReport
from kurogami.engine.store import TreeStore

app = typer.Typer(
    help="Kurogami: agent orchestration for market-entry strategy decisions. "
    "Run with no command to open the interactive home screen.",
    invoke_without_command=True,
)
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
    if name == "fake":
        return FakeSearch()
    if name == "tavily":
        return TavilySearch(os.environ.get("TAVILY_API_KEY", ""))
    raise typer.BadParameter(f"--search {name} is not available; use 'tavily', 'fake' or 'none'.")


_GOAL_FILE_OPTION = typer.Option("--goal-file", exists=True, help="A goal JSON file.")
_GOAL_OPTION = typer.Option("--goal", help="The question itself, in one sentence.")


def _goal_text(goal_file: Path | None, goal: str | None) -> str:
    """Exactly one of --goal-file or --goal."""
    if (goal_file is None) == (goal is None):
        raise typer.BadParameter('give exactly one of --goal "your question" or --goal-file')
    return goal if goal is not None else _load_raw_text(goal_file)  # type: ignore[arg-type]


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
    goal_file: Annotated[Path | None, _GOAL_FILE_OPTION] = None,
    goal: Annotated[str | None, _GOAL_OPTION] = None,
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
    goal_spec = Interpreter(llm_port).run(_goal_text(goal_file, goal))
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


def _snapshot_path(trace_path: Path) -> Path:
    return trace_path.with_suffix(".snapshot.json")


def _save_snapshot(trace_path: Path, goal: str, report: RunReport) -> Path:
    """The full tree (all parents, results, backtracks), so a run can be browsed offline."""
    path = _snapshot_path(trace_path)
    path.write_text(
        json.dumps(
            {
                "goal": goal,
                "outcome": report.aborted_reason or report.breach_reason
                or report.incomplete_reason,
                "snapshot": report.snapshot.model_dump(mode="json"),
            },
            indent=2,
        )
    )
    return path


def _build_runner(
    *,
    llm: str,
    model: str,
    fake_script: Path | None,
    no_cache: bool,
    search: str,
    trace_path: Path,
    max_tokens: int,
    interrupt: InterruptPort,
    observer: RunObserver | None = None,
) -> Runner:
    llm_port = RecordingLLM(
        _build_llm(llm, fake_script, model=model, no_cache=no_cache),
        trace_path.with_suffix(".calls.jsonl"),
    )
    planner = Planner(llm_port)
    return Runner(
        interpreter=Interpreter(llm_port),
        planner=planner,
        executor=Executor(llm_port, search=_build_search(search)),
        rules_checker=RuleChecker(),
        verifier=Verifier(llm_port),
        interrupt=interrupt,
        trace_sink=JsonlTraceSink(trace_path),
        budget=Budget(BudgetLimits(max_tokens_total=max_tokens)),
        localiser=Localiser(llm_port),
        observer=observer,
        check_reviser=planner,
    )


def _new_trace_path(trace_dir: Path) -> Path:
    trace_dir.mkdir(parents=True, exist_ok=True)
    return trace_dir / f"{uuid.uuid4()}.jsonl"


_MAX_TOKENS_OPTION = typer.Option("--max-tokens", help="Run-wide token budget (cost guard).")


def _home_factory(goal: str, settings: Settings) -> tuple[Callable[..., RunReport], Path]:
    """What the home screen calls to start a run: adapters are built here, not in cli/home."""
    trace_path = _new_trace_path(settings.trace_dir)

    def start(bridge: Any) -> RunReport:
        runner = _build_runner(
            llm=settings.llm, model=settings.model, fake_script=None, no_cache=True,
            search=settings.search, trace_path=trace_path, max_tokens=settings.max_tokens,
            interrupt=bridge, observer=bridge,
        )
        report = runner.run(goal)
        _save_snapshot(trace_path, goal, report)
        return report

    return start, _snapshot_path(trace_path)


@app.callback()
def home(ctx: typer.Context) -> None:
    """With no command: the interactive home screen."""
    if ctx.invoked_subcommand is not None:
        return
    settings = Settings(
        llm="openai" if os.environ.get("OPENAI_API_KEY") else "fake",
        model="gpt-4.1-mini",
        search="tavily" if os.environ.get("TAVILY_API_KEY") else "none",
        max_tokens=300_000,
        review=True,
        trace_dir=Path("traces"),
    )
    HomeApp(settings, _home_factory).run()


@app.command(name="run")
def run_cmd(
    goal_file: Annotated[Path | None, _GOAL_FILE_OPTION] = None,
    goal_text: Annotated[str | None, _GOAL_OPTION] = None,
    llm: Annotated[str, typer.Option("--llm")] = "fake",
    model: Annotated[str, typer.Option("--model")] = DEFAULT_MODEL,
    fake_script: Annotated[Path | None, typer.Option("--fake-script")] = None,
    no_cache: Annotated[bool, typer.Option("--no-cache")] = False,
    search: Annotated[str, typer.Option("--search")] = "fake",
    interrupt_at: Annotated[list[str], typer.Option("--interrupt-at")] = [],  # noqa: B006
    trace_dir: Annotated[Path, typer.Option("--trace-dir")] = Path("traces"),
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
    max_tokens: Annotated[int, _MAX_TOKENS_OPTION] = BudgetLimits().max_tokens_total,
) -> None:
    """Run the full loop end to end and write a JSONL trace."""
    goal = _goal_text(goal_file, goal_text)
    trace_path = _new_trace_path(trace_dir)
    runner = _build_runner(
        llm=llm, model=model, fake_script=fake_script, no_cache=no_cache, search=search,
        trace_path=trace_path, max_tokens=max_tokens,
        interrupt=CliInterrupt(interrupt_at) if interrupt_at else ScriptedInterrupt([]),
    )
    report = runner.run(goal)
    snapshot_path = _save_snapshot(trace_path, goal, report)

    if verbose:
        render_tree(report.snapshot, console, title=goal)
    console.print(f"trace written to {trace_path}")
    console.print(f"every LLM call is recorded in {trace_path.with_suffix('.calls.jsonl')}")
    console.print(f"browse it: kurogami tui --open {snapshot_path}")

    if report.aborted_reason:
        console.print(f"[bold red]{report.aborted_reason}[/bold red]")
        raise typer.Exit(code=1)
    if report.budget_breached:
        console.print(f"[bold red]budget breached: {report.breach_reason}[/bold red]")
        raise typer.Exit(code=1)
    if report.incomplete_reason:
        console.print(f"[bold red]incomplete: {report.incomplete_reason}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def tui(
    goal_file: Annotated[Path | None, _GOAL_FILE_OPTION] = None,
    goal_text: Annotated[str | None, _GOAL_OPTION] = None,
    open_snapshot: Annotated[
        Path | None, typer.Option("--open", exists=True, help="Browse a finished run offline.")
    ] = None,
    llm: Annotated[str, typer.Option("--llm")] = "fake",
    model: Annotated[str, typer.Option("--model")] = DEFAULT_MODEL,
    fake_script: Annotated[Path | None, typer.Option("--fake-script")] = None,
    no_cache: Annotated[bool, typer.Option("--no-cache")] = False,
    search: Annotated[str, typer.Option("--search")] = "fake",
    trace_dir: Annotated[Path, typer.Option("--trace-dir")] = Path("traces"),
    max_tokens: Annotated[int, _MAX_TOKENS_OPTION] = BudgetLimits().max_tokens_total,
    no_review: Annotated[
        bool, typer.Option("--no-review", help="Start running as soon as the plan is ready.")
    ] = False,
) -> None:
    """Interactive view: review the plan, pause and steer nodes, watch the tree live."""
    if open_snapshot is not None:
        saved = json.loads(open_snapshot.read_text())
        KurogamiApp(
            saved["goal"], snapshot=TreeSnapshot.model_validate(saved["snapshot"])
        ).run()
        return
    goal = _goal_text(goal_file, goal_text)
    trace_path = _new_trace_path(trace_dir)

    def start(bridge: Any) -> RunReport:
        runner = _build_runner(
            llm=llm, model=model, fake_script=fake_script, no_cache=no_cache, search=search,
            trace_path=trace_path, max_tokens=max_tokens, interrupt=bridge, observer=bridge,
        )
        report = runner.run(goal)
        _save_snapshot(trace_path, goal, report)
        return report

    app_ = KurogamiApp(goal, start_run=start, review_plan=not no_review)
    report = app_.run()
    console.print(f"trace written to {trace_path}")
    if report is None:
        console.print("[yellow]stopped before the run finished[/yellow]")
        raise typer.Exit(code=1)
    console.print(f"browse it again: kurogami tui --open {_snapshot_path(trace_path)}")


@app.command()
def export(
    snapshot: Annotated[Path, typer.Argument(exists=True, help="A run's .snapshot.json.")],
    out: Annotated[Path | None, typer.Option("--out", help="Where to write the page.")] = None,
    open_it: Annotated[bool, typer.Option("--open", help="Open it in the browser.")] = False,
) -> None:
    """Write a run as an interactive HTML graph -- every dependency edge drawn."""
    path = export_run(snapshot, out)
    console.print(f"wrote {path}")
    if open_it:
        webbrowser.open(path.resolve().as_uri())


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
