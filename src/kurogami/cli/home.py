"""`kurogami` with no arguments: ask a question, browse past runs, change settings with /commands.

The run itself is the same RunScreen `kurogami tui` uses; this screen only collects
the question and settings and hands them to a factory built in cli/main.py (the one
place adapters are constructed, CLAUDE.md R1).
"""

import json
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Label, OptionList, Static
from textual.widgets.option_list import Option

from kurogami.cli.export import export_run
from kurogami.cli.tui import RunScreen
from kurogami.contracts import TreeSnapshot
from kurogami.engine.runner import RunReport

_HELP = """[b]Ask[/b] a decision question and press Enter, e.g.
  Should I launch my AI resume builder for Indian college students, and how should I price it?

[b]Commands[/b]
  /model <name>        e.g. /model gpt-4.1        /llm openai|fake
  /search tavily|none  web search for research nodes
  /budget <tokens>     e.g. /budget 400k          /review on|off  plan review before running
  /help   /quit

[b]Keys[/b]  Tab → recent runs · Enter opens one · e opens its graph in the browser"""


@dataclass
class Settings:
    llm: str
    model: str
    search: str
    max_tokens: int
    review: bool
    trace_dir: Path

    def line(self) -> Text:
        text = Text()
        for key, value in [
            ("llm", self.llm), ("model", self.model), ("search", self.search),
            ("budget", f"{self.max_tokens // 1000}k"), ("plan review", "on" if self.review else "off"),
        ]:
            text.append(f"{key} ", style="dim")
            text.append(f"{value}   ", style="bold")
        return text


StartFactory = Callable[[str, Settings], tuple[Callable[..., RunReport], Path]]


def _parse_tokens(raw: str) -> int:
    raw = raw.strip().lower().replace(",", "").replace("_", "")
    return int(float(raw[:-1]) * 1000) if raw.endswith("k") else int(raw)


def recent_runs(trace_dir: Path, limit: int = 30) -> list[tuple[Path, Text]]:
    """Saved runs, newest first, as (snapshot path, one-line label)."""
    rows: list[tuple[float, Path, Text]] = []
    for path in trace_dir.rglob("*.snapshot.json"):
        try:
            saved = json.loads(path.read_text())
            statuses = list(saved["snapshot"]["statuses"].values())
        except (OSError, ValueError, KeyError):
            continue  # a half-written or foreign file; the list simply skips it
        mtime = path.stat().st_mtime
        passed = statuses.count("passed")
        live = len(statuses) - statuses.count("invalidated")
        label = Text()
        if saved.get("outcome"):
            label.append("✗ ", style="bold red")
        else:
            label.append("✓ ", style="green")
        label.append(datetime.fromtimestamp(mtime).astimezone().strftime("%d %b %H:%M  "), style="dim")
        goal = saved.get("goal", "")
        label.append(goal[:70] + ("…" if len(goal) > 70 else ""))
        label.append(f"   {passed}/{live}", style="dim")
        rows.append((mtime, path, label))
    rows.sort(key=lambda r: r[0], reverse=True)
    return [(path, label) for _, path, label in rows[:limit]]


class HomeScreen(Screen[None]):
    DEFAULT_CSS = """
    HomeScreen { align-horizontal: center; }
    #home { width: 110; max-width: 100%; height: 1fr; padding: 1 2; }
    #brand { color: $accent; text-style: bold; margin-bottom: 1; }
    #ask { margin-bottom: 0; }
    #settings { color: $text-muted; margin: 0 1 1 1; }
    #runs-label { text-style: bold; margin-top: 1; }
    #runs { height: 1fr; border: round $panel-lighten-2; }
    #hint { color: $text-muted; }
    """
    BINDINGS = [  # noqa: RUF012 -- Textual reads this class attribute
        Binding("e", "export", "Open graph", show=True),
        Binding("slash", "focus_ask", "Command", show=False),
    ]

    def __init__(self, settings: Settings, factory: StartFactory) -> None:
        super().__init__()
        self.settings = settings
        self._factory = factory
        self._runs: list[Path] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="home"):
            yield Static("黒紙 kurogami — ask a decision question; the Master plans it, "
                         "you steer it.", id="brand")
            yield Input(placeholder="Ask a decision question, or /help", id="ask")
            yield Static(self.settings.line(), id="settings")
            yield Label("Recent runs", id="runs-label")
            yield OptionList(id="runs")
            yield Static("Tab → runs · Enter opens · e opens the graph in your browser · "
                         "/help for commands · ctrl+q quits", id="hint")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_runs()
        self.query_one("#ask", Input).focus()

    def on_screen_resume(self) -> None:
        self.refresh_runs()

    def refresh_runs(self) -> None:
        runs = recent_runs(self.settings.trace_dir)
        self._runs = [path for path, _ in runs]
        options = self.query_one("#runs", OptionList)
        options.clear_options()
        if runs:
            options.add_options([Option(label, id=str(path)) for path, label in runs])
        else:
            options.add_option(Option(Text("No runs yet — ask a question above.", style="dim")))

    @on(Input.Submitted, "#ask")
    def _submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if text.startswith("/"):
            self._command(text)
            return
        start, snapshot_path = self._factory(text, self.settings)
        self.app.push_screen(
            RunScreen(text, start_run=start, review_plan=self.settings.review,
                      snapshot_path=snapshot_path)
        )

    def _command(self, text: str) -> None:
        name, _, arg = text[1:].partition(" ")
        arg = arg.strip()
        try:
            if name == "model" and arg:
                self.settings.model = arg
            elif name == "llm" and arg in ("openai", "fake"):
                self.settings.llm = arg
            elif name == "search" and arg in ("tavily", "none", "fake"):
                self.settings.search = arg
            elif name == "budget" and arg:
                self.settings.max_tokens = _parse_tokens(arg)
            elif name == "review" and arg in ("on", "off"):
                self.settings.review = arg == "on"
            elif name == "help":
                self.notify(_HELP, title="kurogami", timeout=30)
                return
            elif name in ("quit", "exit"):
                self.app.exit()
                return
            else:
                self.notify(f"Unknown or incomplete command: {text}  (try /help)", severity="warning")
                return
        except ValueError:
            self.notify(f"Couldn't read {arg!r} as a number.", severity="error")
            return
        self.query_one("#settings", Static).update(self.settings.line())
        self.notify(f"{name} → {arg}")

    @on(OptionList.OptionSelected, "#runs")
    def _open(self, event: OptionList.OptionSelected) -> None:
        if event.option.id is None:
            return
        path = Path(event.option.id)
        saved = json.loads(path.read_text())
        self.app.push_screen(
            RunScreen(saved["goal"], snapshot=TreeSnapshot.model_validate(saved["snapshot"]),
                      snapshot_path=path)
        )

    def action_export(self) -> None:
        if self.focused is self.query_one("#ask", Input):
            return  # typing "e" in the question box
        options = self.query_one("#runs", OptionList)
        if options.highlighted is None or options.highlighted >= len(self._runs):
            return
        page = export_run(self._runs[options.highlighted])
        webbrowser.open(page.resolve().as_uri())
        self.notify(f"Opened {page.name} in your browser.")

    def action_focus_ask(self) -> None:
        ask = self.query_one("#ask", Input)
        if self.focused is not ask:
            ask.focus()
            ask.value = "/"
            ask.cursor_position = 1


class HomeApp(App[None]):
    TITLE = "kurogami"

    def __init__(self, settings: Settings, factory: StartFactory) -> None:
        super().__init__()
        self._home = HomeScreen(settings, factory)

    def on_mount(self) -> None:
        self.push_screen(self._home)
