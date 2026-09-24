"""The `kurogami` home screen, driven headless: ask, steer settings, reopen past runs."""

import asyncio
import json
from pathlib import Path

from textual.widgets import Input, OptionList

from kurogami.cli.home import HomeApp, HomeScreen, Settings, _parse_tokens, recent_runs
from kurogami.cli.tui import RunScreen
from kurogami.engine.runner import Runner
from tests.cli.test_tui import _Executor, _Interpreter, _Pass, _Planner, _Sink


def _settings(tmp_path: Path) -> Settings:
    return Settings(llm="fake", model="m", search="none", max_tokens=300_000, review=False,
                    trace_dir=tmp_path)


def _factory(asked: list[str]):
    def factory(goal: str, settings: Settings):
        asked.append(goal)

        def start(bridge):
            return Runner(interpreter=_Interpreter(), planner=_Planner(), executor=_Executor(),
                          rules_checker=_Pass(), verifier=_Pass(), interrupt=bridge,
                          trace_sink=_Sink(), observer=bridge).run(goal)

        return start, settings.trace_dir / "x.snapshot.json"

    return factory


async def _until(pilot, condition, timeout: float = 5.0) -> None:
    for _ in range(int(timeout / 0.05)):
        if condition():
            return
        await pilot.pause(0.05)
    raise AssertionError("condition never became true")


def test_typing_a_question_runs_it_and_escape_comes_home(tmp_path):
    asked: list[str] = []

    async def scenario():
        app = HomeApp(_settings(tmp_path), _factory(asked))
        async with app.run_test() as pilot:
            await _until(pilot, lambda: isinstance(app.screen, HomeScreen))
            app.screen.query_one("#ask", Input).value = "Should I launch my yoga app?"
            await pilot.press("enter")
            await _until(pilot, lambda: isinstance(app.screen, RunScreen))
            run = app.screen
            await _until(pilot, lambda: run.state.phase == "finished")
            await pilot.press("escape")
            await _until(pilot, lambda: isinstance(app.screen, HomeScreen))

    asyncio.run(scenario())
    assert asked == ["Should I launch my yoga app?"]


def test_slash_commands_change_the_settings(tmp_path):
    settings = _settings(tmp_path)

    async def scenario():
        app = HomeApp(settings, _factory([]))
        async with app.run_test() as pilot:
            await _until(pilot, lambda: isinstance(app.screen, HomeScreen))
            ask = app.screen.query_one("#ask", Input)
            for command in ["/budget 400k", "/model gpt-4.1", "/review on", "/search tavily"]:
                ask.value = command
                await pilot.press("enter")

    asyncio.run(scenario())
    assert (settings.max_tokens, settings.model, settings.review, settings.search) == (
        400_000, "gpt-4.1", True, "tavily"
    )


def _save(path: Path, goal: str, outcome=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"goal": goal, "outcome": outcome, "snapshot": {
        "root_ids": [], "specs": {}, "statuses": {"a": "passed", "b": "failed"},
        "results": {}, "backtrack_events": []}}))


def test_saved_runs_are_listed_newest_first_and_bad_files_skipped(tmp_path):
    _save(tmp_path / "old.snapshot.json", "old question")
    _save(tmp_path / "sub" / "new.snapshot.json", "new question", outcome="gave up")
    (tmp_path / "broken.snapshot.json").write_text("{not json")
    import os
    os.utime(tmp_path / "old.snapshot.json", (1, 1))
    runs = recent_runs(tmp_path)
    assert [p.name for p, _ in runs] == ["new.snapshot.json", "old.snapshot.json"]
    assert runs[0][1].plain.startswith("✗")
    assert "1/2" in runs[0][1].plain


def test_a_listed_run_opens_offline(tmp_path):
    from tests.cli.test_export import _write_run
    snapshot = _write_run(tmp_path)

    async def scenario():
        app = HomeApp(_settings(tmp_path), _factory([]))
        async with app.run_test() as pilot:
            await _until(pilot, lambda: isinstance(app.screen, HomeScreen))
            runs = app.screen.query_one("#runs", OptionList)
            runs.focus()
            runs.highlighted = 0
            await pilot.press("enter")
            await _until(pilot, lambda: isinstance(app.screen, RunScreen))
            assert app.screen.snapshot_path == snapshot

    asyncio.run(scenario())


def test_token_budgets_read_like_people_write_them():
    assert _parse_tokens("400k") == 400_000
    assert _parse_tokens("1.5k") == 1_500
    assert _parse_tokens("250,000") == 250_000
