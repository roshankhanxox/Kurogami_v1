"""CLI integration, offline. No network, no real LLM.

`interpret` is driven through the real --fake-script path (FakeLLM keyed by exact
prompt text). `plan` and `run` make a dozen-plus calls whose prompts embed earlier
outputs, so exact-text keying would be brittle; for those the composition root's
LLM is replaced by a fake that answers by *which schema is being asked for*. The
Interpreter, Planner, Executor, RuleChecker, Verifier and Runner are all real.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from typer.testing import CliRunner

from kurogami.adapters.llm.cached import CachedLLM
from kurogami.agents._prompt_loader import load_prompt
from kurogami.cli import main as cli_main
from kurogami.cli.main import app
from kurogami.contracts import LLMResponse, TraceRecord

runner = CliRunner()

RAW_TEXT = "Should I launch my invoicing tool?"

# 10 items, one decision sink, longest chain a0-a1-c0-c1-c2-final (depth 5).
_SHAPE = {
    "a0": ("research", []),
    "a1": ("analysis", ["a0"]),
    "a2": ("analysis", ["a1"]),
    "a3": ("analysis", ["a2"]),
    "b0": ("research", []),
    "b1": ("analysis", ["b0"]),
    "c0": ("analysis", ["a1"]),
    "c1": ("synthesis", ["c0", "b1"]),
    "c2": ("analysis", ["c1"]),
    "final": ("decision", ["a3", "c2"]),
}


def _goal_dict() -> dict:
    return {
        "raw_text": RAW_TEXT,
        "product_description": "Invoicing tool for freelance designers.",
        "target_market": "Freelance designers in India",
        "decision_type": "launch",
        "success_definition": "A clear go/no-go.",
    }


def _stage(shape: dict, i: str, seen: frozenset = frozenset()) -> int:
    deps = [d for d in shape[i][1] if d in shape and d not in seen]
    return 1 + max((_stage(shape, d, seen | {i}) for d in deps), default=0)


def _scope(shape: dict = _SHAPE) -> dict:
    return {
        "items": [
            {"id": i, "question": f"{i}?", "kind": kind, "stage": _stage(shape, i), "depends_on": deps}
            for i, (kind, deps) in shape.items()
        ]
    }


def _blueprint(assertions: dict[str, list[str]] | None = None) -> dict:
    assertions = assertions or {}
    return {
        "nodes": [
            {
                "node_id": i,
                "title": i,
                "node_goal": f"goal {i}",
                "generated_prompt": f"prompt for {i} " * 15,
                "pass_condition": {
                    "assertions": assertions.get(i, []),
                    "semantic_check": f"Consistent with {deps[0]}?" if deps else "Answered?",
                },
            }
            for i, (_, deps) in _SHAPE.items()
        ]
    }


class _SchemaFakeLLM:
    """Answers by the requested schema's name; free-text calls get a minimal JSON block."""

    def __init__(self, by_schema: dict[str, Any]) -> None:
        self._by_schema = by_schema
        self.calls: list[str] = []

    def complete(self, *, prompt, system=None, schema=None, temperature=0.0):
        name = schema.__name__ if schema is not None else "text"
        self.calls.append(name)
        if schema is None:
            text = "Analysis.\n```json\n{}\n```"
            return LLMResponse(text=text, tokens_in=1, tokens_out=1, latency_ms=0, model_id="fake")
        parsed = schema.model_validate(self._by_schema[name])
        return LLMResponse(
            text=parsed.model_dump_json(), parsed=parsed, tokens_in=1, tokens_out=1,
            latency_ms=0, model_id="fake",
        )


def _fake(scope: dict | None = None, blueprint: dict | None = None) -> _SchemaFakeLLM:
    return _SchemaFakeLLM(
        {
            "GoalSpec": _goal_dict(),
            "_Scope": scope or _scope(),
            "_Blueprint": blueprint or _blueprint(),
            "_VerifyResponse": {"checked_claims": [], "verdict": "PASS", "reason": None},
            "_GapFill": {"node": None},
            "_CheckReview": {
                "check_is_wrong": False, "explanation": "the answer falls short",
                "replacement_assertions": [],
            },
        }
    )


@pytest.fixture
def use_llm(monkeypatch):
    def install(llm: _SchemaFakeLLM) -> _SchemaFakeLLM:
        monkeypatch.setattr(cli_main, "_build_llm", lambda *args, **kwargs: llm)
        return llm

    return install


def _goal_file(tmp_path: Path) -> Path:
    path = tmp_path / "goal.json"
    path.write_text(json.dumps({"raw_text": RAW_TEXT}))
    return path


def _trace_records(trace_dir: Path) -> list[dict]:
    [trace_file] = [f for f in trace_dir.glob("*.jsonl") if not f.name.endswith(".calls.jsonl")]
    return [json.loads(line) for line in trace_file.read_text().splitlines() if line.strip()]


def test_interpret_command_prints_the_goal_spec(tmp_path):
    interpret_prompt = load_prompt("interpret").format(raw_text=RAW_TEXT)
    script = tmp_path / "script.json"
    script.write_text(json.dumps({interpret_prompt: _goal_dict()}))

    result = runner.invoke(app, ["interpret", "--text", RAW_TEXT, "--fake-script", str(script)])

    assert result.exit_code == 0
    assert "launch" in result.stdout


def test_plan_dry_run_prints_the_whole_planned_tree(tmp_path, use_llm):
    llm = use_llm(_fake())

    result = runner.invoke(
        app,
        ["plan", "--goal-file", str(_goal_file(tmp_path)), "--dry-run", "--trace-dir", str(tmp_path)],
    )

    assert result.exit_code == 0, result.stdout
    assert "10 planned node(s), max depth 5" in result.stdout
    assert "final" in result.stdout
    assert "text" not in llm.calls  # nothing executed


def test_plan_records_every_llm_call(tmp_path, use_llm):
    use_llm(_fake())
    trace_dir = tmp_path / "traces"

    runner.invoke(
        app,
        ["plan", "--goal-file", str(_goal_file(tmp_path)), "--dry-run", "--trace-dir", str(trace_dir)],
    )

    [log] = list(trace_dir.glob("plan-*.calls.jsonl"))
    schemas = [json.loads(line)["schema"] for line in log.read_text().splitlines()]
    assert schemas == ["GoalSpec", "_Scope", "_Blueprint"]


def test_plan_reports_a_plan_that_cannot_be_made(tmp_path, use_llm):
    cyclic = _scope({**_SHAPE, "a0": ("research", ["a3"])})
    use_llm(_fake(scope=cyclic))

    result = runner.invoke(
        app,
        ["plan", "--goal-file", str(_goal_file(tmp_path)), "--dry-run", "--trace-dir", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "planning failed" in result.stdout


def test_run_completes_the_planned_tree(tmp_path, use_llm):
    trace_dir = tmp_path / "traces"
    use_llm(_fake())

    result = runner.invoke(
        app,
        ["run", "--goal-file", str(_goal_file(tmp_path)), "--trace-dir", str(trace_dir), "--verbose"],
    )

    assert result.exit_code == 0, result.stdout
    assert "10 active node(s), max depth 5" in result.stdout
    assert "Scripted · Test model" in result.stdout
    assert "Task dependencies" in result.stdout
    assert "Depends on" in result.stdout
    assert "PASS  a0" in result.stdout
    assert "Final recommendation" in result.stdout
    records = _trace_records(trace_dir)
    assert sorted(r["node_id"] for r in records) == sorted(_SHAPE)
    assert all(r["verifier_verdict"] == "PASS" for r in records)


def test_open_trace_exports_latest_run_with_all_parents_and_calls(tmp_path, use_llm, monkeypatch):
    use_llm(_fake())
    (tmp_path / "plan-only.calls.jsonl").write_text("")
    result = runner.invoke(app, ["run", "--text", RAW_TEXT, "--trace-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "kurogami open-trace" in result.output

    opened = []
    monkeypatch.setattr(cli_main, "open_browser", lambda page: opened.append(page) or True)
    result = runner.invoke(app, ["open-trace", "--trace-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    [page] = opened
    html = page.read_text(encoding="utf-8")
    assert RAW_TEXT in html
    payload = html.split('<script id="run-data" type="application/json">', 1)[1]
    saved = json.loads(payload.split("</script>", 1)[0])
    assert saved["snapshot"]["specs"]["final"]["parent_ids"] == ["a3", "c2"]
    assert len(saved["calls"]) == 23

    result = runner.invoke(app, ["open-trace", "--trace-dir", str(tmp_path), "--no-browser"])
    assert result.exit_code == 0, result.output
    assert opened == [page]


def test_run_requires_exactly_one_goal(tmp_path):
    goal_file = _goal_file(tmp_path)
    for options in ([], ["--text", " "], ["--text", RAW_TEXT, "--goal-file", str(goal_file)]):
        result = runner.invoke(app, ["run", *options, "--trace-dir", str(tmp_path)])
        assert result.exit_code == 2
    goal_file.write_text('{"raw_text": 42}')
    result = runner.invoke(app, ["run", "--goal-file", str(goal_file), "--trace-dir", str(tmp_path)])
    assert result.exit_code == 2
    assert "nonblank raw_text string" in result.output
    assert not list(tmp_path.glob("*.jsonl"))


def test_strict_offline_runs_from_a_recorded_cache_without_openai(tmp_path, monkeypatch):
    cache_dir = tmp_path / "seeded-cache"
    writer = CachedLLM(_fake(), model_id="demo", cache_dir=cache_dir)
    with monkeypatch.context() as patch:
        patch.setattr(cli_main, "_build_llm", lambda *args, **kwargs: writer)
        times = iter([0.0, 1.0] * 60)
        patch.setattr("kurogami.agents.executor.time", SimpleNamespace(monotonic=lambda: next(times)))
        result = runner.invoke(
            app,
            ["run", "--text", RAW_TEXT, "--search", "none", "--trace-dir", str(tmp_path / "seed")],
        )
        assert result.exit_code == 0, result.output

    def unexpected_provider(*args, **kwargs):
        pytest.fail("offline mode constructed a remote provider")

    monkeypatch.setattr(cli_main, "OpenAILLM", unexpected_provider)
    monkeypatch.setattr(cli_main, "TavilySearch", unexpected_provider)
    offline_options = [
        "--llm", "openai", "--offline", "--model", "demo", "--cache-dir", str(cache_dir),
        "--trace-dir", str(tmp_path / "offline"),
    ]
    before = {path.name: path.read_bytes() for path in cache_dir.iterdir()}
    with monkeypatch.context() as patch:
        patch.setattr("kurogami.agents.executor.time", SimpleNamespace(monotonic=lambda: 0.0))
        result = runner.invoke(app, ["run", "--text", RAW_TEXT, *offline_options])
    assert result.exit_code == 0, result.output
    assert "Final recommendation" in result.stdout
    result = runner.invoke(app, ["run", "--text", "Uncached goal", *offline_options])
    assert result.exit_code == 1
    assert "offline cache miss" in result.stdout
    assert "exact constraint" in result.stdout
    assert {path.name: path.read_bytes() for path in cache_dir.iterdir()} == before
    result = runner.invoke(app, ["run", "--text", RAW_TEXT, *offline_options, "--search", "tavily"])
    assert result.exit_code == 2


class _ScenarioLLM(_SchemaFakeLLM):
    def __init__(self, *, plant_fault: bool = False) -> None:
        super().__init__(_fake()._by_schema)
        self.plant_fault = plant_fault
        self.planted = False

    def complete(self, *, prompt, system=None, schema=None, temperature=0.0):
        if schema is None:
            output = "Analysis."
            if "prompt for final " in prompt and "Use email only." in prompt:
                output = "Recommendation: use email only."
            return LLMResponse(text=output, tokens_in=1, tokens_out=1, latency_ms=0, model_id="fake")
        if (self.plant_fault and not self.planted and schema.__name__ == "_VerifyResponse"
                and '"node_id": "a1"' in prompt):
            self.planted = True
            verdict = schema.model_validate({
                "verdict": "FAIL",
                "reason": {
                    "summary": "Planted test fault in the upstream premise",
                    "violated": "semantic",
                    "evidence": "Controlled verifier response",
                    "suspect_node_ids": ["a0"],
                },
            })
            return LLMResponse(
                text=verdict.model_dump_json(), parsed=verdict,
                tokens_in=1, tokens_out=1, latency_ms=0, model_id="fake",
            )
        return super().complete(prompt=prompt, system=system, schema=schema, temperature=temperature)


def test_cli_recovers_from_a_planted_verifier_failure(tmp_path, use_llm):
    use_llm(_ScenarioLLM(plant_fault=True))
    result = runner.invoke(app, ["run", "--text", RAW_TEXT, "--trace-dir", str(tmp_path), "--verbose"])
    assert result.exit_code == 0, result.output
    records = _trace_records(tmp_path)
    assert any(row["node_id"] == "a1" and row["backtrack_target"] == "a0" for row in records)
    assert any(row["node_id"] == "a1~r1" and row["verifier_verdict"] == "PASS" for row in records)
    assert sum(row["node_id"] == "b0" for row in records) == 1
    assert "a1 -> backtrack to a0" in result.stdout
    assert "Final recommendation" in result.stdout


def test_cli_interrupt_changes_the_final_output(tmp_path, use_llm):
    use_llm(_ScenarioLLM())
    result = runner.invoke(
        app,
        ["run", "--text", RAW_TEXT, "--interrupt-at", "a0", "--trace-dir", str(tmp_path)],
        input="Use email only.\n\n",
    )
    assert result.exit_code == 0, result.output
    assert "paused at a0" in result.stdout
    assert "Recommendation: use email only." in result.stdout
    assert any(row["node_id"] == "a0" and row["human_interrupt"] for row in _trace_records(tmp_path))


def test_a_node_that_keeps_failing_gives_up_and_the_rest_of_the_tree_still_runs(tmp_path, use_llm):
    """A root whose assertion is always false. Seen live: such a node used to burn the
    run-wide backtrack budget and stop everything. Now it gets its own retries, gives
    up, and the independent branch (b0 -> b1) still runs; its dependents are skipped.
    """
    trace_dir = tmp_path / "traces"
    use_llm(_fake(blueprint=_blueprint(assertions={"a0": ["False"]})))

    result = runner.invoke(
        app, ["run", "--goal-file", str(_goal_file(tmp_path)), "--trace-dir", str(trace_dir)]
    )

    assert result.exit_code == 1
    assert "incomplete: gave up on a0" in " ".join(result.stdout.split())
    records = _trace_records(trace_dir)
    assert [r["verifier_verdict"] for r in records if r["node_id"] == "a0"] == ["FAIL"] * 3
    passed = {r["node_id"] for r in records if r["verifier_verdict"] == "PASS"}
    assert {"b0", "b1"} <= passed
    assert records[-1]["node_id"] == "__incomplete__"


def test_run_aborts_cleanly_when_planning_fails(tmp_path, use_llm):
    trace_dir = tmp_path / "traces"
    use_llm(_fake(scope=_scope({**_SHAPE, "a0": ("research", ["a3"])})))

    result = runner.invoke(
        app, ["run", "--goal-file", str(_goal_file(tmp_path)), "--trace-dir", str(trace_dir)]
    )

    assert result.exit_code == 1
    assert "planning failed" in result.stdout
    assert [r["node_id"] for r in _trace_records(trace_dir)] == ["__plan__"]


def test_replay_command_renders_the_trace_offline(tmp_path):
    record = TraceRecord(
        run_id=uuid4(),
        seq=1,
        node_id="n_a",
        parent_node_id=None,
        depth=0,
        generated_prompt="p",
        node_goal="g",
        output="o",
        verifier_verdict="PASS",
        failure_reason=None,
        backtrack_target=None,
        tokens_in=1,
        tokens_out=1,
        latency_ms=1,
        human_interrupt=False,
    )
    trace_file = tmp_path / "run.jsonl"
    trace_file.write_text(record.model_dump_json() + "\n")

    result = runner.invoke(app, ["replay", str(trace_file)])

    assert result.exit_code == 0
    assert "n_a" in result.stdout


def test_a_question_can_be_asked_directly_with_goal(use_llm, tmp_path):
    use_llm(_fake())
    result = CliRunner().invoke(
        cli_main.app,
        ["run", "--goal", "Should I launch my yoga app in Pune?", "--llm", "fake",
         "--trace-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.stdout


def test_goal_and_goal_file_are_mutually_exclusive(tmp_path):
    result = CliRunner().invoke(cli_main.app, ["run", "--llm", "fake", "--trace-dir", str(tmp_path)])
    assert result.exit_code != 0
