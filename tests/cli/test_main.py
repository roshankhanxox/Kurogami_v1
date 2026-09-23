"""CLI integration, offline. No network, no real LLM.

`interpret` is driven through the real --fake-script path (FakeLLM keyed by exact
prompt text). `plan` and `run` make a dozen-plus calls whose prompts embed earlier
outputs, so exact-text keying would be brittle; for those the composition root's
LLM is replaced by a fake that answers by *which schema is being asked for*. The
Interpreter, Planner, Executor, RuleChecker, Verifier and Runner are all real.
"""

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from typer.testing import CliRunner

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
            "_VerifyResponse": {"verdict": "PASS", "reason": None},
            "_GapFill": {"node": None},
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
    records = _trace_records(trace_dir)
    assert sorted(r["node_id"] for r in records) == sorted(_SHAPE)
    assert all(r["verifier_verdict"] == "PASS" for r in records)


def test_run_budget_breach_is_a_clean_outcome_not_a_crash(tmp_path, use_llm):
    """A root whose assertion is always false: FAIL -> backtrack to itself -> FAIL
    again, until max_backtracks stops the run cleanly (R7)."""
    trace_dir = tmp_path / "traces"
    use_llm(_fake(blueprint=_blueprint(assertions={"a0": ["False"]})))

    result = runner.invoke(
        app, ["run", "--goal-file", str(_goal_file(tmp_path)), "--trace-dir", str(trace_dir)]
    )

    assert result.exit_code == 1
    assert "budget breached" in result.stdout
    records = _trace_records(trace_dir)
    assert records[0]["verifier_verdict"] == "FAIL"
    assert records[-1]["node_id"] == "__budget__"


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
