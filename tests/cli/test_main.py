"""CLI integration, offline via --fake-script. No network, no real LLM.

FakeLLM keys responses by exact prompt text, so a multi-hop scenario (where
a later prompt embeds an earlier NodeResult, e.g. planner.expand) needs that
embedded JSON to match byte-for-byte. Rather than hand-computing hashes and
word counts, these tests precompute the expected artifacts by running the
same production code (Executor) once in the test's own setup -- since it's
pure and deterministic (latency is monkeypatched to zero), that precomputed
value is guaranteed to equal whatever the CLI's own internal call produces.
"""

import json
from pathlib import Path

from typer.testing import CliRunner

from kurogami.adapters.llm.fake import FakeLLM
from kurogami.agents._prompt_loader import load_prompt
from kurogami.agents.executor import Executor
from kurogami.cli.main import app
from kurogami.contracts import GoalSpec, NodeKind, NodeSpec, PassCondition

runner = CliRunner()


def _write_script(tmp_path: Path, responses: dict) -> Path:
    path = tmp_path / "script.json"
    path.write_text(json.dumps(responses))
    return path


def _goal_file(tmp_path: Path, raw_text: str) -> Path:
    path = tmp_path / "goal.json"
    path.write_text(json.dumps({"raw_text": raw_text}))
    return path


def _goal_dict(raw_text: str) -> dict:
    return {
        "raw_text": raw_text,
        "product_description": "Invoicing tool for freelance designers.",
        "target_market": "Freelance designers in India",
        "decision_type": "market_entry",
        "success_definition": "A clear go/no-go.",
    }


def test_interpret_command_prints_the_goal_spec(tmp_path):
    raw_text = "Should I launch my invoicing tool?"
    goal_dict = _goal_dict(raw_text)
    interpret_prompt = load_prompt("interpret").format(raw_text=raw_text)
    script = _write_script(tmp_path, {interpret_prompt: goal_dict})

    result = runner.invoke(app, ["interpret", "--text", raw_text, "--fake-script", str(script)])

    assert result.exit_code == 0
    assert "market_entry" in result.stdout


def test_plan_dry_run_prints_root_nodes(tmp_path):
    raw_text = "Should I launch my invoicing tool?"
    goal_dict = _goal_dict(raw_text)
    goal_obj = GoalSpec(**goal_dict)
    root_dict = {
        "node_id": "n_a",
        "parent_ids": [],
        "depth": 0,
        "kind": "analysis",
        "title": "root",
        "node_goal": "root goal",
        "generated_prompt": "a" * 50,
        "pass_condition": {"assertions": [], "semantic_check": "ok?"},
    }
    interpret_prompt = load_prompt("interpret").format(raw_text=raw_text)
    plan_prompt = load_prompt("plan").format(goal_json=goal_obj.model_dump_json(indent=2))
    script = _write_script(
        tmp_path,
        {interpret_prompt: goal_dict, plan_prompt: {"nodes": [root_dict]}},
    )
    goal_file = _goal_file(tmp_path, raw_text)

    result = runner.invoke(
        app, ["plan", "--goal-file", str(goal_file), "--fake-script", str(script), "--dry-run"]
    )

    assert result.exit_code == 0
    assert "n_a" in result.stdout
    assert "1 root node" in result.stdout


def test_run_command_fails_backtracks_and_breaches_budget_cleanly(tmp_path):
    """A root node whose assertion is always false: FAIL -> backtrack to self (no
    ancestors) -> requeue -> FAIL again, forever, until the budget stops it.
    No expand() or verify() call is ever reached, so this needs no NodeResult
    to be embedded in a further prompt.

    The runner feeds the backtrack reason back into the retried node's
    context (by design -- a retried node should see why it failed), which
    changes the executor prompt from the second attempt onward. Both prompt
    variants need a registered response.
    """
    raw_text = "Should I launch my invoicing tool?"
    goal_dict = _goal_dict(raw_text)
    goal_obj = GoalSpec(**goal_dict)
    root_dict = {
        "node_id": "n_a",
        "parent_ids": [],
        "depth": 0,
        "kind": "analysis",
        "title": "root",
        "node_goal": "root goal",
        "generated_prompt": "always fails its own assertion",
        "pass_condition": {"assertions": ["False"], "semantic_check": "ok?"},
    }
    root_node = NodeSpec(
        node_id="n_a",
        parent_ids=[],
        depth=0,
        kind=NodeKind.ANALYSIS,
        title="root",
        node_goal="root goal",
        generated_prompt=root_dict["generated_prompt"],
        pass_condition=PassCondition(assertions=["False"], semantic_check="ok?"),
    )
    retried_node = root_node.model_copy(
        update={"context": {"_backtrack_reason": "assertion evaluated to False: False"}}
    )
    # Executor._build_prompt is "private" but pure (no self.* access besides
    # search, unused for an ANALYSIS node), so calling it directly here
    # mirrors exactly what the real second-attempt call will send.
    retried_prompt = Executor(llm=None)._build_prompt(retried_node)

    interpret_prompt = load_prompt("interpret").format(raw_text=raw_text)
    plan_prompt = load_prompt("plan").format(goal_json=goal_obj.model_dump_json(indent=2))
    script = _write_script(
        tmp_path,
        {
            interpret_prompt: goal_dict,
            plan_prompt: {"nodes": [root_dict]},
            "always fails its own assertion": "some executor output",
            retried_prompt: "some executor output",
        },
    )
    goal_file = _goal_file(tmp_path, raw_text)
    trace_dir = tmp_path / "traces"

    result = runner.invoke(
        app,
        [
            "run",
            "--goal-file", str(goal_file),
            "--fake-script", str(script),
            "--trace-dir", str(trace_dir),
        ],
    )

    assert result.exit_code == 1
    assert "budget breached" in result.stdout

    trace_files = list(trace_dir.glob("*.jsonl"))
    assert len(trace_files) == 1
    lines = [line for line in trace_files[0].read_text().splitlines() if line.strip()]
    assert len(lines) >= 2  # at least one FAIL record plus the terminal budget-breach record
    assert '"verifier_verdict":"FAIL"' in lines[0] or '"verifier_verdict": "FAIL"' in lines[0]


def test_run_command_happy_path_single_node(tmp_path, monkeypatch):
    """A root node with no assertions, a PASS verifier, and expand() returning
    zero children -- the simplest possible successful run.
    """
    monkeypatch.setattr("kurogami.agents.executor.time.monotonic", lambda: 0.0)

    raw_text = "Should I launch my invoicing tool?"
    goal_dict = _goal_dict(raw_text)
    goal_obj = GoalSpec(**goal_dict)
    root_dict = {
        "node_id": "n_a",
        "parent_ids": [],
        "depth": 0,
        "kind": "analysis",
        "title": "root",
        "node_goal": "root goal",
        "generated_prompt": "a sufficiently long generated prompt for the root node",
        "pass_condition": {"assertions": [], "semantic_check": "ok?"},
    }
    root_node = NodeSpec(
        node_id="n_a",
        parent_ids=[],
        depth=0,
        kind=NodeKind.ANALYSIS,
        title="root",
        node_goal="root goal",
        generated_prompt=root_dict["generated_prompt"],
        pass_condition=PassCondition(assertions=[], semantic_check="ok?"),
    )
    executor_text = '{"ok": true}'

    # Precompute the NodeResult the real Executor will produce, using the same
    # code the CLI invocation below will independently call -- deterministic
    # given the monkeypatched clock, so this is guaranteed to match exactly.
    setup_llm = FakeLLM(responses={root_node.generated_prompt: executor_text})
    expected_result = Executor(setup_llm).run(root_node)

    interpret_prompt = load_prompt("interpret").format(raw_text=raw_text)
    plan_prompt = load_prompt("plan").format(goal_json=goal_obj.model_dump_json(indent=2))
    verify_prompt = load_prompt("verify").format(
        node_json=root_node.model_dump_json(indent=2),
        result_json=expected_result.model_dump_json(indent=2),
        context_json=json.dumps({}, indent=2),
        semantic_check="ok?",
    )
    expand_prompt = load_prompt("expand").format(
        parent_json=root_node.model_dump_json(indent=2),
        result_json=expected_result.model_dump_json(indent=2),
        goal_json=goal_obj.model_dump_json(indent=2),
    )

    script = _write_script(
        tmp_path,
        {
            interpret_prompt: goal_dict,
            plan_prompt: {"nodes": [root_dict]},
            root_node.generated_prompt: executor_text,
            verify_prompt: {"verdict": "PASS"},
            expand_prompt: {"nodes": []},
        },
    )
    goal_file = _goal_file(tmp_path, raw_text)
    trace_dir = tmp_path / "traces"

    result = runner.invoke(
        app,
        [
            "run",
            "--goal-file", str(goal_file),
            "--fake-script", str(script),
            "--trace-dir", str(trace_dir),
            "--verbose",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "n_a" in result.stdout

    trace_files = list(trace_dir.glob("*.jsonl"))
    assert len(trace_files) == 1
    lines = [line for line in trace_files[0].read_text().splitlines() if line.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["verifier_verdict"] == "PASS"


def test_replay_command_renders_the_trace_offline(tmp_path):
    from uuid import uuid4

    from kurogami.contracts import TraceRecord

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
