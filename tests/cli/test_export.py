"""HTML export: one self-contained page with the whole DAG and every attempt."""

import json

from kurogami.cli.export import export_run
from kurogami.contracts import (
    NodeKind,
    NodeResult,
    NodeSpec,
    NodeStatus,
    PassCondition,
    TraceRecord,
    TreeSnapshot,
)


def _spec(node_id, parents, depth):
    return NodeSpec(node_id=node_id, parent_ids=parents, depth=depth, kind=NodeKind.ANALYSIS,
                    title=node_id, node_goal="g", generated_prompt="p",
                    pass_condition=PassCondition(assertions=[], semantic_check="?"))


def _write_run(tmp_path, output="fine"):
    specs = {"a": _spec("a", [], 0), "b": _spec("b", ["a"], 1)}
    snap = TreeSnapshot(
        root_ids=["a"], specs=specs, statuses={"a": NodeStatus.PASSED, "b": NodeStatus.PASSED},
        results={i: NodeResult(node_id=i, output=output, tokens_in=1, tokens_out=1, latency_ms=1,
                               model_id="f", prompt_version="v") for i in specs},
    )
    snapshot = tmp_path / "run.snapshot.json"
    snapshot.write_text(json.dumps({"goal": "Should I?", "outcome": None,
                                    "snapshot": snap.model_dump(mode="json")}))
    record = TraceRecord(run_id="00000000-0000-0000-0000-000000000000", seq=1, node_id="b",
                         parent_node_id="a", depth=1, generated_prompt="p", node_goal="g",
                         output="first try", verifier_verdict="FAIL", failure_reason="too thin",
                         backtrack_target="a", tokens_in=1, tokens_out=1, latency_ms=1,
                         human_interrupt=False)
    (tmp_path / "run.jsonl").write_text(record.model_dump_json() + "\n")
    return snapshot


def _payload(html: str) -> dict:
    start = html.index('<script id="run-data" type="application/json">') + len(
        '<script id="run-data" type="application/json">')
    return json.loads(html[start:html.index("</script>", start)].replace("<\\/", "</"))


def test_the_page_embeds_the_whole_graph_and_failed_attempts(tmp_path):
    page = export_run(_write_run(tmp_path))
    data = _payload(page.read_text())
    assert page.name == "run.html"
    assert data["snapshot"]["specs"]["b"]["parent_ids"] == ["a"]
    assert data["attempts"]["b"][0]["failure_reason"] == "too thin"


def test_hostile_output_cannot_close_the_data_script(tmp_path):
    html = export_run(_write_run(tmp_path, output="</script><script>alert(1)</script>")).read_text()
    assert "</script><script>alert(1)" not in html
    assert _payload(html)["snapshot"]["results"]["a"]["output"].startswith("</script>")


def test_html_includes_the_recorded_model_calls(tmp_path):
    snapshot = _write_run(tmp_path)
    call = {"seq": 1, "schema": "GoalSpec", "prompt": "Map the goal", "text": "Interpreted"}
    (tmp_path / "run.calls.jsonl").write_text(json.dumps(call) + "\n", encoding="utf-8")
    page = export_run(snapshot).read_text(encoding="utf-8")
    assert _payload(page)["calls"] == [call]


def test_html_formats_readable_output_offline_and_escapes_raw_tags(tmp_path):
    output = (
        "## Next steps\n\n1. **Start here.**\n   - Keep the pilot small.\n\n"
        "First line.\nSecond line.\n\n"
        "```json\n{\n  \"decision\": \"go\"\n}\n```\n\n"
        "<script>alert(1)</script>"
    )
    snapshot = _write_run(tmp_path, output=output)
    (tmp_path / "run.calls.jsonl").write_text(
        json.dumps({"seq": 1, "schema": None, "text": output}) + "\n", encoding="utf-8"
    )
    data = _payload(export_run(snapshot).read_text(encoding="utf-8"))
    formatted = data["formatted"]["results"]["a"]
    assert "<h2>Next steps</h2>" in formatted
    assert "<ol>" in formatted and "<ul>" in formatted
    assert "<strong>Start here.</strong>" in formatted
    assert "First line.<br />\nSecond line." in formatted
    assert '\n  &quot;decision&quot;: &quot;go&quot;\n' in formatted
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in formatted
    assert data["formatted"]["responses"] == [formatted]
