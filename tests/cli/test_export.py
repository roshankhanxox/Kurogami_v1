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
