"""JsonlTraceSink: append-only, one record per line, flushed on every write."""

from uuid import uuid4

from kurogami.adapters.trace.jsonl import JsonlTraceSink
from kurogami.contracts import TraceRecord


def _record(seq: int) -> TraceRecord:
    return TraceRecord(
        run_id=uuid4(),
        seq=seq,
        node_id=f"n_{seq:03d}",
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


def test_emit_appends_one_json_line_per_record(tmp_path):
    path = tmp_path / "run.jsonl"
    sink = JsonlTraceSink(path)

    sink.emit(_record(1))
    sink.emit(_record(2))
    sink.close()

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert TraceRecord.model_validate_json(lines[0]).seq == 1
    assert TraceRecord.model_validate_json(lines[1]).seq == 2


def test_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "run.jsonl"
    sink = JsonlTraceSink(path)
    sink.emit(_record(1))
    sink.close()
    assert path.exists()


def test_appends_across_separate_sink_instances(tmp_path):
    path = tmp_path / "run.jsonl"

    first = JsonlTraceSink(path)
    first.emit(_record(1))
    first.close()

    second = JsonlTraceSink(path)
    second.emit(_record(2))
    second.close()

    lines = path.read_text().splitlines()
    assert len(lines) == 2
