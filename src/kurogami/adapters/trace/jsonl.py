"""Append-only JSONL trace sink. One record per line, flushed on every write."""

from pathlib import Path

from kurogami.contracts import TraceRecord


class JsonlTraceSink:
    """Satisfies TraceSink. Opens the file in append mode; flushes after every emit."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", encoding="utf-8")

    def emit(self, record: TraceRecord) -> None:
        self._file.write(record.model_dump_json() + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()
