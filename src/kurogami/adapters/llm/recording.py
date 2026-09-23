"""RecordingLLM: wraps any LLMPort and appends every call to a JSONL file.

The trace records node executions only; the Master's planning calls (scope,
corrections, blueprint) and every verifier call were invisible. When a live run
fails you need the exact responses, not a summary -- this is that record.
"""

import json
import time
from pathlib import Path

from pydantic import BaseModel

from kurogami.contracts.ports import LLMPort, LLMResponse


class RecordingLLM:
    """Satisfies LLMPort. One JSON line per call: what was asked, what came back."""

    def __init__(self, inner: LLMPort, path: str | Path) -> None:
        self._inner = inner
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0

    @property
    def path(self) -> Path:
        return self._path

    def complete(
        self,
        *,
        prompt: str,
        system: str | None = None,
        schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self._seq += 1
        record: dict[str, object] = {
            "seq": self._seq,
            "schema": schema.__name__ if schema is not None else None,
            "prompt": prompt,
        }
        start = time.monotonic()
        try:
            response = self._inner.complete(
                prompt=prompt, system=system, schema=schema, temperature=temperature
            )
        except Exception as exc:
            record.update(error=f"{type(exc).__name__}: {exc}")
            self._write(record, start)
            raise
        record.update(
            text=response.text,
            parsed=response.parsed.model_dump(mode="json") if response.parsed is not None else None,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            model_id=response.model_id,
        )
        self._write(record, start)
        return response

    def _write(self, record: dict[str, object], start: float) -> None:
        record["wall_ms"] = int((time.monotonic() - start) * 1000)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
