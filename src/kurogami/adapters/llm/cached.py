"""CachedLLM: wraps any real LLM, keyed by hash(prompt+model+temperature+schema).

Demo insurance (ARCHITECTURE.md section 9): run the demo goal a couple of
times during rehearsal against a real LLM through this wrapper; the demo run
is then served from .cache/llm/ -- fast, free, deterministic, and identical
to a live run in every code path. It is a cache, not a mock.
"""

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel

from kurogami.contracts.ports import LLMPort, LLMResponse


class CachedLLM:
    """Satisfies LLMPort by wrapping another LLMPort with an on-disk cache."""

    def __init__(self, inner: LLMPort, *, model_id: str, cache_dir: str | Path = ".cache/llm") -> None:
        self._inner = inner
        self._model_id = model_id
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def complete(
        self,
        *,
        prompt: str,
        system: str | None = None,
        schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        path = self._cache_dir / f"{self._key(prompt, system, schema, temperature)}.json"

        if path.exists():
            cached = json.loads(path.read_text())
            parsed = (
                schema.model_validate(cached["parsed"])
                if schema is not None and cached["parsed"] is not None
                else None
            )
            return LLMResponse(
                text=cached["text"],
                parsed=parsed,
                tokens_in=cached["tokens_in"],
                tokens_out=cached["tokens_out"],
                latency_ms=0,
                model_id=cached["model_id"],
            )

        response = self._inner.complete(prompt=prompt, system=system, schema=schema, temperature=temperature)
        path.write_text(
            json.dumps(
                {
                    "text": response.text,
                    "parsed": response.parsed.model_dump(mode="json") if response.parsed is not None else None,
                    "tokens_in": response.tokens_in,
                    "tokens_out": response.tokens_out,
                    "model_id": response.model_id,
                }
            )
        )
        return response

    def _key(self, prompt: str, system: str | None, schema: type[BaseModel] | None, temperature: float) -> str:
        schema_name = schema.__name__ if schema is not None else ""
        payload = f"{self._model_id}|{temperature}|{schema_name}|{system or ''}|{prompt}"
        return hashlib.sha256(payload.encode()).hexdigest()
