"""OpenAI adapter satisfying LLMPort. Schema-constrained via chat.completions.parse()."""

import time
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

from kurogami.contracts.ports import LLMResponse

DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"


class OpenAIRefusalError(RuntimeError):
    """Raised when the model refuses to produce the requested structured output."""


class OpenAILLM:
    """LLMPort over the real OpenAI API.

    A `client` can be injected for tests -- it only needs to satisfy the two
    methods actually used (`chat.completions.parse`, `chat.completions.create`),
    so tests supply a small stand-in rather than mocking the SDK internals.
    Retries (429/5xx/connection/timeout) are handled by the SDK itself
    (max_retries, default 2) -- ARCHITECTURE.md D1.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_retries: int = 2,
        client: Any = None,
    ) -> None:
        self._model = model
        self._client = client if client is not None else OpenAI(api_key=api_key, max_retries=max_retries)

    def complete(
        self,
        *,
        prompt: str,
        system: str | None = None,
        schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        messages: list[Any] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.monotonic()
        if schema is not None:
            completion = self._client.chat.completions.parse(
                model=self._model,
                messages=messages,
                response_format=schema,
                temperature=temperature,
            )
            message = completion.choices[0].message
            if message.refusal:
                raise OpenAIRefusalError(message.refusal)
            parsed: BaseModel | None = message.parsed
            text = message.content if message.content is not None else parsed.model_dump_json()  # type: ignore[union-attr]
        else:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
            )
            parsed = None
            text = completion.choices[0].message.content or ""
        latency_ms = int((time.monotonic() - start) * 1000)

        usage = completion.usage
        return LLMResponse(
            text=text,
            parsed=parsed,
            tokens_in=usage.prompt_tokens if usage is not None else 0,
            tokens_out=usage.completion_tokens if usage is not None else 0,
            latency_ms=latency_ms,
            model_id=completion.model,
        )
