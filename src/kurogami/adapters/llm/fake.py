"""Deterministic, offline LLMPort implementation. No network, ever."""

import json
from typing import Any

from pydantic import BaseModel

from kurogami.contracts.ports import LLMResponse


class UnregisteredPromptError(KeyError):
    """Raised when FakeLLM is asked to complete a prompt name it has no canned response for."""


class FakeLLM:
    """Satisfies LLMPort with canned, per-prompt-name responses.

    Registered by prompt *name* (not full prompt text), so callers can key
    responses to `prompts/<name>.md` regardless of how context is interpolated
    into the final prompt string.

    A canned response is a BaseModel instance, a plain str, or -- when a
    schema is passed to complete() -- a plain dict/list that gets validated
    against that schema. The dict/list form exists so a scripted conversation
    can be authored as plain JSON (e.g. from a CLI --fake-script file)
    without importing kurogami's pydantic models to build instances by hand.
    """

    def __init__(self, responses: dict[str, BaseModel | str | dict[str, Any] | list[Any]]) -> None:
        self._responses = responses
        self._model_id = "fake-llm"

    def register(self, prompt_name: str, response: BaseModel | str | dict[str, Any] | list[Any]) -> None:
        self._responses[prompt_name] = response

    def complete(
        self,
        *,
        prompt: str,
        prompt_name: str | None = None,
        system: str | None = None,
        schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        key = prompt_name if prompt_name is not None else prompt
        if key not in self._responses:
            raise UnregisteredPromptError(
                f"FakeLLM has no canned response registered for prompt name {key!r}. "
                f"Registered names: {sorted(self._responses)}"
            )
        canned = self._responses[key]

        if schema is not None:
            validated: BaseModel
            if isinstance(canned, schema):
                validated = canned
            elif isinstance(canned, dict | list):
                validated = schema.model_validate(canned)
            else:
                raise TypeError(
                    f"FakeLLM response for {key!r} is {type(canned).__name__}, "
                    f"expected {schema.__name__} or a dict/list to validate against it"
                )
            text = validated.model_dump_json()
            parsed: BaseModel | None = validated
        elif isinstance(canned, BaseModel):
            text = canned.model_dump_json()
            parsed = canned
        elif isinstance(canned, dict | list):
            text = json.dumps(canned)
            parsed = None
        else:
            text = canned
            parsed = None

        return LLMResponse(
            text=text,
            parsed=parsed,
            tokens_in=len(prompt.split()),
            tokens_out=len(text.split()),
            latency_ms=0,
            model_id=self._model_id,
        )
