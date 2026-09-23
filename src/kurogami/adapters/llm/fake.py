"""Deterministic, offline LLMPort implementation. No network, ever."""

from pydantic import BaseModel

from kurogami.contracts.ports import LLMResponse


class UnregisteredPromptError(KeyError):
    """Raised when FakeLLM is asked to complete a prompt name it has no canned response for."""


class FakeLLM:
    """Satisfies LLMPort with canned, per-prompt-name responses.

    Registered by prompt *name* (not full prompt text), so callers can key
    responses to `prompts/<name>.md` regardless of how context is interpolated
    into the final prompt string.
    """

    def __init__(self, responses: dict[str, BaseModel | str]) -> None:
        self._responses = responses
        self._model_id = "fake-llm"

    def register(self, prompt_name: str, response: BaseModel | str) -> None:
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
            if not isinstance(canned, schema):
                raise TypeError(
                    f"FakeLLM response for {key!r} is {type(canned).__name__}, "
                    f"expected {schema.__name__}"
                )
            text = canned.model_dump_json()
            parsed: BaseModel | None = canned
        else:
            text = canned if isinstance(canned, str) else canned.model_dump_json()
            parsed = canned if isinstance(canned, BaseModel) else None

        return LLMResponse(
            text=text,
            parsed=parsed,
            tokens_in=len(prompt.split()),
            tokens_out=len(text.split()),
            latency_ms=0,
            model_id=self._model_id,
        )
