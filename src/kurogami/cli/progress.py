"""Label slow calls in the terminal without changing their inputs or output."""

from pydantic import BaseModel
from rich.console import Console

from kurogami.contracts import LLMPort, LLMResponse, SearchHit, SearchPort

_MODEL_STAGES = {
    "GoalSpec": "Interpreting your goal",
    "_Scope": "Planning the tasks and dependencies",
    "_Blueprint": "Writing task prompts and checks",
    "_VerifyResponse": "Verifying the task output",
    "_GapFill": "Planning a missing prerequisite",
}


class ModelProgress:
    def __init__(self, inner: LLMPort, console: Console) -> None:
        self._inner = inner
        self._console = console

    def complete(
        self,
        *,
        prompt: str,
        system: str | None = None,
        schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        message = (
            _MODEL_STAGES.get(schema.__name__, "Processing model response")
            if schema is not None else "Executing the task"
        )
        with self._console.status(message, spinner="dots", spinner_style="cyan"):
            return self._inner.complete(
                prompt=prompt, system=system, schema=schema, temperature=temperature
            )


class SearchProgress:
    def __init__(self, inner: SearchPort, console: Console) -> None:
        self._inner = inner
        self._console = console

    def search(self, query: str, *, max_results: int = 5) -> list[SearchHit]:
        with self._console.status("Searching public sources", spinner="dots", spinner_style="cyan"):
            return self._inner.search(query, max_results=max_results)
