"""Level 1: sentence -> GoalSpec. Schema-constrained, a single LLM call."""

from kurogami.agents._prompt_loader import load_prompt
from kurogami.contracts import GoalSpec, LLMPort


class Interpreter:
    """Flags ambiguities rather than resolving them; see prompts/interpret.md."""

    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    def run(self, raw_text: str) -> GoalSpec:
        template = load_prompt("interpret")
        prompt = template.format(raw_text=raw_text)
        response = self._llm.complete(prompt=prompt, schema=GoalSpec)
        if not isinstance(response.parsed, GoalSpec):
            raise TypeError(
                f"Interpreter expected a parsed GoalSpec, got {type(response.parsed).__name__}"
            )
        return response.parsed
