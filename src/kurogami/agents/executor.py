"""Level 3: run a node's self-written prompt. Generic -- no domain logic here."""

import hashlib
import json
import time
from typing import Any

from kurogami.agents._prompt_loader import load_prompt
from kurogami.agents._structured import (
    example_value,
    extract_structured,
    required_structured_keys,
    resolved_ancestor_values,
)
from kurogami.contracts import LLMPort, NodeKind, NodeResult, NodeSpec, SearchPort


def _render(prompt_name: str, **fields: str) -> str:
    return load_prompt(prompt_name).format(**fields).rstrip("\n")


class Executor:
    """Sends NodeSpec.generated_prompt (plus assembled context) to the LLM.

    SearchPort is consulted only for kind=RESEARCH nodes (ARCHITECTURE.md B6).

    The executor asks the model in free text, with no schema, so `structured`
    would be empty far more often than not (seen live: an assertion
    referencing structured['market_trends'] raised KeyError every time,
    because nothing ever told the model to produce that key). Rather than
    trust the planner's generated_prompt to independently describe a JSON
    shape that happens to match its own assertions, this extracts the keys
    the assertions actually reference -- deterministically, from their AST, no
    LLM guessing -- and appends an explicit instruction to return exactly
    those keys as a fenced JSON block.
    """

    def __init__(self, llm: LLMPort, search: SearchPort | None = None) -> None:
        self._llm = llm
        self._search = search

    def run(self, node: NodeSpec) -> NodeResult:
        prompt = self._build_prompt(node)

        start = time.monotonic()
        response = self._llm.complete(prompt=prompt)
        latency_ms = int((time.monotonic() - start) * 1000)

        return NodeResult(
            node_id=node.node_id,
            output=response.text,
            structured=self._try_parse_structured(response.text),
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            latency_ms=latency_ms,
            model_id=response.model_id,
            prompt_version=self._prompt_version(node.generated_prompt),
        )

    def _build_prompt(self, node: NodeSpec) -> str:
        parts = [node.generated_prompt]

        if node.context:
            parts.append(_render("execute_context", context_json=json.dumps(node.context, indent=2)))

        if node.injected_constraints:
            constraints = "\n".join(f"- {c}" for c in node.injected_constraints)
            parts.append(_render("execute_constraints", constraints=constraints))

        if node.kind == NodeKind.RESEARCH and self._search is not None:
            hits = self._search.search(node.node_goal)
            results = "\n".join(f"- {hit.title}: {hit.snippet} ({hit.url})" for hit in hits)
            parts.append(_render("execute_search", results=results))

        assertions = node.pass_condition.assertions
        required_keys = self._required_structured_keys(assertions)
        key_rules = ""
        checks = "\n".join(f"- {a}" for a in assertions)
        if required_keys:
            key_rules = _render(
                "execute_keys",
                keys=", ".join(required_keys),
                checks=checks,
                example=json.dumps(
                    {key: example_value(key, assertions) for key in required_keys}, indent=2
                ),
            )
        elif assertions:
            key_rules = _render("execute_checks", checks=checks)
        # Seen live: a check comparing against an ancestor's number is unanswerable if
        # the model has to dig that number out of 30k chars of context -- show it.
        ancestor_values = resolved_ancestor_values(assertions, node.context)
        if ancestor_values:
            key_rules += "\n" + _render(
                "execute_ancestor_values", values="\n".join(f"- {v}" for v in ancestor_values)
            )
        # Seen live: the final decision restated headlines in ~600 words with no
        # figures -- nothing asked it to carry the investigation's numbers through.
        if node.kind == NodeKind.DECISION:
            parts.append(_render("execute_decision"))
        # Always asked for, so any node can report an unplanned prerequisite
        # (the only trigger for runtime gap-filling -- see agents/planner.py).
        parts.append(_render("execute_structured", key_rules=key_rules))

        return "\n".join(parts)

    @staticmethod
    def _required_structured_keys(assertions: list[str]) -> list[str]:
        return required_structured_keys(assertions)

    @staticmethod
    def _try_parse_structured(text: str) -> dict[str, Any]:
        return extract_structured(text)

    @staticmethod
    def _prompt_version(generated_prompt: str) -> str:
        digest = hashlib.sha256(generated_prompt.encode()).hexdigest()[:8]
        return f"generated_prompt@{digest}"
