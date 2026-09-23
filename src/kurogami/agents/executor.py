"""Level 3: run a node's self-written prompt. Generic -- no domain logic here."""

import hashlib
import json
import time
from typing import Any

from kurogami.contracts import LLMPort, NodeKind, NodeResult, NodeSpec, SearchPort


class Executor:
    """Sends NodeSpec.generated_prompt (plus assembled context) to the LLM.

    SearchPort is consulted only for kind=RESEARCH nodes (ARCHITECTURE.md B6).
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
            parts.append("\nAncestor context:\n" + json.dumps(node.context, indent=2))

        if node.injected_constraints:
            constraints = "\n".join(f"- {c}" for c in node.injected_constraints)
            parts.append(
                "\nAdditional constraints injected by a human reviewer, which this "
                f"output MUST honour:\n{constraints}"
            )

        if node.kind == NodeKind.RESEARCH and self._search is not None:
            hits = self._search.search(node.node_goal)
            results = "\n".join(f"- {hit.title}: {hit.snippet} ({hit.url})" for hit in hits)
            parts.append(f"\nSearch results:\n{results}")

        return "\n".join(parts)

    @staticmethod
    def _try_parse_structured(text: str) -> dict[str, Any]:
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _prompt_version(generated_prompt: str) -> str:
        digest = hashlib.sha256(generated_prompt.encode()).hexdigest()[:8]
        return f"generated_prompt@{digest}"
