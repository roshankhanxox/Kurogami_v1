"""Localiser: who is responsible when a cross-node assertion fails.

A deterministic check can say *that* a price broke an ancestor's ceiling, not *why*:
the pricing node may have ignored the ceiling, or an intermediate ancestor (say, a
premium positioning) may have pushed it there. The architecture's claim is that the
failing node is usually not the wrong node, so the rule checker's FAIL is handed to
an LLM that names the suspects; backtrack.locate still intersects them with the real
ancestor set and picks the shallowest.
"""

import json

from pydantic import BaseModel

from kurogami.agents._prompt_loader import load_prompt
from kurogami.agents._structured import ancestor_references
from kurogami.contracts import FailureReason, LLMPort, NodeResult, NodeSpec


class _LocaliseResponse(BaseModel):
    """Decoding envelope. explanation comes first so the model reasons before naming ids."""

    explanation: str
    suspect_node_ids: list[str]


class Localiser:
    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    def localise(
        self, node: NodeSpec, result: NodeResult, reason: FailureReason, context: dict[str, str]
    ) -> FailureReason:
        """The same reason with suspect_node_ids filled in; unchanged unless a cross-node check failed.

        A failed check on the node's own output alone (a missing key, too few items)
        is the node's own fault, so it costs no LLM call.
        """
        failed = [a for a in node.pass_condition.assertions if reason.evidence.startswith(a)]
        if not any(ancestor_references(a) for a in failed):
            return reason
        ancestors = sorted(k for k in context if not k.startswith("_"))
        prompt = load_prompt("localise").format(
            node_json=node.model_dump_json(indent=2, exclude={"context"}),
            output=result.output,
            summary=reason.summary,
            evidence=reason.evidence,
            context_json=json.dumps(context, indent=2),
            ancestor_ids=", ".join(ancestors),
        )
        parsed = self._llm.complete(prompt=prompt, schema=_LocaliseResponse).parsed
        if not isinstance(parsed, _LocaliseResponse):
            raise TypeError(
                f"Localiser expected a parsed _LocaliseResponse, got {type(parsed).__name__}"
            )
        # Only real ancestors; the node itself or an invented id means "the node is at fault".
        suspects = [s for s in parsed.suspect_node_ids if s in ancestors]
        return reason.model_copy(
            update={
                "suspect_node_ids": suspects,
                "summary": f"{reason.summary} -- localiser: {parsed.explanation}",
            }
        )
