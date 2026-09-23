"""Verifier: the harder semantic check, after rules.py's cheap assertions pass."""

import json
from typing import Literal

from pydantic import BaseModel

from kurogami.agents._prompt_loader import load_prompt
from kurogami.contracts import FailureReason, LLMPort, NodeResult, NodeSpec, Verdict


class _VerifyResponse(BaseModel):
    """Decoding envelope only. node_id/checked_by are set by us, not trusted from the model."""

    verdict: Literal["PASS", "FAIL"]
    reason: FailureReason | None = None


class Verifier:
    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    def check(self, node: NodeSpec, result: NodeResult, context: dict[str, str]) -> Verdict:
        template = load_prompt("verify")
        prompt = template.format(
            node_json=node.model_dump_json(indent=2),
            result_json=result.model_dump_json(indent=2),
            context_json=json.dumps(context, indent=2),
            semantic_check=node.pass_condition.semantic_check,
        )
        response = self._llm.complete(prompt=prompt, schema=_VerifyResponse)
        if not isinstance(response.parsed, _VerifyResponse):
            raise TypeError(
                f"Verifier expected a parsed _VerifyResponse, got {type(response.parsed).__name__}"
            )
        parsed = response.parsed
        if parsed.verdict == "FAIL" and parsed.reason is None:
            raise ValueError(f"Verifier returned FAIL for {node.node_id} with no reason")

        return Verdict(
            node_id=node.node_id,
            verdict=parsed.verdict,
            reason=parsed.reason,
            checked_by="llm",
        )
