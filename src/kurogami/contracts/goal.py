"""GoalSpec: the structured form of the user's sentence. Level-1 output."""

from typing import Literal

from pydantic import BaseModel


class GoalSpec(BaseModel):
    """Level-1 output. The structured form of the user's sentence."""

    raw_text: str
    product_description: str
    target_market: str
    decision_type: Literal["market_entry", "positioning", "pricing", "launch"]
    success_definition: str
    known_constraints: list[str] = []
    ambiguities: list[str] = []
