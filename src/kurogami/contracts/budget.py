"""BudgetLimits / BudgetState: the six safety rails from ARCHITECTURE.md §8."""

from pydantic import BaseModel


class BudgetLimits(BaseModel):
    """Every limit is a constructor argument; defaults from ARCHITECTURE.md §8."""

    max_depth: int = 6
    max_nodes: int = 25
    max_backtracks: int = 3
    max_tokens_total: int = 150_000
    max_node_retries: int = 2
    max_retries_per_signature: int = 2
    """Loop-detector cap: same (node_goal, parent_ids) signature re-expanded this many times."""


class BudgetState(BaseModel):
    """Running counters checked against BudgetLimits on every loop iteration."""

    nodes_created: int = 0
    backtracks: int = 0
    tokens_used: int = 0
    node_retries: dict[str, int] = {}
    signature_counts: dict[str, int] = {}
    breached: bool = False
    breach_reason: str | None = None
