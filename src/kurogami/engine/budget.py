"""The six safety rails from ARCHITECTURE.md section 8. A breach is an outcome, not a crash."""

import hashlib

from kurogami.contracts import BudgetLimits, BudgetState


def signature(node_goal: str, parent_ids: list[str]) -> str:
    """Loop-detector signature: hash(node_goal + parent_ids), per ARCHITECTURE.md section 8."""
    payload = node_goal + "|" + ",".join(sorted(parent_ids))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class Budget:
    """Tracks BudgetState against BudgetLimits. Never raises; callers check ok()."""

    def __init__(self, limits: BudgetLimits | None = None, *, max_gap_fills: int = 2) -> None:
        self.limits = limits or BudgetLimits()
        self.state = BudgetState()
        # Lives here, not in the frozen BudgetLimits contract. Exhausting it is not a
        # breach: runtime growth just switches off and the planned tree finishes.
        self.max_gap_fills = max_gap_fills
        self.gap_fills_used = 0

    def ok(self) -> bool:
        return not self.state.breached

    def gap_fills_remaining(self) -> bool:
        return self.gap_fills_used < self.max_gap_fills

    def record_gap_fill(self) -> None:
        self.gap_fills_used += 1

    def record_node_created(
        self, node_id: str, *, depth: int, node_goal: str, parent_ids: list[str]
    ) -> None:
        self.state.nodes_created += 1
        sig = signature(node_goal, parent_ids)
        self.state.signature_counts[sig] = self.state.signature_counts.get(sig, 0) + 1

        if depth > self.limits.max_depth:
            self._breach(f"node {node_id} at depth {depth} exceeds max_depth {self.limits.max_depth}")
        elif self.state.nodes_created > self.limits.max_nodes:
            self._breach(
                f"node count {self.state.nodes_created} exceeds max_nodes {self.limits.max_nodes}"
            )
        elif self.state.signature_counts[sig] > self.limits.max_retries_per_signature:
            self._breach(
                f"node {node_id} re-expanded with an identical signature "
                f"more than max_retries_per_signature ({self.limits.max_retries_per_signature}); "
                "loop detected"
            )

    def record_backtrack(self) -> None:
        self.state.backtracks += 1
        if self.state.backtracks > self.limits.max_backtracks:
            self._breach(
                f"backtracks {self.state.backtracks} exceed max_backtracks {self.limits.max_backtracks}"
            )

    def record_tokens(self, tokens_in: int, tokens_out: int) -> None:
        self.state.tokens_used += tokens_in + tokens_out
        if self.state.tokens_used > self.limits.max_tokens_total:
            self._breach(
                f"tokens used {self.state.tokens_used} exceed "
                f"max_tokens_total {self.limits.max_tokens_total}"
            )

    def record_node_retry(self, node_id: str) -> None:
        count = self.state.node_retries.get(node_id, 0) + 1
        self.state.node_retries[node_id] = count
        if count > self.limits.max_node_retries:
            self._breach(
                f"node {node_id} retried {count} times, exceeding "
                f"max_node_retries {self.limits.max_node_retries}"
            )

    def _breach(self, reason: str) -> None:
        self.state.breached = True
        self.state.breach_reason = reason
