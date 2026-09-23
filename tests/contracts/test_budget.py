"""Round-trip and default-value tests for BudgetLimits, BudgetState."""

from kurogami.contracts import BudgetLimits, BudgetState


def test_budget_limits_defaults_match_architecture_section_8():
    limits = BudgetLimits()
    assert limits.max_depth == 6
    assert limits.max_nodes == 25
    assert limits.max_backtracks == 3
    assert limits.max_tokens_total == 150_000
    assert limits.max_node_retries == 2


def test_budget_limits_round_trip():
    limits = BudgetLimits(max_depth=3, max_nodes=10)
    restored = BudgetLimits.model_validate_json(limits.model_dump_json())
    assert restored == limits


def test_budget_state_starts_clean():
    state = BudgetState()
    assert state.nodes_created == 0
    assert state.backtracks == 0
    assert state.tokens_used == 0
    assert state.breached is False
    assert state.breach_reason is None


def test_budget_state_round_trip():
    state = BudgetState(
        nodes_created=12,
        backtracks=1,
        tokens_used=4200,
        node_retries={"n_004": 1},
        signature_counts={"abc123": 1},
        breached=True,
        breach_reason="max_backtracks exceeded",
    )
    restored = BudgetState.model_validate_json(state.model_dump_json())
    assert restored == state
