"""Budget breach must stop the loop cleanly -- it never raises."""

from kurogami.contracts import BudgetLimits
from kurogami.engine.budget import Budget


def test_budget_stops_cleanly_not_crash():
    budget = Budget(BudgetLimits(max_nodes=2))
    assert budget.ok()

    budget.record_node_created("n_001", depth=0, node_goal="g1", parent_ids=[])
    assert budget.ok()

    budget.record_node_created("n_002", depth=1, node_goal="g2", parent_ids=["n_001"])
    assert budget.ok()  # exactly at the limit, not yet over

    budget.record_node_created("n_003", depth=2, node_goal="g3", parent_ids=["n_002"])
    assert not budget.ok()
    assert budget.state.breached is True
    assert "max_nodes" in budget.state.breach_reason


def test_budget_depth_breach():
    budget = Budget(BudgetLimits(max_depth=2))
    budget.record_node_created("n_001", depth=3, node_goal="g", parent_ids=[])
    assert not budget.ok()
    assert "depth" in budget.state.breach_reason


def test_budget_backtrack_breach():
    budget = Budget(BudgetLimits(max_backtracks=1))
    budget.record_backtrack()
    assert budget.ok()
    budget.record_backtrack()
    assert not budget.ok()


def test_budget_token_breach():
    budget = Budget(BudgetLimits(max_tokens_total=100))
    budget.record_tokens(60, 60)
    assert not budget.ok()


def test_budget_node_retry_breach():
    budget = Budget(BudgetLimits(max_node_retries=1))
    budget.record_node_retry("n_004")
    assert budget.ok()
    budget.record_node_retry("n_004")
    assert not budget.ok()


def test_budget_loop_detector_breach_on_identical_signature():
    budget = Budget(BudgetLimits(max_retries_per_signature=1))
    budget.record_node_created("n_a", depth=1, node_goal="same goal", parent_ids=["n_root"])
    assert budget.ok()
    budget.record_node_created("n_b", depth=1, node_goal="same goal", parent_ids=["n_root"])
    assert not budget.ok()


def test_budget_never_raises_once_already_breached():
    budget = Budget(BudgetLimits(max_nodes=0))
    budget.record_node_created("n_001", depth=0, node_goal="g", parent_ids=[])
    budget.record_node_created("n_002", depth=0, node_goal="g2", parent_ids=[])  # would raise if unguarded
    assert not budget.ok()


def test_gap_fill_cap_is_not_a_breach():
    budget = Budget(max_gap_fills=1)
    assert budget.gap_fills_remaining()
    budget.record_gap_fill()
    assert not budget.gap_fills_remaining()
    assert budget.ok()  # growth just switches off; the planned tree still finishes
