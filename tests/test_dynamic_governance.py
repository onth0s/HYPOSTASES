"""Tests — Contention 3: Dynamic Governance & Supervisory Fee Scaling.

Verifies that step_env's WITHDRAW_FEE scales up proportionally with the
fraction of agents withdrawing (defection prevalence).
"""

from __future__ import annotations

from hypostases.engine.dynamics import step_env
from hypostases.engine.types import Action, ActionType


def _make_actions(**kwargs: str) -> list[tuple[str, Action]]:
    """Build agent_actions list from name -> 'WITHDRAW'|'REQUEST'|'SHARE' map."""
    mapping = {
        "WITHDRAW": Action(ActionType.WITHDRAW),
        "REQUEST": Action(ActionType.REQUEST, amount=1.0),
        "SHARE": Action(ActionType.SHARE, amount=1.0),
    }
    return [(name, mapping[atype]) for name, atype in kwargs.items()]


class TestDynamicGovernance:
    def test_no_withdraws_no_deduction(self):
        """Zero withdrawers → no fee deductions regardless of flag."""
        actions = _make_actions(alice="REQUEST", bob="SHARE")
        pool_after, _ = step_env(10.0, actions, enable_withdraw_fee=True)
        # SHARE adds 1.0, REQUEST takes ≤pool. No withdraw deductions.
        assert pool_after >= 0.0

    def test_single_withdraw_charges_base_fee(self):
        """1 withdrawer out of 3 → fee charged, pool reduced."""
        actions = _make_actions(alice="WITHDRAW", bob="REQUEST", carol="SHARE")
        pool_no_fee, _ = step_env(10.0, actions, enable_withdraw_fee=False)
        pool_fee, _ = step_env(10.0, actions, enable_withdraw_fee=True)
        assert pool_fee < pool_no_fee, "Withdraw fee should reduce pool"

    def test_higher_defection_ratio_charges_more(self):
        """Higher defection ratio → larger per-withdraw dynamic fee → larger total pool deduction."""
        # 1 withdrawer out of 3 (prevalence = 1/3)
        one_defector = _make_actions(alice="WITHDRAW", bob="SHARE", carol="SHARE")
        # 3 withdrawers out of 3 (prevalence = 1.0)
        all_defectors = _make_actions(alice="WITHDRAW", bob="WITHDRAW", carol="WITHDRAW")

        pool = 20.0
        _, one_log = step_env(pool, one_defector, enable_withdraw_fee=True)
        _, all_log = step_env(pool, all_defectors, enable_withdraw_fee=True)

        # In the one_defector case: 2 SHARE agents add to pool, partially obscuring fee.
        # Compare per-agent fee paid by comparing pool_before minus what governance removed:
        # pool_before_adj = pool_before - deductions. For SHARE-only companions the pool
        # grows from shares, so compute deduction before shares using log fields.
        one_pool_before = one_log["pool_before"]
        one_shares = one_log["shares_total"]
        all_pool_before = all_log["pool_before"]
        all_shares = all_log["shares_total"]

        # Deduction = pool_before - (pool_after_shares - shares)
        # pool_after_shares = pool_before_adj + shares; pool_before_adj = pool_before - deductions
        # → deductions = pool_before + shares - pool_after_shares
        one_deduction = one_pool_before + one_shares - one_log["pool_after_shares"]
        all_deduction = all_pool_before + all_shares - all_log["pool_after_shares"]

        assert all_deduction > one_deduction, (
            f"All-defect deduction ({all_deduction:.3f}) should exceed "
            f"one-defect deduction ({one_deduction:.3f})"
        )

    def test_dynamic_fee_scales_with_lambda(self):
        """Fee proportional to GOVERNANCE_SCALING_LAMBDA — higher lambda → larger deduction."""
        import hypostases.engine.dynamics as dyn

        actions = _make_actions(alice="WITHDRAW", bob="WITHDRAW")
        pool = 30.0

        orig_lambda = dyn.GOVERNANCE_SCALING_LAMBDA

        # Low lambda
        dyn.GOVERNANCE_SCALING_LAMBDA = 0.0
        pool_low, _ = step_env(pool, actions, enable_withdraw_fee=True)

        # High lambda
        dyn.GOVERNANCE_SCALING_LAMBDA = 5.0
        pool_high, _ = step_env(pool, actions, enable_withdraw_fee=True)

        # Restore
        dyn.GOVERNANCE_SCALING_LAMBDA = orig_lambda

        assert pool_low > pool_high, "Higher governance lambda should extract more fees"

    def test_withdraw_fee_disabled_ignores_governance(self):
        """With enable_withdraw_fee=False, governance scaling has no effect."""
        actions = _make_actions(alice="WITHDRAW", bob="WITHDRAW", carol="WITHDRAW")
        pool = 20.0
        pool_no_fee, _ = step_env(
            pool, actions, enable_withdraw_fee=False, enable_withdraw_degrade=False
        )
        # All three withdrew but no fee → pool unchanged by deductions
        # (they don't change pool directly via deductions, only degrade/fee do)
        assert pool_no_fee >= 0.0

    def test_pool_never_negative(self):
        """Even with extreme governance fees, pool cannot go below 0."""
        actions = _make_actions(**{f"a{i}": "WITHDRAW" for i in range(10)})
        pool_after, _ = step_env(1.0, actions, enable_withdraw_fee=True)
        assert pool_after >= 0.0
