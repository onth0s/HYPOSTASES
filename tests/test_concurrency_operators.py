"""Unit tests for engine concurrency operator resolvers."""

from hypostases.engine.dynamics import (
    _resolve_lottery,
    _resolve_priority,
    _resolve_pro_rata,
    _resolve_shares_first,
)


def test_resolve_shares_first_under_capacity():
    requests = [("A", 4.0), ("B", 3.0)]
    pool_final, pool_after_shares, granted = _resolve_shares_first(
        pool_before_adj=10.0, shares_total=2.0, requests=requests
    )
    assert pool_after_shares == 12.0
    assert granted == {"A": 4.0, "B": 3.0}
    assert pool_final == 5.0


def test_resolve_shares_first_oversubscribed():
    requests = [("A", 10.0), ("B", 10.0)]
    pool_final, pool_after_shares, granted = _resolve_shares_first(
        pool_before_adj=5.0, shares_total=5.0, requests=requests
    )
    assert pool_after_shares == 10.0
    assert granted == {"A": 5.0, "B": 5.0}
    assert pool_final == 0.0


def test_resolve_pro_rata_does_not_use_step_shares_for_requests():
    requests = [("A", 10.0)]
    pool_final, pool_after_shares, granted = _resolve_pro_rata(
        pool_before_adj=5.0, shares_total=4.0, requests=requests
    )
    assert pool_after_shares == 5.0
    assert granted == {"A": 5.0}
    assert pool_final == 4.0


def test_resolve_priority_respects_ordering():
    requests = [("A", 8.0), ("B", 8.0)]
    priorities = {"A": 1.0, "B": 5.0}  # B has higher priority
    pool_final, _pool_after_shares, granted = _resolve_priority(
        pool_before_adj=10.0, shares_total=0.0, requests=requests, priorities=priorities
    )
    assert granted["B"] == 8.0
    assert granted["A"] == 2.0
    assert pool_final == 0.0


def test_resolve_lottery_deterministic_with_rng(rng):
    requests = [("A", 6.0), ("B", 6.0)]
    pool_final, _pool_after_shares, granted = _resolve_lottery(
        pool_before_adj=8.0, shares_total=0.0, requests=requests, rng=rng
    )
    assert sum(granted.values()) == 8.0
    assert pool_final == 0.0
