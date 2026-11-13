"""Tests for chess multiprocessing utilities and DomainWorkerPool."""

from __future__ import annotations

from hypostases.plugins.domains.chess.multiprocess_utils import (
    DomainWorkerPool,
    init_worker_process,
)


def _sample_worker_task(x: int) -> int:
    return x * x


def test_domain_worker_pool_lifecycle():
    with DomainWorkerPool(max_workers=2, initializer=init_worker_process) as pool:
        f1 = pool.submit(_sample_worker_task, 4)
        f2 = pool.submit(_sample_worker_task, 7)
        assert f1.result(timeout=5) == 16
        assert f2.result(timeout=5) == 49


def test_init_worker_process():
    # Smoke test to ensure signal handler runs without exception
    init_worker_process()
