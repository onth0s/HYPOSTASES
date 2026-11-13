"""Domain worker pool context manager and helpers for isolated game dispatch."""

from __future__ import annotations

import contextlib
import multiprocessing
import signal
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any


def init_worker_process() -> None:
    """Worker process initializer ignoring SIGINT so only main process handles Ctrl+C cleanly."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def terminate_pool_workers(executor: ProcessPoolExecutor) -> None:
    """Force-kills all pool worker processes so Ctrl+C or crashes leave no orphans."""
    with contextlib.suppress(Exception):
        executor.shutdown(wait=False, cancel_futures=True)
    for attr in ("_call_queue", "_result_queue"):
        queue = getattr(executor, attr, None)
        if queue is not None:
            with contextlib.suppress(Exception):
                queue.cancel_join_thread()
    with contextlib.suppress(Exception):
        for p in multiprocessing.active_children():
            p.terminate()


class DomainWorkerPool:
    """Managed ProcessPoolExecutor wrapper with worker initialization and cleanup."""

    def __init__(
        self,
        max_workers: int | None = None,
        initializer: Callable[[], None] = init_worker_process,
    ) -> None:
        self.max_workers = max_workers
        self.initializer = initializer
        self.executor: ProcessPoolExecutor | None = None

    def __enter__(self) -> DomainWorkerPool:
        self.executor = ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=self.initializer,
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.executor is not None:
            if exc_type is not None:
                terminate_pool_workers(self.executor)
            else:
                self.executor.shutdown(wait=True)
            self.executor = None

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        """Submit a task to the managed worker pool."""
        if self.executor is None:
            raise RuntimeError("DomainWorkerPool is not running. Use as a context manager.")
        return self.executor.submit(fn, *args, **kwargs)

    def terminate(self) -> None:
        """Immediately terminate all child worker processes."""
        if self.executor is not None:
            terminate_pool_workers(self.executor)
            self.executor = None


# Aliases for backward compatibility
_init_worker_process = init_worker_process
_terminate_pool_workers = terminate_pool_workers
