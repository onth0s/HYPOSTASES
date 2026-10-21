"""Multiprocessing utility helpers for chess training and evaluation runners."""

from __future__ import annotations

import contextlib
import multiprocessing
import signal
from concurrent.futures import ProcessPoolExecutor


def init_worker_process() -> None:
    """Worker process initializer ignoring SIGINT so only main process handles Ctrl+C cleanly."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def terminate_pool_workers(executor: ProcessPoolExecutor) -> None:
    """Force-kills all pool worker processes so Ctrl+C or crashes leave no orphans.

    Workers are spawned with SIGINT ignored and the runner's SIGINT handler exits via
    os._exit(130), which would otherwise orphan them. The teardown order avoids two
    CPython 3.10 hazards when children are terminated mid-flight:

    1. ``shutdown`` MUST be called first so the executor's manager thread stops
       respawning replacement workers after a child is terminated.
    2. ``cancel_join_thread`` detaches the internal call/result queue feeders from
       interpreter-atexit finalization so they cannot block there.

    Known limitation: on CPython 3.10 a feeder thread blocked in ``send_bytes`` on a
    full pipe (its reader worker just died) can never be joined, and the non-daemon
    ``_ExecutorManagerThread`` will block on ``join_thread`` until interpreter shutdown
    hangs. This function therefore deliberately does NOT join the manager thread; the
    caller MUST terminate the process via ``os._exit`` rather than returning through
    normal interpreter shutdown.
    """
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


# Alias for backward compatibility with private helper naming
_init_worker_process = init_worker_process
_terminate_pool_workers = terminate_pool_workers
