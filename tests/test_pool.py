"""The spawn-based worker pool: results in order, progress logging, the hung-pool watchdog, and the entry-point guard."""

import runpy
import sys

from c44 import pipeline


def _double(x):
    return x * 2


def _hang_in_child(x):
    """Hangs in every worker PROCESS (as a deadlocked forked worker would) but works in the main process's threads."""
    import multiprocessing as mp
    import time
    if mp.current_process().name != "MainProcess":
        time.sleep(120)
    return x * 2


def test_run_pool_returns_results_in_order_and_logs_progress():
    logs = []
    out = pipeline.run_pool(_double, list(range(25)), n=3, log=logs.append, label="t", every=10, timeout=120)
    assert out == [2 * i for i in range(25)]
    assert any("[t] 20/25" in m for m in logs) and any("[t] 25/25" in m for m in logs)


def test_run_pool_watchdog_finishes_a_hung_pool_in_threads():
    logs, seen = [], []
    out = pipeline.run_pool(_hang_in_child, list(range(6)), n=2, log=logs.append, label="hang", timeout=4, on_result=seen.append)
    assert out == [2 * i for i in range(6)] and sorted(seen) == out
    assert any("looks hung" in m for m in logs)


def _slow(x):
    import time
    time.sleep(3)
    return x + 1


def test_run_pool_logs_a_heartbeat_while_waiting_and_does_not_fall_back(monkeypatch):
    monkeypatch.setattr(pipeline, "POOL_HEARTBEAT", 1.0)
    logs = []
    out = pipeline.run_pool(_slow, [1, 2], n=2, log=logs.append, label="hb", timeout=30)
    assert out == [2, 3]
    assert any("still working" in m for m in logs) and not any("looks hung" in m for m in logs)


def test_entry_point_does_not_run_the_cli_when_imported_by_a_worker(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["c44", "robust", "--n", "1"])
    runpy.run_module("c44.__main__", run_name="__mp_main__")          # unguarded, this would parse argv and start the stage
