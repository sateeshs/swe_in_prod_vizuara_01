"""Tests for parallel_rollouts module."""

from __future__ import annotations

import threading
import time

import pytest

from parallel_rollouts import RolloutResult, run_rollouts_parallel


def _make_run_one(delay: float = 0.0, reward: float = 0.5):
    """Factory for a simple run_one callable."""

    def run_one(idx: int) -> tuple[dict, float]:
        if delay:
            time.sleep(delay)
        return (
            {
                "messages": [{"role": "user", "content": "fix"}],
                "patch": f"patch-{idx}",
                "calls": ["ls", "cat file.py"],
            },
            reward,
        )

    return run_one


class TestRunRolloutsParallel:
    def test_returns_correct_count(self) -> None:
        results = run_rollouts_parallel(_make_run_one(), group_size=4, max_workers=2)
        assert len(results) == 4

    def test_results_ordered_by_index(self) -> None:
        results = run_rollouts_parallel(_make_run_one(), group_size=6, max_workers=3)
        indices = [r.index for r in results]
        assert indices == list(range(6))

    def test_each_result_has_rollout_data(self) -> None:
        results = run_rollouts_parallel(_make_run_one(), group_size=3, max_workers=2)
        for r in results:
            assert isinstance(r, RolloutResult)
            assert r.rollout["patch"].startswith("patch-")
            assert r.reward == 0.5
            assert r.duration_s >= 0

    def test_parallel_is_faster_than_serial(self) -> None:
        delay = 0.1
        group_size = 4
        serial_min = delay * group_size  # 0.4s if sequential

        t0 = time.monotonic()
        results = run_rollouts_parallel(
            _make_run_one(delay=delay), group_size=group_size, max_workers=group_size
        )
        elapsed = time.monotonic() - t0

        # With full parallelism, should take ~0.1s, not ~0.4s
        assert elapsed < serial_min * 0.8

    def test_max_workers_capped_at_group_size(self) -> None:
        # Should not error when max_workers > group_size
        results = run_rollouts_parallel(
            _make_run_one(), group_size=2, max_workers=100
        )
        assert len(results) == 2

    def test_single_worker_runs_sequentially(self) -> None:
        results = run_rollouts_parallel(
            _make_run_one(), group_size=3, max_workers=1
        )
        assert len(results) == 3

    def test_propagates_exceptions(self) -> None:
        def failing_run_one(idx: int):
            if idx == 2:
                raise RuntimeError("rollout 2 crashed")
            return {"messages": [], "patch": "", "calls": []}, 0.0

        with pytest.raises(RuntimeError, match="rollout 2 crashed"):
            run_rollouts_parallel(failing_run_one, group_size=4, max_workers=4)

    def test_threads_are_concurrent(self) -> None:
        """Verify multiple threads actually run at the same time."""
        active_threads: list[str] = []
        lock = threading.Lock()
        peak_concurrent = [0]

        def tracking_run_one(idx: int):
            tid = threading.current_thread().name
            with lock:
                active_threads.append(tid)
                peak_concurrent[0] = max(
                    peak_concurrent[0], len(set(active_threads))
                )
            time.sleep(0.05)
            return {"messages": [], "patch": "", "calls": []}, 0.0

        run_rollouts_parallel(tracking_run_one, group_size=4, max_workers=4)
        # With 4 workers and sleep, we should see >1 concurrent thread
        assert peak_concurrent[0] > 1

    def test_duration_recorded(self) -> None:
        results = run_rollouts_parallel(
            _make_run_one(delay=0.05), group_size=2, max_workers=2
        )
        for r in results:
            assert r.duration_s >= 0.04  # at least ~50ms
