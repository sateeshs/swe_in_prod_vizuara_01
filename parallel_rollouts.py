"""Parallel rollout execution for SWE-RL GRPO training.

Runs multiple agent rollouts concurrently using ``ThreadPoolExecutor``.
Each rollout gets its own Docker container (or MockEnv) so there is no
shared mutable state between workers.

We use threads (not processes) because the bottleneck is Docker I/O and
model inference — both of which release the GIL.  The model and tokenizer
are read-only during generation so sharing them across threads is safe.

Usage::

    results = run_rollouts_parallel(
        model, tok, inst, fail_to_pass,
        group_size=6, max_workers=4,
        env_factory=lambda: DockerEnv(...),
    )
    # results: list[RolloutResult]
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from lang_config import EnvProtocol


@dataclass(frozen=True)
class RolloutResult:
    """Output of one parallel rollout."""

    index: int
    rollout: dict[str, Any]
    reward: float
    duration_s: float


def run_rollouts_parallel(
    run_one: Callable[[int], tuple[dict[str, Any], float]],
    group_size: int,
    max_workers: int = 4,
) -> list[RolloutResult]:
    """Execute *group_size* rollouts in parallel.

    Parameters
    ----------
    run_one : callable(index) -> (rollout_dict, reward)
        A function that runs a single rollout and returns the rollout dict
        and its reward.  Called once per rollout index.  Must be thread-safe
        (each call should create its own env).
    group_size : int
        Number of rollouts to run.
    max_workers : int
        Maximum concurrent rollouts.  Capped at *group_size*.

    Returns
    -------
    list[RolloutResult]
        Results ordered by rollout index (0 .. group_size-1).
    """
    effective_workers = min(max_workers, group_size)
    results: dict[int, RolloutResult] = {}

    def _worker(idx: int) -> RolloutResult:
        t0 = time.monotonic()
        rollout, reward = run_one(idx)
        elapsed = round(time.monotonic() - t0, 3)
        return RolloutResult(
            index=idx,
            rollout=rollout,
            reward=reward,
            duration_s=elapsed,
        )

    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
        futures = {pool.submit(_worker, i): i for i in range(group_size)}
        for future in as_completed(futures):
            result = future.result()  # propagates exceptions
            results[result.index] = result
            print(
                f"  rollout {result.index}: "
                f"{len(result.rollout.get('calls', []))} commands, "
                f"patch {'yes' if result.rollout.get('patch') else 'no'}, "
                f"reward {result.reward}, "
                f"{result.duration_s:.1f}s"
            )

    # Return in index order
    return [results[i] for i in range(group_size)]
