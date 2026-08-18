"""Append-only JSONL logger for SWE-RL rollout trajectories.

Each line is a self-contained JSON object capturing one rollout:
messages, patch, commands, reward, and metadata (task, timing, env mode).

Usage::

    logger = RolloutLogger("runs/2024-08-18.jsonl")
    logger.log(rollout, reward=0.75, task_id="django__django-12345",
               rollout_index=0, env_mode="docker")
    logger.close()

Or as a context manager::

    with RolloutLogger("runs/out.jsonl") as logger:
        logger.log(rollout, reward=1.0, task_id="...", rollout_index=0)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any, Mapping


@dataclass(frozen=True)
class RolloutRecord:
    """One logged rollout — serialised as a single JSONL line."""

    task_id: str
    rollout_index: int
    env_mode: str
    reward: float
    patch: str
    calls: list[str]
    messages: list[dict[str, str]]
    timestamp: float
    duration_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class RolloutLogger:
    """Append-only JSONL writer for rollout trajectories.

    Creates parent directories automatically.  Safe to call ``log``
    from multiple threads — writes are serialised through ``_fh.write``
    which holds the GIL for the duration of a single ``write`` call on
    buffered IO.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[str] = open(self._path, "a", encoding="utf-8")
        self._count = 0

    # ── public API ───────────────────────────────────────────────────

    def log(
        self,
        rollout: Mapping[str, Any],
        *,
        reward: float,
        task_id: str,
        rollout_index: int,
        env_mode: str = "mock",
        duration_s: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> RolloutRecord:
        """Append one rollout record and return it."""
        record = RolloutRecord(
            task_id=task_id,
            rollout_index=rollout_index,
            env_mode=env_mode,
            reward=reward,
            patch=rollout.get("patch", ""),
            calls=list(rollout.get("calls", [])),
            messages=list(rollout.get("messages", [])),
            timestamp=time.time(),
            duration_s=duration_s,
            extra=extra or {},
        )
        line = json.dumps(asdict(record), ensure_ascii=False)
        self._fh.write(line + "\n")
        self._fh.flush()
        self._count += 1
        return record

    @property
    def count(self) -> int:
        """Number of records written so far."""
        return self._count

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        """Flush and close the underlying file."""
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    # ── context manager ──────────────────────────────────────────────

    def __enter__(self) -> RolloutLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_rollouts(path: str | Path) -> list[RolloutRecord]:
    """Read all records from a JSONL file.  Useful for post-hoc analysis."""
    records: list[RolloutRecord] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(RolloutRecord(**data))
    return records
