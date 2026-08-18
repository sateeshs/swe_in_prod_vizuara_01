"""Tests for rollout_logger module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from rollout_logger import RolloutLogger, RolloutRecord, read_rollouts


def _sample_rollout(patch: str = "--- a/f\n+++ b/f") -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Fix the bug."},
            {"role": "assistant", "content": "```bash\nls\n```"},
        ],
        "patch": patch,
        "calls": ["ls"],
    }


class TestRolloutLogger:
    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        log_path = tmp_path / "deep" / "nested" / "run.jsonl"
        logger = RolloutLogger(log_path)
        logger.close()
        assert log_path.parent.exists()

    def test_writes_valid_jsonl(self, tmp_path: Path) -> None:
        log_path = tmp_path / "out.jsonl"
        with RolloutLogger(log_path) as logger:
            logger.log(
                _sample_rollout(),
                reward=0.75,
                task_id="django__django-12345",
                rollout_index=0,
                env_mode="docker",
            )

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["reward"] == 0.75
        assert record["task_id"] == "django__django-12345"
        assert record["env_mode"] == "docker"

    def test_appends_multiple_records(self, tmp_path: Path) -> None:
        log_path = tmp_path / "out.jsonl"
        with RolloutLogger(log_path) as logger:
            for i in range(3):
                logger.log(
                    _sample_rollout(),
                    reward=float(i) / 2,
                    task_id="task-1",
                    rollout_index=i,
                )

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 3

    def test_count_tracks_writes(self, tmp_path: Path) -> None:
        log_path = tmp_path / "out.jsonl"
        with RolloutLogger(log_path) as logger:
            assert logger.count == 0
            logger.log(_sample_rollout(), reward=1.0, task_id="t", rollout_index=0)
            assert logger.count == 1
            logger.log(_sample_rollout(), reward=0.5, task_id="t", rollout_index=1)
            assert logger.count == 2

    def test_returns_record(self, tmp_path: Path) -> None:
        log_path = tmp_path / "out.jsonl"
        with RolloutLogger(log_path) as logger:
            record = logger.log(
                _sample_rollout(),
                reward=0.5,
                task_id="t",
                rollout_index=0,
                env_mode="mock",
                duration_s=1.23,
            )
        assert isinstance(record, RolloutRecord)
        assert record.reward == 0.5
        assert record.duration_s == 1.23
        assert record.env_mode == "mock"

    def test_extra_metadata(self, tmp_path: Path) -> None:
        log_path = tmp_path / "out.jsonl"
        with RolloutLogger(log_path) as logger:
            logger.log(
                _sample_rollout(),
                reward=1.0,
                task_id="t",
                rollout_index=0,
                extra={"model": "qwen-0.5b", "temperature": 1.0},
            )

        record = json.loads(log_path.read_text().strip())
        assert record["extra"]["model"] == "qwen-0.5b"
        assert record["extra"]["temperature"] == 1.0

    def test_timestamp_populated(self, tmp_path: Path) -> None:
        log_path = tmp_path / "out.jsonl"
        with RolloutLogger(log_path) as logger:
            logger.log(_sample_rollout(), reward=0.0, task_id="t", rollout_index=0)

        record = json.loads(log_path.read_text().strip())
        assert record["timestamp"] > 0

    def test_appends_to_existing_file(self, tmp_path: Path) -> None:
        log_path = tmp_path / "out.jsonl"
        # First session
        with RolloutLogger(log_path) as logger:
            logger.log(_sample_rollout(), reward=0.1, task_id="t", rollout_index=0)
        # Second session appends
        with RolloutLogger(log_path) as logger:
            logger.log(_sample_rollout(), reward=0.9, task_id="t", rollout_index=1)

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_empty_patch(self, tmp_path: Path) -> None:
        log_path = tmp_path / "out.jsonl"
        with RolloutLogger(log_path) as logger:
            logger.log(
                _sample_rollout(patch=""),
                reward=0.0,
                task_id="t",
                rollout_index=0,
            )
        record = json.loads(log_path.read_text().strip())
        assert record["patch"] == ""

    def test_path_property(self, tmp_path: Path) -> None:
        log_path = tmp_path / "out.jsonl"
        logger = RolloutLogger(log_path)
        assert logger.path == log_path
        logger.close()


class TestReadRollouts:
    def test_round_trips(self, tmp_path: Path) -> None:
        log_path = tmp_path / "out.jsonl"
        with RolloutLogger(log_path) as logger:
            logger.log(_sample_rollout(), reward=0.5, task_id="t1", rollout_index=0)
            logger.log(_sample_rollout(), reward=1.0, task_id="t1", rollout_index=1)

        records = read_rollouts(log_path)
        assert len(records) == 2
        assert records[0].reward == 0.5
        assert records[1].reward == 1.0
        assert all(isinstance(r, RolloutRecord) for r in records)

    def test_empty_file(self, tmp_path: Path) -> None:
        log_path = tmp_path / "empty.jsonl"
        log_path.write_text("")
        assert read_rollouts(log_path) == []
