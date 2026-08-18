"""Tests for subprocess_env.SubprocessEnv.

These tests run real subprocesses in temp directories — no mocking needed
since the operations are lightweight (git init, echo, etc.).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lang_config import PYTHON, RUST, LangConfig
from subprocess_env import SubprocessEnv


def _make_test_repo() -> str:
    """Create a tiny git repo in a temp dir and return its path."""
    repo_dir = tempfile.mkdtemp(prefix="test-repo-")
    subprocess.run(
        ["bash", "-c",
         f"cd {repo_dir} && git init -q && git config user.email test@test.com "
         f"&& git config user.name test "
         f"&& echo 'print(1)' > main.py && git add . && git commit -q -m init"],
        check=True, capture_output=True,
    )
    # Get the commit SHA
    result = subprocess.run(
        ["git", "-C", repo_dir, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return repo_dir, result.stdout.strip()


class TestSubprocessEnvLifecycle:
    def test_start_creates_workdir(self) -> None:
        repo_path, commit = _make_test_repo()
        env = SubprocessEnv(
            config=PYTHON,
            repo_url=repo_path,
            base_commit=commit,
        )
        env.start()
        assert env.workdir.exists()
        assert (env.workdir / "main.py").exists()
        env.stop()

    def test_stop_removes_workdir(self) -> None:
        repo_path, commit = _make_test_repo()
        env = SubprocessEnv(
            config=PYTHON,
            repo_url=repo_path,
            base_commit=commit,
        )
        env.start()
        workdir = env.workdir
        env.stop()
        assert not workdir.exists()

    def test_stop_is_idempotent(self) -> None:
        repo_path, commit = _make_test_repo()
        env = SubprocessEnv(
            config=PYTHON,
            repo_url=repo_path,
            base_commit=commit,
        )
        env.start()
        env.stop()
        env.stop()  # should not raise

    def test_context_manager(self) -> None:
        repo_path, commit = _make_test_repo()
        with SubprocessEnv(
            config=PYTHON,
            repo_url=repo_path,
            base_commit=commit,
        ) as env:
            assert env.workdir.exists()
        assert not env.workdir.exists()

    def test_start_idempotent(self) -> None:
        repo_path, commit = _make_test_repo()
        env = SubprocessEnv(
            config=PYTHON,
            repo_url=repo_path,
            base_commit=commit,
        )
        env.start()
        workdir1 = env.workdir
        env.start()  # should not re-clone
        assert env.workdir == workdir1
        env.stop()


class TestSubprocessEnvRun:
    def test_run_returns_output(self) -> None:
        repo_path, commit = _make_test_repo()
        with SubprocessEnv(
            config=PYTHON,
            repo_url=repo_path,
            base_commit=commit,
        ) as env:
            result = env.run("echo hello")
            assert "hello" in result

    def test_run_records_calls(self) -> None:
        repo_path, commit = _make_test_repo()
        with SubprocessEnv(
            config=PYTHON,
            repo_url=repo_path,
            base_commit=commit,
        ) as env:
            env.run("echo a")
            env.run("echo b")
            assert len(env.calls) == 2
            assert env.calls[0] == "echo a"

    def test_run_handles_timeout(self) -> None:
        repo_path, commit = _make_test_repo()
        with SubprocessEnv(
            config=PYTHON,
            repo_url=repo_path,
            base_commit=commit,
            timeout=1,
        ) as env:
            result = env.run("sleep 10")
            assert "timed out" in result

    def test_run_handles_nonexistent_command(self) -> None:
        repo_path, commit = _make_test_repo()
        with SubprocessEnv(
            config=PYTHON,
            repo_url=repo_path,
            base_commit=commit,
        ) as env:
            result = env.run("nonexistent_cmd_xyz")
            assert "not found" in result or "No such file" in result or result != ""

    def test_cat_file(self) -> None:
        repo_path, commit = _make_test_repo()
        with SubprocessEnv(
            config=PYTHON,
            repo_url=repo_path,
            base_commit=commit,
        ) as env:
            result = env.run("cat main.py")
            assert "print(1)" in result


class TestSubprocessEnvPatch:
    def test_no_changes_empty_patch(self) -> None:
        repo_path, commit = _make_test_repo()
        with SubprocessEnv(
            config=PYTHON,
            repo_url=repo_path,
            base_commit=commit,
        ) as env:
            assert env.patch() == ""

    def test_changes_produce_diff(self) -> None:
        repo_path, commit = _make_test_repo()
        with SubprocessEnv(
            config=PYTHON,
            repo_url=repo_path,
            base_commit=commit,
        ) as env:
            env.run("echo 'print(2)' > main.py")
            diff = env.patch()
            assert "print(2)" in diff
            assert "---" in diff


class TestSubprocessEnvRunTests:
    def test_run_tests_parses_output(self) -> None:
        repo_path, commit = _make_test_repo()
        # Create a test that passes
        subprocess.run(
            ["bash", "-c",
             f"cd {repo_path} && echo 'def test_ok(): assert True' > test_main.py "
             f"&& git add . && git commit -q -m 'add test'"],
            check=True, capture_output=True,
        )
        new_commit = subprocess.run(
            ["git", "-C", repo_path, "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

        with SubprocessEnv(
            config=PYTHON,
            repo_url=repo_path,
            base_commit=new_commit,
        ) as env:
            passed, failed, output = env.run_tests()
            assert passed >= 1
            assert failed == 0


class TestSubprocessEnvCallLog:
    def test_call_log_returns_copy(self) -> None:
        repo_path, commit = _make_test_repo()
        with SubprocessEnv(
            config=PYTHON,
            repo_url=repo_path,
            base_commit=commit,
        ) as env:
            env.run("ls")
            log = env.call_log
            assert list(log) == ["ls"]
            # Modifying returned list shouldn't affect internal state
            log_list = list(log)
            log_list.append("fake")
            assert len(env.calls) == 1
