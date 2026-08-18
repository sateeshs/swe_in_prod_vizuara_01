"""Tests for docker_env.DockerEnv.

Docker calls are mocked — these tests verify the orchestration logic,
not that Docker itself works.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from docker_env import DockerEnv, _WORKDIR
from lang_config import PYTHON, RUST


def _make_env(config=PYTHON, **kwargs) -> DockerEnv:
    """Create a DockerEnv without starting it."""
    return DockerEnv(
        config=config,
        repo_url="https://github.com/test/repo.git",
        base_commit="abc123",
        **kwargs,
    )


class TestLifecycle:
    @patch.object(DockerEnv, "_docker")
    @patch.object(DockerEnv, "_exec")
    def test_start_creates_container(self, mock_exec: MagicMock, mock_docker: MagicMock) -> None:
        mock_docker.return_value = "container123\n"
        env = _make_env()
        env.start()

        # Should call docker run
        mock_docker.assert_called_once()
        args = mock_docker.call_args
        assert "run" in args[0]
        assert PYTHON.docker_image in args[0]

    @patch.object(DockerEnv, "_docker")
    @patch.object(DockerEnv, "_exec")
    def test_start_clones_and_checkouts(self, mock_exec: MagicMock, mock_docker: MagicMock) -> None:
        mock_docker.return_value = "container123\n"
        env = _make_env()
        env.start()

        exec_calls = [c.args[0] for c in mock_exec.call_args_list]
        assert any("git clone" in c for c in exec_calls)
        assert any("git checkout" in c and "abc123" in c for c in exec_calls)

    @patch.object(DockerEnv, "_docker")
    @patch.object(DockerEnv, "_exec")
    def test_start_applies_test_patch(self, mock_exec: MagicMock, mock_docker: MagicMock) -> None:
        mock_docker.return_value = "container123\n"
        env = _make_env(test_patch="--- a/file\n+++ b/file\n@@ -1 +1 @@\n-old\n+new")
        env.start()

        exec_calls = [c.args[0] for c in mock_exec.call_args_list]
        assert any("git apply" in c for c in exec_calls)

    @patch.object(DockerEnv, "_docker")
    @patch.object(DockerEnv, "_exec")
    def test_start_skips_test_patch_when_empty(self, mock_exec: MagicMock, mock_docker: MagicMock) -> None:
        mock_docker.return_value = "container123\n"
        env = _make_env(test_patch="")
        env.start()

        exec_calls = [c.args[0] for c in mock_exec.call_args_list]
        assert not any("git apply" in c for c in exec_calls)

    @patch.object(DockerEnv, "_docker")
    @patch.object(DockerEnv, "_exec")
    def test_stop_removes_container(self, mock_exec: MagicMock, mock_docker: MagicMock) -> None:
        mock_docker.return_value = "container123\n"
        env = _make_env()
        env.start()

        mock_docker.reset_mock()
        env.stop()

        mock_docker.assert_called_once()
        args = mock_docker.call_args[0]
        assert "rm" in args
        assert "-f" in args

    @patch.object(DockerEnv, "_docker")
    @patch.object(DockerEnv, "_exec")
    def test_stop_is_idempotent(self, mock_exec: MagicMock, mock_docker: MagicMock) -> None:
        mock_docker.return_value = "container123\n"
        env = _make_env()
        env.start()

        env.stop()
        env.stop()  # should not raise

    @patch.object(DockerEnv, "_docker")
    @patch.object(DockerEnv, "_exec")
    def test_context_manager(self, mock_exec: MagicMock, mock_docker: MagicMock) -> None:
        mock_docker.return_value = "container123\n"
        with _make_env() as env:
            assert env._container_id == "container123"
        assert env._container_id == ""


class TestRun:
    @patch.object(DockerEnv, "_docker")
    @patch.object(DockerEnv, "_exec")
    def test_run_records_call(self, mock_exec: MagicMock, mock_docker: MagicMock) -> None:
        mock_docker.return_value = "cid\n"
        env = _make_env()
        env.start()

        mock_exec.reset_mock()
        mock_exec.return_value = "output"
        result = env.run("ls -la")

        assert "ls -la" in env.calls
        assert result == "output"

    @patch.object(DockerEnv, "_docker")
    @patch.object(DockerEnv, "_exec")
    def test_run_handles_timeout(self, mock_exec: MagicMock, mock_docker: MagicMock) -> None:
        mock_docker.return_value = "cid\n"
        env = _make_env(timeout=5)
        env.start()

        mock_exec.reset_mock()
        mock_exec.side_effect = [TimeoutError(), None]
        # Manually simulate what run() does on SubprocessError
        import subprocess
        mock_exec.side_effect = subprocess.TimeoutExpired("cmd", 5)
        result = env.run("sleep 999")

        assert "timed out" in result

    @patch.object(DockerEnv, "_docker")
    @patch.object(DockerEnv, "_exec")
    def test_run_handles_error(self, mock_exec: MagicMock, mock_docker: MagicMock) -> None:
        mock_docker.return_value = "cid\n"
        env = _make_env()
        env.start()

        mock_exec.reset_mock()
        import subprocess
        mock_exec.side_effect = subprocess.SubprocessError("container crashed")
        result = env.run("bad command")

        assert "container crashed" in result


class TestPatch:
    @patch.object(DockerEnv, "_docker")
    @patch.object(DockerEnv, "_exec")
    def test_patch_returns_git_diff(self, mock_exec: MagicMock, mock_docker: MagicMock) -> None:
        mock_docker.return_value = "cid\n"
        env = _make_env()
        env.start()

        mock_exec.reset_mock()
        mock_exec.return_value = "--- a/file\n+++ b/file\n"
        diff = env.patch()

        mock_exec.assert_called_with("git diff")
        assert "---" in diff

    @patch.object(DockerEnv, "_docker")
    @patch.object(DockerEnv, "_exec")
    def test_patch_returns_empty_on_error(self, mock_exec: MagicMock, mock_docker: MagicMock) -> None:
        mock_docker.return_value = "cid\n"
        env = _make_env()
        env.start()

        mock_exec.reset_mock()
        import subprocess
        mock_exec.side_effect = subprocess.SubprocessError("no git")
        assert env.patch() == ""


class TestMultiLanguage:
    @patch.object(DockerEnv, "_docker")
    @patch.object(DockerEnv, "_exec")
    def test_rust_uses_rust_image(self, mock_exec: MagicMock, mock_docker: MagicMock) -> None:
        mock_docker.return_value = "cid\n"
        env = _make_env(config=RUST)
        env.start()

        args = mock_docker.call_args[0]
        assert RUST.docker_image in args

    @patch.object(DockerEnv, "_docker")
    @patch.object(DockerEnv, "_exec")
    def test_setup_commands_run(self, mock_exec: MagicMock, mock_docker: MagicMock) -> None:
        from lang_config import LangConfig
        custom = LangConfig(
            name="custom",
            docker_image="ubuntu:22.04",
            build_cmd="make",
            test_cmd="make test",
            file_extensions=(".c",),
            useful_commands=("ls",),
            setup_commands=("apt-get update", "apt-get install -y gcc"),
        )
        mock_docker.return_value = "cid\n"
        env = _make_env(config=custom)
        env.start()

        exec_calls = [c.args[0] for c in mock_exec.call_args_list]
        assert "apt-get update" in exec_calls
        assert "apt-get install -y gcc" in exec_calls
