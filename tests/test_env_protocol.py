"""Tests for EnvProtocol, NodeJS config, reward_test, and parse helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from docker_env import DockerEnv, _parse_test_counts, reward_test
from lang_config import (
    NODEJS,
    PYTHON,
    RUST,
    EnvProtocol,
    get_config,
    list_languages,
)
from utils import MockEnv


# ── EnvProtocol conformance ─────────────────────────────────────────


class TestEnvProtocolConformance:
    """Both MockEnv and DockerEnv must satisfy EnvProtocol."""

    def test_mock_env_satisfies_protocol(self) -> None:
        env: EnvProtocol = MockEnv(["test::one"])
        assert hasattr(env, "run")
        assert hasattr(env, "patch")
        assert hasattr(env, "calls")

    def test_mock_env_run_returns_str(self) -> None:
        env: EnvProtocol = MockEnv(["test::one"])
        result = env.run("ls")
        assert isinstance(result, str)

    def test_mock_env_patch_returns_str(self) -> None:
        env: EnvProtocol = MockEnv(["test::one"])
        result = env.patch()
        assert isinstance(result, str)

    def test_mock_env_calls_is_list(self) -> None:
        env: EnvProtocol = MockEnv(["test::one"])
        env.run("ls")
        assert isinstance(env.calls, list)
        assert len(env.calls) == 1

    def test_docker_env_has_protocol_methods(self) -> None:
        env = DockerEnv(
            config=PYTHON,
            repo_url="https://github.com/test/repo.git",
            base_commit="abc123",
        )
        assert hasattr(env, "run")
        assert hasattr(env, "patch")
        assert hasattr(env, "calls")


# ── NodeJS config ───────────────────────────────────────────────────


class TestNodeJSConfig:
    def test_nodejs_by_name(self) -> None:
        assert get_config("nodejs") is NODEJS

    def test_nodejs_alias_node(self) -> None:
        assert get_config("node") is NODEJS

    def test_nodejs_alias_javascript(self) -> None:
        assert get_config("javascript") is NODEJS

    def test_nodejs_alias_js(self) -> None:
        assert get_config("js") is NODEJS

    def test_nodejs_alias_typescript(self) -> None:
        assert get_config("typescript") is NODEJS

    def test_nodejs_alias_ts(self) -> None:
        assert get_config("ts") is NODEJS

    def test_nodejs_image_version(self) -> None:
        assert "22.13.1" in NODEJS.docker_image

    def test_nodejs_test_cmd(self) -> None:
        assert NODEJS.test_cmd == "npm test"

    def test_nodejs_build_cmd(self) -> None:
        assert NODEJS.build_cmd == "npm run build"

    def test_nodejs_extensions(self) -> None:
        assert ".js" in NODEJS.file_extensions
        assert ".ts" in NODEJS.file_extensions

    def test_nodejs_setup_installs_deps(self) -> None:
        assert "npm install" in NODEJS.setup_commands

    def test_nodejs_in_list_languages(self) -> None:
        assert "nodejs" in list_languages()

    def test_nodejs_useful_commands_include_jest(self) -> None:
        assert any("jest" in cmd for cmd in NODEJS.useful_commands)


# ── _parse_test_counts ──────────────────────────────────────────────


class TestParseTestCounts:
    def test_pytest_output(self) -> None:
        output = "===== 3 passed, 2 failed in 1.23s ====="
        assert _parse_test_counts(output) == (3, 2)

    def test_pytest_all_pass(self) -> None:
        output = "===== 5 passed in 0.5s ====="
        assert _parse_test_counts(output) == (5, 0)

    def test_pytest_all_fail(self) -> None:
        output = "===== 4 failed in 2.1s ====="
        assert _parse_test_counts(output) == (0, 4)

    def test_jest_output(self) -> None:
        output = "Tests:  3 passed, 1 failed, 4 total"
        assert _parse_test_counts(output) == (3, 1)

    def test_cargo_output(self) -> None:
        output = "test result: ok. 10 passed; 0 failed; 0 ignored"
        assert _parse_test_counts(output) == (10, 0)

    def test_no_test_output(self) -> None:
        output = "some random output"
        assert _parse_test_counts(output) == (0, 0)

    def test_empty_output(self) -> None:
        assert _parse_test_counts("") == (0, 0)


# ── reward_test ─────────────────────────────────────────────────────


def _make_started_env(**kwargs) -> DockerEnv:
    """Create a DockerEnv with a fake container id (skip start)."""
    env = DockerEnv(
        config=PYTHON,
        repo_url="https://github.com/test/repo.git",
        base_commit="abc123",
        **kwargs,
    )
    env._container_id = "fake-container"
    return env


class TestRewardTest:
    @patch.object(DockerEnv, "_exec")
    def test_returns_zero_when_no_patch(self, mock_exec: MagicMock) -> None:
        env = _make_started_env()
        mock_exec.return_value = ""  # git diff returns empty
        score = reward_test(env, ["test::one", "test::two"])
        assert score == 0.0

    @patch.object(DockerEnv, "_exec")
    def test_returns_one_when_all_pass(self, mock_exec: MagicMock) -> None:
        env = _make_started_env()
        mock_exec.side_effect = [
            "--- a/f\n+++ b/f\n",  # git diff (non-empty patch)
            "===== 2 passed in 0.5s =====",  # test run
        ]
        score = reward_test(env, ["test::one", "test::two"])
        assert score == 1.0

    @patch.object(DockerEnv, "_exec")
    def test_returns_partial_credit(self, mock_exec: MagicMock) -> None:
        env = _make_started_env()
        mock_exec.side_effect = [
            "--- a/f\n+++ b/f\n",  # git diff
            "===== 1 passed, 1 failed in 0.5s =====",  # test run
        ]
        score = reward_test(env, ["test::one", "test::two"])
        assert score == 0.5

    @patch.object(DockerEnv, "_exec")
    def test_returns_zero_when_all_fail(self, mock_exec: MagicMock) -> None:
        env = _make_started_env()
        mock_exec.side_effect = [
            "--- a/f\n+++ b/f\n",  # git diff
            "===== 3 failed in 0.5s =====",  # test run
        ]
        score = reward_test(env, ["test::one", "test::two", "test::three"])
        assert score == 0.0

    @patch.object(DockerEnv, "_exec")
    def test_returns_zero_when_no_test_output(self, mock_exec: MagicMock) -> None:
        env = _make_started_env()
        mock_exec.side_effect = [
            "--- a/f\n+++ b/f\n",  # git diff
            "error: could not run tests",  # unparseable output
        ]
        score = reward_test(env, ["test::one"])
        assert score == 0.0


# ── DockerEnv.run_tests ─────────────────────────────────────────────


class TestRunTests:
    @patch.object(DockerEnv, "_exec")
    def test_runs_default_test_cmd(self, mock_exec: MagicMock) -> None:
        env = _make_started_env()
        mock_exec.return_value = "===== 3 passed in 0.5s ====="
        passed, failed, output = env.run_tests()

        mock_exec.assert_called_once_with("python -m pytest")
        assert passed == 3
        assert failed == 0

    @patch.object(DockerEnv, "_exec")
    def test_runs_specific_tests(self, mock_exec: MagicMock) -> None:
        env = _make_started_env()
        mock_exec.return_value = "===== 1 passed, 1 failed ====="
        passed, failed, output = env.run_tests(["test::a", "test::b"])

        call_cmd = mock_exec.call_args[0][0]
        assert "test::a" in call_cmd
        assert "test::b" in call_cmd
        assert passed == 1
        assert failed == 1


# ── DockerEnv clone fix ─────────────────────────────────────────────


class TestCloneFix:
    @patch.object(DockerEnv, "_docker")
    @patch.object(DockerEnv, "_exec")
    def test_clones_to_tmp_then_copies(
        self, mock_exec: MagicMock, mock_docker: MagicMock
    ) -> None:
        mock_docker.return_value = "container123\n"
        env = DockerEnv(
            config=PYTHON,
            repo_url="https://github.com/test/repo.git",
            base_commit="abc123",
        )
        env.start()

        clone_call = mock_exec.call_args_list[0].args[0]
        assert "/tmp/_repo" in clone_call
        assert "cp -a" in clone_call
