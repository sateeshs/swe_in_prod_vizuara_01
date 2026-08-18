"""Tests for utils.MockEnv."""

import sys
from pathlib import Path

import pytest

# Add project root to path so utils can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import TARGET, MockEnv, first_bash_block


FAILING_TESTS = [
    "astropy/modeling/tests/test_separable.py::test_custom_model_separable",
    "astropy/modeling/tests/test_separable.py::test_cstack",
]


class TestFirstBashBlock:
    def test_extracts_bash_block(self) -> None:
        text = "Let me check.\n```bash\nls\n```"
        assert first_bash_block(text) == "ls"

    def test_extracts_sh_block(self) -> None:
        text = "```sh\ncat file.py\n```"
        assert first_bash_block(text) == "cat file.py"

    def test_extracts_unfenced_code_block(self) -> None:
        text = "```\nls -la\n```"
        assert first_bash_block(text) == "ls -la"

    def test_returns_none_when_no_block(self) -> None:
        assert first_bash_block("just some text") is None

    def test_extracts_first_of_multiple(self) -> None:
        text = "```bash\nfirst\n```\n```bash\nsecond\n```"
        assert first_bash_block(text) == "first"

    def test_multiline_command(self) -> None:
        text = "```bash\ncat > f.py <<'EOF'\nprint('hi')\nEOF\n```"
        result = first_bash_block(text)
        assert result is not None
        assert "cat > f.py" in result
        assert "print('hi')" in result


class TestMockEnvLs:
    def test_ls_lists_target(self) -> None:
        env = MockEnv(FAILING_TESTS)
        out = env.run("ls")
        assert TARGET in out

    def test_ls_includes_fixtures(self) -> None:
        env = MockEnv(FAILING_TESTS)
        out = env.run("ls")
        assert "setup.py" in out


class TestMockEnvCat:
    def test_cat_existing_file(self) -> None:
        env = MockEnv(FAILING_TESTS)
        out = env.run(f"cat {TARGET}")
        assert "_cstack" in out

    def test_cat_missing_file(self) -> None:
        env = MockEnv(FAILING_TESTS)
        out = env.run("cat nonexistent.py")
        assert "No such file" in out


class TestMockEnvGrep:
    def test_grep_finds_match(self) -> None:
        env = MockEnv(FAILING_TESTS)
        out = env.run(f'grep -n "cright" {TARGET}')
        assert "cright" in out

    def test_grep_no_match(self) -> None:
        env = MockEnv(FAILING_TESTS)
        out = env.run(f'grep -n "nonexistent_pattern_xyz" {TARGET}')
        assert "no matches" in out


class TestMockEnvPytest:
    def test_pytest_reports_failures(self) -> None:
        env = MockEnv(FAILING_TESTS)
        out = env.run("python -m pytest")
        assert "FAILED" in out
        assert "2 failed" in out

    def test_pytest_includes_test_names(self) -> None:
        env = MockEnv(FAILING_TESTS)
        out = env.run("python -m pytest")
        for test in FAILING_TESTS:
            assert test in out


class TestMockEnvWrite:
    def test_heredoc_write_updates_fs(self) -> None:
        env = MockEnv(FAILING_TESTS)
        env.run(f"cat > {TARGET} <<'EOF'\nnew content\nEOF")
        assert env.fs[TARGET] == "new content"

    def test_heredoc_write_returns_confirmation(self) -> None:
        env = MockEnv(FAILING_TESTS)
        out = env.run(f"cat > {TARGET} <<'EOF'\nline1\nline2\nEOF")
        assert "wrote" in out.lower()
        assert "2 lines" in out


class TestMockEnvPatch:
    def test_no_change_returns_empty(self) -> None:
        env = MockEnv(FAILING_TESTS)
        assert env.patch() == ""

    def test_change_returns_unified_diff(self) -> None:
        env = MockEnv(FAILING_TESTS)
        env.run(f"cat > {TARGET} <<'EOF'\nfixed content\nEOF")
        diff = env.patch()
        assert "---" in diff
        assert "+++" in diff

    def test_patch_shows_before_and_after(self) -> None:
        env = MockEnv(FAILING_TESTS)
        env.run(f"cat > {TARGET} <<'EOF'\nfixed\nEOF")
        diff = env.patch()
        assert f"a/{TARGET}" in diff
        assert f"b/{TARGET}" in diff


class TestMockEnvCallLog:
    def test_tracks_commands(self) -> None:
        env = MockEnv(FAILING_TESTS)
        env.run("ls")
        env.run(f"cat {TARGET}")
        assert len(env.calls) == 2
        assert env.calls[0] == "ls"

    def test_unknown_command(self) -> None:
        env = MockEnv(FAILING_TESTS)
        out = env.run("unknown_cmd")
        assert "command not found" in out
