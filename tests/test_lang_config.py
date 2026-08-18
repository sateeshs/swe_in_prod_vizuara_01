"""Tests for lang_config module."""

import pytest

from lang_config import (
    CSHARP,
    PYTHON,
    RUST,
    LangConfig,
    build_system_prompt,
    get_config,
    list_languages,
)


class TestGetConfig:
    def test_python_by_name(self) -> None:
        assert get_config("python") is PYTHON

    def test_csharp_by_name(self) -> None:
        assert get_config("csharp") is CSHARP

    def test_csharp_alias_dotnet(self) -> None:
        assert get_config("dotnet") is CSHARP

    def test_csharp_alias_c_hash(self) -> None:
        assert get_config("c#") is CSHARP

    def test_rust_by_name(self) -> None:
        assert get_config("rust") is RUST

    def test_case_insensitive(self) -> None:
        assert get_config("Python") is PYTHON
        assert get_config("RUST") is RUST

    def test_strips_whitespace(self) -> None:
        assert get_config("  csharp  ") is CSHARP

    def test_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Unknown language"):
            get_config("java")

    def test_unknown_error_lists_known(self) -> None:
        with pytest.raises(KeyError, match="csharp"):
            get_config("fortran")


class TestListLanguages:
    def test_returns_canonical_names(self) -> None:
        langs = list_languages()
        assert "python" in langs
        assert "csharp" in langs
        assert "rust" in langs

    def test_no_aliases(self) -> None:
        langs = list_languages()
        assert "dotnet" not in langs
        assert "c#" not in langs


class TestLangConfigFrozen:
    def test_cannot_mutate(self) -> None:
        with pytest.raises(AttributeError):
            PYTHON.name = "not-python"  # type: ignore[misc]


class TestBuildSystemPrompt:
    def test_contains_language_name(self) -> None:
        prompt = build_system_prompt(PYTHON, "src/main.py")
        assert "python" in prompt.lower()

    def test_contains_target_file(self) -> None:
        prompt = build_system_prompt(RUST, "src/lib.rs")
        assert "src/lib.rs" in prompt

    def test_contains_useful_commands(self) -> None:
        prompt = build_system_prompt(CSHARP, "Program.cs")
        assert "dotnet build" in prompt
        assert "dotnet test" in prompt

    def test_contains_heredoc_write(self) -> None:
        prompt = build_system_prompt(PYTHON, "app.py")
        assert "cat > app.py <<'EOF'" in prompt

    def test_contains_bash_fence(self) -> None:
        prompt = build_system_prompt(RUST, "main.rs")
        assert "```bash" in prompt

    def test_custom_config(self) -> None:
        custom = LangConfig(
            name="go",
            docker_image="golang:1.22",
            build_cmd="go build ./...",
            test_cmd="go test ./...",
            file_extensions=(".go",),
            useful_commands=("go test ./...",),
        )
        prompt = build_system_prompt(custom, "main.go")
        assert "go" in prompt
        assert "go test" in prompt
