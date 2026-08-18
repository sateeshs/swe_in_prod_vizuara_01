"""Tests for transpile module.

Covers: TranspileRecord, prompt builder, reward scoring,
JSONL I/O, and C++ source+test file pairing.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from transpile import (
    TranspileRecord,
    build_transpile_prompt,
    estimate_complexity,
    pair_source_and_tests,
    reward_transpile,
    write_transpile_tasks,
)


# ── sample C++ source + tests ─────────────────────────────────────

CPP_SOURCE = textwrap.dedent("""\
    #include <cmath>
    #include <stdexcept>

    double compute_period(double semi_major_axis, double mu) {
        if (semi_major_axis <= 0) {
            throw std::invalid_argument("semi_major_axis must be positive");
        }
        return 2.0 * M_PI * sqrt(pow(semi_major_axis, 3) / mu);
    }

    double compute_velocity(double radius, double mu) {
        if (radius <= 0) {
            throw std::invalid_argument("radius must be positive");
        }
        return sqrt(mu / radius);
    }
""")

CPP_TEST = textwrap.dedent("""\
    #include <gtest/gtest.h>
    #include "orbit.h"

    TEST(OrbitTest, PositiveSemiMajor) {
        double period = compute_period(7000e3, 3.986e14);
        EXPECT_NEAR(period, 5828.5, 1.0);
    }

    TEST(OrbitTest, NegativeSemiMajorThrows) {
        EXPECT_THROW(compute_period(-1.0, 3.986e14), std::invalid_argument);
    }

    TEST(OrbitTest, VelocityAtLEO) {
        double v = compute_velocity(7000e3, 3.986e14);
        EXPECT_NEAR(v, 7546.0, 10.0);
    }
""")


# ── TranspileRecord ───────────────────────────────────────────────


class TestTranspileRecord:
    def test_frozen(self) -> None:
        record = TranspileRecord(
            repo="nasa/cfs",
            source_file="src/orbit.cpp",
            source_code=CPP_SOURCE,
            source_lang="cpp",
            target_lang="rust",
            test_code=CPP_TEST,
            test_names=["OrbitTest.PositiveSemiMajor"],
            instance_id="nasa__cfs-orbit.cpp",
        )
        with pytest.raises(AttributeError):
            record.repo = "changed"  # type: ignore[misc]

    def test_fields(self) -> None:
        record = TranspileRecord(
            repo="owner/repo",
            source_file="src/nav.cpp",
            source_code="int main() {}",
            source_lang="cpp",
            target_lang="rust",
            test_code="TEST(Nav, Init) {}",
            test_names=["Nav.Init"],
            instance_id="owner__repo-nav.cpp",
        )
        assert record.source_lang == "cpp"
        assert record.target_lang == "rust"
        assert len(record.test_names) == 1

    def test_default_target_is_rust(self) -> None:
        record = TranspileRecord(
            repo="o/r",
            source_file="f.cpp",
            source_code="",
            source_lang="cpp",
            target_lang="rust",
            test_code="",
            test_names=[],
            instance_id="o__r-f.cpp",
        )
        assert record.target_lang == "rust"


# ── JSONL Writer ───────────────────────────────────────────────────


class TestWriteTranspileTasks:
    def test_writes_jsonl(self, tmp_path: Path) -> None:
        out = tmp_path / "transpile.jsonl"
        records = [
            TranspileRecord(
                repo="o/r", source_file="a.cpp", source_code="int x;",
                source_lang="cpp", target_lang="rust",
                test_code="TEST(A, B) {}", test_names=["A.B"],
                instance_id="o__r-a.cpp",
            ),
        ]
        count = write_transpile_tasks(records, out)
        assert count == 1

        data = json.loads(out.read_text().strip())
        assert data["source_file"] == "a.cpp"
        assert data["target_lang"] == "rust"

    def test_empty_records(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.jsonl"
        count = write_transpile_tasks([], out)
        assert count == 0
        assert not out.exists()

    def test_appends(self, tmp_path: Path) -> None:
        out = tmp_path / "tasks.jsonl"
        out.write_text('{"existing": true}\n')
        records = [
            TranspileRecord(
                repo="o/r", source_file="b.cpp", source_code="",
                source_lang="cpp", target_lang="rust",
                test_code="", test_names=[], instance_id="o__r-b.cpp",
            ),
        ]
        write_transpile_tasks(records, out)
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2


# ── Prompt Builder ─────────────────────────────────────────────────


class TestBuildTranspilePrompt:
    def test_contains_source_code(self) -> None:
        prompt = build_transpile_prompt(
            source_code=CPP_SOURCE,
            source_lang="cpp",
            target_lang="rust",
            test_hints=["OrbitTest.PositiveSemiMajor"],
        )
        assert "compute_period" in prompt
        assert "compute_velocity" in prompt

    def test_contains_target_language(self) -> None:
        prompt = build_transpile_prompt(
            source_code="int x;",
            source_lang="cpp",
            target_lang="rust",
        )
        assert "rust" in prompt.lower()
        assert "Rust" in prompt

    def test_contains_instructions(self) -> None:
        prompt = build_transpile_prompt(
            source_code="int x;",
            source_lang="cpp",
            target_lang="rust",
        )
        assert "idiomatic" in prompt.lower()
        assert "cargo" in prompt.lower()

    def test_includes_test_hints(self) -> None:
        prompt = build_transpile_prompt(
            source_code="int x;",
            source_lang="cpp",
            target_lang="rust",
            test_hints=["OrbitTest.PositiveSemiMajor", "OrbitTest.Negative"],
        )
        assert "OrbitTest.PositiveSemiMajor" in prompt

    def test_no_test_hints(self) -> None:
        prompt = build_transpile_prompt(
            source_code="int x;",
            source_lang="cpp",
            target_lang="rust",
        )
        # Should still be valid without test hints
        assert "```" in prompt

    def test_contains_bash_fence(self) -> None:
        prompt = build_transpile_prompt(
            source_code="int x;",
            source_lang="cpp",
            target_lang="rust",
        )
        assert "```bash" in prompt


# ── Reward Function ────────────────────────────────────────────────


class TestRewardTranspile:
    def _mock_env(self, patch: str, build_output: str, test_output: str) -> MagicMock:
        env = MagicMock()
        env.patch.return_value = patch
        env.run.side_effect = [build_output, test_output]
        env.config = MagicMock()
        env.config.name = "rust"
        return env

    def test_empty_patch_returns_zero(self) -> None:
        env = MagicMock()
        env.patch.return_value = ""
        assert reward_transpile(env) == 0.0

    def test_build_fails_returns_zero(self) -> None:
        env = self._mock_env(
            patch="diff --git a/src/lib.rs ...",
            build_output="error[E0308]: mismatched types",
            test_output="",
        )
        # Build failure indicated by "error" in output
        result = reward_transpile(env)
        assert result == 0.0

    def test_build_succeeds_tests_pass(self) -> None:
        env = self._mock_env(
            patch="diff --git a/src/lib.rs ...",
            build_output="Compiling orbit v0.1.0\n    Finished",
            test_output="test result: ok. 3 passed; 0 failed; 0 ignored",
        )
        result = reward_transpile(env)
        assert result == 1.0

    def test_build_succeeds_partial_tests(self) -> None:
        env = self._mock_env(
            patch="diff --git a/src/lib.rs ...",
            build_output="Compiling orbit v0.1.0\n    Finished",
            test_output="test result: FAILED. 2 passed; 1 failed; 0 ignored",
        )
        result = reward_transpile(env)
        assert 0.5 < result < 1.0

    def test_build_succeeds_no_tests(self) -> None:
        env = self._mock_env(
            patch="diff --git a/src/lib.rs ...",
            build_output="Compiling orbit v0.1.0\n    Finished",
            test_output="no tests to run",
        )
        result = reward_transpile(env)
        assert result == 0.5  # compiles but no tests


# ── Source-Test Pairing ────────────────────────────────────────────


class TestPairSourceAndTests:
    def test_pairs_cpp_with_test(self) -> None:
        files = [
            "src/orbit.cpp",
            "src/orbit.h",
            "tests/orbit_test.cpp",
            "src/nav.cpp",
            "README.md",
        ]
        pairs = pair_source_and_tests(files, "cpp")
        # orbit.cpp should pair with orbit_test.cpp
        source_files = [p[0] for p in pairs]
        assert "src/orbit.cpp" in source_files

    def test_no_test_file_excluded(self) -> None:
        files = [
            "src/main.cpp",
            "src/utils.cpp",
        ]
        pairs = pair_source_and_tests(files, "cpp")
        assert len(pairs) == 0

    def test_header_files_excluded(self) -> None:
        files = [
            "src/orbit.h",
            "tests/orbit_test.cpp",
        ]
        pairs = pair_source_and_tests(files, "cpp")
        # Headers alone shouldn't be source files
        assert len(pairs) == 0

    def test_empty_files(self) -> None:
        assert pair_source_and_tests([], "cpp") == []


# ── Complexity Estimator ──────────────────────────────────────────


class TestEstimateComplexity:
    def test_short_file_is_low(self) -> None:
        source = "int main() { return 0; }"
        assert estimate_complexity(source) == "low"

    def test_medium_file(self) -> None:
        source = "\n".join([f"void func_{i}() {{}}" for i in range(60)])
        assert estimate_complexity(source) == "medium"

    def test_long_file_is_high(self) -> None:
        source = "\n".join([f"void func_{i}() {{}}" for i in range(200)])
        assert estimate_complexity(source) == "high"

    def test_template_heavy_is_high(self) -> None:
        source = "\n".join([
            "template<typename T>",
            "class Container {",
            "    T data;",
            "    template<typename U>",
            "    void convert(U val) {}",
            "};",
        ] * 5)
        assert estimate_complexity(source) in ("medium", "high")
