"""Tests for Angular and Next.js MFE lang configs, system prompts, and parsers."""

from __future__ import annotations

import pytest

from docker_env import _parse_test_counts
from lang_config import (
    ANGULAR,
    NEXTJS,
    NODEJS,
    build_system_prompt,
    get_config,
    list_languages,
)


# ── Angular config ──────────────────────────────────────────────────


class TestAngularConfig:
    def test_get_by_name(self) -> None:
        assert get_config("angular") is ANGULAR

    def test_alias_ng(self) -> None:
        assert get_config("ng") is ANGULAR

    def test_alias_angular20(self) -> None:
        assert get_config("angular20") is ANGULAR

    def test_case_insensitive(self) -> None:
        assert get_config("Angular") is ANGULAR
        assert get_config("NG") is ANGULAR

    def test_image_is_node(self) -> None:
        assert "node" in ANGULAR.docker_image
        assert "22.13.1" in ANGULAR.docker_image

    def test_build_cmd(self) -> None:
        assert "ng build" in ANGULAR.build_cmd

    def test_test_cmd(self) -> None:
        assert "ng test" in ANGULAR.test_cmd
        assert "ChromeHeadless" in ANGULAR.test_cmd

    def test_extensions(self) -> None:
        assert ".ts" in ANGULAR.file_extensions
        assert ".html" in ANGULAR.file_extensions
        assert ".scss" in ANGULAR.file_extensions

    def test_setup_installs_deps(self) -> None:
        assert any("npm install" in cmd for cmd in ANGULAR.setup_commands)

    def test_setup_installs_chromium(self) -> None:
        assert any("chromium" in cmd for cmd in ANGULAR.setup_commands)

    def test_in_list_languages(self) -> None:
        assert "angular" in list_languages()

    def test_useful_commands_include_federation_search(self) -> None:
        assert any("Federation" in cmd for cmd in ANGULAR.useful_commands)

    def test_useful_commands_include_lint(self) -> None:
        assert any("lint" in cmd for cmd in ANGULAR.useful_commands)


# ── Next.js config ──────────────────────────────────────────────────


class TestNextJSConfig:
    def test_get_by_name(self) -> None:
        assert get_config("nextjs") is NEXTJS

    def test_alias_next(self) -> None:
        assert get_config("next") is NEXTJS

    def test_alias_next_dot_js(self) -> None:
        assert get_config("next.js") is NEXTJS

    def test_case_insensitive(self) -> None:
        assert get_config("NextJS") is NEXTJS
        assert get_config("NEXT") is NEXTJS

    def test_image_is_node(self) -> None:
        assert "node" in NEXTJS.docker_image
        assert "22.13.1" in NEXTJS.docker_image

    def test_build_cmd(self) -> None:
        assert NEXTJS.build_cmd == "npm run build"

    def test_test_cmd(self) -> None:
        assert NEXTJS.test_cmd == "npm test"

    def test_extensions(self) -> None:
        assert ".tsx" in NEXTJS.file_extensions
        assert ".jsx" in NEXTJS.file_extensions
        assert ".ts" in NEXTJS.file_extensions

    def test_setup_installs_deps(self) -> None:
        assert "npm install" in NEXTJS.setup_commands

    def test_in_list_languages(self) -> None:
        assert "nextjs" in list_languages()

    def test_useful_commands_include_vitest(self) -> None:
        assert any("vitest" in cmd for cmd in NEXTJS.useful_commands)

    def test_useful_commands_include_federation_search(self) -> None:
        assert any("Federation" in cmd for cmd in NEXTJS.useful_commands)

    def test_useful_commands_include_next_config(self) -> None:
        assert any("next.config" in cmd for cmd in NEXTJS.useful_commands)


# ── distinct from nodejs ────────────────────────────────────────────


class TestDistinctFromNodeJS:
    def test_angular_is_not_nodejs(self) -> None:
        assert ANGULAR is not NODEJS
        assert ANGULAR.name == "angular"

    def test_nextjs_is_not_nodejs(self) -> None:
        assert NEXTJS is not NODEJS
        assert NEXTJS.name == "nextjs"

    def test_ts_alias_is_nodejs_not_angular(self) -> None:
        # "ts" should still map to generic nodejs, not angular
        assert get_config("ts") is NODEJS


# ── MFE system prompts ──────────────────────────────────────────────


class TestAngularSystemPrompt:
    def test_contains_angular(self) -> None:
        prompt = build_system_prompt(ANGULAR, "src/app/app.component.ts")
        assert "angular" in prompt.lower()

    def test_contains_target_file(self) -> None:
        prompt = build_system_prompt(ANGULAR, "src/app/app.component.ts")
        assert "src/app/app.component.ts" in prompt

    def test_contains_mfe_context(self) -> None:
        prompt = build_system_prompt(ANGULAR, "src/app/app.component.ts")
        assert "Micro Frontend" in prompt
        assert "Module Federation" in prompt

    def test_contains_standalone_hint(self) -> None:
        prompt = build_system_prompt(ANGULAR, "src/app/app.component.ts")
        assert "standalone" in prompt.lower()

    def test_contains_signals_hint(self) -> None:
        prompt = build_system_prompt(ANGULAR, "src/app/app.component.ts")
        assert "signal()" in prompt

    def test_contains_ng_build(self) -> None:
        prompt = build_system_prompt(ANGULAR, "src/app/app.component.ts")
        assert "ng build" in prompt

    def test_contains_bash_fence(self) -> None:
        prompt = build_system_prompt(ANGULAR, "src/app/app.component.ts")
        assert "```bash" in prompt


class TestNextJSSystemPrompt:
    def test_contains_nextjs(self) -> None:
        prompt = build_system_prompt(NEXTJS, "app/page.tsx")
        assert "nextjs" in prompt.lower()

    def test_contains_target_file(self) -> None:
        prompt = build_system_prompt(NEXTJS, "app/page.tsx")
        assert "app/page.tsx" in prompt

    def test_contains_mfe_context(self) -> None:
        prompt = build_system_prompt(NEXTJS, "app/page.tsx")
        assert "Micro Frontend" in prompt
        assert "NextFederationPlugin" in prompt

    def test_contains_server_component_hint(self) -> None:
        prompt = build_system_prompt(NEXTJS, "app/page.tsx")
        assert "Server Components" in prompt
        assert "use client" in prompt

    def test_contains_app_router_hint(self) -> None:
        prompt = build_system_prompt(NEXTJS, "app/page.tsx")
        assert "App Router" in prompt

    def test_no_mfe_context_for_python(self) -> None:
        from lang_config import PYTHON
        prompt = build_system_prompt(PYTHON, "main.py")
        assert "Micro Frontend" not in prompt
        assert "Module Federation" not in prompt


# ── Karma test output parsing ───────────────────────────────────────


class TestKarmaParser:
    def test_karma_success(self) -> None:
        output = "Executed 5 of 5 SUCCESS (0.123 secs / 0.089 secs)"
        passed, failed = _parse_test_counts(output)
        assert passed == 5
        assert failed == 0

    def test_karma_failure(self) -> None:
        output = "Executed 5 of 5 (2 FAILED) (0.123 secs / 0.089 secs)"
        passed, failed = _parse_test_counts(output)
        assert failed == 2

    def test_karma_specs_no_failure(self) -> None:
        output = "3 specs, 0 failures"
        passed, failed = _parse_test_counts(output)
        assert passed == 3
        assert failed == 0

    def test_karma_specs_with_failure(self) -> None:
        output = "5 specs, 2 failures"
        passed, failed = _parse_test_counts(output)
        assert failed == 2


# ── Vitest test output parsing ──────────────────────────────────────


class TestVitestParser:
    def test_vitest_passed(self) -> None:
        output = " Tests  3 passed (3)\n Duration  1.23s"
        passed, failed = _parse_test_counts(output)
        assert passed == 3
        assert failed == 0

    def test_vitest_mixed(self) -> None:
        output = " Tests  2 passed | 1 failed (3)\n Duration  0.89s"
        passed, failed = _parse_test_counts(output)
        assert passed == 2
        assert failed == 1

    def test_vitest_all_fail(self) -> None:
        output = " Tests  3 failed (3)\n Duration  0.5s"
        passed, failed = _parse_test_counts(output)
        assert passed == 0
        assert failed == 3
