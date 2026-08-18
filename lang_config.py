"""Language configurations for multi-language SWE-RL training.

Each LangConfig bundles the toolchain details a DockerEnv needs to spin up
the right container and the prompt details the agent needs to issue the
right commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence


class EnvProtocol(Protocol):
    """Minimal interface shared by MockEnv and DockerEnv.

    Any environment that can ``run`` commands and produce a ``patch`` diff
    satisfies this protocol.  The ``calls`` attribute tracks every command
    issued to the environment.
    """

    calls: list[str]

    def run(self, cmd: str) -> str: ...

    def patch(self) -> str: ...


@dataclass(frozen=True)
class LangConfig:
    """Immutable configuration for one target language."""

    name: str
    docker_image: str
    build_cmd: str
    test_cmd: str
    file_extensions: tuple[str, ...]
    useful_commands: tuple[str, ...]

    # Optional: extra packages to install inside the container before the agent runs
    setup_commands: tuple[str, ...] = ()


# ── built-in configs ────────────────────────────────────────────────

PYTHON = LangConfig(
    name="python",
    docker_image="python:3.11-slim",
    build_cmd="pip install -e .",
    test_cmd="python -m pytest",
    file_extensions=(".py",),
    useful_commands=(
        "ls",
        "cat {target}",
        'grep -n "pattern" {target}',
        "python -m pytest",
    ),
)

CSHARP = LangConfig(
    name="csharp",
    docker_image="mcr.microsoft.com/dotnet/sdk:8.0",
    build_cmd="dotnet build",
    test_cmd="dotnet test",
    file_extensions=(".cs", ".csproj", ".sln"),
    useful_commands=(
        "ls",
        "cat {target}",
        'grep -rn "pattern" {target}',
        "dotnet build",
        "dotnet test",
        "dotnet test --filter {test_name}",
    ),
)

RUST = LangConfig(
    name="rust",
    docker_image="rust:1.79-slim",
    build_cmd="cargo build",
    test_cmd="cargo test",
    file_extensions=(".rs", ".toml"),
    useful_commands=(
        "ls",
        "cat {target}",
        'grep -rn "pattern" {target}',
        "cargo build",
        "cargo test",
        "cargo test {test_name}",
    ),
)

NODEJS = LangConfig(
    name="nodejs",
    docker_image="node:22.13.1-slim",
    build_cmd="npm run build",
    test_cmd="npm test",
    file_extensions=(".js", ".ts", ".mjs", ".cjs", ".json"),
    useful_commands=(
        "ls",
        "cat {target}",
        'grep -rn "pattern" {target}',
        "npm run build",
        "npm test",
        "npx jest {test_name}",
        "node -e 'console.log(process.version)'",
    ),
    setup_commands=(
        "npm install",
    ),
)

ANGULAR = LangConfig(
    name="angular",
    docker_image="node:22.13.1-slim",
    build_cmd="npx ng build",
    test_cmd="npx ng test --watch=false --browsers=ChromeHeadless",
    file_extensions=(".ts", ".html", ".scss", ".css", ".json"),
    useful_commands=(
        "ls",
        "cat {target}",
        'grep -rn "pattern" {target}',
        "npx ng build",
        "npx ng test --watch=false --browsers=ChromeHeadless",
        "npx ng test --watch=false --include={test_name}",
        "npx ng lint",
        "cat angular.json",
        "cat package.json",
        "cat tsconfig.json",
        'find src -name "*.module.ts" -o -name "*.component.ts"',
        'grep -rn "loadChildren\\|loadComponent\\|Federation" src/',
    ),
    setup_commands=(
        "npm install",
        "apt-get update -qq && apt-get install -y -qq chromium > /dev/null 2>&1 || true",
        "export CHROME_BIN=$(which chromium || which chromium-browser || echo /usr/bin/chromium)",
    ),
)

CPP = LangConfig(
    name="cpp",
    docker_image="gcc:13-bookworm",
    build_cmd="mkdir -p build && cd build && cmake .. && make -j$(nproc)",
    test_cmd="cd build && ctest --output-on-failure",
    file_extensions=(".c", ".cpp", ".h", ".hpp", ".cmake"),
    useful_commands=(
        "ls",
        "cat {target}",
        'grep -rn "pattern" {target}',
        "mkdir -p build && cd build && cmake .. && make -j$(nproc)",
        "cd build && ctest --output-on-failure",
        "cd build && ctest -R {test_name} --output-on-failure",
        "cat CMakeLists.txt",
        'find . -name "CMakeLists.txt" -o -name "*.cmake"',
        'find . -name "*_test.cpp" -o -name "*Test.cpp"',
    ),
    setup_commands=(
        "apt-get update -qq && apt-get install -y -qq cmake make git > /dev/null 2>&1",
    ),
)

FPRIME = LangConfig(
    name="fprime",
    docker_image="python:3.11-slim",
    build_cmd="fprime-util generate && fprime-util build",
    test_cmd="fprime-util check",
    file_extensions=(".cpp", ".hpp", ".fpp", ".py"),
    useful_commands=(
        "ls",
        "cat {target}",
        'grep -rn "pattern" {target}',
        "fprime-util generate",
        "fprime-util build",
        "fprime-util check",
        "cat CMakeLists.txt",
        'find . -name "*.fpp" -o -name "*.cpp" -o -name "*.hpp"',
        'grep -rn "component\\|port\\|topology" {target}',
    ),
    setup_commands=(
        "pip install fprime-tools fprime-gds > /dev/null 2>&1",
        "apt-get update -qq && apt-get install -y -qq cmake make g++ > /dev/null 2>&1",
    ),
)

NEXTJS = LangConfig(
    name="nextjs",
    docker_image="node:22.13.1-slim",
    build_cmd="npm run build",
    test_cmd="npm test",
    file_extensions=(".ts", ".tsx", ".js", ".jsx", ".css", ".json"),
    useful_commands=(
        "ls",
        "cat {target}",
        'grep -rn "pattern" {target}',
        "npm run build",
        "npm test",
        "npx jest {test_name}",
        "npx vitest run {test_name}",
        "npm run lint",
        "cat next.config.js || cat next.config.mjs || cat next.config.ts",
        "cat package.json",
        "cat tsconfig.json",
        'find app pages src -name "*.tsx" -o -name "*.ts" 2>/dev/null',
        'grep -rn "NextFederationPlugin\\|ModuleFederationPlugin\\|remotes\\|exposes" next.config.* webpack.config.* 2>/dev/null',
    ),
    setup_commands=(
        "npm install",
    ),
)

_REGISTRY: Mapping[str, LangConfig] = {
    "python": PYTHON,
    "csharp": CSHARP,
    "c#": CSHARP,
    "dotnet": CSHARP,
    "rust": RUST,
    "nodejs": NODEJS,
    "node": NODEJS,
    "javascript": NODEJS,
    "js": NODEJS,
    "typescript": NODEJS,
    "ts": NODEJS,
    "angular": ANGULAR,
    "ng": ANGULAR,
    "angular20": ANGULAR,
    "nextjs": NEXTJS,
    "next": NEXTJS,
    "next.js": NEXTJS,
    "cpp": CPP,
    "c++": CPP,
    "c": CPP,
    "cmake": CPP,
    "fprime": FPRIME,
    "f-prime": FPRIME,
    "f'": FPRIME,
}


def get_config(lang: str) -> LangConfig:
    """Look up a language config by name (case-insensitive).

    Raises ``KeyError`` with the list of known languages when *lang* is unknown.
    """
    key = lang.strip().lower()
    if key not in _REGISTRY:
        known = sorted({c.name for c in _REGISTRY.values()})
        raise KeyError(f"Unknown language {lang!r}. Known: {known}")
    return _REGISTRY[key]


def list_languages() -> Sequence[str]:
    """Return canonical language names."""
    return sorted({c.name for c in _REGISTRY.values()})


_MFE_CONTEXT: Mapping[str, str] = {
    "angular": (
        "\nThis is an Angular Micro Frontend (MFE) project. Key patterns:\n"
        "- Module Federation: check webpack.config.ts or angular.json for "
        "exposes/remotes configuration\n"
        "- Shared dependencies: @angular/core, @angular/common, @angular/router "
        "are typically shared singletons\n"
        "- Standalone components: Angular 20 uses standalone by default — "
        "no NgModules unless legacy\n"
        "- Signals: prefer signal(), computed(), effect() over BehaviorSubject\n"
        "- Routing: loadComponent() for lazy standalone routes, "
        "loadChildren() for module-based\n"
        "- Check angular.json 'projects' for multi-project workspace layout\n"
    ),
    "nextjs": (
        "\nThis is a Next.js Micro Frontend (MFE) project. Key patterns:\n"
        "- Module Federation: check next.config.js/mjs for "
        "NextFederationPlugin or ModuleFederationPlugin\n"
        "- App Router (app/) vs Pages Router (pages/) — check which is used\n"
        "- Server Components are default in app/ — add 'use client' only "
        "when needed (hooks, browser APIs, interactivity)\n"
        "- Shared dependencies: react, react-dom are typically shared singletons\n"
        "- Dynamic imports: next/dynamic for client-side lazy loading\n"
        "- Remote entry: check for remoteEntry.js references in config\n"
        "- API routes: app/api/ (Route Handlers) or pages/api/\n"
    ),
}


def build_system_prompt(config: LangConfig, target_file: str) -> str:
    """Build the agent system prompt for a given language and target file.

    Mirrors the structure of ``utils.SYSTEM`` but adapts the useful-commands
    block to the language toolchain.  Includes MFE-specific context for
    Angular and Next.js projects.
    """
    commands = "\n".join(
        f"  {cmd.format(target=target_file, test_name='TEST_NAME')}"
        for cmd in config.useful_commands
    )

    mfe_section = _MFE_CONTEXT.get(config.name, "")

    return (
        "You are a software engineering agent. "
        f"You are in a {config.name} repository.\n"
        "Fix the bug described in the issue.\n\n"
        f"{mfe_section}\n"
        "Respond with exactly ONE bash command per message, in a fenced block:\n\n"
        "```bash\nyour command here\n```\n\n"
        f"Useful commands:\n{commands}\n\n"
        f"To rewrite a file:\n"
        "```bash\n"
        f"cat > {target_file} <<'EOF'\n"
        "...full new contents...\n"
        "EOF\n"
        "```\n\n"
        "One short sentence of reasoning, then exactly one command block."
    )
