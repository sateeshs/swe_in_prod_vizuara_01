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


def build_system_prompt(config: LangConfig, target_file: str) -> str:
    """Build the agent system prompt for a given language and target file.

    Mirrors the structure of ``utils.SYSTEM`` but adapts the useful-commands
    block to the language toolchain.
    """
    commands = "\n".join(
        f"  {cmd.format(target=target_file, test_name='TEST_NAME')}"
        for cmd in config.useful_commands
    )

    return (
        "You are a software engineering agent. "
        f"You are in a {config.name} repository.\n"
        "Fix the bug described in the issue.\n\n"
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
