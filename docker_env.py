"""Docker-based environment for multi-language SWE-RL training.

Provides the same interface as ``utils.MockEnv`` — ``run()``, ``patch()``,
and ``self.calls`` — but executes commands in a real Docker container with
the correct language toolchain installed.

Usage::

    from lang_config import get_config

    cfg = get_config("rust")
    with DockerEnv(cfg, repo_url="https://github.com/user/repo.git",
                   base_commit="abc123") as env:
        obs = env.run("cargo test")
        diff = env.patch()
"""

from __future__ import annotations

import atexit
import subprocess
import uuid
from dataclasses import dataclass, field
from typing import Sequence

import re

from lang_config import LangConfig


_DEFAULT_TIMEOUT = 120  # seconds per command
_WORKDIR = "/workspace"


@dataclass
class DockerEnv:
    """A shell environment backed by a Docker container.

    Parameters
    ----------
    config : LangConfig
        Language configuration (image, build/test commands, etc.).
    repo_url : str
        Git URL of the repository to clone inside the container.
    base_commit : str
        Commit SHA to checkout after cloning.
    test_patch : str
        Optional unified diff to apply before the agent starts (the grader's
        held-out tests).  Applied with ``git apply``.
    timeout : int
        Max seconds per ``run()`` call.  Prevents the agent from hanging.
    """

    config: LangConfig
    repo_url: str
    base_commit: str
    test_patch: str = ""
    timeout: int = _DEFAULT_TIMEOUT
    calls: list[str] = field(default_factory=list)

    _container_id: str = field(default="", init=False, repr=False)

    # ── lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        """Create and prepare the container."""
        if self._container_id:
            return

        name = f"swe-rl-{uuid.uuid4().hex[:12]}"

        # Start detached container
        self._container_id = self._docker(
            "run", "-d",
            "--name", name,
            "-w", _WORKDIR,
            self.config.docker_image,
            "sleep", "infinity",
        ).strip()

        # Register cleanup so a crash doesn't leave containers behind
        atexit.register(self.stop)

        # Clone into a temp dir first, then move contents into workdir.
        # Docker sets -w /workspace at container start, and that dir already
        # exists — git clone refuses to clone into a non-empty directory.
        self._exec(
            f"git clone --quiet {self.repo_url} /tmp/_repo"
            f" && cp -a /tmp/_repo/. {_WORKDIR}/"
            f" && rm -rf /tmp/_repo"
        )
        self._exec(f"git checkout --quiet {self.base_commit}")

        # Apply held-out test patch if provided
        if self.test_patch:
            self._exec(
                f"git apply --allow-empty -",
                input_data=self.test_patch,
            )

        # Run language-specific setup
        for cmd in self.config.setup_commands:
            self._exec(cmd)

    def stop(self) -> None:
        """Remove the container. Safe to call multiple times."""
        if not self._container_id:
            return
        try:
            self._docker("rm", "-f", self._container_id)
        except subprocess.SubprocessError:
            pass
        self._container_id = ""

    def __enter__(self) -> DockerEnv:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # ── public interface (matches MockEnv) ───────────────────────────

    def run(self, cmd: str) -> str:
        """Execute *cmd* inside the container and return its combined output.

        Mirrors ``MockEnv.run`` — always returns a string, never raises.
        """
        self.calls.append(cmd)
        try:
            return self._exec(cmd)
        except subprocess.TimeoutExpired:
            return f"[docker env] command timed out after {self.timeout}s"
        except subprocess.SubprocessError as exc:
            return str(exc)

    def patch(self) -> str:
        """Return the ``git diff`` of all changes the agent made."""
        try:
            return self._exec("git diff").strip()
        except subprocess.SubprocessError:
            return ""

    def run_tests(self, test_names: Sequence[str] | None = None) -> tuple[int, int, str]:
        """Run the test suite and return ``(passed, failed, raw_output)``.

        If *test_names* is given, only those tests are executed (appended to
        the configured ``test_cmd``).  Otherwise the full suite runs.
        """
        cmd = self.config.test_cmd
        if test_names:
            cmd = f"{cmd} {' '.join(test_names)}"
        try:
            output = self._exec(cmd)
        except subprocess.SubprocessError as exc:
            output = str(exc)

        passed, failed = _parse_test_counts(output, self.config.name)
        return passed, failed, output

    @property
    def call_log(self) -> Sequence[str]:
        """Read-only view of commands executed."""
        return list(self.calls)

    # ── internals ────────────────────────────────────────────────────

    def _docker(self, *args: str) -> str:
        """Run a ``docker`` CLI command on the host."""
        result = subprocess.run(
            ["docker", *args],
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=True,
        )
        return result.stdout

    def _exec(self, cmd: str, *, input_data: str | None = None) -> str:  # noqa: C901
        """Run *cmd* inside the container via ``docker exec``."""
        proc_args = ["docker", "exec"]
        if input_data:
            proc_args += ["-i"]
        proc_args += [self._container_id, "bash", "-c", cmd]

        result = subprocess.run(
            proc_args,
            input=input_data,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        # Combine stdout and stderr like a real terminal
        output = result.stdout
        if result.stderr:
            output = f"{output}{result.stderr}" if output else result.stderr
        return output


# ── test result parsing ─────────────────────────────────────────────

# Patterns for extracting pass/fail counts from test runner output.
#
# Supported runners:
#   pytest:   "3 passed, 2 failed in 1.23s"
#   jest:     "Tests:  3 passed, 1 failed, 4 total"
#   cargo:    "test result: ok. 10 passed; 0 failed"
#   dotnet:   "Passed!  3" / "Failed!  1"
#   karma:    "3 specs, 1 failure" or "Executed 5 of 5 SUCCESS"
#             or "Executed 5 of 5 (1 FAILED)"
#   vitest:   "Tests  3 passed | 1 failed" or "3 passed (3)"
#
_PASS_PATTERNS = (
    re.compile(r"(\d+)\s+passed"),
    re.compile(r"Passed!\s*-?\s*(\d+)", re.I),
    re.compile(r"test result: ok\.\s*(\d+)\s+passed"),
    re.compile(r"Executed\s+(\d+)\s+of\s+\d+\s+SUCCESS", re.I),
    re.compile(r"(\d+)\s+specs?,\s+0\s+failures?"),
)

_FAIL_PATTERNS = (
    re.compile(r"(\d+)\s+failed"),
    re.compile(r"Failed!\s*-?\s*(\d+)", re.I),
    re.compile(r"test result:.*?(\d+)\s+failed"),
    re.compile(r"(\d+)\s+failures?"),
    re.compile(r"Executed\s+\d+\s+of\s+\d+\s+\((\d+)\s+FAILED\)", re.I),
)


def _parse_test_counts(output: str, lang: str = "") -> tuple[int, int]:
    """Extract (passed, failed) counts from test runner output."""
    passed = 0
    failed = 0
    for pat in _PASS_PATTERNS:
        m = pat.search(output)
        if m:
            passed = int(m.group(1))
            break
    for pat in _FAIL_PATTERNS:
        m = pat.search(output)
        if m:
            failed = int(m.group(1))
            break
    return passed, failed


def reward_test(
    env: DockerEnv,
    fail_to_pass: Sequence[str],
) -> float:
    """Score a rollout by running the held-out tests.

    Returns a float in [0, 1]:
      - 1.0 = all ``fail_to_pass`` tests now pass
      - 0.0 = none pass (or patch is empty)
      - Partial credit = fraction of tests that pass

    This replaces ``reward_random`` from the original walkthrough.
    """
    diff = env.patch()
    if not diff:
        return 0.0

    passed, failed, _output = env.run_tests(list(fail_to_pass))
    total = passed + failed
    if total == 0:
        return 0.0
    return round(passed / total, 3)
