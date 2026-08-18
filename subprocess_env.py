"""Subprocess-based environment for Docker-free SWE-RL execution.

Satisfies the same ``EnvProtocol`` as ``MockEnv`` and ``DockerEnv`` but
runs commands via ``subprocess.run`` in an isolated temp directory.
Works on Kaggle, Colab, and any Linux machine without Docker.

Usage::

    from lang_config import get_config

    cfg = get_config("rust")
    with SubprocessEnv(cfg, repo_url="https://github.com/user/repo.git",
                       base_commit="abc123") as env:
        obs = env.run("cargo test")
        diff = env.patch()
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from docker_env import _parse_test_counts
from lang_config import LangConfig


_DEFAULT_TIMEOUT = 120


@dataclass
class SubprocessEnv:
    """A shell environment backed by subprocess in an isolated temp directory.

    Parameters
    ----------
    config : LangConfig
        Language configuration (build/test commands, etc.).
    repo_url : str
        Git URL of the repository to clone.
    base_commit : str
        Commit SHA to checkout after cloning.
    test_patch : str
        Optional unified diff to apply before the agent starts.
    timeout : int
        Max seconds per ``run()`` call.
    """

    config: LangConfig
    repo_url: str
    base_commit: str
    test_patch: str = ""
    timeout: int = _DEFAULT_TIMEOUT
    calls: list[str] = field(default_factory=list)

    _workdir: Path = field(default=Path(), init=False, repr=False)
    _started: bool = field(default=False, init=False, repr=False)

    # ── lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        """Clone the repo into a temp directory and prepare."""
        if self._started:
            return

        self._workdir = Path(tempfile.mkdtemp(prefix="swe-rl-"))

        # Clone and checkout
        self._shell(f"git clone --quiet {self.repo_url} {self._workdir}")
        self._shell(f"git checkout --quiet {self.base_commit}")

        # Apply held-out test patch
        if self.test_patch:
            self._shell(
                "git apply --allow-empty -",
                input_data=self.test_patch,
            )

        # Run language-specific setup
        for cmd in self.config.setup_commands:
            self._shell(cmd)

        self._started = True

    def stop(self) -> None:
        """Remove the temp directory. Safe to call multiple times."""
        if not self._started:
            return
        try:
            shutil.rmtree(self._workdir, ignore_errors=True)
        except Exception:
            pass
        self._started = False

    def __enter__(self) -> SubprocessEnv:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ── public interface (matches EnvProtocol) ───────────────────────

    def run(self, cmd: str) -> str:
        """Execute *cmd* in the workdir and return combined output."""
        self.calls.append(cmd)
        try:
            return self._shell(cmd)
        except subprocess.TimeoutExpired:
            return f"[subprocess env] command timed out after {self.timeout}s"
        except subprocess.SubprocessError as exc:
            return str(exc)

    def patch(self) -> str:
        """Return ``git diff`` of all changes the agent made."""
        try:
            return self._shell("git diff").strip()
        except subprocess.SubprocessError:
            return ""

    def run_tests(
        self, test_names: Sequence[str] | None = None
    ) -> tuple[int, int, str]:
        """Run the test suite and return ``(passed, failed, raw_output)``."""
        cmd = self.config.test_cmd
        if test_names:
            cmd = f"{cmd} {' '.join(test_names)}"
        try:
            output = self._shell(cmd)
        except subprocess.SubprocessError as exc:
            output = str(exc)

        passed, failed = _parse_test_counts(output, self.config.name)
        return passed, failed, output

    @property
    def call_log(self) -> Sequence[str]:
        """Read-only view of commands executed."""
        return list(self.calls)

    @property
    def workdir(self) -> Path:
        """Path to the isolated working directory."""
        return self._workdir

    # ── internals ────────────────────────────────────────────────────

    def _shell(self, cmd: str, *, input_data: str | None = None) -> str:
        """Run *cmd* via bash in the workdir."""
        result = subprocess.run(
            ["bash", "-c", cmd],
            cwd=self._workdir,
            input=input_data,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        output = result.stdout
        if result.stderr:
            output = f"{output}{result.stderr}" if output else result.stderr
        return output
