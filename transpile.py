"""C++ to Rust transpilation pipeline for SWE-RL training.

Mines C++ repos for source+test pairs, generates transpilation tasks
where the agent must produce equivalent idiomatic Rust, and scores
with a partial-credit reward function (compile + tests pass).

Usage::

    # Mine a C++ repo for transpilation tasks
    python transpile.py mine --repo JSBSim-Team/jsbsim --output transpile.jsonl

    # Dry run
    python transpile.py mine --repo nasa/cfs --max-files 20 --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from docker_env import _parse_test_counts
from lang_config import EnvProtocol


# ── data model ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class TranspileRecord:
    """One transpilation task — serialised as a single JSONL line."""

    repo: str
    source_file: str
    source_code: str
    source_lang: str
    target_lang: str
    test_code: str
    test_names: list[str]
    instance_id: str


def make_transpile_id(repo: str, source_file: str) -> str:
    """Build instance ID: ``owner__repo-filename``."""
    return f"{repo.replace('/', '__')}-{Path(source_file).name}"


# ── JSONL writer ───────────────────────────────────────────────────


def write_transpile_tasks(records: Sequence[TranspileRecord], path: Path) -> int:
    """Append *records* as JSONL lines.  Returns count written."""
    if not records:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for record in records:
            line = json.dumps(asdict(record), ensure_ascii=False)
            fh.write(line + "\n")
        fh.flush()
    return len(records)


# ── prompt builder ─────────────────────────────────────────────────


def build_transpile_prompt(
    source_code: str,
    source_lang: str,
    target_lang: str,
    test_hints: Sequence[str] | None = None,
) -> str:
    """Build a system prompt for transpilation tasks.

    The agent receives the C++ source and must produce idiomatic Rust
    that compiles and passes equivalent tests.
    """
    test_section = ""
    if test_hints:
        test_list = "\n".join(f"  - {name}" for name in test_hints)
        test_section = (
            f"\nThe original {source_lang} code has these tests that must "
            f"pass in the Rust version:\n{test_list}\n"
            "Write equivalent Rust tests using #[test] functions.\n"
        )

    return (
        "You are a software engineering agent specializing in code transpilation.\n"
        f"Your task: convert the following {source_lang.upper()} code to "
        f"idiomatic {target_lang.capitalize()}.\n\n"
        "Requirements:\n"
        f"- Produce idiomatic {target_lang.capitalize()}, not a line-by-line translation\n"
        "- Use proper error handling (Result<T, E> instead of exceptions)\n"
        "- Use Rust naming conventions (snake_case functions, CamelCase types)\n"
        "- The code must compile with `cargo build`\n"
        "- The tests must pass with `cargo test`\n"
        "- Include a Cargo.toml with any needed dependencies\n"
        f"{test_section}\n"
        f"## Original {source_lang.upper()} Source\n\n"
        f"```{source_lang}\n{source_code}\n```\n\n"
        "Respond with exactly ONE bash command per message, in a fenced block:\n\n"
        "```bash\nyour command here\n```\n\n"
        "Useful commands:\n"
        "  cargo init --name transpiled\n"
        '  cat > src/lib.rs <<\'EOF\'\n  ...Rust code...\n  EOF\n'
        '  cat > Cargo.toml <<\'EOF\'\n  ...manifest...\n  EOF\n'
        "  cargo build\n"
        "  cargo test\n\n"
        "Start by creating the Rust project structure, then write the code, "
        "then build and test. One short sentence of reasoning, then exactly "
        "one command block."
    )


# ── reward function ────────────────────────────────────────────────

_BUILD_ERROR = re.compile(r"error\[E\d+\]|^error:|cannot find", re.MULTILINE)
_FINISHED = re.compile(r"Finished|Compiling.*Finished", re.DOTALL)


def reward_transpile(env: EnvProtocol) -> float:
    """Score a transpilation rollout.

    Returns a float in [0, 1]:
      - 0.0 = empty patch or build fails
      - 0.5 = builds successfully but no tests pass
      - 0.5 + (passed/total * 0.5) = builds + partial/full test pass
      - 1.0 = builds + all tests pass
    """
    diff = env.patch()
    if not diff:
        return 0.0

    # Check if code compiles
    build_output = env.run("cargo build 2>&1")
    if _BUILD_ERROR.search(build_output):
        return 0.0

    # Run tests
    test_output = env.run("cargo test 2>&1")
    passed, failed = _parse_test_counts(test_output, "rust")

    total = passed + failed
    if total == 0:
        return 0.5  # compiles but no tests found/ran

    test_score = passed / total
    return round(0.5 + (test_score * 0.5), 3)


# ── source-test pairing ───────────────────────────────────────────

# Source file extensions (not headers, not test files).
_SOURCE_EXTENSIONS = {
    "cpp": (".cpp", ".cc", ".cxx", ".c"),
}

_HEADER_EXTENSIONS = (".h", ".hpp", ".hxx")

# Test file patterns per language.
_TEST_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "cpp": (
        re.compile(r"_test\.c(?:pp|c|xx)?$"),
        re.compile(r"Test\.c(?:pp|c|xx)?$"),
        re.compile(r"test_.*\.c(?:pp|c|xx)?$"),
        re.compile(r"_tests\.c(?:pp|c|xx)?$"),
    ),
}


def _is_source_file(path: str, lang: str) -> bool:
    """Return True if *path* is a source file (not header, not test)."""
    exts = _SOURCE_EXTENSIONS.get(lang, ())
    if not any(path.endswith(ext) for ext in exts):
        return False
    if any(path.endswith(ext) for ext in _HEADER_EXTENSIONS):
        return False
    # Exclude test files
    patterns = _TEST_PATTERNS.get(lang, ())
    if any(pat.search(path) for pat in patterns):
        return False
    return True


def _find_test_for_source(source: str, all_files: list[str], lang: str) -> str | None:
    """Find the test file that corresponds to *source*."""
    stem = Path(source).stem  # e.g., "orbit" from "src/orbit.cpp"
    patterns = _TEST_PATTERNS.get(lang, ())

    for f in all_files:
        if not any(pat.search(f) for pat in patterns):
            continue
        # Check if the test file name contains the source stem
        test_stem = Path(f).stem
        if stem in test_stem:
            return f
    return None


def pair_source_and_tests(
    files: list[str],
    lang: str,
) -> list[tuple[str, str]]:
    """Pair source files with their test files.

    Returns list of ``(source_path, test_path)`` tuples.
    Only includes pairs where both exist.
    """
    pairs: list[tuple[str, str]] = []
    for f in files:
        if not _is_source_file(f, lang):
            continue
        test_file = _find_test_for_source(f, files, lang)
        if test_file:
            pairs.append((f, test_file))
    return pairs


# ── complexity estimator ──────────────────────────────────────────

_TEMPLATE_PATTERN = re.compile(r"template\s*<")


def estimate_complexity(source_code: str) -> str:
    """Estimate transpilation complexity: low, medium, or high.

    Based on line count and template usage.
    """
    lines = source_code.strip().split("\n")
    line_count = len(lines)
    template_count = len(_TEMPLATE_PATTERN.findall(source_code))

    if line_count > 150 or template_count > 3:
        return "high"
    if line_count > 50 or template_count > 1:
        return "medium"
    return "low"


# ── GitHub mining ──────────────────────────────────────────────────


def _check_gh() -> None:
    """Verify ``gh`` CLI is available."""
    try:
        subprocess.run(
            ["gh", "--version"],
            capture_output=True, check=True, timeout=10,
        )
    except FileNotFoundError:
        print(
            "Error: 'gh' CLI not found. Install from https://cli.github.com/",
            file=sys.stderr,
        )
        sys.exit(1)
    except subprocess.SubprocessError as exc:
        print(f"Error checking gh CLI: {exc}", file=sys.stderr)
        sys.exit(1)


def _gh(args: list[str], timeout: int = 30) -> str:
    """Run a ``gh`` command and return stdout."""
    result = subprocess.run(
        ["gh", *args],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _list_repo_files(repo: str, ref: str = "HEAD") -> list[str]:
    """List all files in a repo at a given ref."""
    raw = _gh([
        "api", f"repos/{repo}/git/trees/{ref}",
        "--jq", ".tree[].path",
        "-q", "recurse=1",
    ], timeout=60)
    return [line.strip() for line in raw.strip().split("\n") if line.strip()]


def _get_file_content(repo: str, path: str, ref: str = "HEAD") -> str:
    """Fetch a file's content from a repo."""
    return _gh([
        "api", f"repos/{repo}/contents/{path}",
        "--jq", ".content",
        "-H", "Accept: application/vnd.github.v3.raw",
    ], timeout=30)


def _get_default_branch_sha(repo: str) -> str:
    """Get the SHA of the default branch HEAD."""
    raw = _gh([
        "api", f"repos/{repo}",
        "--jq", ".default_branch",
    ], timeout=15)
    branch = raw.strip()
    sha_raw = _gh([
        "api", f"repos/{repo}/git/ref/heads/{branch}",
        "--jq", ".object.sha",
    ], timeout=15)
    return sha_raw.strip()


def mine_transpile_tasks(
    repo: str,
    max_files: int = 50,
    *,
    complexity_filter: str = "",
    dry_run: bool = False,
) -> list[TranspileRecord]:
    """Mine *repo* for C++ source+test pairs as transpilation tasks."""
    commit = _get_default_branch_sha(repo)
    all_files = _list_repo_files(repo, commit)
    pairs = pair_source_and_tests(all_files, "cpp")

    if max_files:
        pairs = pairs[:max_files]

    records: list[TranspileRecord] = []
    skipped = 0

    for source_path, test_path in pairs:
        try:
            source_code = _get_file_content(repo, source_path, commit)
            test_code = _get_file_content(repo, test_path, commit)
        except RuntimeError:
            skipped += 1
            continue

        if complexity_filter:
            complexity = estimate_complexity(source_code)
            if complexity != complexity_filter:
                skipped += 1
                continue

        # Extract test names from the test file content
        test_names = _extract_cpp_test_names(test_code)
        if not test_names:
            skipped += 1
            continue

        instance_id = make_transpile_id(repo, source_path)

        record = TranspileRecord(
            repo=repo,
            source_file=source_path,
            source_code=source_code,
            source_lang="cpp",
            target_lang="rust",
            test_code=test_code,
            test_names=test_names,
            instance_id=instance_id,
        )

        if dry_run:
            complexity = estimate_complexity(source_code)
            print(f"  [dry-run] {instance_id}: {len(test_names)} tests, "
                  f"{complexity} complexity, {len(source_code)} chars")
        else:
            records.append(record)

    print(f"  {repo}: {len(records)} transpile tasks, {skipped} skipped")
    return records


# Google Test / Catch2 extraction from file content (not diff).
_GTEST_PATTERN = re.compile(r"TEST(?:_F|_P)?\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)")
_CATCH2_PATTERN = re.compile(r'TEST_CASE\s*\(\s*"([^"]+)"')


def _extract_cpp_test_names(test_content: str) -> list[str]:
    """Extract test names from C++ test file content."""
    names: list[str] = []
    for m in _GTEST_PATTERN.finditer(test_content):
        names.append(f"{m.group(1)}.{m.group(2)}")
    for m in _CATCH2_PATTERN.finditer(test_content):
        names.append(m.group(1))
    return names


# ── CLI ────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="C++ to Rust transpilation pipeline for SWE-RL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python transpile.py mine --repo JSBSim-Team/jsbsim
              python transpile.py mine --repo nasa/cfs --max-files 20 --dry-run
              python transpile.py mine --repo nasa/cfs --complexity low
        """),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    mine_parser = sub.add_parser("mine", help="Mine a repo for transpilation tasks.")
    mine_parser.add_argument(
        "--repo", action="append", required=True,
        help="GitHub repo (owner/name). Can be repeated.",
    )
    mine_parser.add_argument(
        "--output", default="transpile_tasks.jsonl",
        help="Output JSONL path (default: transpile_tasks.jsonl).",
    )
    mine_parser.add_argument(
        "--max-files", type=int, default=50,
        help="Max source files to process per repo (default: 50).",
    )
    mine_parser.add_argument(
        "--complexity", choices=["low", "medium", "high"], default="",
        help="Filter by complexity level.",
    )
    mine_parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be mined without writing.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "mine":
        _check_gh()

        all_records: list[TranspileRecord] = []

        for repo in args.repo:
            print(f"Mining {repo} for transpile tasks...")
            records = mine_transpile_tasks(
                repo=repo,
                max_files=args.max_files,
                complexity_filter=args.complexity,
                dry_run=args.dry_run,
            )
            all_records.extend(records)

        if args.dry_run:
            print(f"\n[dry-run] Would write {len(all_records)} tasks "
                  f"to {args.output}")
            return

        if all_records:
            output_path = Path(args.output)
            count = write_transpile_tasks(all_records, output_path)
            print(f"\nWrote {count} transpile tasks to {output_path}")
        else:
            print("\nNo transpile tasks found.")


if __name__ == "__main__":
    main()
