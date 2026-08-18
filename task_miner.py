"""Mine GitHub repos for SWE-RL training tasks.

Finds merged bug-fix PRs that include test changes, separates test
patches from code patches, extracts failing test names, and writes
JSONL in the format our training pipeline expects.

Requires the ``gh`` CLI to be installed and authenticated.

Usage::

    # Mine a single Rust repo
    python task_miner.py --repo BurntSushi/ripgrep --lang rust

    # Mine multiple C# repos, cap at 50 PRs each
    python task_miner.py --repo dotnet/runtime --repo dotnet/aspnetcore \\
        --lang csharp --max-prs 50 --output tasks.jsonl

    # Dry run — print what would be mined without writing
    python task_miner.py --repo tokio-rs/tokio --lang rust --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


# ── data model ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class TaskRecord:
    """One training task — serialised as a single JSONL line."""

    repo: str
    base_commit: str
    test_patch: str
    fail_to_pass: list[str]
    problem_statement: str
    lang: str
    instance_id: str
    patch: str


def make_instance_id(repo: str, pr_number: int) -> str:
    """Build a SWE-bench-style instance ID: ``owner__repo-PR``."""
    return f"{repo.replace('/', '__')}-{pr_number}"


# ── JSONL writer ───────────────────────────────────────────────────


def write_tasks(records: Sequence[TaskRecord], path: Path) -> int:
    """Append *records* as JSONL lines to *path*.  Returns count written."""
    if not records:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for record in records:
            line = json.dumps(asdict(record), ensure_ascii=False)
            fh.write(line + "\n")
        fh.flush()
    return len(records)


# ── diff splitting ─────────────────────────────────────────────────

# Patterns that identify test files per language.
_TEST_FILE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "rust": (
        re.compile(r"_test\.rs$"),
        re.compile(r"tests/.*\.rs$"),
        re.compile(r"test_.*\.rs$"),
    ),
    "csharp": (
        re.compile(r"Tests?\.cs$"),
        re.compile(r"_test\.cs$", re.I),
        re.compile(r"\.Tests?/.*\.cs$"),
        re.compile(r"tests?/.*\.cs$", re.I),
    ),
}

# Regex to extract the file path from a "diff --git a/... b/..." header.
_DIFF_HEADER = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)


def _is_test_file(path: str, lang: str) -> bool:
    """Return True if *path* looks like a test file for *lang*."""
    patterns = _TEST_FILE_PATTERNS.get(lang, ())
    return any(pat.search(path) for pat in patterns)


def split_diff(unified_diff: str, lang: str) -> tuple[str, str]:
    """Split a unified diff into ``(test_patch, code_patch)``.

    Each returned string is itself a valid unified diff (or empty).
    """
    if not unified_diff.strip():
        return "", ""

    # Split into per-file chunks.  Each chunk starts with "diff --git".
    chunks = re.split(r"(?=^diff --git )", unified_diff, flags=re.MULTILINE)

    test_chunks: list[str] = []
    code_chunks: list[str] = []

    for chunk in chunks:
        chunk = chunk.rstrip()
        if not chunk:
            continue
        m = _DIFF_HEADER.match(chunk)
        if not m:
            continue
        file_path = m.group(2)
        if _is_test_file(file_path, lang):
            test_chunks.append(chunk)
        else:
            code_chunks.append(chunk)

    test_patch = "\n".join(test_chunks) if test_chunks else ""
    code_patch = "\n".join(code_chunks) if code_chunks else ""
    return test_patch, code_patch


# ── test name extraction ──────────────────────────────────────────

# Rust: #[test] or #[tokio::test] followed by fn <name>
_RUST_TEST_FN = re.compile(
    r"^\+\s*#\[(tokio::)?test\]"   # added line with test attribute
    r".*?"                          # optional lines between attribute and fn
    r"^\+\s*(?:async\s+)?fn\s+(\w+)",
    re.MULTILINE | re.DOTALL,
)

# Simpler line-by-line approach for Rust (more reliable across chunks).
_RUST_TEST_ATTR = re.compile(r"^\+\s*#\[(tokio::)?test\]")
_RUST_FN_DECL = re.compile(r"^\+\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)")

# C#: [Fact], [Test], [Theory], [TestCase(...)] followed by method declaration.
_CSHARP_TEST_ATTR = re.compile(
    r"^\+\s*\[(Fact|Test|Theory|TestCase|TestMethod)(?:\(.*?\))?\]"
)
_CSHARP_METHOD = re.compile(
    r"^\+\s*(?:public|private|internal|protected)?\s*"
    r"(?:static\s+)?(?:async\s+)?(?:Task\s+|void\s+|[\w<>\[\],\s]+\s+)"
    r"(\w+)\s*\("
)


def extract_test_names(test_patch: str, lang: str) -> list[str]:
    """Extract added test function/method names from a test patch."""
    if not test_patch.strip():
        return []

    lines = test_patch.split("\n")
    names: list[str] = []

    if lang == "rust":
        saw_test_attr = False
        for line in lines:
            if _RUST_TEST_ATTR.match(line):
                saw_test_attr = True
                continue
            if saw_test_attr:
                m = _RUST_FN_DECL.match(line)
                if m:
                    names.append(m.group(1))
                    saw_test_attr = False
                elif not line.startswith("+"):
                    saw_test_attr = False

    elif lang == "csharp":
        saw_test_attr = False
        for line in lines:
            if _CSHARP_TEST_ATTR.match(line):
                saw_test_attr = True
                continue
            if saw_test_attr:
                m = _CSHARP_METHOD.match(line)
                if m:
                    names.append(m.group(1))
                    saw_test_attr = False
                elif line.startswith("+") and line.strip() == "+":
                    continue  # blank added line, keep looking
                elif line.startswith("+") and _CSHARP_TEST_ATTR.match(line):
                    continue  # stacked attributes like [TestCase]
                elif not line.startswith("+"):
                    saw_test_attr = False

    return names


# ── GitHub PR mining ───────────────────────────────────────────────


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


def _gh_json(args: list[str], timeout: int = 30) -> str:
    """Run a ``gh`` command and return its stdout."""
    result = subprocess.run(
        ["gh", *args],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _list_bug_prs(repo: str, max_prs: int) -> list[dict]:
    """List merged bug-fix PRs for *repo* using ``gh``."""
    raw = _gh_json([
        "pr", "list",
        "--repo", repo,
        "--state", "merged",
        "--search", "label:bug fix in:title",
        "--limit", str(max_prs),
        "--json", "number,title,body,mergeCommit,baseRefName",
    ], timeout=60)
    return json.loads(raw)


def _get_pr_diff(repo: str, pr_number: int) -> str:
    """Fetch the unified diff for a PR."""
    return _gh_json([
        "pr", "diff", str(pr_number),
        "--repo", repo,
    ], timeout=60)


def _get_merge_base(repo: str, pr_number: int) -> str:
    """Get the base commit SHA (merge base) for a PR."""
    raw = _gh_json([
        "pr", "view", str(pr_number),
        "--repo", repo,
        "--json", "mergeCommit",
    ], timeout=30)
    data = json.loads(raw)
    oid = data.get("mergeCommit", {}).get("oid", "")
    if not oid:
        raise ValueError(f"No merge commit found for {repo}#{pr_number}")
    # The base commit is the parent of the merge commit.
    # Use gh api to get the parent.
    raw_commit = _gh_json([
        "api", f"repos/{repo}/commits/{oid}",
        "--jq", ".parents[0].sha",
    ], timeout=30)
    return raw_commit.strip()


def mine_repo(
    repo: str,
    lang: str,
    max_prs: int = 100,
    *,
    dry_run: bool = False,
) -> list[TaskRecord]:
    """Mine *repo* for training tasks.  Returns a list of TaskRecords."""
    if lang not in _TEST_FILE_PATTERNS:
        known = sorted(_TEST_FILE_PATTERNS.keys())
        raise KeyError(f"Unknown language {lang!r}. Known: {known}")

    prs = _list_bug_prs(repo, max_prs)
    records: list[TaskRecord] = []
    skipped = 0

    for pr in prs:
        pr_number = pr["number"]
        title = pr.get("title", "")
        body = pr.get("body", "") or ""

        try:
            diff = _get_pr_diff(repo, pr_number)
        except RuntimeError:
            skipped += 1
            continue

        test_patch, code_patch = split_diff(diff, lang)

        # Skip PRs without both test and code changes.
        if not test_patch or not code_patch:
            skipped += 1
            continue

        test_names = extract_test_names(test_patch, lang)
        if not test_names:
            skipped += 1
            continue

        try:
            base_commit = _get_merge_base(repo, pr_number)
        except (RuntimeError, ValueError):
            skipped += 1
            continue

        instance_id = make_instance_id(repo, pr_number)
        problem_statement = f"{title}\n\n{body}".strip()

        record = TaskRecord(
            repo=repo,
            base_commit=base_commit,
            test_patch=test_patch,
            fail_to_pass=test_names,
            problem_statement=problem_statement,
            lang=lang,
            instance_id=instance_id,
            patch=code_patch,
        )

        if dry_run:
            print(f"  [dry-run] {instance_id}: {len(test_names)} tests, "
                  f"{len(code_patch)} chars patch")
        else:
            records.append(record)

    print(f"  {repo}: {len(records)} tasks mined, {skipped} PRs skipped")
    return records


# ── CLI ────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mine GitHub repos for SWE-RL training tasks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python task_miner.py --repo BurntSushi/ripgrep --lang rust
              python task_miner.py --repo dotnet/runtime --lang csharp --max-prs 50
              python task_miner.py --repo tokio-rs/tokio --lang rust --dry-run
        """),
    )
    parser.add_argument(
        "--repo", action="append", required=True,
        help="GitHub repo (owner/name). Can be repeated.",
    )
    parser.add_argument(
        "--lang", required=True, choices=sorted(_TEST_FILE_PATTERNS.keys()),
        help="Target language.",
    )
    parser.add_argument(
        "--output", default="tasks.jsonl",
        help="Output JSONL path (default: tasks.jsonl).",
    )
    parser.add_argument(
        "--max-prs", type=int, default=100,
        help="Max merged PRs to scan per repo (default: 100).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be mined without writing.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    _check_gh()

    all_records: list[TaskRecord] = []

    for repo in args.repo:
        print(f"Mining {repo} ({args.lang})...")
        records = mine_repo(
            repo=repo,
            lang=args.lang,
            max_prs=args.max_prs,
            dry_run=args.dry_run,
        )
        all_records.extend(records)

    if args.dry_run:
        print(f"\n[dry-run] Would write {len(all_records)} tasks to {args.output}")
        return

    if all_records:
        output_path = Path(args.output)
        count = write_tasks(all_records, output_path)
        print(f"\nWrote {count} tasks to {output_path}")
    else:
        print("\nNo tasks found. Try different repos or labels.")


if __name__ == "__main__":
    main()
