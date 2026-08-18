# Plan: Task Miner for Rust/C# Training Data

**Complexity**: Medium

## Summary
Build `task_miner.py` that scrapes merged bug-fix PRs from GitHub repos (Rust, C#), separates test patches from code patches, extracts failing test names, and outputs JSONL matching our SWE-RL training pipeline format.

## Patterns to Mirror
| Category | Source | Pattern |
|---|---|---|
| Naming | `lang_config.py:30` | `@dataclass(frozen=True)` for immutable records |
| Errors | `lang_config.py:194` | `raise KeyError(f"Unknown ... Known: {known}")` |
| JSONL I/O | `rollout_logger.py:86-88` | `json.dumps(asdict(record))` + flush per write |
| CLI | `swe_grpo_one_step.py:19` | `argparse` with descriptive help |
| Tests | `tests/test_angular_nextjs.py` | pytest classes grouped by feature |

## Files to Change
| File | Action | Why |
|---|---|---|
| `task_miner.py` | CREATE | Core mining logic + CLI |
| `tests/test_task_miner.py` | CREATE | Unit tests for diff splitting, test name extraction, record building |

## Tasks
### Task 1: Data Model + JSONL Writer
- **Action**: `TaskRecord` frozen dataclass + `write_tasks()` JSONL writer
- **Mirror**: `rollout_logger.py` pattern
- **Validate**: `python -m pytest tests/test_task_miner.py -k TaskRecord`

### Task 2: Diff Splitter
- **Action**: `split_diff(unified_diff, lang)` → `(test_patch, code_patch)`
- **Mirror**: Language-aware file patterns from `lang_config.py`
- **Validate**: `python -m pytest tests/test_task_miner.py -k split_diff`

### Task 3: Test Name Extractor
- **Action**: `extract_test_names(test_patch, lang)` → `list[str]`
- **Mirror**: Regex patterns like `docker_env.py` `_PASS_PATTERNS`
- **Validate**: `python -m pytest tests/test_task_miner.py -k extract_test`

### Task 4: GitHub PR Miner
- **Action**: `mine_repo(repo, lang, max_prs)` using `gh` CLI
- **Validate**: `python -m pytest tests/test_task_miner.py -k mine`

### Task 5: CLI Entry Point
- **Action**: argparse with `--repo`, `--lang`, `--output`, `--max-prs`, `--dry-run`
- **Validate**: `python task_miner.py --help`

## Validation
```bash
python -m pytest tests/test_task_miner.py -v
```

## Risks
| Risk | Likelihood | Mitigation |
|---|---|---|
| `gh` CLI not installed | Medium | Check at startup, clear error |
| PRs without test changes | High | Filter early, skip gracefully |
| Rate limiting | Medium | `--max-prs` cap |
| Exotic test patterns | Low | Start common, log skipped |

## Acceptance
- [ ] All tasks complete
- [ ] Validation passes
- [ ] Patterns mirrored, not reinvented
