# Plan: C++ to Rust Transpilation Mode

**Complexity**: Medium

## Summary
Add a transpilation pipeline that trains the agent to convert C++ aerospace code to idiomatic Rust. Mines C++ repos for source+test pairs, generates transpilation tasks, validates with cargo build/test, and scores with a partial-credit reward function.

## Patterns to Mirror
| Category | Source | Pattern |
|---|---|---|
| Data model | `task_miner.py:37` | `@dataclass(frozen=True)` for `TranspileRecord` |
| Prompts | `lang_config.py:232` | `build_system_prompt()` — returns formatted string |
| Reward | `docker_env.py:247` | `reward_test()` — returns float [0, 1] |
| JSONL I/O | `task_miner.py:61` | `write_tasks()` — `json.dumps(asdict(record))` |
| Tests | `tests/test_task_miner.py` | pytest classes, sample data as module constants |

## Files to Change
| File | Action | Why |
|---|---|---|
| `transpile.py` | CREATE | TranspileRecord, prompt builder, reward, mine, CLI |
| `tests/test_transpile.py` | CREATE | Tests for all transpile functions |

## Acceptance
- [ ] All tasks complete
- [ ] Validation passes
- [ ] Patterns mirrored, not reinvented
