# SWE-RL Architecture Diagrams

## Outer Loop (Training Loop)

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING LOOP                            │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  Task     │    │  Load Policy │    │  Parallel Rollouts    │  │
│  │  Queue    │───▶│  (+ 4-bit?) │───▶│  (ThreadPoolExecutor) │  │
│  │  (JSONL)  │    │              │    │                       │  │
│  └──────────┘    └──────────────┘    │  ┌─────┐ ┌─────┐     │  │
│       ▲                              │  │ R1  │ │ R2  │ ... │  │
│       │                              │  └──┬──┘ └──┬──┘     │  │
│       │                              └─────┼───────┼────────┘  │
│       │                                    ▼       ▼            │
│       │                              ┌─────────────────┐       │
│       │                              │  reward_test()   │       │
│       │                              │  per rollout     │       │
│       │                              │  → [0.0 … 1.0]  │       │
│       │                              └────────┬────────┘       │
│       │                                       ▼                 │
│       │                              ┌─────────────────┐       │
│       │                              │  GRPO Update     │       │
│       │                              │                  │       │
│       │                              │  1. Group rewards│       │
│       │                              │  2. Advantages   │       │
│       │                              │     A = r - mean │       │
│       │                              │  3. REINFORCE    │       │
│       │                              │     loss         │       │
│       │                              │  4. KL penalty   │       │
│       │                              │  5. optimizer    │       │
│       │                              │     step         │       │
│       │                              └────────┬────────┘       │
│       │                                       ▼                 │
│       │                              ┌─────────────────┐       │
│       │          next task           │  Save Checkpoint │       │
│       └──────────────────────────────│  Log to JSONL    │       │
│                                      └─────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### How It Works

1. **Task Queue**: Load training tasks from JSONL (mined from GitHub PRs)
2. **Load Policy**: Initialize the language model, optionally with 4-bit quantization
3. **Parallel Rollouts**: Run multiple agent rollouts concurrently using ThreadPoolExecutor
4. **Reward**: Score each rollout by running held-out tests (`reward_test()` returns 0.0-1.0)
5. **GRPO Update**: Group Relative Policy Optimization compares rollouts within a group,
   computes advantages relative to group mean, applies REINFORCE loss with KL penalty
6. **Checkpoint**: Save model weights and log trajectories, then loop to next task

---

## Inner Loop (Single Rollout)

```
┌────────────────────────────────────────────────────────────┐
│                    SINGLE ROLLOUT (run_agent)               │
│                                                            │
│  ┌────────────────────┐                                    │
│  │ System Prompt       │   build_system_prompt(config,     │
│  │ (lang-specific +    │   target_file) + MFE context      │
│  │  MFE context)       │                                   │
│  └────────┬───────────┘                                    │
│           ▼                                                │
│  ┌────────────────────┐                                    │
│  │ messages = [system, │                                   │
│  │   user: issue_text] │                                   │
│  └────────┬───────────┘                                    │
│           ▼                                                │
│  ┌─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐     │
│  │         TURN LOOP (max_turns=10)                  │     │
│  │                                                   │     │
│  │  ┌──────────────┐    ┌─────────────────────┐      │     │
│  │  │ Model        │    │ Parse ```bash        │      │     │
│  │  │ Inference    │───▶│ code block          │      │     │
│  │  │ (generate)   │    │ from response       │      │     │
│  │  └──────────────┘    └──────────┬──────────┘      │     │
│  │                                 ▼                  │     │
│  │                      ┌─────────────────────┐      │     │
│  │                      │ env.run(command)     │      │     │
│  │                      │                     │      │     │
│  │                      │ DockerEnv: docker   │      │     │
│  │                      │   exec <container>  │      │     │
│  │                      │ SubprocessEnv:      │      │     │
│  │                      │   bash -c in tmpdir │      │     │
│  │                      └──────────┬──────────┘      │     │
│  │                                 ▼                  │     │
│  │                      ┌─────────────────────┐      │     │
│  │                      │ Observe output      │      │     │
│  │                      │ Append to messages  │      │     │
│  │                      └──────────┬──────────┘      │     │
│  │                                 │                  │     │
│  │                        ┌────────┴────────┐        │     │
│  │                        │ No bash block?  │        │     │
│  │                        │ → break (done)  │        │     │
│  │                        └─────────────────┘        │     │
│  └─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘     │
│           ▼                                                │
│  ┌────────────────────┐                                    │
│  │ env.patch()         │  → git diff of all changes        │
│  │ return (messages,   │                                   │
│  │         patch)      │                                   │
│  └────────────────────┘                                    │
└────────────────────────────────────────────────────────────┘
```

### How It Works

1. **System Prompt**: Built from `LangConfig` with language-specific commands and
   optional MFE context (Angular/Next.js patterns)
2. **Message Init**: Start with system prompt + user message containing the issue text
3. **Turn Loop**: Up to 10 turns of model inference → parse bash command → execute → observe
4. **Environment**: Commands run in either DockerEnv (real container) or SubprocessEnv (tmpdir)
5. **Termination**: Loop ends when model produces no bash block (signals it's done)
6. **Output**: Return the conversation messages and the git diff patch

---

## End-to-End Pipeline (Training to Inference)

```
                        TRAINING PHASE
                        ══════════════

  ┌───────────┐     ┌──────────────┐     ┌──────────────┐
  │ Task Miner│     │ Training     │     │ Trained      │
  │           │────▶│ Loop         │────▶│ Model        │
  │ GitHub PRs│     │ (GRPO)       │     │ Checkpoint   │
  │ → JSONL   │     │              │     │              │
  └───────────┘     └──────────────┘     └──────┬───────┘
                                                │
       ┌────────────────────────────────────────┘
       ▼
                        INFERENCE PHASE
                        ═══════════════

  ┌───────────┐     ┌──────────────┐     ┌──────────────┐
  │ New Issue  │     │ run_agent()  │     │ Generated    │
  │ (problem   │────▶│ with trained │────▶│ Patch        │
  │  statement)│     │ model        │     │ (git diff)   │
  └───────────┘     │              │     └──────────────┘
                    │ DockerEnv /  │
                    │ SubprocessEnv│
                    └──────────────┘
```

### Training Phase

1. **Task Miner** scrapes merged bug-fix PRs from GitHub repos (Rust, C#)
2. Outputs JSONL with: repo, base_commit, test_patch, fail_to_pass, problem_statement, patch
3. **Training Loop** runs GRPO over these tasks, producing a trained checkpoint

### Inference Phase

1. A new issue (bug report) arrives as a problem statement
2. The trained model runs `run_agent()` in a sandbox environment
3. Output is a git diff patch that fixes the bug

---

## Task Miner Data Flow

```
  GitHub API                    task_miner.py                  Pipeline
  ═════════                    ═══════════════                 ════════

  ┌─────────────┐
  │ Merged PRs  │   gh pr list --state merged --label bug
  │ with tests  │──────────────────┐
  └─────────────┘                  ▼
                          ┌────────────────┐
  ┌─────────────┐         │ For each PR:   │
  │ PR diff     │◀────────│                │
  │ (full)      │         │ 1. Fetch diff  │
  └──────┬──────┘         └────────────────┘
         ▼
  ┌──────────────┐
  │ split_diff() │
  │              │
  │ Test files:  │──▶ test_patch     ──▶ ┌──────────────┐
  │  *_test.rs   │                       │              │
  │  *Tests.cs   │   extract_test_names  │  TaskRecord  │
  │              │──▶ fail_to_pass  ──▶  │  (JSONL)     │
  │ Code files:  │                       │              │
  │  *.rs, *.cs  │──▶ patch         ──▶  │  {repo,      │
  └──────────────┘                       │   base_commit│
                                         │   test_patch │
  ┌─────────────┐                        │   fail_to_pass
  │ PR body     │──▶ problem_statement ▶ │   patch,     │
  │ + title     │                        │   lang,      │
  └─────────────┘                        │   instance_id│
                                         │  }           │
  ┌─────────────┐                        │              │
  │ base_commit │──▶ merge_base     ──▶  │              │
  │ (PR parent) │                        └──────┬───────┘
  └─────────────┘                               │
                                                ▼
                                         tasks.jsonl
                                         (training input)
```

### How It Works

1. **Query GitHub**: Find merged PRs labeled "bug" (or with bug-fix keywords) in target repos
2. **Fetch Diff**: Get the full unified diff for each PR
3. **Split Diff**: Separate test file changes from code file changes using language-specific
   patterns (e.g., `*_test.rs`, `*Tests.cs`)
4. **Extract Test Names**: Parse test function names from the test patch
   - Rust: `#[test] fn test_name()`
   - C#: `[Fact] public void TestName()` / `[Test] public void TestName()`
5. **Build Record**: Combine repo, base_commit, patches, test names, and PR description
6. **Output JSONL**: One line per task, ready for the training pipeline

---

## Component Map

```
┌─────────────────────────────────────────────────────────┐
│                      CLI Layer                          │
│  swe_grpo_one_step.py          task_miner.py (planned)  │
└────────┬──────────────────────────────┬─────────────────┘
         │                              │
         ▼                              ▼
┌─────────────────┐          ┌──────────────────┐
│  Agent Runtime   │          │  Data Generation  │
│                  │          │                   │
│  run_agent()     │          │  mine_repo()      │
│  build_system    │          │  split_diff()     │
│    _prompt()     │          │  extract_test     │
│                  │          │    _names()        │
└────────┬────────┘          └──────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                   Environment Layer                      │
│                                                         │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────┐    │
│  │ MockEnv  │  │  DockerEnv   │  │ SubprocessEnv  │    │
│  │ (tests)  │  │  (container) │  │ (Kaggle/Colab) │    │
│  └──────────┘  └──────────────┘  └────────────────┘    │
│         ▲              ▲                ▲                │
│         └──────────────┴────────────────┘                │
│                   EnvProtocol                            │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                   Config Layer                           │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  LangConfig   │  │ ModelConfig  │  │ RolloutLogger│  │
│  │  (lang_config │  │ (model_     │  │ (rollout_    │  │
│  │   .py)        │  │  config.py) │  │  logger.py)  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                         │
│  Languages: python, rust, csharp, nodejs, angular,      │
│             nextjs                                       │
│  Models: qwen-0.5b/1.5b/3b/7b, deepseek-1.3b/6.7b,    │
│          codellama-7b, starcoder2-3b                     │
└─────────────────────────────────────────────────────────┘
```
