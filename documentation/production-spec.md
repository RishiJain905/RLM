# Production spec

> **Step 1 of [production-build-plan.md](production-build-plan.md).** Written 2026-10-10. This spec pins the interfaces and settings that every later step is coded against: the bridge API, prompt text, reward, episode limits, training settings, difficulty controller, code layout and checkpoint manifest.
>
> **Status:** approved by Rishi on 2026-10-10, including the additions marked *(mine)*. Step 1 is complete. Nothing has been built, run, committed or pushed, and no W&B credit has been spent.
>
> **Spec v2 (2026-10-12):** §12 adds the two edit families from [code-editing-plan.md](code-editing-plan.md), approved by Rishi. It replaces prompt v1 with v2 (§12.2), the step 7 mix (§12.7) and the inference split (§12.11). Sections 1-11 are otherwise unchanged and still govern the read families.
>
> **Sources:** [rl-run-research.md](rl-run-research.md) (the source of truth for the research), the archived 20-step code (`rlm_rl/archive/rlm_env.py`, `repl_worker.py`, `ticket_tasks.py`, `scripts/archive/run_rlsd_vs_grpo.py`), and ART 0.5.20's own source in `.venv`. Values marked *(mine)* are my defaults, not the research's. Values marked *(verify)* still need checking against the live system.

## 0. Decisions applied

| # | Decision | Answer | Decided |
|---|---|---|---|
| 1a | Bridge function names | `llm`, `llm_batch`, `submit`, `context` | Rishi, 2026-10-10 |
| 1b | Partial credit | Calibrate against the blind guess | Rishi, 2026-10-10 |
| 1c | CISPO importance-weight clip | Restore ART's default | Rishi, 2026-10-10 |
| E-a to E-g | Edit families, real-task source, binary edit reward, budget, normal-agent arm, feature tasks, second-seed timing | Every recommended option in code-editing-plan.md §15; E7 probe dropped; reserve $10 (§12) | Rishi, 2026-10-12 |
| LR | Learning rate | 1e-5 for the second seed. Production is 2e-5 if the second seed's stability checks pass (§5.2), otherwise 1e-5. It falls back to 1e-5 on instability. | Rishi, 2026-10-10 |

These decisions are still open and touch this spec. Each section notes where.
- **3b**, how the sandbox reaches the bridge: it changes only the transport in §1.4, not the API.
- **3c**, which packages are in the sandbox: it decides whether the prompt gets one extra line (§2).
- **7a and 7c** (both decided 2026-10-11), the domain mix: it sets the task sampling weights in §6. 7c dropped cyber, so the mix is 45% aggregation, 40% code generators and 15% repo localization.
- **11a**, which model serves sub-calls once deployed: in training, sub-calls are always served by the frozen base model.

---

## 1. Bridge API v1 (frozen)

One module, `rlm_rl/bridge.py`, defines this API, the prompt text and the wire protocol. The training environment and the Part 2 harness both import it, and a test diffs their system prompts (H10).

### 1.1 What the model sees in the REPL

```python
context: str                                  # the whole input, read-only
llm(prompt: str) -> str                       # one sub-call
llm_batch(prompts: list[str]) -> list[str]    # sub-calls in parallel, outputs in input order
submit(answer) -> None                        # final answer; the episode ends after this cell
```

Nothing else is reachable from the sandbox. There is no network, no file access outside scratch, and no import of project modules (H2-H4).

### 1.2 Rules

- **Budget.** Each episode gets 30 sub-calls. `llm_batch` with n prompts uses n of them. A call that would go over the budget uses nothing and raises `RuntimeError("sub-call budget exhausted (30 total)")` inside the REPL. The model sees that error like any other.
- **Prompt size.** A prompt longer than 24,000 characters is cut to 24,000 and ends with `[prompt truncated]`.
- **Sub-call model.** In training this is the frozen base model, Qwen3.8-27B, with these settings:

  | Setting | Value |
  |---|---|
  | Temperature | 0.3 |
  | Max output tokens | 1,024 |
  | Thinking | off |

  The settings match the 20-step run, except that thinking is now pinned off explicitly (§5.1).
- **`submit`.**
  - The answer is sent as JSON. Supported types are int, float, str, or a list of those; anything else is converted with `str()`.
  - The episode ends when the cell that called `submit` finishes. If that cell calls it more than once, the last value counts.
  - *(mine)* The 20-step run turned every answer into a string, which breaks list answers.
- **Infra failures** are a sub-call that still fails after retries, a 5xx error, an unreachable bridge, or a kernel crash not caused by model code.
  - The episode stops and is flagged `infra_error`, and it never reaches training (H5).
  - The model is never shown a fake answer. In the 20-step run it was shown `"[sub-call failed: rate limited]"`.

### 1.3 Concurrency

All root turns and sub-calls share one limiter, capped at 8 requests in flight. Retries use jittered exponential backoff, with up to 6 tries starting at 2 s. Retries running out means `infra_error` (H6).

### 1.4 Wire protocol

- Requests go one JSON message per line from the REPL to the host: `{"op": "llm", "prompts": [...]}`.
- The host replies with `{"outputs": [...]}` or `{"error": "..."}`.
- `submit` never crosses the bridge. Its value comes back with the cell result, as in the 20-step run.
- The transport is a Unix socket mounted into a `--network none` container, or a relay, depending on **3b**. The messages are the same either way.

---

## 2. Prompt text v1 (pinned)

> Superseded by prompt v2 (§12.2) once it is built. Every checkpoint records which version its run used (§8).

The bridge module exports this text with `PROMPT_VERSION = "v1"`. The hash of all three texts below is saved with every checkpoint (§8).

### 2.1 System prompt

```text
You are a Recursive Language Model. You answer a question about a long text that you cannot see directly. The text is stored in a persistent Python REPL as the string variable `context`.

Each turn, write a short plan and then exactly one ```python code block. The REPL runs it and shows you what it prints (truncated to 2000 characters). Variables persist between turns.

Available in the REPL:
- `context`: the full text (str)
- `llm(prompt: str) -> str`: ask a language model; it can read about 20,000 characters per call
- `llm_batch(prompts: list[str]) -> list[str]`: ask several prompts in parallel; each prompt counts as one sub-call
- `submit(answer)`: give your final answer and end the episode

You have at most 12 turns and 30 sub-calls in total. Look at the structure of `context` before processing it, decompose the work, use code for exact filtering and arithmetic, and use sub-calls for judgments that need reading comprehension. Call `submit()` once you are confident.
```

Changes from the 20-step prompt:
- `llm_query` is renamed to `llm`, per 1a.
- The limits go from 6 turns and 20 sub-calls to 12 turns and 30 sub-calls.
- The `llm_batch` line now says each prompt counts as one sub-call.

If 3c settles on the standard library only, one line gets added before the last paragraph: `Only the Python standard library is available.`

### 2.2 First user message (per task)

```text
Question: {question}

`context` holds {n_chars:,} characters in {n_lines} lines. The first 3 lines are:
{first_3_lines}
```

This is unchanged from the 20-step run.

### 2.3 Observation after each turn

This keeps the 20-step format, with the new limits:

```text
REPL output:
{stdout, cut to 2000 characters, then "[... N more characters truncated]"}
Error:
{last 6 lines of the traceback, only if there was one}
({turns_left} turns left, {subcalls_left} sub-calls left)
Your next turn is the last one: it must call submit() with your best answer.   <- only when 1 turn is left
```

If nothing was printed, the first line reads `REPL output: (nothing printed)`.

A reply with no code block gets `Error: No ```python code block found. Reply with exactly one ```python block.` and uses up the turn.

---

## 3. Reward

Every rollout gets one scalar reward. It is computed on the host, in this order:

1. **No `submit`, or an answer that can't be parsed: 0.**
2. **Exact answer: 1.** What counts as exact depends on the answer type:

   | Answer type | Exact means |
   |---|---|
   | Integer | Equal, after parsing the first number in the answer and removing `$` and `,` |
   | Decimal | Within 0.005 |
   | Entity (host, account, function name) | Equal after trimming whitespace and quotes, ignoring case |
   | List | The same set, after normalizing each item the same way |

3. **Near miss: calibrated partial credit (1b).**
   - **Numbers:** `0.8 × max(0, 1 − err / e_blind)`.
     - `err` is the distance between the submitted answer and the correct one.
     - `e_blind` is the distance between the blind guess and the correct one.
     - The blind guess is whatever an answer gets from metadata alone, without reading. For the ticket tasks, that's `round(n_lines × the target class's share)`.
     - If `e_blind = 0`, a near miss earns 0.
   - **Lists:** `0.8 × max(0, (F1 − F1_blind) / (1 − F1_blind))` *(mine)*. This applies the same idea to set-F1. `F1_blind` is the set-F1 of the generator's blind strategy, for example submitting every candidate entity visible in the prompt.
   - **Entities:** no partial credit. A single host or account is either right or wrong.
   - Every generator must provide its blind guess. The host computes it, and it never enters the sandbox.
4. **A hack detector fired: 0.** This happens *before* the group mean, so the hack becomes a negative example (H11). The rules are in §7.
5. **`infra_error`:** the episode leaves its group and is not scored (H5).

Exact accuracy is logged next to the reward, and it is the headline metric. There is no length penalty or process penalty at the start (research §4.1).

**Check before training (H1):** a blind guesser scores no more than its exact-hit rate on every task family.

---

## 4. Episode limits

> Edit families use their own limits (§12.3).

| Limit | 20-step run | Production |
|---|---|---|
| Sub-calls per episode | 20 | 30 |
| Root turns | 6 | 12 |
| Root output tokens per turn | 1,536 | 3,072 |
| Sub-call prompt size | 24,000 chars | 24,000 chars |
| Sub-call output tokens | 1,024 | 1,024 |
| REPL output shown | 2,000 chars | 2,000 chars; 8,000 from prompt v2 (§12.3) |
| Traceback shown | last 6 lines | last 6 lines |
| Cell timeout | 120 s, including time queued for sub-calls | 120 s of compute; time blocked on the bridge doesn't count |
| After a timeout or crash | REPL restarted, variables lost | Same, and the model is told |

Turn and token limits are caps, not targets.

---

## 5. Training settings

### 5.1 The train call

```python
await backend.train(
    model, groups,
    learning_rate=LR,              # §5.2
    loss_fn="cispo",
    epsilon=1.0, epsilon_high=4.0, # ART 0.5.20's CISPO defaults, pinned explicitly (1c)
    scale_rewards=False,
    kl_penalty_coef=0.0,           # KL to base is logged, not penalized (§5.3)
    packed_sequence_length=PSL,    # set in step 6 from measured root transcript lengths
)
```

**What 1c changes:**
- ART 0.5.20 clips CISPO's importance weight to [1 − ε, 1 + ε_high], which is [0, 5] at the defaults (`art/loss.py`).
- The 20-step run passed `epsilon = epsilon_high = 0`, so every weight was forced to exactly 1. That's why its `clipped_token_fraction` read ~0.97 on every step.
- With the defaults restored, only weights above 5 get clipped. The clipped fraction should drop to near 0, and that becomes the new baseline to watch.
- We pass both values explicitly instead of relying on server defaults.

| Setting | Value | Note |
|---|---|---|
| Base model | `Qwen/Qwen3.8-27B`, LoRA | |
| ART | 0.5.20, pinned | |
| Episodes per step | 8 tasks × 4 rollouts | |
| Root sampling temperature | 1.0 for training and eval | Every earlier eval used 1.0, so numbers stay comparable |
| Thinking | off | Pinned with `chat_template_kwargs={"enable_thinking": False}` on every root call and sub-call *(verify)*. The 20-step run relied on the server default. |
| Logprobs | `logprobs=True` and `prompt_logprobs=0` on root calls | Needed to rebuild token ids, which W&B's gateway strips |
| Learning rate | 1e-5, then per §5.2 | ART's serverless default is 5e-6, so we always pass it |
| Reward std scaling | off (`scale_rewards=False`) | |
| KL penalty | 0 | ART supports `kl_penalty_coef` if the fallback is ever needed (§5.3) |

### 5.2 Learning-rate rule (agreed 2026-10-10)

1. **Second seed: 1e-5.** It has to match the first run to count as a replication.
2. **Before production, read the second seed's stability curves.** Production starts at **2e-5** only if all of these held over its 20 steps:
   - entropy is roughly flat, with the steps 16-20 average at least 80% of the steps 1-5 average *(proposed threshold)*
   - grad norm never goes above 3× its median *(proposed)*
   - `probs_corr` stays above 0.98; the first 20-step run sat at ~0.997 *(proposed)*

   If any of these fails, production stays at 1e-5. I'll show you the curves and the call before production starts.
3. **Fallback during production at 2e-5.** We roll back to the last good checkpoint and continue at 1e-5, logged as a notice on the dashboard, if either of these happens:
   - entropy falls more than 30% within 10 steps *(proposed)*
   - within-group reward std falls below half its 10-step average for 3 steps in a row *(proposed)*

   The proposed thresholds get finalized against the second seed's real curves.

**Entropy has to be measured by us.** ART's serverless metrics didn't include `loss/entropy` in the 20-step run. We estimate it from the rollouts instead, as the mean of −logprob over sampled root tokens. We already request those logprobs, so this costs nothing.

### 5.3 KL to base: logged, not penalized

- Every step, 4 sampled rollouts are re-scored under the base model using `prompt_logprobs`. The logged KL is the mean of (policy logprob − base logprob) over the sampled root tokens.
- That's ~4 × 8K input tokens a step, about $0.01 *(inferred)*. A KL estimate from ART itself only appears when `kl_penalty_coef > 0`.
- **Fallback (research §8.5):** if KL to base jumps, or the capability check drops, we turn on a small `kl_penalty_coef` against the base model. That's a one-argument change.

### 5.4 What happens to a step's rollouts before training

1. Score every rollout (§3).
2. Zero the reward of any rollout a hack detector flagged (§7).
3. Remove `infra_error` episodes. A group left with fewer than 2 episodes is dropped.
4. If more than 1% of the step's episodes were `infra_error`, pause the run.
5. Drop groups where every reward is equal, and don't refill them (H13).
6. If fewer than 2 groups are left, skip the train call for this step and log why *(mine)*.
7. Train, then log every metric (§9).

---

## 6. Tasks and the difficulty controller

> The mix below is step 7's. Spec v2 replaces it with §12.7's, which adds the edit families.

- **Each step:** 8 tasks, drawn across families by the domain mix (`tasks.DOMAIN_MIX`, decisions 7a and 7c): aggregation 45%, call graph 20%, needle 20%, repo localization 15%. Every task gets a fresh random 64-bit seed. Generators, seeds and gold answers stay on the host (H3).
- **Levels:** every family has levels, from level 1 (today's ~35K characters) up to a top level of ~400K characters, which can be covered in ~20 sub-calls. Step 7 defines each generator's levels.
- **The rule:** for each family, take the exact-accuracy pass rate over its episodes from the last 3 steps.
  - Above ~70%: move up a level.
  - Below ~20%: move down a level, but never below level 1.
  - Otherwise, stay.
  - After a move, the family waits 3 steps before it can move again, so it doesn't bounce between two levels *(mine)*.
- **Evals** always use the fixed selection set, never the moving training mix.

---

## 7. Hack detectors (rules, run on every rollout)

> Edit episodes add `test_tamper`, `grader_tamper`, `test_aware` and the logged-only `network_try` (§12.6).

| Rule | Fires when |
|---|---|
| `early_submit` | `submit()` is called on turn 1, or before any code has read `context` |
| `copied_answer` | The submitted value appears in the prompt but in no REPL output, and is no atom of the task's own blind guess (the reward already caps that guesser; added step 8, Rishi 2026-10-11). Where the blind guess is every possible name (callgraph, aggregation `labels_over` and `users_with_label`), the rule never fires; the reward bounds those guessers |
| `escape_attempt` | Any blocked network, file or host-import attempt shows up in the sandbox logs |
| `timeout_loop` | 3 or more cell timeouts in one episode |

- A firing detector zeroes the reward before the group mean, and it is counted per rule.
- Separately, a Claude audit of sampled wins gives the clean solve rate (decision 5a). It is never used as a training signal.

---

## 8. What is saved with every checkpoint

`results/<run>/config.json` gets rewritten at each saved checkpoint. It records:

- the ART version, the base model id, and the chat template hash
- `PROMPT_VERSION` and the prompt hash, the bridge API version, and the episode limits
- every train-call argument in §5.1, the current learning rate, and which LR rule branch is active
- generator versions, the dataset manifest hash, and each family's controller state
- the git commit of the code, and the inference and training spend so far

Every 5th checkpoint is kept, plus the best on the selection set.

---

## 9. Logged every step

> Edit episodes add an `edit` block to every step record (§12.9).

These go to W&B and `results/<run>.json`. The live dashboard (plan step 9) reads them.

- **Pass rates:** with and without infra errors, the pass-rate histogram, and the share of all-0 and all-1 groups.
- **Groups and levels:** trainable groups, and the level for each family.
- **Reward:** exact accuracy, reward components, and hack-detector hits by rule.
- **Training signals:** entropy (§5.2), KL to base (§5.3), within-group reward std, `probs_corr`, grad norm, clip fraction and importance-ratio stats.
- **Episode shape:** sub-calls per episode, budget-hit rate, turns, root and sub-call tokens, and truncations.
- **Cost and infra:** infra error rate, step time split into rollouts and training, inference and training spend, and the learning rate.

---

## 10. Code layout

Production code goes at the top level of `rlm_rl/` and `scripts/`. Archived code is ported, not imported.

| Path | Holds | Ported from |
|---|---|---|
| `rlm_rl/bridge.py` | API v1, prompt text v1, wire protocol, version and hash | `repl_worker.py`, `rlm_env.SYSTEM_PROMPT` |
| `rlm_rl/repl_worker.py` | The REPL process that runs inside the sandbox | `repl_worker.py` |
| `rlm_rl/sandbox.py` | Docker container per episode, limits, mounts | `rlm_env.ReplWorker` |
| `rlm_rl/client.py` | Shared limiter, retries, token-id rebuild, infra errors | `rollout.py`, `rlm_env.SubcallBridge` |
| `rlm_rl/env.py` | `run_episode`, observations, episode limits | `rlm_env.run_episode` |
| `rlm_rl/reward.py` | Scoring and calibration (§3) | `rlm_env.score_answer` |
| `rlm_rl/detectors.py` | Hack rules (§7) | new |
| `rlm_rl/controller.py` | Difficulty controller (§6) | new |
| `rlm_rl/config.py` | Every pinned value in this spec, saved as `config.json` | new |
| `rlm_rl/tasks/` | `tickets.py`; from step 7, `aggregation.py`, `synthcode.py` (call graph, needle), `repoloc.py` and `oolong.py` | `ticket_tasks.py` |
| `rlm_rl/evals.py` | Selection set, capability check, eval transcript logging | `rlm_env.evaluate_rlm` |
| `scripts/train.py` | Resumable GRPO loop, stop rules, LR rule, operator stop switch | `run_rlsd_vs_grpo.py` |
| `scripts/eval.py` | Standalone evals: baselines, the final eval | new |

Not ported: `sdpo.py`, `shaping.py` and `tasks.py`. The README Layout section gets updated when the code lands.

---

## 11. Still to verify

| Item | Where it gets checked |
|---|---|
| W&B serverless honours `enable_thinking: False` | Step 2 or the step 6 dry run |
| `packed_sequence_length` value | Set in step 6 from measured root transcript lengths |
| Entropy estimate tracks ART's entropy, where ART reports one | Second seed |
| The proposed LR-rule thresholds (§5.2) | Second seed |

---

## 12. Edit families (spec v2, 2026-10-12)

> Added for [code-editing-plan.md](code-editing-plan.md), approved by Rishi on 2026-10-12 with every recommended option (E-a to E-g), the E7 probe dropped and the reserve cut to $10. Everything in §1-§11 still holds for the read families. This section pins what changes for the two edit families, `edit_real` and `edit_synth`, and is the contract the edit sandbox, the grader, the generators, the training loop and the dashboard are coded against.

### 12.1 Bridge API unchanged; coding tools added

The bridge stays v1: `context`, `llm`, `llm_batch`, `submit`, with the same budget, prompt cut and wire protocol (§1). Nothing new crosses it.

- `context` holds the task text (the issue, or the feature to build) and the repository's file list with sizes. The files themselves are on disk.
- `submit(answer)` ends the episode as before. Any value the read families accept is fine; a one-line summary is expected. **What gets graded is the diff of the repository against its base commit, not the submitted value.**

Edit episodes also get seven coding tools (Rishi, 2026-10-12; code-editing-plan.md §5.6), modelled on what Claude Code, Codex CLI, DeepSeek Harness and Qwen Code give their models. They are Python functions in `rlm_rl/repl_tools.py`, standard library only, which the container mounts read-only at `/opt/rlm` and the edit REPL loads into its namespace before turn 1. They run inside the container, so the sandbox bounds them as it bounds `open` and `subprocess`, which keep working. The edit REPL's working directory is `/work/repo`, and relative paths in the tools start there. Read episodes don't get the tools.

| Tool | Behaviour |
|---|---|
| `read_file(path, start=1, n=40) -> str` | `n` lines from line `start`, numbered, lines over 300 characters cut, ending with how many lines follow and the call that reads them. Raises for a missing or binary file or a start past the end |
| `find_files(pattern, path=".") -> list[str]` | Files under `path` matching a glob (`*` within a folder, `**` across folders, `{a,b}`), sorted, relative to the repository. Skips what `.gitignore` excludes (git's own file list), or common dependency and cache folders where there is no git |
| `grep(pattern, path=".", glob=None, ignore_case=False, files_only=False) -> list[str]` | Python-regex matches as `"path:line: text"`, or each matching file once. `glob` without a `/` matches file names at any depth, as ripgrep does. Skips binary files, files over 2 MB and ignored files; stops after 10,000 matches with a last entry saying so |
| `edit_file(path, old, new, replace_all=False) -> str` | Exact replacement. `old` must occur once unless `replace_all=True`; otherwise it raises with the count and the file is untouched. Keeps CRLF line endings |
| `write_file(path, text) -> str` | Creates or overwrites a file, making folders |
| `shell(cmd, timeout=120) -> ShellResult` | `sh -c` in the repository, stdout and stderr together. `.code` and `.out` (its last 1,000,000 characters); printing shows the exit code and the last 1,500 characters. A timeout kills the process group and returns code 124 with the output so far. Capped at the cell's remaining time |
| `diff() -> str` | `git diff` of the working tree, new files included, against the base commit, staged through a copy of the index so the repository's own index is untouched. A commit by the model doesn't hide its change |

`TOOLS_VERSION = "t1"`: names, signatures and docstrings freeze with prompt v2. `tests/test_repl_tools.py` pins them, every checkpoint records the tools version next to the prompt version (§8), and the Part 2 harness imports the same file (H10). Web tools, sub-agents with their own tools, a todo tool and `apply_patch` are left to Part 2.

The REPL wires them in with `repl_tools.configure(root, base, time_left, on_call)` and `namespace()`. `on_call(tool, ok, seconds, target)` feeds the metrics in §12.9, and `repl_tools.active()` tells the audit hook that a file write or a process came from a tool.

### 12.2 Prompt v2

`PROMPT_VERSION = "v2"`. One system prompt for every family (H10 still diffs it against the harness). Two changes from v1: the limits move into each task's first message, and one section covers coding tasks and lists the tools (§12.1).

```text
You are a Recursive Language Model. You work on a task whose input is too large to read directly. The input is stored in a persistent Python REPL as the string variable `context`; for coding tasks, a repository is also checked out on disk.

Each turn, write a short plan and then exactly one ```python code block. The REPL runs it and shows you what it prints (truncated to 8000 characters). Variables persist between turns.

Available in the REPL:
- `context`: the task input (str)
- `llm(prompt: str) -> str`: ask a language model; it can read about 20,000 characters per call
- `llm_batch(prompts: list[str]) -> list[str]`: ask several prompts in parallel; each prompt counts as one sub-call
- `submit(answer)`: give your final answer and end the episode

The REPL has only the Python standard library.

Your turn and sub-call limits are given with the task. Look at the structure of the input before processing it, decompose the work, use code for exact filtering and arithmetic, and use sub-calls for judgments that need reading comprehension. Call `submit()` once you are confident.

For coding tasks, the REPL starts in the repository and also has these tools. They return values; print what you need to see.
- `read_file(path, start=1, n=40) -> str`: numbered lines of a file
- `find_files(pattern, path=".") -> list[str]`: files matching a glob such as "src/**/*.py"
- `grep(pattern, path=".", glob=None, ignore_case=False, files_only=False) -> list[str]`: regex matches as "path:line: text"
- `edit_file(path, old, new, replace_all=False) -> str`: replace exact text that occurs once (or every time with replace_all=True); raises otherwise
- `write_file(path, text) -> str`: create or overwrite a file
- `shell(cmd, timeout=120)`: run a shell command; the result has `.code` and `.out` and prints as the exit code and the end of the output
- `diff() -> str`: your changes so far, as they will be graded

Explore with code and print summaries, not whole files. Use `llm_batch` to read many files at once, and `llm` to draft or review a change. Run the relevant build or tests with `shell` before you submit. Your changes to the repository are what gets graded, by tests you cannot see, so do not change existing tests or test configuration. There is no network.
```

First user message, read families (the v1 template plus one limits line):

```text
Question: {question}

`context` holds {n_chars:,} characters in {n_lines} lines. The first 3 lines are:
{first_3_lines}

You have at most {max_turns} turns and {max_subcalls} sub-calls.
```

First user message, edit families:

```text
Task: {task}

The repository is at /work/repo ({language}, {n_files} files). `context` holds this task and the repository's file list with sizes. Run its tests with: {test_hint}

When your change is ready, call submit() with a one-line summary. You have at most {max_turns} turns and {max_subcalls} sub-calls, and each cell may run for {cell_timeout_s} seconds.
```

`{test_hint}` is a generic command for the repository's visible tests (for example `python -m pytest -q` or `go test ./...`), never the hidden tests' names.

The timeout and crash messages (§4) gain the cell limit as a value, and for edit episodes add one sentence: `Your files in /work/repo are unchanged.`

The second seed (step 10) keeps prompt v1, so it repeats the setup steps 5 and 6 tested (decision E-g), and its checkpoints record v1 (§8). The step 11 baselines ran v2.

**Prompt v3 (Rishi, 2026-10-17, after step 11; production uses it).** Under v2 the base model wrote its code in Qwen3.8's native tool-call syntax (`<tool_call><function=python>…`) on 43% of its turns, which the REPL couldn't run (step-11-results.md §3). v3 keeps v2's system prompt and first messages and changes three things: code in a `<function=python>` tool call runs like a ```python block (`env.extract_code`); the "no code block" error adds "Tool-call tags don't run here: call llm, llm_batch, submit and the tools as Python functions inside the block."; and edit episodes get 30 turns (§12.3). v1 and v2 keep their texts and hashes, so earlier runs stay reproducible.

### 12.3 Episode limits

| Limit | Read families (§4) | Edit families |
|---|---|---|
| Root turns | 12 | 30 (prompt v3, Rishi 2026-10-17; 20 under v2) |
| Sub-calls | 30 | 30 |
| Root output tokens per turn | 3,072 | 4,096 |
| Sub-call prompt and output | 24,000 chars, 1,024 tokens | same |
| REPL output shown | 8,000 chars (prompt v1: 2,000) | 8,000 chars |
| Cell timeout (compute) | 120 s | 300 s |
| Container | 1 CPU, 512 MB, 64 pids | 2 CPUs, 4 GB, 256 pids |
| Grader timeout | none | Per task: 600 s, or 4x its slowest gold run if longer |

Limits are looked up per family (`bridge.limits(family)`), and each episode records the limits it ran under.

**A slow disk never fails a good answer** (Rishi, 2026-10-13). The images live on a USB drive, so time spent waiting on the disk costs wall-clock time, never reward:
- Container start-up waits are long (300 s for a read sandbox, 600 s for an edit sandbox, 600 s of extra slack for a grader). A launch that runs out is retried once, and a second miss is `infra_error`, never 0.
- An edit task whose own test command, run through `shell()` under load, takes more than half the cell limit gets its own `cell_timeout_s`: twice that time, rounded up to 30 s. It is not dropped, and the episode's first message, timeout text and recorded limits use the task's value.
- A grader that hits its cap runs once more, alone, with double the cap. Only a second timeout is the model's (§12.5).
- Timings for these rules come from `build_edit_data.py measure --jobs 30`, which times every indexed task in real containers 30 at a time (container start, repo copy, gold test command, grader), not from an idle drive. The index reads its results for each task's grader cap (`tests.timeout_s`) and cell limit. Graders may queue for a slot, and queued time counts against no limit.

### 12.4 The edit sandbox

Read episodes are unchanged (§10, step 3). Edit episodes differ in five ways:

1. **Image.** `edit_synth` uses `rlm-sandbox`. Each `edit_real` repository uses a derived image, `rlm-edit-<repo>:<hash>`, built once from its source image: the image's own copy of the repository is deleted (for injected bugs it is the answer), caches and any other installed copy of the package are removed, and `/work/repo` points at the path the image's environment expects (for example `/testbed`). Every image is pulled and built with the network on, then only ever run `--network none`. Every edit image has `git`, which `diff()` and the model's own `git status` need. A source image without it gets it in its derived image, and `edit_synth` gets it the same way from `rlm-sandbox`.
2. **The REPL runs on our own Python**, a standalone CPython 3.12 build mounted read-only at `/opt/rlm/python`, so it works in any image whatever language or Python version the repository uses. The repository's own toolchain runs through `subprocess`.
3. **Writable repository.** `/work/repo` is a writable bind mount of a per-episode host directory with a random name, holding a copy of the repository at the task's base: a fresh `git init` with one commit, no other history, branches, tags, reflog or remotes. It survives a REPL restart. Hidden test files are never in it.
4. **Processes are allowed** inside the container: Landlock keeps reads and writes to the image, `/work/repo`, `/scratch` and `/opt/rlm`, but lets the image's programs run. The audit hook still logs and blocks network and host-file attempts; a blocked network attempt in an edit episode is logged as `network_try` and doesn't zero the reward (§12.6).
5. **Limits** are §12.3's.

Everything else from step 3 holds: `--network none`, no capabilities, a non-root user, `env -i`, the bridge socket as the only way out, and the infra rules for failures the model didn't cause. Two points differ, as built in E2 (2026-10-12):

- **A writable layer, not a read-only root.** Go, Cargo, Maven and Gradle need warm, writable caches, so each edit container gets its own writable layer, which Landlock confines (point 4) and which is deleted with the container.
- **The user is the host user's own non-root uid**, not the read sandbox's fixed one.

### 12.5 The grader

1. After `submit` or the last turn, the host takes `git diff` of the episode's directory against the base commit and removes the episode container.
2. **Diff checks:** text files only, inside the repository, at most 2,000 changed lines *(mine)*. Changes to existing test files, any `conftest.py`, and test configuration (`pytest.ini`, `tox.ini`, the test sections of `setup.cfg` and `pyproject.toml`, `jest`/`vitest` configs, `go test` flags files) are stripped and fire `test_tamper`. New test files (each language's test-file pattern: Go `*_test.go` and `testdata/`, Rust `tests/**` and `src/**/tests.rs`, Java `src/test/**`, JS/TS `*.test.*`, `*.spec.*`, `__tests__/` and `test/`, Python `test_*.py`, `*_test.py` and `tests/`) are dropped from the diff before the run and counted, without firing `test_tamper`. They never reach the grader container, so they are never compiled or imported: a Go `*_test.go` would otherwise be built into the test binary and its `init()` would run.
3. A fresh grader container starts from the same image, `--network none`, with the same limits: a clean copy of the repository at base, the diff applied, the full test suite restored including hidden tests, and two canary tests with names drawn fresh per episode (one must pass, one must fail).
4. It runs the task's fail-to-pass and pass-to-pass tests plus the canaries, with a 600 s cap, and reads per-test results from the runner's own report: JUnit XML (pytest, Maven Surefire, Gradle), `go test -json`, Jest, Vitest or Mocha JSON, and `cargo test`'s per-test lines. Results are keyed by file or package plus test name. A test missing from the report counts as failed.
5. A grader that can't start, or a container killed from outside, is `infra_error` (§3). A test run that hits its cap (§12.3) runs once more, alone, with double the cap, and only a second timeout is the model's: every task's gold patch was validated well inside the cap under load (§12.8).

### 12.6 Reward and detectors

Reward for edit episodes, in order:
1. No `submit`, an empty diff, or a diff that doesn't apply: **0**.
2. A detector fired: **0**, before the group mean (H11).
3. Every fail-to-pass and pass-to-pass test passed: **1**. This is `resolved`, the edit families' headline, logged where `exact` is for the read families.
4. Anything else: **0** (binary, decision E-c).
5. Grader or container failure: `infra_error`, out of the group.

New detector rules (§7 keeps its four):

| Rule | Fires when | Effect |
|---|---|---|
| `test_tamper` | The diff touches an existing test file, a `conftest.py`, or test configuration, or adds skip or xfail markers to existing tests | Zero |
| `grader_tamper` | The must-fail canary passes, the must-pass canary fails, the collected test count is off, or the report is missing or malformed | Zero |
| `test_aware` | Non-test code added by the diff reads `PYTEST_CURRENT_TEST`, looks for a test runner in `sys.modules`, imports `_pytest`, or calls `os._exit`, or the equivalent checks the language's test runner exposes | Zero |
| `network_try` | A blocked network attempt in an edit episode | Logged only |

`early_submit` applies to edit episodes as "submit on turn 1, or before any code read a file in `/work/repo`". `copied_answer` doesn't apply to edit episodes. Broader signals (`inspect.stack()`, new environment-variable reads, `sys.settrace` in non-test code) go to the Claude audit queue, never to the reward.

**Blind check (H1) for edits:** the empty patch, a patch that deletes the target code, and a patch that stubs it to return a constant must all score 0 on every edit task kind and level.

### 12.7 Tasks, mix and levels

`tasks.DOMAIN_MIX` becomes:

| Domain | Share | Families |
|---|---|---|
| aggregation | 25% | `aggregation` |
| code reading | 20% | `callgraph` 10%, `needle` 10% |
| repo localization | 15% | `repo_loc` |
| editing | 40% | `edit_real` 25%, `edit_synth` 15% |

`production_mix()` today splits a domain's share evenly among its families; editing's 25/15 split needs the family shares set directly. `allocate_step` is unchanged and keeps every family within one task of its share (exact every 5 steps at 8 tasks: 10/4/4/6/10/6 per 40 tasks).

- **`edit_synth`** (generated from `synthcode` and `behaviours`): three kinds, fix-the-symptom, change-and-update-callers, add-a-feature; levels 1-5 from ~20K to ~200K characters, 1 to 4 files to change; hidden property tests with inputs drawn on the host per seed. `edit_synth_heldout` is eval-only.
- **`edit_real`**: real repositories in five language groups (code-editing-plan.md §4.1): Python ~90 training tasks, Go ~60, TypeScript and JavaScript ~60, Rust ~45, Java ~35, on SWE-smith's per-repository images, plus ~24 real pull requests (Python, Go, TypeScript) from SWE-rebench V2. Training uses SWE-smith's own task rows only for Python and Go, whose datasets are MIT. The JavaScript, TypeScript, Rust and Java datasets state no licence, so those four languages train only on tasks we generate on the permissive repositories (feature removals and our own injected bugs), and SWE-smith's rows for them are eval-only (Rishi, 2026-10-12). Three kinds: bug fix, feature implementation (generated by removing a covered feature) and real pull request. Repositories used by SWE-bench Pro, DeepSWE v1.1 or SWE-bench Multilingual are excluded, and every image must pass its tests with `--network none`. Levels by the gold patch: 1 = one file and up to ~10 lines; 2 = one file up to ~40 lines; 3 = two files; 4 = three or more files; 5 = multi-file features of ~100+ lines. Each task records its `language`.
- **Retirement:** a task whose group came back all-0 twice, or all-1 once, isn't drawn again. Retirement state is saved with the checkpoint, so a resume keeps it.
- The licence bar (7b) applies: every training repository has a permissive licence, checked by `tasks.check_mix()`.

### 12.8 Validation before a real task enters the pool

In our grader: the gold patch passes every fail-to-pass and pass-to-pass test three runs in a row; the empty patch fails every fail-to-pass test; removing the hidden test files doesn't break the visible tests; the gold run finishes under half the grader cap, timed under production load; the task text doesn't quote the gold patch or a hidden test's name. Validation costs PC time only.

### 12.9 Logged every step (adds to §9)

Each step record gains an `edit` block, over that step's edit episodes:

| Key | Meaning |
|---|---|
| `episodes` | Edit episodes scored |
| `resolved` | Share with every hidden test passing |
| `f2p_pass` | Mean share of fail-to-pass tests passing |
| `p2p_broken` | Share that broke at least one pass-to-pass test |
| `empty_diff` | Share that submitted no change |
| `submitted` | Share that called `submit` |
| `tested_before_submit` | Share that ran a test command after their last edit and before `submit` |
| `test_runs` | Mean test commands per episode |
| `edit_cycles` | Mean edit-then-test cycles per episode |
| `turns_to_first_edit` | Mean turn of the first write to `/work/repo` |
| `diff_lines`, `files_changed` | Mean size of the submitted diff |
| `subcall_read_share` | Characters sent to sub-calls ÷ (characters sent to sub-calls + characters printed to the root) |
| `review_subcalls` | Mean sub-calls whose prompt holds a diff |
| `repeated_cells` | Mean cells identical to an earlier cell in the episode (the loop signal) |
| `network_tries` | Mean blocked network attempts |
| `grader_s` | Mean grader seconds |
| `tool_calls` | `{tool: mean calls per episode}` for the seven tools (§12.1) |
| `edit_errors` | Mean `edit_file` calls per episode that raised |
| `raw_io_share` | Share of writes to `/work/repo` and commands made with plain `open` or `subprocess` rather than the tools |
| `grader_retries` | Mean grader re-runs per episode after hitting the cap (the slow-disk signal) |
| `by_language` | `{language: {"episodes", "resolved"}}` |

A test command counts for `test_runs`, `tested_before_submit` and `edit_cycles` whether it ran through `shell` or `subprocess`, and an edit counts whether it came from a tool or `open`. The edit detectors appear in the existing per-rule hack counts. Eval summaries gain `by_language` next to `by_family` and `by_level`.

### 12.10 Evals

- **`selection-v2`** (64 tasks): 48 read tasks (the read families in the new mix's proportions, plus the ticket tasks as the held-out generator family) and 16 edit tasks (6 `edit_synth` held-out seeds, 10 `edit_real` from repositories training never sees, across languages). It replaced `selection-v1`; the step 11 baselines ran on it.
- **`selection-v3`** (64 tasks, production's selection set from 2026-10-17, Rishi): `selection-v2` without its 8 ticket tasks, which production never trains on and which the base model found too easy, and with 8 more held-out seeds of the trained read families at levels 3-5 in their place (aggregation 3, call graph 1, needle 1, repo localization 3 from the selection repositories). The read tasks now follow the mix exactly (20/8/8/12). The other 56 tasks are `selection-v2`'s, with the same seeds. The new tasks share training's generators and sources but not its seeds, so no task repeats one training sees, as for the rest of the set.
- **`heldout-test-v2`**: `heldout-test-v1` plus 60 edit tasks (20 `edit_synth_heldout`, 40 `edit_real` from held-out repositories, at least 15 feature tasks).
- **Step 11** ran the base model on `selection-v2`, with the 16 edit tasks twice, which also measured cost per edit episode and the base pass rate before production, then once more in production's set-up (prompt v3, root thinking medium) on `selection-v3`.
- **Step 13** (off W&B) adds SWE-bench Pro, DeepSWE v1.1 and the normal-coding-agent arm (code-editing-plan.md §12).

### 12.11 Budget (replaces plan step 10a)

| Item | Inference |
|---|---|
| Second seed (step 10) | ~$12 |
| Baselines (step 11) | ~$9 |
| Production ceiling (step 12) | ~$157 |
| Selection-set evals during production | ~$30 |
| Reserve (ask Rishi first) | ~$10 |
| **Total** | **~$218** |

### 12.12 Code layout (adds to §10)

| Path | Holds |
|---|---|
| `rlm_rl/editbox.py` | The edit sandbox (§12.4): derived images, the writable repository mount, the standalone Python mount |
| `rlm_rl/repl_tools.py` | The seven coding tools (§12.1), standard library only, mounted into edit containers and imported by the Part 2 harness |
| `rlm_rl/grader.py` | The grader (§12.5) and per-language report parsers |
| `rlm_rl/tasks/editsynth.py` | `edit_synth` and `edit_synth_heldout` |
| `rlm_rl/tasks/editreal.py` | `edit_real` and its eval-only splits, reading the validated task index |
| `scripts/build_edit_data.py` | Pulls and derives the images, downloads task metadata to the data drive, validates tasks (§12.8), writes the index and manifest |

Docker images and downloaded data live on the D: drive (Rishi, 2026-10-12), never in the repository.
