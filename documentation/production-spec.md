# Production spec

> **Step 1 of [production-build-plan.md](production-build-plan.md).** Written 2026-10-10. This spec pins the interfaces and settings that every later step is coded against: the bridge API, prompt text, reward, episode limits, training settings, difficulty controller, code layout and checkpoint manifest.
>
> **Status:** approved by Rishi on 2026-10-10, including the additions marked *(mine)*. Step 1 is complete. Nothing has been built, run, committed or pushed, and no W&B credit has been spent.
>
> **Sources:** [rl-run-research.md](rl-run-research.md) (the source of truth for the research), the archived 20-step code (`rlm_rl/archive/rlm_env.py`, `repl_worker.py`, `ticket_tasks.py`, `scripts/archive/run_rlsd_vs_grpo.py`), and ART 0.5.20's own source in `.venv`. Values marked *(mine)* are my defaults, not the research's. Values marked *(verify)* still need checking against the live system.

## 0. Decisions applied

| # | Decision | Answer | Decided |
|---|---|---|---|
| 1a | Bridge function names | `llm`, `llm_batch`, `submit`, `context` | Rishi, 2026-10-10 |
| 1b | Partial credit | Calibrate against the blind guess | Rishi, 2026-10-10 |
| 1c | CISPO importance-weight clip | Restore ART's default | Rishi, 2026-10-10 |
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

| Limit | 20-step run | Production |
|---|---|---|
| Sub-calls per episode | 20 | 30 |
| Root turns | 6 | 12 |
| Root output tokens per turn | 1,536 | 3,072 |
| Sub-call prompt size | 24,000 chars | 24,000 chars |
| Sub-call output tokens | 1,024 | 1,024 |
| REPL output shown | 2,000 chars | 2,000 chars |
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

| Rule | Fires when |
|---|---|
| `early_submit` | `submit()` is called on turn 1, or before any code has read `context` |
| `copied_answer` | The submitted value appears in the prompt but in no REPL output, and is no atom of the task's own blind guess (the reward already caps that guesser; added step 8, Rishi 2026-10-11) |
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
