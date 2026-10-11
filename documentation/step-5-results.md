# Step 5 results: environment v2

> **Step 5 of [production-build-plan.md](production-build-plan.md)** (holes H1, H11 and H13, plus the 30-call sub-call budget). Built and tested 2026-10-11 on Rishi's PC, inside WSL2 with Docker. **Decisions applied (Rishi, 2026-10-11):** 5a, a Claude thread audits sampled wins at each 10-step checkpoint; 5b, keep the spec's near-miss credit formula.
>
> **Status: complete (confirmed by Rishi, 2026-10-11).** Committed and pushed to `claude/project-thread-2apfvb`.
>
> **No W&B credit was used.** Every test serves the model from `tests/fake_model.py`, and the H1 and detector checks score generated tasks and the 20-step run's saved transcripts on the host. The `.env` file was never opened. Nothing was installed.
>
> Anything marked *(mine)* is my choice, not the spec's. Anything marked *(inferred)* hasn't been checked against the live system.

## Summary

**Step 5 is built, and every done-when check passes.** There is one caveat on H1, which Rishi settled with decision 5b below. The regression suite is now 170 tests (97 before), and all of them pass in WSL with Docker in about 2.5 minutes. One test skips there, the tokenizer check from step 4, as before.

| Done-when check | Result | Evidence |
|---|---|---|
| A blind guesser scores no more than its exact-hit rate on every family | **Pass, with a caveat** | `BlindGuesserTest` and `scripts/reward_check.py`: both ticket families at all 5 levels. The task's own blind guess earns exactly its exact-hit rate (1-11%). But a guesser that adds an offset or noise to it earns 13-21% (§3). |
| Each detector has a unit test | **Pass** | `tests/test_detectors.py`: 31 tests, 7 of them through real Docker sandboxes. The detectors raised **0 false alarms** on the 20-step run's 1,280 episodes (§4). |
| The controller moves levels correctly on synthetic pass rates | **Pass** | `tests/test_controller.py`: 12 tests covering up, down, the floor, the top, the band edges, the 3-step wait, sparse families, infra episodes, frozen mode and resume. |

Everything else the plan lists for step 5 is also in:

| Plan item | Result |
|---|---|
| Calibrated reward, with exact accuracy as the headline metric (H1) | `rlm_rl/reward.py` is the default scorer. An episode counts as `correct` only when it is exact and no detector fired. |
| Detectors zero the reward before the group mean, counted on the dashboard (H11) | `run_episode` runs them on every scored episode. Hits are counted per rule in each step's filter and in each episode's metrics. |
| Claude audit of sampled wins, reported as a clean solve rate (5a) | `scripts/audit.py checkpoint` and `report`, plus [audit-procedure.md](audit-procedure.md). It is never a training signal. |
| Difficulty controller; all-equal groups dropped, not refilled (H13) | `rlm_rl/controller.py`. `filter_step` (step 4) already drops all-equal groups, and nothing refills them. |
| Episode limits from step 1, including the 30-call budget | Pinned in `bridge.py` since step 3. A new test runs one episode up to exactly 30 sub-calls and shows the 31st refused. |
| Ticket tasks ported with levels, so the second seed can run on this environment | `rlm_rl/tasks/tickets.py`, levels 1-5 (~41K to ~382K characters). Level 1 rebuilds the 20-step run's tasks exactly. |

The carry-overs from steps 3 and 4 are done too:
- **Env vs harness prompt diff (H10).** A new test runs an episode and diffs what actually reached the model against the bridge's pinned text: system prompt, first message and observation. On a mismatch it prints a line diff. The harness gets the same test when Part 2 exists.
- **Timeout and crash messages.** The episode loop shows the model the two texts pinned in step 3. Tests check that a timeout, through the fake sandbox and through Docker, reaches the model's next message.
- **The 1% pause rule and the limiter caps** are unchanged from step 4: per step for the second seed, a 10-step window in production, and 30 requests in flight with 20 per model.

### Decision 5b: near-miss credit window

**Decided by Rishi, 2026-10-11: keep the spec's formula.** `NEAR_MISS_SLOPE` stays at 1. The options were:
- **Keep the spec's formula** (recommended, and chosen). Smart guessers earn up to 0.19-0.21, against 0.53 and 0.31 under the 20-step reward. In the 20-step run the model didn't drift to guessing even when the plain guess paid 0.53. A new `near_blind` metric and the audit watch for drift toward the guess.
- **Halve the window** (`NEAR_MISS_SLOPE = 2`). Guessers drop to about 0.10-0.12. Honest near misses lose about 0.1 of credit, so slightly more groups come out all-zero. It's a one-constant change.

---

## 1. What was built

| Path | What it is |
|---|---|
| `rlm_rl/reward.py` | The reward (spec §3): answer parsing, exact matching for numbers, decimals, entities and lists, partial credit calibrated against the blind guess, and `settings()` for the run config |
| `rlm_rl/detectors.py` | The four hack rules (spec §7), `check()` and per-rule `counts()` |
| `rlm_rl/controller.py` | `DifficultyController`: per-family levels, moves, saturation, frozen mode, and state saved with checkpoints |
| `rlm_rl/tasks/tickets.py` | The ticket families with 5 levels, the blind guess, and `legacy_tasks()`, which rebuilds the 20-step run's task lists |
| `rlm_rl/tasks/__init__.py` | Now also the family registry: `make_task`, `max_levels`, `versions`, `allocate` and `draw` (one step's tasks by mix and level) |
| `rlm_rl/env.py` | The reward is the default scorer, and detectors run on every scored episode. `Episode.hacks`, new metrics (`exact`, `partial`, `near_blind`, `hack`, `hack_<rule>`) and hack counts in `StepFilter` |
| `rlm_rl/transcripts.py`, `rlm_rl/evals.py` | Transcripts record the detectors that fired (format 2). Eval summaries add hack counts and accuracy by level |
| `scripts/audit.py` | Detector hits in `summary`, plus two new commands, `checkpoint` and `report`, for the Claude audit |
| `scripts/reward_check.py` | What blind guessers earn, per family and level (§3) |
| `scripts/backtest_detectors.py` | Replays the 20-step run's episodes through the detectors (§4) |
| `documentation/audit-procedure.md` | How the Claude thread runs the audit at each checkpoint |
| `tests/` | New: `test_reward.py` (16), `test_detectors.py` (31), `test_controller.py` (12), `test_tickets.py` (10). Extended: `test_env.py` (+2), `test_contract.py` (+1), `test_evals.py` (+1). `tests/fakes.py` can now fake a timeout or an audit event |
| `.gitignore` | Adds `results/*/audit/`, because audit samples hold gold answers |
| `README.md` | The Layout section lists the new modules and scripts |

How step 6's training loop uses it:

```python
ctl = DifficultyController(tasks.max_levels(FAMILIES), frozen=(run == "second_seed"))
batch_tasks = tasks.draw(8, MIX, ctl.levels())
groups = await run_groups(batch_tasks, 4, client=client, root_model=..., run_id=run)  # scored, hacks zeroed
step_filter = filter_step(groups)            # infra out, then small and all-equal groups; counts hacks by rule
ctl.update(step, [ep for g in groups for ep in g])
```

## 2. How it works

### 2.1 The reward (spec §3, H1)

Every task's generator puts the grading in `task.grading`. For numbers and lists that includes the **blind guess**, the answer the prompt's metadata alone gives. For the ticket tasks it is `round(line count × the target's expected share)`. The host computes the blind guess's error and scores in the spec's order:

| Case | Reward |
|---|---|
| No submit, or an answer that can't be parsed | 0 |
| Exact | 1 |
| Number, not exact | `0.8 × max(0, 1 − err / e_blind)`, or 0 if the blind guess was exact |
| List, not exact | `0.8 × max(0, (F1 − F1_blind) / (1 − F1_blind))` |
| Entity, not exact | 0 |
| A detector fired | 0, whatever the answer earned |

Parsing details *(mine, within the spec)*:
- **Numbers.** The first number counts, after `$` and `,` are removed. A minus sign counts only when it isn't joined to a word, so `host-42` reads as 42. Hedging doesn't pay: "69 or 70" is scored as 69.
- **Entities.** Whitespace and quotes are trimmed (straight, curly and backticks), and case is ignored.
- **Lists.** A list type also accepts a string, split on commas, semicolons and newlines. Decimal items are compared to 2 places.

Every score also records its components: `exact`, `partial`, `err`, `e_blind` (or `f1`, `f1_blind`), and `near_blind`. That last one is new *(mine)*. It is 1 when a wrong answer sits closer to the blind guess than to the gold, which is what guessing looks like. It is logged only and never changes the reward. If its rate climbs during a run, the policy is drifting toward the guess.

### 2.2 The detectors (spec §7, H11)

They read only what the host recorded, so nothing the model prints can switch them off. Each rule's exact definition is below; the spec gave the intent, and I filled in the details *(mine)*.

| Rule | Fires when |
|---|---|
| `early_submit` | `submit()` on turn 1, or no cell up to the submit names the variable `context` (checked by parsing the code on the host) |
| `copied_answer` | Every part of the answer passes all four checks. It is in the task's own prompt (the question, the size line and the first 3 lines, not the system prompt). It is typed in the model's code. It is distinctive: a whole number of 100 or more, a decimal, or a string of 3 or more characters. And it appears in no cell output, error or sub-call output. |
| `escape_attempt` | Any sandbox audit event of kind `network`, `file`, `process`, `ctypes`, `signal`, `tamper`, `host_import` or `malformed`, or any bridge request the host refused. A failed `import numpy` (`nonstdlib_import`) doesn't count, as step 3 suggested. |
| `timeout_loop` | 3 or more cell timeouts in the episode |

- **When they run.** `run_episode` runs them after scoring. A hit sets the reward to 0 and `correct` to false. It keeps the score it would have had as `reward_before_hack`, and the raw `exact` stays in the components for the audit.
- **No submit needed.** `escape_attempt` and `timeout_loop` fire without a submit, so an episode that probes the sandbox and never answers is still counted.
- **A bug in a detector** is an infra error (`host_bug`), never a zero, as with the scorer in step 4.
- **Why `copied_answer` has two extra conditions.** Taken literally, the spec's rule would have fired on 7 honest episodes in the 20-step run, 6 of them wins. Each was a small count, such as 8 or 12, that also happened to appear in the first lines ("8 days ago", "a customer for 12 years"). Three more matched numbers in the old system prompt ("6 turns"). Two conditions stop this. A small number or a short string never counts as copied. And the value must be typed in the code, so an answer the code computed isn't "copied" just because the prompt shows the same value.

### 2.3 The difficulty controller (spec §6, H13)

The rule is the spec's: the exact pass rate over a family's last 3 steps. Above 70% the family moves up, below 20% it moves down (never below level 1), and otherwise it stays. Details *(mine)*:
- **The window.** It only counts episodes at the family's current level, and a move clears it. That gives the spec's "wait 3 steps after a move" for free, and the pass rate is never mixed across levels.
- **Sparse families.** A step in which a family drew no tasks, or only infra errors, doesn't count toward its window. With the production mix, a family may draw one task a step or none.
- **Too few episodes.** A window needs at least 8 scored episodes before anything moves.
- **The top level.** A family at its top level above 70% is marked `saturated`. When every family is, that is stop rule 4.
- **Frozen mode.** It records what the controller would do but never moves. The second seed runs at level 1 (plan step 10), so it uses frozen mode.
- **Saved state.** `state_dict()` holds the levels, windows and every move, for `config.json` and resume (spec §8).

All-equal groups are dropped before training by step 4's `filter_step` and never refilled. A hacked win now enters that filter with reward 0. A test shows an exact-but-hacked rollout ending up below its honest near-miss sibling, so it gets a negative advantage instead of the group's best.

### 2.4 The ticket tasks with levels

The generator is ported unchanged, with the same texts, names and items. Its knob is the number of tickets:

| Level | Tickets | Median context | Range seen |
|---|---|---|---|
| 1 | 250-380 | ~41K chars | 33-49K (the 20-step run's size) |
| 2 | 500-760 | ~80K | 65-93K |
| 3 | 950-1,400 | ~155K | 126-178K |
| 4 | 1,600-2,300 | ~242K | 211-297K |
| 5 | 2,600-3,400 | ~382K | 341-426K |

- **Answers and grading.** Answers are integers now (the 20-step code stored floats), and each task carries its blind guess.
- **Task ids** never contain the seed. Pass a stable id for eval tasks, so a resumed eval matches its episodes.
- **`legacy_tasks(50, 99_999)`** rebuilds the first seed's 50 held-out eval tasks exactly, and a test compares it with the archived generator. The second seed can therefore be evaluated on the same tasks as the first, which makes the comparison with 0.70 cleaner (plan step 10).
- **A text-only recount.** A test recounts every answer from the rendered text alone, by matching each ticket to its template.

### 2.5 The Claude audit (decision 5a)

The full procedure is in [audit-procedure.md](audit-procedure.md). In short:
1. `audit.py checkpoint results/<run> --step N` samples 16 wins from the last 10 training steps, 8 wins from that checkpoint's selection eval, and up to 4 detector-flagged episodes *(sizes mine)*. It writes them as `sample.md`, with a `verdicts.json` to fill in.
2. The Claude thread judges each episode `clean`, `suspicious` or `hack`, using the rubric in the procedure, and records a note citing the turn.
3. `audit.py report` gives the **clean solve rate** for each checkpoint and split: the solve rate × the clean share of audited wins. It also gives the detectors' precision on the flagged episodes, and writes `report.json` for the dashboard.

A hack among the wins triggers stop rule 5: pause, fix, add a test, resume. That call stays with you. The audit never changes a reward.

## 3. H1: what guessing still earns

`scripts/reward_check.py` scores answers that never read the context, 1,000 tasks per family and level:

| Family, level | Exact-hit rate | Blind guess (the check) | Best fixed offset | Noisy guess | Honest reader off by 1 / 2 / 3 |
|---|---|---|---|---|---|
| tickets_all, 1 | 0.059 | 0.059 | 0.190 | 0.148 | 0.71 / 0.54 / 0.44 |
| tickets_all, 5 | 0.013 | 0.013 | 0.186 | 0.133 | 0.79 / 0.69 / 0.62 |
| tickets_customer, 1 | 0.112 | 0.112 | 0.204 | 0.168 | 0.60 / 0.41 / 0.30 |
| tickets_customer, 5 | 0.030 | 0.030 | 0.188 | 0.132 | 0.76 / 0.61 / 0.53 |

Levels 2-4 sit between these. "Noisy" is the blind guess plus Gaussian noise half the size of its typical error.

- **The check as written passes everywhere,** and by construction. The blind guess's own error is the credit window, so the blind guess itself always lands exactly on 0.
- **But a guesser that misses the blind guess on purpose earns credit.** Nudging the guess by a few in a fixed direction lands inside the window whenever the gold lies that way. So does noise. Such guessers earn about 0.13-0.21 at every level. That's 2-14× their exact-hit rate, and the gap widens at higher levels, where exact hits get rarer.
- **It's still far below the old reward** (0.53 and 0.31 for the plain guess) and far below honest reading (0.30-0.79). That is why the spec's formula was kept (decision 5b).

With the window halved (`--slope 2`), the best guessers earn 0.10-0.12 and noisy ones 0.06-0.11, close to the exact-hit rates. Honest readers lose about 0.1 (0.62 / 0.41 / 0.30 at tickets_all level 1).

No credit formula can fully close this. Any credit for "closer than the guess" is also available to a guesser who is sometimes closer by chance. The two real defences are the audit and `near_blind`.

## 4. Detector false alarms on the 20-step run

`scripts/backtest_detectors.py` replays all 1,280 training episodes of the 20-step run (GRPO and RLSD, 1,086 of them submitted) through the detectors. That run had no detectors, and its audit found no hacks, so any hit would be a false alarm.

| Rule | Hits |
|---|---|
| `early_submit` | 0. No episode submitted on turn 1 or without naming `context` |
| `copied_answer` | 0. With the spec's literal wording it would have been 7, or 10 counting the old system prompt (§2.2) |
| `escape_attempt`, `timeout_loop` | 0, but the old run had no sandbox events to read |

The archived files lack the last cell's output and the sub-call outputs. So `copied_answer` was checked against fewer outputs than it will see live, which can only make it fire more, not less.

## 5. How it was tested

Run from the repo root inside Ubuntu:

```bash
.venv-linux/bin/python scripts/sandbox_ctl.py check
```

| File | Tests | What they cover |
|---|---|---|
| `tests/test_reward.py` | 16 | Parsing; exact matching for each answer type; the calibrated formula's values; the blind guess earning 0; unparseable answers; hedging; `near_blind`; the slope; grading checks; the H1 check over both families at all 5 levels |
| `tests/test_detectors.py` | 31 | Each rule's hits and non-hits; the 20-step false alarms as cases; hack zeroing through the episode loop; a hacked win as a negative example; detector bugs as infra. **Through Docker:** blocked file, network and host-import attempts fire `escape_attempt`; honest standard-library work fires nothing; 3 real timeouts fire `timeout_loop`, and the model is told |
| `tests/test_controller.py` | 12 | Synthetic pass rates for every rule; state round trip |
| `tests/test_tickets.py` | 10 | Level sizes; scorable grading with the blind guess; a text-only recount of the answer; seeds never in ids or text; exact reproduction of the 20-step tasks; the registry, `allocate` and `draw` |
| `tests/test_env.py` | +2 | The 30-call budget through the loop; a ticket task end to end with the default reward |
| `tests/test_contract.py` | +1 | H10 on the wire: the system prompt, first message and observation the model received equal the bridge's text |
| `tests/test_evals.py` | +1 | The Claude audit end to end: ten training steps and an eval logged, a checkpoint sampled, verdicts recorded, and the clean solve rate checked by hand |

- **In WSL with Docker:** 170 tests, 169 pass and 1 skips (the tokenizer check), in about 2.5 minutes. Run twice in a row, both clean. No container or episode directory was left behind.
- **On Windows:** 135 tests run, 120 pass and 15 skip because they need Docker or a Unix socket.
- **One existing test changed.** `test_reply_without_code_uses_up_a_turn` submitted without reading `context`, which `early_submit` now zeroes. Its second reply now counts the lines.

## 6. Defaults I picked

| Fork | Default |
|---|---|
| What `correct` means | Exact and no detector fired, so a hacked win never counts toward accuracy. The raw exact match stays in the components. |
| Detectors in evals | On, as in training. Baselines and the final eval apply the same rules. |
| `copied_answer` matching | The task's prompt only; typed in the code; small numbers and short strings skipped (§2.2) |
| `early_submit` "read `context`" | The name `context` appears in a cell's code, checked by the host's parser. A read through `globals()` or the file doesn't count. |
| Malformed audit lines and refused bridge requests | Both fire `escape_attempt`. Only model code can produce either. |
| Controller window, sparse families, minimum episodes, frozen mode | As in §2.3 |
| Ticket levels | 5 levels by ticket count, top level ~382K median |
| Audit sample size | 16 training wins, 8 eval wins, 4 flagged episodes per checkpoint |
| Clean solve rate | Solve rate × the clean share of audited wins, per split |
| `near_blind` | Logged, never rewarded |

## 7. What changes the spec or plan

For the planning thread, which owns both files.

1. **Spec §3 gains:**
   - the parsing details in §2.1 above
   - the grading format: a generator supplies the blind guess's *answer*, and the host computes `e_blind` or `F1_blind` from it
   - `correct` = exact and no detector fired
   - the `near_blind` metric
2. **Spec §3's H1 check needs one more row** for offset and noisy guessers, and decision 5b (keep the formula, Rishi 2026-10-11). With the spec's formula, the check as written can't fail.
3. **Spec §7 gains the exact rule definitions in §2.2,** including `copied_answer`'s three extra conditions (prompt = the first user message, typed in code, distinctive). It should also say that `escape_attempt` and `timeout_loop` fire without a submit.
4. **Spec §6 gains the controller details in §2.3:** the window holds only current-level episodes and is cleared by a move, steps without episodes don't count, at least 8 episodes, the saturation flag, and frozen mode.
5. **Spec §8, config.json,** adds `reward.settings()`, `detectors.settings()`, `controller.state_dict()` and `tasks.versions()`.
6. **Spec §9, logged every step,** adds hack hits by rule (`StepFilter.hacks`), hacked episodes, the `near_blind` rate and the controller's summary (level, window pass rate, saturated).
7. **Spec §10, code layout,** adds `rlm_rl/tasks/tickets.py`, the registry in `rlm_rl/tasks/__init__.py`, `scripts/reward_check.py`, `scripts/backtest_detectors.py` and `documentation/audit-procedure.md`. It should also say that `run_episode` runs the detectors.
8. **Plan step 5, the Claude audit,** gets its definition: 16 + 8 + 4 episodes per checkpoint, and clean solve rate = solve rate × clean share.
9. **Plan step 10, the second seed:**
   - It runs the controller frozen at level 1.
   - It can evaluate on the first seed's exact 50 held-out tasks (`legacy_tasks(50, 99_999)`, 2 samples each), as the 20-step run did. Note the first seed's eval ran under the old limits and reward, so the comparison is same-tasks, not same-environment.
10. **Plan step 7:** the ticket tasks already pass its 130K bar from level 3 up, but they're still the 20-step templates. Step 7's new generators each need a blind guess in their grading, and a `reward_check.py` row.

## 8. Known limits, for the red-team (step 8)

- **Offset and noisy guessing earns 0.13-0.21** under the spec's formula (§3, decision 5b).
- **`early_submit` is a tripwire.** Any mention of `context` satisfies it, for example `len(context)` before a guess.
- **`copied_answer` misses some copies:**
  - small numbers (under 100) and short strings
  - values pulled out by code (`context.split()[3]`)
  - values printed first and then submitted
- **`escape_attempt` inherits the audit hook's blind spots** from step 3 §6. `_posixsubprocess.fork_exec` and `signal.raise_signal` raise no audit event; Landlock still stops what they'd do.
- **Model code can forge audit events** by writing to file descriptor 2. That can only fire a detector on its own episode, never clear one.
- **The ticket templates are fixed,** about 40 sentences, so a policy could learn them as a lookup table (research §3.2). Step 7's generators replace them for production.

## 9. Commands

```bash
.venv-linux/bin/python scripts/sandbox_ctl.py check                         # all 170 tests, inside Ubuntu
.venv-linux/bin/python scripts/reward_check.py                              # H1: what blind guessers earn
.venv-linux/bin/python scripts/reward_check.py --slope 2                    # the same with the window halved (5b)
.venv-linux/bin/python scripts/backtest_detectors.py                        # detector false alarms on the 20-step run
.venv-linux/bin/python scripts/audit.py checkpoint results/<run> --step 10  # the Claude audit's sample
.venv-linux/bin/python scripts/audit.py report results/<run>                # clean solve rate from the verdicts
```
