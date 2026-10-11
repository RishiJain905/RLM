# Step 8 results: red-team

> Step 8 of [production-build-plan.md](production-build-plan.md). Run 2026-10-11 on Rishi's PC, WSL (Ubuntu) + Docker Desktop 4.93 (engine 29.8.1), branch `claude/project-thread-9nnh3n`.
> **No W&B credit used.** Every check is local: fake model (`tests/fake_model.py`), host-side scoring, the 20-step run's saved trajectories, and generated tasks. The `.env` file was never opened. **Committed and pushed to `claude/project-thread-9nnh3n` with Rishi's approval, 2026-10-11** (spec §7 change and commit both approved).

## Summary

- **Full suite: 246 tests, 245 pass, 1 skip** (the step 4 tokenizer check, skips in WSL): 241 tests / 240 pass / 628 s at
  baseline, then 246 / 245 / 610 s at the final re-run with the five new regression tests (2 detector + 3 reward).

## Baseline (§2)

- `sandbox_ctl.py check`: **241 tests, 240 pass, 1 skip, 628 s**, green on first run.
- `backtest_detectors.py`: **0 detector hits over 1,280 episodes** (GRPO 640 + RLSD 640, 1,086 submitted) — 0 false alarms, as in step 5 §4. Re-run after the copied_answer fix: still 0.
- `data_check.py`: **all four gates pass** — H8 top-level medians 371K-404K chars; H1 blind ≤ exact-hit on every family and level; H7 zero shared templates/behaviours/sources/items; 7b licence bar clean; code-only share 36% (L1) falling to ~20% (L5). Output written to `manifests/data-check.json`; the only diff against the committed copy is `seconds_per_task` timing noise.
- `reward_check.py`: the tables below. (n: tickets 1,000, aggregation 400, callgraph 150, needle/repo_loc 400 per level.)

## Area A — isolation

| Check | Result | Evidence (test name) |
|---|---|---|
| `.env`, `.art/`, repo, home, `/proc` unreadable inside | green | `FilesTest.test_host_files_and_proc_are_unreadable`, `LayersTest` |
| Writes outside `/scratch` fail | green | `FilesTest.test_writes_only_in_scratch` |
| No network, DNS, `pip`, `curl` | green | `NetworkTest.test_no_network` |
| No project module importable | green | `ImportsAndSecretsTest.test_project_modules_are_not_importable` |
| No key or seed visible inside | green | `ImportsAndSecretsTest.test_no_keys_or_seed_inside` |
| Bridge is the only way out, `llm`/`llm_batch` only | green | `NetworkTest.test_bridge_is_the_way_out`, `EpisodeBridgeTest` |
| One fresh container per episode, nothing left behind | green | every sandbox test + `ScaleTest.test_parallel_episodes` |
| Env and harness share one bridge prompt (H10) | green | `test_contract.OneSystemPromptTest`, `WirePromptTest` |

- Layer tests green: **yes** (all in the baseline and final runs). Stray containers/dirs: **none** — `sandbox_ctl.py clean` removed 0 containers, no `rlm-*` container on the Docker list, 0 episode directories under `/tmp/rlm-sandbox-<uid>/`.
- Residual items confirmed contained (each tripwire gap still backed by its wall): fork/exec without an audit event is stopped by Landlock (`LayersTest` proves exec is blocked) and capped by `--pids-limit`; a forged death report/audit event can only hurt its own episode, and `FailureTest.test_model_crashes_are_not_infra` / `test_self_inflicted_kills_are_the_models` still attribute every REPL death to the model; `/etc` is readable at the Landlock layer but the audit hook blocks Python-level reads and the file tests are green.

## Area E — grader and infra integrity

| Situation | Routes to | Test | Pass? |
|---|---|---|---|
| Cell over 120 s compute | model told, restarted, variables lost — not a win | `test_timeout_restarts_and_tells_the_model` | green |
| REPL crash / OOM / self-kill | model's outcome | `test_model_crashes_are_not_infra`, `test_self_inflicted_kills_are_the_models` | green |
| Sub-call fails after retries | `InfraError`, never a fake answer | `test_subcall_infra_error_discards_the_episode` | green |
| Host bug in sub-call code | `InfraError` | `test_host_bug_in_a_subcall_is_infra` | green |
| Container can't start / killed from outside | `InfraError` | `test_container_that_cannot_start_is_infra`, `test_killed_from_outside_is_infra` | green |
| Scorer bug | `host_bug`, never a zero | `test_env.InfraTest.test_host_bug_is_infra_not_a_zero` | green |
| Detector itself raises | `host_bug`, never a zero or a win | `test_detectors.ZeroedBeforeTheGroupMeanTest.test_a_detector_bug_is_infra_not_a_zero` | green |

- Infra episodes leave their group before advantages: **confirmed** — `FilterTest.test_spec_5_4` (infra removed, groups under 2 dropped, all-equal dropped) and `InfraTest.test_none_reach_training` (a full step of forced 429s/kills: every flagged episode counted, none trained on).
- The infra-pause rule is intact: `FilterTest.test_infra_pause` (25% > 1% pauses; 1/320 under a 10-step window doesn't) and `test_pause_rule_per_run` (per step for the second seed, 10-step window in production).

## Area B — reward and parser

### Guessing (§5.1)

All tables from `scripts/reward_check.py` (slope 1, the spec):

| Family | Blind guess vs exact-hit (L1→L5) | Best fixed offset/subset | Noisy (0.5x/1x) | Honest reader off by 1 |
|---|---|---|---|---|
| tickets_all | 0.059 = 0.059 → 0.013 = 0.013 | 0.180-0.197 | 0.119-0.151 | 0.70 → 0.79 |
| tickets_customer | 0.112 = 0.112 → 0.030 = 0.030 | 0.188-0.206 | 0.135-0.174 | 0.59 → 0.74 |
| aggregation | 0.040 = 0.040 → 0.013 = 0.013 | 0.118-0.160 | 0.060-0.114 | 0.53 → 0.60 |
| callgraph | 0.000 = 0.000, every level | 0.003-0.011 | ≤0.005 | 0.63 → 0.76 |
| needle | 0.000 = 0.000, every level (entity: no guess) | 0.000 | 0.000 | 0.000 (a wrong name earns nothing) |
| repo_loc | 0.292 = 0.292 → 0.077 = 0.077 | 0.104-0.297 | 0.068-0.165 | 0.09 → 0.23 |

- **The blind guess earns exactly its exact-hit rate everywhere** (done-when passes on every family and level; all rows "pass").
- Offset/noisy guessers stay in the **documented 0.13-0.21 band** on the ticket families (0.118-0.206) and well below honest readers. `repo_loc`'s best fixed strategy (0.297 at L1) is the **issue-names-file skill**, not a hack — see Area D.
- **`near_blind` is flat, not climbing**: a 4-step production dry run on the production families (fake model, $0) logged `reward.near_blind` = 0.000, 0.062, 0.000, 0.000; detector hits 0, infra 0, capability 100%.
- With `--slope 2` (the 5b comparison): best fixed offsets drop to **0.100-0.122** (tickets) and 0.055-0.095 (aggregation), and honest readers lose ~0.1. Matches step 5 §3. **5b keeps slope 1** — no change proposed.

### Parser edge cases (§5.2)

| Edge case | Documented | Tested | Behaves as spec'd |
|---|---|---|---|
| Several numbers ("69 or 70") | first counts, hedging doesn't pay | `IntegerTest.test_first_number_only` | yes |
| Number joined to a word (`host-42`) | reads 42, not −42 | `ParseTest.test_first_number` | yes |
| `$` and `,` in a number | stripped before compare | `ParseTest.test_first_number` (`$1,234.50`) | yes |
| A string where a list is expected | split on commas/semicolons/newlines | `OtherTypesTest.test_list_by_set_f1_...` (`"a, b; c\nd"`) | yes |
| A list where a single entity is expected | no partial credit | `OtherTypesTest.test_entity_has_no_partial_credit` (`["ws-07"]` → 0) | yes |
| Unparseable answer | 0, never a crash | `IntegerTest.test_unparseable_or_missing_is_zero` | yes |
| Over-long / wrong-type value | rejected in the REPL, not scored | `ReplTest.test_submit_keeps_types` (200K string), `test_every_submitted_value_is_a_valid_answer` | yes |

**Pass:** every documented rule has a test, and no submit value makes the scorer raise — a scorer/detector bug routes to `infra`/`host_bug` (Area E), never a silent zero.

## Area C — detectors

- **False alarms (backtest): 0** hits over 1,280 episodes, before and after the fix.
- **Unit tests: 33 green** (31 before step 8 + 2 new), including the 7 that run through real Docker sandboxes (`DockerDetectorTest`).
- Per-rule coverage confirmed:
  - `early_submit`: fires on turn-1 submit and submit without any code naming `context` (incl. obfuscated `globals()` reads); quiet on honest episodes; bounded tripwire (`len(context)` satisfies it) — backed by `near_blind` + the audit.
  - `copied_answer`: fires on typed prompt values in no output; quiet on computed, printed, sub-call-derived and chance-matching answers; **new**: quiet on the task's own blind guess (the reward already caps it); lists need every atom copied. Known misses (small numbers <100, code-extracted, print-then-submit) are by design and recorded below.
  - `escape_attempt`: fires on every audit kind (network, file, process, ctypes, signal, tamper, host_import, malformed) and refused bridge requests, **without a submit**; `nonstdlib_import` doesn't fire.
  - `timeout_loop`: fires at 3+ cell timeouts, and the model is told.

## Area D — per-family

- **Aggregation (45%):** blind guess earns exactly its exact-hit rate (0.013-0.040); the best fixed subset guesser ≤0.160 and noisy ≤0.114, all below honest reading (0.53+); `near_blind` flat in the dry run. **Label-noise ceiling noted, not a fail**: arXiv titles are the noisiest source (step 7 §11) — that's OOLONG's nature; the controller keeps the pass rate in band.
- **Code (40%):** `callgraph`'s blind guess (every function in the package) earns **0.000-0.011**; `needle` guessers **0.000** at every level (a name can't be guessed from metadata). `copied_answer` fired **0** times on 24 honest callgraph qualified-name answers and 12 needle answers in the probe — names the code computes aren't "copied".
- **Repo loc (15%), the main watch item:**
  - **(a) Guessable issues = skill, not hack.** The issue names exactly the gold files for 40% of level-1 tasks falling to 5% at level 5 (`data_check.py`, matches step 7 §4.2 exactly). The larger sample (`reward_check.py`, 400/level) gives 0.29 → 0.08 — the same falling curve, tails slightly fatter than the n=20 documentation but never *higher* in the way a data regression would show. No action.
  - **(b) `copied_answer` false alarms — the expected finding.** A probe ran 104 honest-style episodes (oracle policy: read `context` on turn 1, submit the exact gold on turn 2, print nothing) through the real episode loop on the production families. **Before the fix: 13 honest wins zeroed** — 9 repo_loc wins where the issue names the gold path or the first-3-lines sample shows it, 2 repo_loc wins where only the sample shows it, and 2 aggregation wins (topics the question itself lists). **After the fix: 1 remains** (aggregation `most_active_user`: a user id visible in the prompt's first-3-lines sample, typed as a literal, never printed — genuinely indistinguishable from copying by the rule's inputs). Residual, recorded below; no rule change proposed for it.

## Findings and fixes

| # | Area | Finding | Fix (rule/parser/sandbox) | Regression test | Status |
|---|---|---|---|---|---|
| 1 | C/D (H11) | `copied_answer` zeroed **13/104 honest wins** in the probe: repo_loc paths the issue or sample names, aggregation topics the question lists. Confirmed step 7 §11's *(inferred)* risk as real. | **Detector rule** (spec §7): skip any atom of the task's own blind guess (`rlm_rl/detectors.py` `_blind_atoms`; the reward already caps that guesser, H1). Post-fix: 0/60 repo_loc probe false alarms, backtest still 0/1,280. | `tests/test_detectors.py` `test_not_for_the_task_s_own_blind_guess` (fails before: `['copied_answer'] != []`) + `test_still_fires_when_the_answer_is_not_the_blind_guess` | **Fixed; spec §7 change approved by Rishi 2026-10-11** |
| 2 | B/D (H1 tooling) | `scripts/reward_check.py` **crashed on 4 of 6 production families** (KeyError `blind` on entity tasks; `list − list` TypeError) — written in step 5, never extended for step 7's types. | **Parser/tooling**: type-dispatched scoring through the real `score_answer`; lists get fixed-prefix strategies, numbers keep step 5's fixed-offset semantics; noise scaled by the family-typical error. No production code touched. | `tests/test_reward.py` `RewardCheckScriptTest` (3 tests; the pre-fix shapes raise KeyError/TypeError) | **Fixed** |
| 3 | C/D (H11) | The one residual probe false alarm (aggregation `most_active_user`, user id in the prompt sample typed unprinted). Honest episode, but the rule's inputs can't tell it from a copy. | None proposed: tightening further (e.g. ignoring first-3-line strings) would also blind the rule to copying a count from the sample. `near_blind` + the Claude audit watch it. | — | **Accepted residual; FYI to Rishi** |

Files changed (committed and pushed to `claude/project-thread-9nnh3n`, Rishi's approval 2026-10-11): `rlm_rl/detectors.py`, `scripts/reward_check.py`, `tests/test_detectors.py` (+2 tests), `tests/test_reward.py` (+1 test class, 3 tests), `manifests/data-check.json` (regenerated, timing-only diff), `documentation/production-spec.md` (§7 `copied_answer` row) and this file. Probe artifacts kept in `.art/step8/` (git-ignored). The dry-run scaffold (`results/dry/nb4*`) was removed after recording.

## Known residual items (accepted, not bugs)

- Offset/noisy guessing earns 0.118-0.206 on the ticket families and ≤0.165 elsewhere (inside the accepted 0.13-0.21 band; 5b keeps slope 1). `near_blind` (flat in the dry run) and the audit watch for drift.
- `repo_loc`'s issue-names-file answers are a real skill (40%→5% exact-hit) and pay as skill; the reward's H1 guarantee holds on every level.
- `copied_answer` misses small numbers (<100), code-extracted values, and print-then-submit by design, and still fires on 1/104 honest probe episodes (the `most_active_user` residual above).
- The audit hook's tripwire gaps (fork/exec, self-signals) — Landlock and `--pids-limit` are the wall; `LayersTest`/`FailureTest` green.
- Pretraining memory on LCA code: step 11's baselines measure it, not step 8.
- Label noise caps aggregation accuracy at high levels (arXiv noisiest) — OOLONG's nature; the controller keeps the rate in band.

## Decisions for Rishi

1. **The `copied_answer` spec §7 change** (skip the task's own blind-guess atoms) — **approved by Rishi, 2026-10-11**; spec §7's `copied_answer` row now carries the condition.
2. **`reward_check.py` rewrite** — tooling only; the numbers it now prints are the first full-family H1 table (step 5's script only ever covered the two ticket families at n=1,000; the others ran at n≤400 for time).
3. **Nothing else changed**: no near-miss window change (slope stays 1), no new detector rules beyond finding 1, no audit verdicts, no live probe.
4. **§9 live probe: not run; all checks done locally at $0.** A live probe would need your go-ahead and a named budget (~$0.016/episode).
5. **Commit approval: granted 2026-10-11.** Committed and pushed to `claude/project-thread-9nnh3n` (one commit, message "Apply step 8: copied_answer skips the task's blind guess").

## Commands (all $0, all run)

```bash
.venv-linux/bin/python scripts/sandbox_ctl.py check              # 241 baseline → 246 final, 240 → 245 pass, 1 skip
.venv-linux/bin/python scripts/reward_check.py                   # full H1 table (tickets n=1000; others 150-400)
.venv-linux/bin/python scripts/reward_check.py --slope 2         # 5b comparison
.venv-linux/bin/python scripts/backtest_detectors.py             # 0 hits / 1,280 episodes
.venv-linux/bin/python scripts/data_check.py                      # H8/H1/H7/7b pass
.venv-linux/bin/python scripts/sandbox_ctl.py clean               # removed 0 containers
.venv-linux/bin/python scripts/train.py start --kind production --run nb4 --lr 2e-5 --dry-run --max-steps 4   # near_blind flat
.venv-linux/bin/python .art/step8/probe_copied.py                 # the honest-episode probe (Area D)
```

Not done during the checks, by the plan's rules: any `train.py --live`, anything touching `.env`, any W&B call. The one commit and push happened after the checks, with Rishi's approval (2026-10-11). §13's public benchmarks are untouched (step 13's job).