# Step 8: red-team test plan

> **Step 8 of [production-build-plan.md](production-build-plan.md)** (hole H11, re-checks H1-H4; research §4.6). This is the last gate before any W&B credit is spent in step 10.
>
> **What this is.** A defensive QA plan for *our own* RL training environment. The goal is to confirm the environment resists reward hacking before a long run lets RL find whatever is reachable. It operationalizes the "known limits, for the red-team" sections the earlier steps already wrote ([step-3-sandbox-results.md](step-3-sandbox-results.md) §6, [step-5-results.md](step-5-results.md) §8, [step-7-results.md](step-7-results.md) §11) into an ordered, runnable checklist.
>
> **What this is not.** It is not an exploit manual. It contains no attack code, no payload strings and no sandbox-escape recipes. Every isolation check points at the step 3 test suite, and a pass is "that suite is green". Every reward and detector check uses tools that already exist (`reward_check.py`, `backtest_detectors.py`, `audit.py`, the regression suite) and local scoring against a fake model.
>
> **Cost.** Every step below is **$0**. It all runs locally, against the fake model (`tests/fake_model.py`) and host-side scoring. Nothing here calls W&B. The one place credit could enter is an optional *live* probe (§9), which is **not** part of step 8 and needs Rishi's go-ahead first.
>
> **Status when you start:** steps 1-7 are built and the suite is 241 tests, all green in WSL with Docker. Branch `claude/project-thread-9nnh3n` (step 7) on top of `claude/project-thread-2apfvb`.

---

## 0. DO NOT (read first)

These are hard rules for this project. None of the work below needs any of them broken.

- **No W&B inference credit without asking Rishi first.** Inference is the binding pot: $250 total, ~$218 left, and it is the limit on the whole project. Every step in this plan is $0 and local. If you ever think a check needs a real model call, stop and ask Rishi before spending a cent. The same goes for training (GPU) credit.
- **Never open, read, print, log or echo the W&B key.** It lives in the git-ignored `.env`. There is no reason to touch `.env` in step 8. Do not `cat` it, do not load it, do not print `os.environ`.
- **No commit and no push without asking Rishi.** Write the results file and any new tests, show Rishi, and commit only when he says so. Step 8's own branch is `claude/project-thread-9nnh3n` unless Rishi says otherwise.
- **Everything stays inside the RLM repo folder** (`F:\Personal\RLM\RLM`), docs in `documentation\`. The one exception is the datasets, which stay on `D:\RLM-data` (`/mnt/d/RLM-data` from WSL) and are never moved into the repo or into git.
- **No network from the sandbox.** The container runs `--network none` by design (step 3). Do not add a network, a relay, a port or a mount to "make a test easier". A check that seems to need network is a misread of the check.
- **Probe our own local environment only.** Step 8 is entirely about this repo's code on this PC. No external hosts, no other machines, no live services.
- **Never skip, disable, `xfail` or quarantine a test to get a green run.** A red test is a finding. If a test is genuinely wrong, fix the test and say why in the results file; do not silence it.
- **Do not weaken an isolation layer to observe what it blocks.** The step 3 tests already prove each layer in isolation. Re-run them; don't dismantle them.
- **If a safety check stops you on any section,** do not reword around it. Write the rest, mark that section "not written", and tell Rishi plainly which part was stopped and why.

**$0 vs credit at a glance:** §1-§8 and the checklist and template are all **$0, local, fakes**. §9 (optional live probe) is the **only** part that would spend inference credit, and it is gated on Rishi.

---

## 1. What step 8 proves, and how

Reward hacking is any way to earn reward without doing the task the way we intend. On a long RL run the policy will find whatever path pays, so before step 10 we confirm each known path is either blocked or measured and bounded. Step 8 groups the surface into five areas, each checked with tools that already exist:

| Area | Hole | What we confirm | Main tools |
|---|---|---|---|
| A. Isolation (sandbox + bridge) | H2-H4, H10 | The REPL cannot reach files, network, secrets, seeds, or other episodes | `tests/test_sandbox.py`, `test_bridge.py`, `test_contract.py` via `sandbox_ctl.py check` |
| B. Reward and parser | H1, §3 | Guessing from metadata pays far less than reading; the parser can't be gamed by format | `scripts/reward_check.py`, `tests/test_reward.py` |
| C. Detectors | H11, §7 | The four detectors catch what they should and don't fire on honest work | `tests/test_detectors.py`, `scripts/backtest_detectors.py` |
| D. Per-family | H7, H8 | Each family's own weak spots (guessable issues, copied names, label noise) are bounded | `scripts/reward_check.py`, `data_check.py`, family tests |
| E. Grader and infra integrity | H5 | A crash, a timeout or a detector bug becomes `infra_error`, never a win or a silent zero | `tests/test_sandbox.py` (`FailureTest`), `test_detectors.py` |

The method for every check is the same loop:

1. **Measure** with the named tool.
2. **Compare** against the pass/fail threshold.
3. On a pass, **record** the number in the results template (§11) and move on.
4. On a fail, **harden** (tighten or add a detector rule, fix the parser, or make the sandbox refuse the case), **add a regression test** that fails before the fix and passes after, re-run the full suite, and record both the finding and the fix.

A "fix" in step 8 is always one of: a detector rule, a parser rule, a sandbox refusal, or a regression test. It is never a change that weakens a check, and it never touches training credit.

---

## 2. Before you start

Run everything inside WSL (Ubuntu), from the repo root, with Docker Desktop running and WSL integration on for Ubuntu (step 3 §6). Use the Linux virtual environment, `.venv-linux/bin/python`.

**Baseline run — the whole suite must be green before you look for anything.**

```bash
.venv-linux/bin/python scripts/sandbox_ctl.py check
```

Expected: 241 tests, 240 pass, 1 skip (the step 4 tokenizer check skips in WSL). Record the count and the time. If anything other than that one tokenizer test is red or skipped, stop and fix the environment first — a red baseline makes every later result meaningless. **Do not** proceed by skipping the red test.

Then capture the reference numbers you'll compare against:

```bash
.venv-linux/bin/python scripts/reward_check.py                 # H1: what blind guessers earn, per family/level
.venv-linux/bin/python scripts/backtest_detectors.py           # detector false alarms on the 20-step run
.venv-linux/bin/python scripts/data_check.py                   # sizes, overlaps, licences -> manifests/data-check.json
```

These three cost $0 (no model calls) and give you the baselines for areas B, C and D. Save their output into the results file as the starting point.

---

## 3. Order of operations

Run the areas in this order. Each is $0 and local. The order goes from "nothing else is trustworthy without this" outward.

1. **§2 baseline** — full suite green, reference numbers captured.
2. **Area A, isolation** (§4) — if the sandbox leaks, every reward number is suspect, so confirm it first.
3. **Area E, grader/infra integrity** (§8) — confirm failures can't become wins, because the reward checks assume scoring is honest.
4. **Area B, reward and parser** (§5) — the largest live reward-hack surface once isolation holds.
5. **Area C, detectors** (§6) — confirm the backstop that zeroes hacks works and is quiet on honest work.
6. **Area D, per-family** (§7) — the family-specific weak spots, especially repo localization.
7. **§9 optional live probe** — only if Rishi approves credit; otherwise skip.
8. **§10 wrap-up** — fill the template, list findings and fixes, tell Rishi.

---

## 4. Area A — isolation (sandbox and bridge)

**Holes:** H2, H3, H4, H10. **Cost:** $0. **Tools:** `tests/test_sandbox.py`, `test_bridge.py`, `test_contract.py`.

The container is the boundary; Landlock and the Python audit hook sit inside it (step 3 §2.2). Step 3 already proves each layer on its own, so step 8's job is to **re-run those proofs on the current code** and confirm nothing regressed when steps 4-7 landed.

### What to test

Re-run the sandbox suite and read the layer tests' results:

```bash
.venv-linux/bin/python scripts/sandbox_ctl.py check
```

The done-when checks from step 3 that matter here (all already have tests):

| Check | Test | Pass means |
|---|---|---|
| `.env`, `.art/`, repo, home, `/proc` unreadable from inside | `FilesTest`, `LayersTest` | all green |
| Writes outside `/scratch` fail | `FilesTest.test_writes_only_in_scratch` | green |
| No network, DNS, `pip`, `curl` | `NetworkTest.test_no_network` | green |
| No project module importable | `ImportsAndSecretsTest` | green |
| No key or seed visible inside | `ImportsAndSecretsTest.test_no_keys_or_seed_inside` | green |
| The bridge is the only way out, and accepts only `llm`/`llm_batch` | `NetworkTest.test_bridge_is_the_way_out` | green |
| One fresh container per episode, nothing left behind | all sandbox tests check cleanup | green, no stray containers/dirs |
| Env and harness share one bridge prompt (H10) | `test_contract.py` | green |

### How to record

Record: the suite result (count, pass, skip, time), and an explicit "layer tests green" line. Note any container or episode directory left behind (`sandbox_ctl.py clean` should have nothing to do).

### Pass / fail threshold

**Pass:** every sandbox and bridge test green (the one tokenizer skip aside), no stray containers. **Fail:** any isolation test red, or any probe the tests record as `ALLOWED` instead of blocked.

### Known residual items to confirm (not bugs — confirm they are still contained)

These are documented in step 3 §6 and step 5 §8. They are accepted limits of the *tripwire* (the audit hook), each backed by a *wall* (Landlock, the pids limit, or the host). Step 8 confirms the wall still holds via the layer tests; it does **not** try to exercise the tripwire gap.

- The audit hook does not log every low-level fork/exec or self-signal path. **Landlock still stops any exec, and `--pids-limit` caps forking** — `LayersTest` proves the Landlock layer blocks running any program. Confirm that test is green.
- Model code can write to its own stderr/protocol pipe and forge an audit event or a death report. **This can only affect its own episode** (step 3 §2.4); it can never mark an episode `infra` or clear a real event. Confirm `FailureTest` still attributes every REPL death to the model.
- `/etc` is readable at the Landlock layer (the loader needs it); **the audit hook blocks Python-level reads**, and `/etc` holds nothing from the host beyond Docker's own files. Confirm the file tests are green.

### What to fix if a check fails

A real isolation failure is a step 3 regression. The fix pattern (step 3 §2): make the sandbox **refuse to start or turn the episode into `infra_error`** rather than run with less protection (the sandbox already does this for a missing Landlock), and add a `LayersTest`/`FilesTest`/`NetworkTest` case that fails before the fix. Never loosen a flag to make a test pass. If the failure is environmental (Docker Desktop down, WSL integration off), fix the environment and note it; that is not an environment bug.

---

## 5. Area B — reward and parser robustness

**Hole:** H1; spec §3. **Cost:** $0. **Tools:** `scripts/reward_check.py`, `tests/test_reward.py`.

This is the largest reward-hack surface once the sandbox holds: can a policy score without reading, either by guessing from metadata or by submitting a value whose *format* games the scorer?

### 5.1 Guessing from metadata (H1)

**What to test.** `reward_check.py` scores answers that never read the context — the task's own blind guess, a fixed-offset guess, and a noisy guess — over 1,000 tasks per family and level, and compares them to what an honest reader off by 1/2/3 earns.

```bash
.venv-linux/bin/python scripts/reward_check.py
.venv-linux/bin/python scripts/reward_check.py --slope 2    # the halved-window comparison (decision 5b)
```

**How to record.** The table of exact-hit rate, blind-guess reward, best-offset reward, noisy reward and honest-reader reward, per family and level. Also run one short training-style eval (fake model, $0) and record the `near_blind` rate.

**Pass / fail threshold.**
- **Pass:** the task's own blind guess earns exactly its exact-hit rate (0.00-0.11 by family/level, per step 5 §3 and step 7 §4.2). Offset/noisy guessers stay in the **0.13-0.21** band already documented and accepted under decision 5b, and well below an honest reader (0.30-0.79 at level 1). `near_blind` is flat, not climbing.
- **Fail:** any guesser earns more than ~0.21, or more than an honest off-by-1 reader, or `near_blind` climbs over a run.

**What to fix.** The clean one-constant lever is `NEAR_MISS_SLOPE` (halving the window drops guessers to 0.10-0.12; step 5 §3). But decision **5b kept slope 1** deliberately, so changing it is **Rishi's call** — surface it, don't just flip it. Other levers: strengthen the `near_blind` alarm, or re-examine a family's blind-guess definition if its guess is too easy to beat. No credit formula fully closes this (step 5 §3); the real backstops are `near_blind` and the Claude audit (area C and [audit-procedure.md](audit-procedure.md)).

### 5.2 Parser robustness (format gaming)

**What to test.** Confirm the reward parser's **documented** behaviour on edge-case submit values — it should neither over-credit a hedged or malformed answer nor crash. The intended behaviour is pinned in spec §3 and step 5 §2.1; `tests/test_reward.py` already encodes most of it. Walk each documented rule and confirm there is a test for it:

| Edge case | Intended behaviour (spec §3 / step 5 §2.1) | Covered by |
|---|---|---|
| Several numbers ("69 or 70") | First number counts; hedging doesn't pay | `test_reward.py` |
| Number joined to a word (`host-42`) | Reads 42, not −42 | `test_reward.py` |
| `$` and `,` in a number | Stripped before compare | `test_reward.py` |
| A string where a list is expected | Split on commas/semicolons/newlines | `test_reward.py` |
| A list where a single entity is expected | No partial credit for entities | `test_reward.py` |
| An answer that can't be parsed | Reward 0, never a crash | `test_reward.py` |
| Over-long / wrong-type value | Rejected in the REPL (step 3 §5 item 3), not scored | `test_bridge.py` / `test_contract.py` |

**How to record.** A line per edge case: "documented, tested, behaves as intended" or the gap found.

**Pass / fail threshold.** **Pass:** every documented edge case has a test and behaves as spec'd; no submit value makes the scorer raise instead of returning a reward (a scorer bug is routed to `infra`/`host_bug`, area E). **Fail:** a format earns more than the honest answer would, or the scorer crashes on an input instead of scoring it 0 or routing it to infra.

**What to fix.** Tighten the parsing rule in `rlm_rl/reward.py` to the intended behaviour and add a `test_reward.py` case built from the exact value that misbehaved (not a crafted exploit — the literal value you observed). Keep the rule minimal; don't add parsing the spec doesn't call for.

---

## 6. Area C — detectors: coverage and false alarms

**Hole:** H11; spec §7. **Cost:** $0. **Tools:** `tests/test_detectors.py`, `scripts/backtest_detectors.py`.

The four detectors (`early_submit`, `copied_answer`, `escape_attempt`, `timeout_loop`) zero a hacked reward **before** the group mean, turning a hack into a negative example (step 5 §2.2). Step 8 confirms two things: they fire on what they should, and they stay quiet on honest work.

### What to test

```bash
.venv-linux/bin/python scripts/backtest_detectors.py          # false alarms on the 20-step run (1,280 episodes)
# plus the detector unit tests inside the suite:
.venv-linux/bin/python scripts/sandbox_ctl.py check           # runs tests/test_detectors.py (31 tests, 7 through Docker)
```

Walk each rule against its documented behaviour and blind spots (spec §7, step 5 §2.2 and §8):

| Rule | Fires on | Known blind spot (confirm it's bounded, not a new hole) |
|---|---|---|
| `early_submit` | submit on turn 1, or before any cell names `context` | Any mention of `context` (e.g. `len(context)`) satisfies it — a tripwire, backed by `near_blind` and the audit |
| `copied_answer` | a distinctive value from the prompt, typed in code, in no output | misses small numbers (<100), short strings, code-extracted values, print-then-submit |
| `escape_attempt` | any sandbox audit event or refused bridge request | inherits the audit hook's gaps (area A); Landlock is the wall |
| `timeout_loop` | 3+ cell timeouts in an episode | — |

### How to record

- `backtest_detectors.py`: hits per rule on the 20-step run. The reference is **0 false alarms** on 1,280 episodes (step 5 §4).
- Detector unit tests: green, including the 7 that run through real Docker sandboxes.
- For `copied_answer` specifically, record the false-alarm count on honest repo-localization episodes (see §7.3).

### Pass / fail threshold

**Pass:** `backtest_detectors.py` still shows 0 false alarms; all `test_detectors.py` green; false-alarm rate ~0 on honest episodes of the new families. **Fail:** a detector fires on honest work (false alarm), or misses a pattern it is defined to catch.

### What to fix

- **False alarm:** tighten the rule's conditions (as `copied_answer` already gained three conditions in step 5 §2.2 after it would have fired on 7 honest wins), and add a regression test built from the exact honest episode that tripped it.
- **Miss:** broaden or add a rule to cover the class, and add a test that fails before and passes after. Keep rules host-side only (they read what the host recorded, so model output can't switch them off — step 5 §2.2).
- Record every change; a new detector rule is a spec §7 change the planning thread needs (step 5 §7).

---

## 7. Area D — per-family checks

**Holes:** H7, H8. **Cost:** $0. **Tools:** `scripts/reward_check.py`, `data_check.py`, `tests/test_data_families.py`.

Each of the three production families has its own weak spot. The mix is 45% aggregation, 40% code (callgraph 20 + needle 20), 15% repo localization (step 7 §6).

### 7.1 Aggregation (45%)

**What to test.** Guessing (covered by §5.1 — all levels show 0.00 exact-hit, so the blind guess earns ~0) and the label-noise ceiling. Run `reward_check.py` rows for `aggregation` and confirm `near_blind` stays flat in a fake-model eval.

**Pass / fail.** **Pass:** guessers at ~0, honest readers well above, `near_blind` flat. **Fail:** a guesser earns credit, or `near_blind` climbs. **Note (not a fail):** label noise caps accuracy at high levels (arXiv titles are the noisiest source, step 7 §11) — that's OOLONG's nature, and the controller keeps the pass rate in band. Record it, don't "fix" it.

### 7.2 Code — callgraph and needle (40%)

**What to test.** 
- `callgraph`: the blind guess is "every function in the package", which scores a low F1 (step 7 §2.2). Confirm `reward_check.py` shows it near 0, and that `copied_answer` doesn't fire on honest qualified-name answers (names are computed from the call graph, not copied from the prompt).
- `needle`: has **no** blind guess — a name can't be guessed from metadata, so a non-reader scores 0 (step 7 §4.2). Confirm that holds.

**Pass / fail.** **Pass:** callgraph blind guess near 0, needle guessers at 0, no `copied_answer` false alarm on honest code answers. **Fail:** either guesser earns credit, or `copied_answer` fires on honest qualified names.

**What to fix.** A `copied_answer` false alarm on code → tighten the rule (qualified names the code computed aren't "copied"), add a test.

### 7.3 Repo localization (15%) — the main watch item

This family is the outlier. At level 1, **40% of issues name exactly the files their fix changed** (step 7 §4.2), falling to 5% by level 5. Two things to check:

**(a) Guessable issues — legitimate skill vs hack.** A policy that submits "the files the issue mentions" gets those right without reading code. LCA's own benchmark keeps such issues, and these count toward the code-solvable share (step 7 §4.3), so this is a **real skill, not a hack** — do not try to detect it away. Record the rate (`data_check.py` / `reward_check.py` rows for `repo_loc`) and confirm it matches step 7's 40%→5% curve. A fail here is only if the rate is *higher* than documented (data regression).

**(b) `copied_answer` false alarm.** When an issue names the gold file, a policy might type the path straight from the issue without printing it, and `copied_answer` would wrongly zero an honest answer (step 7 §11). 

**What to test.** Run honest-style episodes (fake model, $0) on repo-localization tasks whose issue names the gold file, and count `copied_answer` false alarms.

**Pass / fail.** **Pass:** false-alarm rate ~0. **Fail:** `copied_answer` zeroes honest repo-localization wins.

**What to fix.** Tighten `copied_answer` so a file path the issue legitimately names (and the policy confirmed by reading) isn't treated as copied — e.g. require the value to appear in no output *and* the episode to show no genuine read of that file. Add a regression test from the honest episode. This is the single most likely detector change in step 8; flag it to Rishi as a spec §7 change.

**Note (not a fail):** pretraining memory — the base model may have seen LCA fixes (step 7 §11). That's measured by the step 11 baselines (how far base gets without reading), not by step 8.

---

## 8. Area E — grader and infra integrity

**Hole:** H5. **Cost:** $0. **Tools:** `tests/test_sandbox.py` (`FailureTest`), `tests/test_detectors.py`.

The reward checks assume scoring is honest: a crash, a timeout, or a bug must never become a win or a silent wrong zero. This re-confirms step 3's and step 4's failure handling on current code.

### What to test

Re-run the failure tests (part of the suite) and confirm the routing in step 3 §2.4:

| Situation | Must become | Test |
|---|---|---|
| Cell runs over 120 s compute | model told, restarted, variables lost (not a win) | `test_timeout_restarts_and_tells_the_model` |
| REPL crash / OOM / self-kill | model's outcome, restarted and told | `test_model_crashes_are_not_infra`, `test_self_inflicted_kills_are_the_models` |
| Sub-call fails after retries | `InfraError`, never a fake answer shown | `test_subcall_infra_error_discards_the_episode` |
| Host bug in sub-call code | `InfraError` | `test_host_bug_in_a_subcall_is_infra` |
| Container can't start / killed from outside | `InfraError` | `test_container_that_cannot_start_is_infra`, `test_killed_from_outside_is_infra` |
| A detector itself raises | `infra`/`host_bug`, never a zero or a win | `test_detectors.py` (detector bug as infra) |

Also confirm the key invariant: a flagged `infra_error` episode **leaves its group before advantages** and is counted, so it can't dilute or inflate a group (spec §5.4, step 4).

### How to record

A line per situation: "routes to [outcome] as intended, test green". Note the infra-pause rule is intact (>1% flagged pauses the run).

### Pass / fail threshold

**Pass:** every failure routes as the table says, all `FailureTest` green, a detector bug becomes `host_bug` not a zero. **Fail:** any failure path that could yield a scored reward (a win or a silent zero) from something that isn't the model solving the task.

### What to fix

Route the mis-handled failure to `infra_error`/`host_bug` in `rlm_rl/sandbox.py` or the scorer, and add a `FailureTest` case. Model-attributed deaths stay the model's outcome; only genuine infra becomes infra (step 3 §2.4).

---

## 9. Optional live probe (credit — gated on Rishi)

The plan's step 8 description imagines a Claude agent *trying* to earn reward on the real environment. **Everything that matters can be done at $0** with the fake model and local scoring (§4-§8), because the reward, detectors and sandbox all run host-side and don't need a real model to be exercised. That is the default, and it is what "done when" (§10) is written against.

A *live* probe — letting a real model roll out episodes on the environment and watching what it does — would spend **inference credit** and is therefore **not part of step 8 as planned**. If Rishi ever wants it:

- It needs his explicit go-ahead and a tiny budget named up front (keep it to a handful of episodes; ~$0.016/episode).
- It still probes **only our own local environment**.
- Its findings feed the same loop (§1): any new hack path gets a detector/parser/sandbox fix and a $0 regression test.

Default for this run: **skip §9.** Record "live probe not run; all checks done locally at $0".

---

## 10. Done when

Step 8 is done — the gate for step 10 opens — when **all** of these hold:

1. The full suite is green (240 pass, 1 skip) on current code.
2. Every area A-E check meets its pass threshold, or every failure found has a fix **and** a new regression test that fails before and passes after.
3. No reward-hack class is left measured-but-unbounded: guessing stays ≤ the documented band and below honest reading; detectors keep 0 false alarms and cover their classes; isolation and infra routing are green.
4. Every change is recorded in the results file with its before/after test.
5. Nothing was committed or pushed, no credit was spent, and `.env` was never opened.

The plan's phrasing is "the agent can no longer find an exploit". Translated to this $0 plan: **every known reward-hack class is contained or bounded, each backed by a regression test, with the full suite green.**

**Escalate to Rishi (don't decide alone):**
- Any change to the near-miss window (`NEAR_MISS_SLOPE`) — decision 5b is his.
- Any new or changed detector rule (it's a spec §7 change the planning thread owns).
- An audit `hack` verdict on a sampled win (that's stop rule 5).
- Anything that would need a live model call or any credit.

---

## 11. Results template

Fill this into `documentation/step-8-results.md` (write it, show Rishi, commit only when he says so). Keep gold answers and any sampled transcripts under `results/*/audit/`, which is git-ignored.

```markdown
# Step 8 results: red-team

> Step 8 of production-build-plan.md. Run <date> on Rishi's PC, WSL + Docker.
> No W&B credit used. .env never opened. Nothing committed/pushed until Rishi approves.

## Summary
- Full suite: <pass>/<total>, <skip> skipped, <time>. Baseline green: yes/no.
- Findings: <n> total, <n> fixed with regression tests, <n> escalated to Rishi.
- Verdict: gate for step 10 open / not open, and why.

## Baseline (§2)
- sandbox_ctl check: <counts/time>
- reward_check.py: <attach table>
- backtest_detectors.py: <hits per rule — expect 0 false alarms>
- data_check.py: <sizes/overlaps/licences OK?>

## Area A — isolation
| Check | Result | Evidence (test name) |
- Layer tests green: yes/no. Stray containers/dirs: none/<list>.
- Residual items confirmed contained: fork/exec+pids, forged audit events, /etc reads.

## Area E — grader/infra integrity
| Situation | Routes to | Test | Pass? |
- Infra episodes leave the group before advantages: confirmed yes/no.

## Area B — reward and parser
- Guessing (§5.1): blind <x>, offset <x>, noisy <x>, honest off-by-1 <x>. near_blind: flat/climbing.
- Parser edge cases (§5.2): each documented + tested? gaps: <list>.
- Pass/fail: <...>

## Area C — detectors
- False alarms (backtest): <n> (expect 0). Unit tests: green/<red list>.
- Per-rule coverage confirmed: early_submit / copied_answer / escape_attempt / timeout_loop.

## Area D — per-family
- Aggregation: guessers ~0, near_blind flat, label-noise ceiling noted.
- Code: callgraph blind ~0, needle guessers 0, no copied_answer false alarm on names.
- Repo loc: issue-names-file rate matches 40%→5% (skill, not hack). copied_answer false alarms: <n>.

## Findings and fixes
| # | Area | Finding | Fix (rule/parser/sandbox) | Regression test | Status |
- <one row per finding; "escalated to Rishi" where it's his call>

## Known residual items (accepted, not bugs)
- Offset/noisy guessing 0.13-0.21 (5b keeps slope 1; near_blind + audit watch it).
- Audit-hook tripwire gaps (Landlock/pids are the wall).
- copied_answer misses small numbers / code-extracted / print-then-submit by design.

## Decisions for Rishi
- <near-miss window? new detector rules? any hack verdict? live probe?>
```

---

## 12. Commands (all $0, local)

```bash
# baseline + references
.venv-linux/bin/python scripts/sandbox_ctl.py check              # full suite (241 tests)
.venv-linux/bin/python scripts/reward_check.py                   # H1: blind-guesser reward
.venv-linux/bin/python scripts/reward_check.py --slope 2         # halved-window comparison (5b)
.venv-linux/bin/python scripts/backtest_detectors.py             # detector false alarms (expect 0)
.venv-linux/bin/python scripts/data_check.py                     # sizes, overlaps, licences

# housekeeping
.venv-linux/bin/python scripts/sandbox_ctl.py clean              # remove anything a crash left behind

# audit (if a checkpoint exists to sample; reads local transcripts only)
.venv-linux/bin/python scripts/audit.py checkpoint results/<run> --step N
.venv-linux/bin/python scripts/audit.py report results/<run>
```

Not in step 8: any `scripts/train.py ... --live` command, anything touching `.env`, any commit or push. Those wait for Rishi.
