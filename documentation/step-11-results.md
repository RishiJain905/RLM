# Step 11 results: baselines

> **Step 11 of [production-build-plan.md](production-build-plan.md)** (research §8.2, spec §12.10). Run on 2026-10-16 and 17 UTC on Rishi's PC, end to end from a Claude thread while Rishi was away: episodes in WSL and Docker, the dashboard and this write-up from Windows.
>
> **Decisions applied:**
> - **11a** (Rishi, 2026-10-16): sub-calls use the base model, as in training. Every arm here does.
> - **11b** (Rishi, 2026-10-16): one extra arm, the base model as an RLM with root thinking at `medium`. Sub-calls stay off.
> - The set-up the coordinator passed on: prompt v2, the 8,000-character REPL output cap, the step 10 limiter (at most 12 requests in flight to the base model, starts at least 0.25 s apart, 8 tries from 4 s), and a $12 hard ceiling on W&B inference. The ~$2 base eval on the second seed's held-out tasks was not run; it waits on Rishi's answer in the project chat.
>
> **This write-up as a page:** [Baseline Run Results](https://claude.ai/artifact/TfAMvuzFTTKCJqrk5nm25P).
>
> **Dashboards:** step 11 has its own page, [Step 11 Baselines](https://claude.ai/artifact/KLzyWe43mQoHtV4Mn35r3q). Step 10's page was frozen first as [Step 10 Dashboard](https://claude.ai/artifact/Cto75Tz4h4A4KXjthBAXhy). The original live link ([Jrm5PZd6VQNJT7cpkM9ou3](https://claude.ai/artifact/Jrm5PZd6VQNJT7cpkM9ou3)) was not touched. Both new pages are private until shared from their Share menu.
>
> **W&B inference:** $5.89 measured by the script ($2.40 RLM arm, $1.09 direct reading, $2.39 medium thinking). I also counted up to $2.15 more against the ceiling, so $8.04 in all, inside the $12 ceiling. That extra is the diagnosis probes and failed tries W&B may have billed (§8). The billing page decides which it is (§9).
>
> **Status:** complete. Rishi answered the three calls on 2026-10-17 (prompt v3, medium thinking, 30 edit turns) and chose to replace the ticket tasks (`selection-v3`). The base model was rerun in production's set-up: 51.6% (§13). Training with thinking still has to be wired before production (§13).
>
> Anything marked *(mine)* is my choice, not the plan's. Anything marked *(inferred)* hasn't been checked.

## Summary

**Update 2026-10-17:** Rishi took all three calls (prompt v3, medium thinking, 30 edit turns) and replaced the ticket tasks (`selection-v3`). In that set-up the base model scores 51.6%, the step-0 number production starts from (§13). Step 11 counted $9.99 of its $12.

**Medium thinking clears the bar: +17 points over thinking off (59.4% vs 42.2% over the 64 tasks, ± 6.8, paired by task).** Reading gains 15 points and editing 25. Production would think at `medium` under 11b's rule, at about 1.25× the cost per step. Part of the gain is that thinking mostly stops a format failure, described next.

**The base model often answers in its native tool-call syntax instead of a ```python block, and those turns do nothing** (§3). With thinking off this hit 18 of 48 read episodes and 17 of 32 edit episodes, mostly on the very first turn. Episodes that did it were right 17-24% of the time, against 57-67% for those that didn't. It is a prompt v2 effect: the system prompt now lists the coding tools, and Qwen3.8 is trained to call listed tools with `<tool_call>` tags. The second seed's prompt v1 almost never triggered it. Half of these turns carry runnable Python inside the tags.

**Reading directly beat the base RLM on this set** (59% vs 43% over the 44 read tasks both finished; 54% for medium thinking). The gap is all at levels 1-2, where the text is 35-100K characters and simply fits. At levels 3-5 the RLM is ahead, 55% vs 45%, on small numbers. Every selection-set context fits the model's window, so this set can't show the RLM's real advantage; the final eval's 1M-character tasks can.

**Edit episodes cost far less than planned, so the $157 buys more steps.** An edit episode costs $0.035 (synth $0.024, real $0.042), against the plan's guesses of $0.06 and $0.15. At production's mix a step costs about $0.95 with thinking off and about $1.19 at medium. That is ~165 or ~130 steps for $157, against the plan's 75 *(inferred: training usually lengthens episodes)*. Steps also look shorter than feared, about 15-20 minutes rather than 30-35 *(inferred)*.

**Rate limiting is fixed, but W&B now fails about 27% of tries with HTTP 500** (§8). There were 0 rate-limit (429) responses in 2,645 tries, so the step 10 limiter works. There were 712 HTTP 500s ("Failed to generate completions"); retries recovered every RLM episode (0 of 144 lost). Direct reading lost its 4 largest contexts to 500s and 120-second proxy timeouts (524).

| Done-when check (plan step 11) | Result | Evidence |
|---|---|---|
| Both baselines logged with full transcripts, and the thinking arm | **Done** | 80 RLM, 44 of 48 direct and 64 medium-thinking episodes, every try in `results/step11-v1/transcripts/val/` |
| Cost per edit episode measured; production's step count re-estimated | **Done: cheaper** | $0.035 an edit episode (plan $0.06-0.15). ~$0.95 a step, ~165 steps for $157 (plan ~$2.10, ~75) |
| Over ~20% of edit episodes at the 20-turn cap unresolved → 30 turns | **Over: 56%** (thinking off), **25%** (medium) | 18 of 32, mostly tool-call loops; 5 of 15 (33%) among episodes without them (§4) |
| Base under ~15% on `edit_real` level 1 → lower `edit_real` share for 20 steps | **Pass: 40%** (off), **80%** (medium) | 4 of 10 off (5 before a detector false positive), 4 of 5 medium |
| 11b settled: medium only at about +12 points | **+17.2 ± 6.8** | 64 tasks paired by task; read +14.6 ± 8.4, edit +25.0 ± 10.2 (§6) |

---

## 1. What ran

| Arm | What | Episodes | Accuracy | Cost |
|---|---|---|---|---|
| `rlm` | Base Qwen3.8-27B as an RLM, environment v2, prompt v2, thinking off. The 16 edit tasks run twice | 80 (48 read + 32 edit) | **42.5% ± 5.5** | $2.40 |
| `direct` | The base model reading the whole context in one call, thinking off, read tasks only | 44 of 48 (4 lost, §8) | **59.1% ± 7.4** | $1.09 |
| `rlm_medium` | As `rlm`, with root thinking at `medium`; edit tasks once | 64 (48 read + 16 edit) | **59.4% ± 6.1** | $2.39 |

The task set is `selection-v2`, production's selection set (spec §12.10). It has 48 read tasks: aggregation 17, call graph 7, needle 7, repo localization 9, tickets 8, across levels 1-5. It also has 16 edit tasks: 10 `edit_real` from held-out repositories (Go 4, JavaScript 3, Python 3; 5 at level 1) and 6 generated `edit_synth` tasks (Python, levels 1-5). The contexts run from 34K to 423K characters.

**Direct reading** *(mine)*: one system and user message carrying the question, the whole context between `<context>` tags, the question again, and "end your reply with `FINAL ANSWER: …`". The model gets 6,144 output tokens at temperature 1.0, as the RLM's root does, and may reason in writing first. The answer after the last `FINAL ANSWER:` is scored by the same scorer as a submitted value. No detectors run, since there is no REPL. 10 of the 44 direct episodes wrote past 6,144 tokens without an answer, which counts as wrong.

**Timeline (UTC).**

| Time | What happened |
|---|---|
| 22:41 | Rishi's go-ahead; I read the plan and wrote the script |
| 22:53 | $0 dry run of all three arms, 192 episodes, clean |
| 23:01 | Step 10's page frozen as its own artifact |
| 23:07 | Session 1 live: `rlm` and `direct` together |
| 23:21 | 90% of both arms done. The tool-call failure and W&B's 500s found (§3, §8) |
| 23:43 | Session 1 stopped by hand: the 4 largest direct contexts kept failing (§8), and the last RLM episode, a Go edit task, was mid-cell |
| 23:48 | Session 2: the last `rlm` episode and `rlm_medium`. The spend gate refused 34 of the medium arm's episodes at once (§10); fixed |
| 00:24 | Session 3: the 34 medium episodes. Done 00:30 |

## 2. Results

**By family** (correct of scored):

| Family | RLM, off | Direct | RLM, medium |
|---|---|---|---|
| aggregation | 8 / 17 | 5 / 15 | 10 / 17 |
| callgraph | 1 / 7 | 6 / 7 | 3 / 7 |
| needle | 2 / 7 | 4 / 7 | 3 / 7 |
| repo_loc_select | 4 / 9 | 7 / 9 | 5 / 9 |
| tickets_all | 1 / 4 | 1 / 3 | 2 / 4 |
| tickets_customer | 4 / 4 | 3 / 3 | 4 / 4 |
| **Read, all** | **20 / 48 (42%)** | **26 / 44 (59%)** | **27 / 48 (56%)** |
| edit_synth | 8 / 12 | | 5 / 6 |
| edit_real_select | 6 / 20 | | 6 / 10 |
| **Edit, all** | **14 / 32 (44%)** | | **11 / 16 (69%)** |

**Read tasks by level:**

| Level | RLM, off | Direct | RLM, medium |
|---|---|---|---|
| 1 (~35-50K chars) | 5 / 12 | 10 / 12 | 7 / 12 |
| 2 (~65-100K) | 3 / 12 | 7 / 12 | 6 / 12 |
| 3 (~115-170K) | 5 / 10 | 4 / 10 | 5 / 10 |
| 4 (~240-280K) | 4 / 7 | 3 / 6 | 4 / 7 |
| 5 (~350-425K) | 3 / 7 | 2 / 4 | 5 / 7 |

The ticket tasks are the held-out generator family production never trains on. The base model gets `tickets_customer` right every time in every arm, and `tickets_all` 1-2 times in 4.

## 3. The tool-call failure

Prompt v2's system prompt lists seven coding tools as Python signatures (spec §12.1). Qwen3.8 is trained to call tools that a system prompt lists, in its own XML syntax. So on many turns the base model writes `<tool_call>` tags instead of a ```python block. Sometimes it writes `<tool_call><function=python>` around perfectly good code. Sometimes it writes the tags around nothing, then repeats the empty tags turn after turn. Either way the REPL finds no ```python block, returns the "No ```python code block found" error and burns the turn.

| | RLM, off: read | RLM, off: edit | RLM, medium: read | RLM, medium: edit |
|---|---|---|---|---|
| Episodes with tool-call syntax | 18 of 48 | 17 of 32 | 6 of 48 | 12 of 16 |
| Turns with it | 181 of 438 (41%) | 245 of 559 (44%) | 27 of 378 (7%) | 147 of 266 (55%) |
| It started on the first turn | 15 of 18 | 15 of 17 | 4 of 6 | 10 of 12 |
| Correct, with it | 3 of 18 (17%) | 4 of 17 (24%) | 2 of 6 (33%) | 8 of 12 (67%) |
| Correct, without it | 17 of 30 (57%) | 10 of 15 (67%) | 25 of 42 (60%) | 3 of 4 (75%) |

- **The tool-call failure is most of the gap between the RLM and direct reading.** Without it the RLM arm's read accuracy (57%) is about direct reading's (59%).
- **It is new with prompt v2.** The second seed ran prompt v1, which lists no tools: 31 of its 4,916 training turns (0.6%) had the tags, against 426 of this run's 997 (43%).
- **Of the first tool-call turn in each episode,** 16 of 35 held runnable Python inside `<function=python>`, 4 held a ```python block (which ran), and 15 held no code at all.
- **Thinking mostly prevents it on read tasks** (6 of 48 episodes), but not on edit tasks (12 of 16), where it recovers: most of those episodes still finished.

**What it means for production** *(mine)*. Training would see an untrained model wasting a third to half of its turns on syntax the environment can't run. GRPO would likely train it away, since those episodes score lower, but the first steps' gains would then be format learning, and they would flatter the trained model against this base number. It is cheap to fix before production (§11, decision A).

## 4. Editing (plan step 11's two edit gates)

| | RLM, off (32) | RLM, medium (16) |
|---|---|---|
| Resolved: `edit_real` | 6 / 20 (30%) | 6 / 10 (60%) |
| … level 1 | **4 / 10 (40%)**; 5 / 10 before a detector false positive | **4 / 5 (80%)** |
| … by language | Go 4/8, JavaScript 2/6, Python 0/6 | Go 3/4, JavaScript 1/3, Python 2/3 |
| Resolved: `edit_synth` | 8 / 12 (67%) | 5 / 6 (83%) |
| Hit the 20-turn cap | 19 (59%) | 8 (50%) |
| **At the cap and unresolved** | **18 (56%)** | **4 (25%)** |
| … among episodes without tool-call syntax | 5 of 15 (33%) | 1 of 4 |
| Submitted | 16 (50%) | 13 (81%) |
| Ran the tests before submitting | 44% | 56% |
| Sub-calls an episode | 0.1 | 0 |
| Turns an episode | 17.5 | 16.6 |

- **The turn-cap gate is over 20%, so by the plan's rule edit turns go to 30.** The cause is mostly the tool-call loops, though: fixing those (decision A) should bring most of these episodes in under 20. Even without loops, 5 of 15 hit the cap, which is still over 20%. Edit episodes cost $0.035, so 10 more turns would cost little *(inferred)*.
- **The `edit_real` level 1 gate passes** at 40% (thinking off) and 80% (medium), so `edit_real` keeps its 25% share from the start.
- **The base model never uses sub-calls on edit tasks** (2 sub-calls in 48 edit episodes). The RLM habit for editing, surveying with `llm_batch` and reviewing a change with `llm`, would have to come entirely from training.
- **A detector false positive zeroed one resolved JavaScript episode.** The `test_aware` check fired on a scratch `reproduce.js` the model left in the repository, which ends with `setTimeout(() => process.exit(0), 3000)`. That isn't test-detection code. Three `escape_attempt` hits (two blocked file writes, one blocked process) all landed on wrong answers, so they changed nothing.

## 5. Direct reading against the RLM

| Read tasks both finished (44) | Direct | RLM, off | RLM, medium |
|---|---|---|---|
| All | 59.1% | 43.2% (−15.9 ± 9.7) | 54.5% (−4.5 ± 9.7) |
| Levels 1-2 (24) | 70.8% | 33.3% | 54.2% |
| Levels 3-5 (20) | 45.0% | 55.0% (+10 ± 14) | 55.0% (+10 ± 14) |

- **Every selection-set context fits the served window,** up to 178K prompt tokens in this run. The plan's direct-reading baseline assumed degradation within the window; at these sizes there is little. Call graph (6 of 7 direct, 1 of 7 RLM) and repo localization (7 of 9 vs 4 of 9) are where reading the whole text wins most.
- **The RLM pulls ahead only at levels 3-5,** and the sample there is too small to call (± 14).
- **What this means for the claim that the RLM setup helps:** for the untrained model at these sizes, it doesn't yet. The claim rests on training, and on the final eval's 1M-character tasks, which don't fit any window. The 4 lost direct episodes were the largest contexts (260-423K characters), exactly where direct reading should fall behind (§8). Step 13, off W&B, can run them without the proxy timeout.

## 6. 11b: medium thinking

| | Thinking off | Medium | Difference (paired by task) |
|---|---|---|---|
| All 64 tasks | 42.2% | 59.4% | **+17.2 ± 6.8** |
| Read (48) | 41.7% | 56.3% | +14.6 ± 8.4 |
| Edit (16) | 43.8% | 68.8% | +25.0 ± 10.2 |
| Thinking tokens a read episode | 0 | 3,600 | |
| Root output tokens a read episode | 2,170 | 5,272 | 2.4× |
| Cost a read / edit episode | $0.026 / $0.035 | $0.036 / $0.042 | 1.35× / 1.2× |
| Production step at spec §12.7's mix | ~$0.95 | ~$1.19 | 1.25× |

- **By the plan's rule production thinks at `medium`:** +17 points is above the ~12-point bar, at about 2.5 standard errors.
- **How much is the format failure?** On episodes without tool-call syntax the two arms are closer on reading (60% medium vs 57% off). So a fair share of medium's read gain is that thinking stops the bad first turn. Editing gains with or without it.
- **Thinking sometimes eats the whole turn.** 22 of 644 medium turns (3.4%) hit the 3,072- or 4,096-token cap, 17 of them spent entirely on thinking with no reply, in 13 episodes (6 still correct). Thinking off had none. Production at `medium` would see the same, so a larger read-turn cap is worth considering *(mine)*. Medium cost 1.25× a step, less than step 2's 1.6× guess.
- **Production would train on the thinking tokens too** *(inferred from step 2 §6: logprobs cover them and the token-id rebuild works)*. That is untested in a train call, so a fake-model dry run with thinking on comes before production if Rishi picks medium.

## 7. Cost and time

**Per episode (measured, thinking off):** read $0.026 (aggregation $0.033, call graph $0.024, needle $0.008, repo localization $0.027, tickets $0.030). Edit synth $0.024 and edit real $0.042, against the plan's guesses of $0.06 and $0.15. Edit episodes are cheap because their turns are short (1,760 output tokens an episode, against the plan's ~11K). Prompts are uncached: 74K root prompt tokens an edit episode, $0.03 of the $0.04.

**Production re-estimated** (spec §12.7's mix, 32 episodes a step, the RLM arm's cost per family):

| | Thinking off | Medium |
|---|---|---|
| A step, families at the selection set's levels | $0.95 | $1.19 |
| A step, at level 1 costs | $0.70 | $0.98 |
| Steps for the $157 ceiling | ~165 (~225 at level 1 costs) | ~130 (~160) |
| A selection-set eval (64 tasks) | $1.83 | $2.39 |
| Plan (code-editing-plan.md §10.2) | ~$2.10 a step, ~75 steps, ~$3.50 an eval | |

*(inferred)* Training changes behaviour: in step 10 sub-calls rose ~20% over 20 steps, and longer transcripts cost more. A trained model that tests its edits will also run longer. A working range is ~$1-1.5 a step, so ~100-150 steps, well over the plan's 75. Two stop rules (plateau, saturation) will likely end the run before the money does. The $30 eval budget covers 12-16 selection evals.

**Time (re-measured with the step 10 limiter).**

| | Median | p90 | Max |
|---|---|---|---|
| RLM read episode | 231 s | 316 s | 728 s |
| Medium read episode | 208 s | 334 s | 407 s |
| `edit_synth` episode | 75 s | | 164 s |
| `edit_real` episode (Go: 334-608 s) | 238 s | 435 s | 608 s |
| Go grader | 3 min | | |
| Root call, retries included | 4.6 s | 17 s | 360 s |

All 80 RLM episodes started together, more than two production steps' worth, and 90% were done in 14 minutes. A production step is its slowest episode plus grading and training, so about 15-20 minutes *(inferred)*, against the plan's 30-35. The base model rarely ran the slow Go test suites (Go tests here took 1-2 minutes, not the 15-30 measured under the data build's load). A trained model that tests every change will be slower. **~130-165 steps at 15-20 minutes is 35-55 hours of the PC running.**

## 8. Infra: rate limits and W&B's errors

| | Tries | 429 (rate limited) | 5xx | Episodes lost |
|---|---|---|---|---|
| Session 1: `rlm` | 1,459 | 0 | 363 | 0 |
| Session 1: `direct` | 86 | 0 | 42 | 4 |
| Session 2: `rlm`, `rlm_medium` | 679 | 0 | 195 (all HTTP 500) | 0 |
| Session 3: `rlm_medium` | 421 | 0 | 112 (all HTTP 500) | 0 |
| **All** | **2,645** | **0** | **712 (27%)** | **4 of 192** |

- **The step 10 limiter fixed rate limiting:** 0 429s in 2,645 tries (step 10 had 3,103). The cost is pacing time: a medium-arm session spent 17 minutes in total, summed over episodes, waiting for start times.
- **W&B failed 27% of tries with HTTP 500, "Failed to generate completions".** Step 10 saw almost none (0-1 a step). The client counted statuses only from session 2 on; there, every failure was a 500. Retries (8 tries from 4 s) recovered every RLM episode. *(inferred)* It may be load on W&B's side tonight, or our 12-in-flight cap; production's per-step infra numbers will show it.
- **Requests that run longer than 120 seconds fail with Cloudflare's 524.** I measured this with a probe: the largest context (126K prompt tokens) answered in 52 s with a 16-token answer, but timed out at 120 s with a 4,900-token answer. Direct reading of the largest contexts needs that long, so 4 direct episodes (260-423K characters) failed 8 tries each. Streaming would avoid the timeout, but W&B's gateway can't parse a streamed reply through ART's client (it returned a 500 "Failed to parse response"). RLM turns finish well inside 120 s.
- **The largest successful prompt was 177,892 tokens,** so the served window is over 128K.
- **Billing of failed tries is unknown.** A 524'd request may have finished on W&B's side and been billed. I counted an upper bound against the ceiling: $1.92 for the 32 failed direct tries on the 4 largest contexts, priced as full requests, and $0.23 for the probes. The billing page settles it (§9).

## 9. The billing check (for Rishi)

The balance before step 11 was about $201.02 (step 10's $201.33 less its $0.31 follow-ups). If W&B bills only completed requests, the page should now show about **$195.1 left** ($5.89 here plus ~$0.06 of completed probes). If it bills failed tries in full, it could be as low as about **$193.0**. A reading between those also says how much a 5xx costs, which matters for production at 27% failed tries.

**Answer (Rishi, 2026-10-17): $193.81 left**, so $7.21 billed: W&B billed about $1.32 of the failed tries and probes, about 60% of the upper bound. The ledger now holds the billed figure.

## 10. What I built and fixed (all $0, tested)

| File | What |
|---|---|
| `rlm_rl/baselines.py`, `scripts/baselines.py` | The baseline runner train.py lacked: arms `rlm`, `direct`, `rlm_medium`, run together through one limiter. Resumable (finished episodes skip, infra episodes rerun). A hard spend ceiling over all arms and sessions, with `project` to price what's left before spending. `--dry-run` on the fake model server |
| `rlm_rl/client.py` | Root thinking for one arm (`root_thinking="medium"`; `xhigh` refused), checked from the rendered prompt. A plain `answer()` call for direct reading. Thinking tokens counted. Failed tries counted by HTTP status |
| `rlm_rl/env.py` | Each turn records its thinking tokens and text |
| `rlm_rl/baseline_dashboard.py`, `rlm_rl/baselines_page.html`, `scripts/dashboard.py baselines` | The step 11 page, in its own folder (`results/dashboard-step11/`), scanned by the same check before every publish |
| `rlm_rl/dashboard.py`, `scripts/dashboard.py check --deep --eval-set` | **A gap in the page scan, fixed:** it recognised edit tasks by the training family names only, so `edit_real_select` and `edit_real_test` tasks were never checked for their task ids or repositories. Production's page runs the selection set every 10 steps, so this mattered there too. The scan now covers them (16 task ids and 3 repositories here; the deep check 76 and 9) |
| `tests/test_baselines.py` (18 tests), `tests/fake_model.py` | Direct parsing and scoring, thinking requests and checks, the ceiling (including waiting instead of dropping), resume, infra retries, status counts. The fake server honours per-request thinking |

**A bug the live run found:** the spend gate refused episodes whose cautious default reservation didn't fit while others were in flight. At the start of the medium arm it dropped 34 of 64 that way. Now an episode waits for in-flight ones (whose measured costs replace the defaults) and is refused only if it can't fit with nothing in flight. A test covers it, and session 3 ran the 34.

**Tests:** every module the changes touch passes in WSL on the final code: `test_baselines`, `test_client`, `test_env`, `test_evals`, `test_contract`, `test_dashboard`, `test_monitor`, `test_training` and `test_edit_env`, 147 tests (1 skipped, the uncached tokenizer) in 20 minutes. The rest of the suite wasn't rerun.

## 11. Decisions for Rishi before production

*Answered by Rishi on 2026-10-17: A as recommended, B medium, C 30 turns. §13 has what was built and the rerun.*

**A. The tool-call failure** (§3)
- **Accept `<function=python>…</function>` as a code block and say so in the "no code block" error: recommended (mine).** It's a few lines in `env.extract_code` and one sentence in the error text, $0. Then rerun the RLM arm (~$2.40) for a clean step-0 number.
- **Fix the prompt instead (prompt v3):** tell the model that tools are Python functions called inside the block and never with tool-call tags. That changes the frozen prompt text, so the second seed's and this run's numbers stop being comparable.
- **Leave it:** GRPO probably trains it away, but its first gains would be format learning.

**B. Thinking for the production root (11b's result)**
- **`medium`: recommended by 11b's rule** (+17 points, ~1.25× a step, ~130 steps for $157). It needs a fake-model dry run that trains on thinking tokens first.
- **Off:** cheaper, ~165 steps. After fix A the gap may shrink, since part of medium's gain is avoiding the format failure. That would take one rerun of both arms (~$4.80) to know.

**C. Edit turns: 20 → 30?** The plan's rule says yes (56% unresolved at the cap). After fix A it would likely fall under 20%, so I'd decide after the rerun *(mine)*. At $0.035 an edit episode, 30 turns is affordable either way.

**Also before production** (no decision needed unless Rishi objects):
- Fix the `test_aware` false positive on scratch scripts that call `process.exit`.
- The base eval on the second seed's held-out tasks (~$2) is still open in the project chat.

## 12. How the base results compare with the second seed's 83.3%

They can't be compared directly. The second seed's 83.3% is on `heldout-seed1`: 100 episodes of the two ticket families at level 1, prompt v1, a 2,000-character output cap. Step 11's selection set holds only 8 ticket tasks across levels 1-5. On those the base model got `tickets_customer` 4 of 4 in every arm and `tickets_all` 1 of 4 (off) or 2 of 4 (medium). That is consistent with the second seed's reading that `tickets_customer` is easy and `tickets_all` is where the room is (step 10 §4). Whether the 13-point gap above the first seed's 70% is environment v2 still needs the ~$2 base eval on `heldout-seed1`, which waits on Rishi.

## 13. Rishi's decisions and the rerun in production's set-up (2026-10-17)

**Rishi's answers (02:07 UTC):**
- **A, the tool-call failure:** fix it as recommended. Built as **prompt v3**.
- **B, thinking:** production's root thinks at `medium`.
- **C, edit turns:** 30.
- **Server errors:** acceptable as long as retries recover and the run doesn't stop.
- **Billing:** $193.81 left, so step 11's first run cost **$7.21 on W&B**. The script measured $5.89, so W&B billed about $1.32 for failed tries and the probes. My $2.15 upper bound held. Production's own estimates will run about 20% under the bill while 500s stay this common.
- **Commit and push:** done (`b7d8814` on the PC's branch).
- **The ticket tasks:** replace both with held-out tasks related to the training data that can't leak. Built as **`selection-v3`**.

**What changed** ($0, tested):

| Change | Where | What it does |
|---|---|---|
| Prompt v3, now the default | `bridge.py`, `env.py` | Code in a `<function=python>` tool call runs like a ```python block. The "no code block" error adds "Tool-call tags don't run here: call llm, llm_batch, submit and the tools as Python functions inside the block." Edit episodes get 30 turns. v2's system prompt and first messages are unchanged. v1 and v2 keep their texts and hashes, so step 10 and step 11's first run stay reproducible |
| `selection-v3`, production's selection set | `evals.py`, `config.py` | `selection-v2`'s 56 other tasks, unchanged (same seeds and task ids), plus 8 held-out tasks from the trained read families at levels 3-5: aggregation L3, L4, L5; call graph L4; needle L5; repo localization L3, L4, L4. The read tasks now follow the mix exactly (20/8/8/12) |
| A sandbox bug, found by the rerun | `editbox.py` | An edit cell's wall-clock cap was a fixed 1,800 s, under the raised cell limits of slow tasks (JavaScript 3,570 s, Go up to 2,760 s). A cell that ran or hung past 30 minutes became an infra error instead of the model's own timeout. In production that would have been a dropped episode counted toward the 1% infra pause. The cap is now the cell limit plus 1,800 s |
| `prod_v3` arm, the page across runs | `baselines.py`, `baseline_dashboard.py` | The base model in production's set-up. The step 11 page shows both runs |

Tests: 198 across every module these touch, in WSL. Three that pinned v2's defaults (20 edit turns, `selection-v2`) were updated to v3's and pass, and new tests cover the tool-call parser, v3's limits and the wall cap.

**No leak in `selection-v3`.** Its new tasks come from the same generators and sources as training, on fresh 64-bit seeds kept in `private/`. So they're the same kind of task as training, but no context, question or answer repeats one a training step draws, which is how the rest of the selection set works. Repo localization reads only the three selection repositories, which training never sees. The final claim still rests on step 13's held-out test, which uses held-out sources.

**The rerun: the base model in production's set-up** (`prod_v3`: prompt v3, root thinking medium, 30 edit turns, on `selection-v3`, 64 episodes, run `step11-v3`):

| | Production set-up (v3) | Medium thinking (v2), §6 | Thinking off (v2) |
|---|---|---|---|
| All 64 | **51.6% ± 6.2** | 59.4% | 42.5% |
| Read (48) | 43.8% | 56.3% | 41.7% |
| Edit (16) | 75.0% | 68.8% | 43.8% (32 eps) |
| Paired, on the 56 tasks both sets share | 53.6% | 57.1% (−3.6 ± 6.2) | 39.3% (+14.3 ± 6.1) |
| The 8 new tasks | 3 of 8 | | |
| Turns with no runnable code | 72 of 675 (11%) | 143 of 644 (22%) | 340 of 997 (34%) |

- **The production set-up's step-0 number is 51.6% on `selection-v3`.** Production's own checkpoint-0 eval will measure it again.
- **The fix works mechanically, but it adds nothing on top of medium thinking** (−3.6 ± 6.2 on the shared tasks, noise). 67 tool-call turns ran their code, and turns without runnable code halved. Medium thinking had already avoided most of the failure on read tasks, where the arms differ most (45% vs 53% on the 40 shared read tasks, ± 7.5).
- **The 8 new tasks are hard (3 of 8),** so unlike `tickets_customer` they can move the checkpoint pick.
- **30 edit turns:** 3 of 16 edit episodes ended unresolved at the cap (19%, under the 20% line), against 25% at 20 turns. Resolved held at 75% (`edit_real` 6 of 10, level 1 3 of 5; `edit_synth` 6 of 6). Edit episodes cost more: `edit_real` $0.090 an episode, against $0.051 at 20 turns.
- **Cost:** read $0.035 and edit $0.067 an episode, so a production step is ~$1.52 (~$1.28 at level-1 costs). That buys **~100-120 steps for $157**, and a selection eval costs $2.73. Add ~20% if failed tries keep being billed at step 11's rate.
- **Infra:** 0 rate-limit errors, 25% HTTP 500 (274 of 1,108 tries), one 524, all recovered. One episode was lost to the sandbox bug above and rerun. The model had started a `node` script that never exited, and the bug turned that into an infra error at 30 minutes.
- **Time:** 64 episodes in 38 minutes, including the hung JavaScript episode (35 minutes before the cap fired) and its rerun. Most edit episodes ran 2-13 minutes.

**Spend:** step11-v3 measured $2.78, so step 11 counts **$9.99** against its $12 ceiling ($7.21 billed + $2.78 measured). If W&B bills this run's failed tries at the first run's rate, the billing page should read about $190.4.

**Left before production (step 12):**
1. **Train with root thinking at `medium`** (Rishi's decision B). Not built yet:
   - the training loop's client and pinned config need the setting;
   - the KL monitor re-scores whole conversations with thinking off, so it would skip every thinking turn as misaligned. It needs to re-score each turn with its thinking included, which costs about 5× today's ~$0.01 a step.

   ART 0.5.20 handles thinking turns itself, as far as I can tell from its code: a turn whose prompt drops the earlier thinking becomes its own training sequence, so the tokens trained are the tokens sampled. Then a fake-model dry run with thinking on.
2. **Optional:** 3.4% of medium-thinking turns hit the token cap, most of them read turns spent entirely on thinking (§6). A 4,096-token read turn would fix most of that for a little cost.
3. **The `test_aware` false positive** on scratch scripts that call `process.exit` (§4) is still open. It is a red-team detector, so I left it for Rishi.

## 14. Files

| Path | What |
|---|---|
| `results/step11-v1/transcripts/val/{rlm,direct,rlm_medium}/step0000.jsonl` | Every try, infra ones included (git-ignored) |
| `results/step11-v1/summary.json`, `spend.json`, `progress.json`, `run.log` | The summary, the spend ledger (set to W&B's billed $7.21), the last session's request counts, the console log |
| `results/dashboard-step11/page.html`, `results/dashboard-step10/page.html` | The two pages as published (git-ignored) |
| `results/dry/step11-dry1/` | The $0 dry run |
| `results/step11-v3/` | The rerun in production's set-up (`prod_v3` on `selection-v3`); its spend ledger carries the first run's billed $7.21 |
