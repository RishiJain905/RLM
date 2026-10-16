# Step 10 results: the second seed

> **Step 10 of [production-build-plan.md](production-build-plan.md)** (hole H12, research §7.3, §11). Run on 2026-10-16 on Rishi's PC: training in WSL, the dashboard loop and this write-up from a Claude thread on Windows.
>
> **Decisions applied:**
> - **10a-10c** (taken 2026-10-12, decisions E-d and E-g): the inference split, run the second seed, run it now.
> - **10d** (Rishi, 2026-10-16): test resuming from an earlier checkpoint once the seed finishes, about $0.50.
> - The run used `config.second_seed`: learning rate 1e-5, fixed; prompt v1; the controller frozen at level 1; thinking off.
> - During the run (Rishi, 2026-10-16): if a ceiling stops it, resume with a higher one and let it finish. After it: **production starts at 2e-5.** Spread out sub-call requests in production, even if that makes it slower.
>
> **Dashboard:** [claude.ai/artifact/Jrm5PZd6VQNJT7cpkM9ou3](https://claude.ai/artifact/Jrm5PZd6VQNJT7cpkM9ou3), now shared with anyone who has the link. It shows the finished seed (versions 22-67 of the page cover this step).
>
> **W&B inference spent:** $16.22 on W&B's billing page ($201.33 left of $250). The run estimated $16.40: $15.74 for the seed (training $13.77, eval $1.97) and $0.66 for the 10d step (§8). The plan budgeted ~$12 plus ~$0.50.
>
> **Status:** complete. Rishi's follow-ups (resuming from a saved LoRA, sub-call spacing, the REPL output cap) are being built before step 10 is closed.
>
> Anything marked *(mine)* is my choice, not the plan's. Anything marked *(inferred)* hasn't been checked.

## Summary

**The second seed reached 83.3% on the held-out set (80 of 96, ±3.8), against the first seed's 70%.** Four of the 100 eval episodes were lost to W&B rate limiting; counting them as wrong gives 80%. The run trained cleanly: no collapse signal, `probs_corr` at the first run's 0.997, and every stability check passed, so **production starts at 2e-5** (Rishi confirmed).

**The replication check is not a clean pass.** The plan asked for the held-out score to land within about 1 SE of 0.70, or for the gap to be explained. It landed 13 points above, and the explanation is inferred: environment v2 makes the untrained model stronger (§4). One $2 eval of the base model on these held-out tasks would settle it, and I recommend adding it to step 11's baselines.

**10d found a real problem: resuming from an earlier checkpoint doesn't work on W&B serverless** (§8). The forked branch trained from a fresh LoRA instead of checkpoint 15, so production's rollback and its 2e-5 → 1e-5 fallback would silently restart from the base model. The plan's KL check couldn't have caught it; comparing the LoRA weights did.

**Three more things need fixing before production** (§10): W&B rate limiting (429s on 19-47% of requests from step 2 on, and 1.5% of episodes lost overall), the 2,000-character REPL output cap (it led to the one suspicious audited win), and the escape rule firing on harmless `/tmp` scratch writes.

| Done-when check (plan step 10) | Result | Evidence |
|---|---|---|
| Held-out accuracy within ~1 SE of the first seed's 0.70, or the gap explained | **Above, explanation inferred** | 0.833 ± 0.038 (n 96). +13 points is 3.5 of this run's SE, 2.2 SE of the difference. Base-model rollouts at step 1 passed 62.5% vs the first seed's 53% (§4) |
| Grad norm and `probs_corr` look like the first run's | **Pass** | Grad norm median 0.52, max 0.82 (first seed 0.48 / 1.14). `probs_corr` 0.9964-0.9975 (first seed ~0.997) |
| Stability curves decide production's LR: 2e-5 if they pass spec §5.2 | **Pass: 2e-5** | Entropy steps 16-20 = 86% of steps 1-5 (80% needed); grad norm max 1.6× its median (under 3×); `probs_corr` min 0.9964 (above 0.98) |
| Infra errors stay under 1% | **Fail: 1.5%** | 7 of 640 training episodes and 4 of 100 eval episodes, all W&B 429s after 6 tries; 7 of the 11 in step 20 (§6) |
| No detector fires unexpectedly | **Two false alarms** | Both `escape_attempt` hits were `/tmp` scratch writes, judged clean by the audit (§5) |
| The dashboard's cost panel matches W&B's billing page | **Pass** | W&B: $16.22 spent ($201.33 left). The run's estimate: $16.40, 1.1% high (§9) |

---

## 1. What ran

| Time (ET) | What happened |
|---|---|
| 11:20-12:05 | Pre-flight checks (§2), dashboard wiped to a "waiting" page |
| 12:19 | Rishi started `train.py start --kind second_seed --run second-seed-v1 --live` in WSL |
| 12:28 | Step 1 landed; the dashboard republished after every step and notice from then on |
| 14:37 | Step 20 done; the run stopped at checkpoint 20 because the eval's cost estimate ($3.24) was over its $3 ceiling. I resumed it with `--ceiling-eval 4`, as Rishi had said to |
| 14:45 | Held-out eval done (83.3% over 96); the run finished. Audit samples were written at checkpoints 10 and 20 |
| 14:54 | After saving a copy of the finished run, the 10d fork test started (§8) |

The 20 steps took 136 minutes (median 6.2 a step), and 2 h 26 m with the eval and the resume. The estimate given before the start was 2.5 hours, up to 3.

## 2. Pre-flight checks (all $0)

| Check | Result |
|---|---|
| The plan thread's four additions (10d, baseline script, step 11 thresholds, edit wins in the audit) | Brought in from the cloud branch; Rishi committed them as `cae28c7` |
| Docker and WSL up, no leftover `rlm-` containers, sandbox folder owned by Rishi | Pass. Docker Desktop was off and I started it; 0 sandbox- or edit-labelled containers; `/tmp/rlm-sandbox-1000` owned by `ytrjx` |
| Full test suite in WSL (`scripts/sandbox_ctl.py check`) | **Pass:** 412 tests in 31 min, 1 skipped (the Qwen tokenizer isn't in the local Hugging Face cache) |
| One fake-model dry run of the second seed (`--dry-run`) | **Pass:** `seed-preflight-1016`, 20 steps and the held-out eval, no infra errors or detector hits. The dashboard's deep check of a page built from it was clean (376 seeds, 279 eval tasks) |
| W&B sign-in, and the run name unused | Pass (read-only API calls): signed in; the `rlm-production` project didn't exist yet |
| Inference balance from W&B's billing page | Rishi: $32.45 of $250 used, so $217.55 left; GPU $8.02 of $1,000 |

**Container start-up is slower than in step 3.** The suite's scale test started 32 sandboxes at once in 83 s (median start 60 s); a rerun gave 78 s (56 s). Step 3 measured 3-7 s. It stays far under the 300 s start timeout, so it cost time, not episodes. Docker's disk is on the USB data drive *(inferred cause)*.

## 3. Training

| Step | Pass | Trainable groups | Entropy | Reward std | KL | Grad norm | probs_corr | Sub-calls | Turns | Time (min) | Cost | 429s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.62 | 7/8 | 0.259 | 0.359 | 0.0016 | 0.45 | 0.9970 | 3.9 | 7.6 | 8.7 | $0.67 | 0 |
| 2 | 0.59 | 6/8 | 0.307 | 0.268 | 0.0026 | 0.50 | 0.9974 | 2.4 | 8.2 | 7.6 | $0.68 | 175 |
| 3 | 0.66 | 6/8 | 0.295 | 0.281 | 0.0014 | 0.67 | 0.9974 | 3.5 | 7.7 | 11.0 | $0.65 | 176 |
| 4 | 0.56 | 5/8 | 0.292 | 0.275 | 0.0027 | 0.82 | 0.9975 | 3.0 | 8.1 | 9.9 | $0.71 | 177 |
| 5 | 0.66 | 6/8 | 0.277 | 0.297 | 0.0020 | 0.68 | 0.9973 | 2.6 | 8.2 | 7.0 | $0.75 | 122 |
| 6 | 0.75 | 6/8 | 0.241 | 0.304 | 0.0029 | 0.72 | 0.9971 | 3.4 | 8.5 | 7.3 | $0.78 | 186 |
| 7 | 0.56 | 7/8 | 0.253 | 0.269 | 0.0041 | 0.37 | 0.9964 | 2.9 | 7.4 | 7.5 | $0.63 | 127 |
| 8 | 0.75 | 8/8 | 0.259 | 0.303 | 0.0025 | 0.38 | 0.9969 | 2.9 | 7.8 | 4.3 | $0.69 | 141 |
| 9 | 0.72 | 4/8 | 0.243 | 0.174 | 0.0036 | 0.71 | 0.9965 | 4.4 | 6.8 | 6.4 | $0.66 | 165 |
| 10 | 0.66 | 5/8 | 0.253 | 0.242 | 0.0025 | 0.55 | 0.9970 | 3.5 | 7.5 | 6.0 | $0.68 | 120 |
| 11 | 0.69 | 4/8 | 0.248 | 0.193 | 0.0033 | 0.65 | 0.9973 | 4.0 | 7.8 | 6.1 | $0.68 | 188 |
| 12 | 0.53 | 6/8 | 0.253 | 0.304 | 0.0037 | 0.49 | 0.9969 | 5.0 | 7.5 | 8.6 | $0.76 | 280 |
| 13 | 0.81 | 4/8 | 0.265 | 0.190 | 0.0018 | 0.68 | 0.9973 | 4.3 | 8.2 | 6.1 | $0.78 | 191 |
| 14 | 0.78 | 5/8 | 0.261 | 0.237 | 0.0061 | 0.49 | 0.9969 | 2.3 | 7.3 | 5.8 | $0.64 | 88 |
| 15 | 0.75 | 4/8 | 0.241 | 0.193 | 0.0039 | 0.41 | 0.9973 | 2.9 | 7.5 | 5.4 | $0.70 | 78 |
| 16 | 0.66 | 6/8 | 0.249 | 0.286 | 0.0031 | 0.55 | 0.9969 | 2.5 | 7.5 | 5.3 | $0.60 | 98 |
| 17 | 0.56 | 6/8 | 0.265 | 0.303 | 0.0023 | 0.43 | 0.9970 | 3.3 | 7.8 | 5.6 | $0.70 | 108 |
| 18 | 0.62 | 6/8 | 0.261 | 0.310 | 0.0080 | 0.39 | 0.9973 | 3.2 | 7.6 | 6.4 | $0.69 | 108 |
| 19 | 0.62 | 4/8 | 0.216 | 0.181 | 0.0056 | 0.71 | 0.9969 | 5.2 | 8.0 | 5.6 | $0.67 | 271 |
| 20 | 0.88 | 2/8 | 0.233 | 0.083 | 0.0032 | 0.44 | 0.9971 | 4.1 | 8.0 | 5.2 | $0.67 | 304 |

Pass is over episodes that ran properly (infra episodes left out); step 20's 0.88 is over its 25 good episodes. Clip fraction was 0 or ~2e-5 on every step, the new normal with CISPO's clip restored.

**What the curves say:**
- **Training pass rate moved little:** 0.62 (steps 1-5) to 0.66 (steps 16-20), within noise (z ≈ 0.8). By family, `tickets_all` went 0.43 → 0.51, and `tickets_customer` stayed at 0.81-0.82. The first seed went 0.53 → 0.71 on the same kind of tasks.
- **`tickets_customer` is saturated at level 1.** Groups where all 4 rollouts are right rose from 1.8 to 3.0 a step, and they give GRPO nothing. Trainable groups fell from 6.0 to 4.8 a step; step 20 trained on 2. The controller wanted to move it to level 2 from step 3 on, which production allows.
- **Entropy was flat** after a bump at steps 2-5 (base 0.259, steps 16-20 average 0.245).
- **KL to base rose slowly**, from 0.002 to 0.0045 on average (max 0.008 at step 18). It's small, but worth watching at 2e-5.
- **Behaviour barely moved:** 3.1 → 3.7 sub-calls an episode and about 7.8 turns throughout, against the first seed's ~5.2. Sub-calls were already common at step 1: 81% of episodes used one, against 38% in the first seed.

## 4. The held-out eval

| Family | Correct | Accuracy | Turns | Sub-calls | Failures |
|---|---|---|---|---|---|
| `tickets_all` | 33 of 46 scored | 0.72 ± 0.07 | 10.1 | 5.1 | 11 wrong, 2 no submit (at the 12-turn cap), 4 infra |
| `tickets_customer` | 47 of 50 | 0.94 ± 0.03 | 5.3 | 2.2 | 3 wrong |
| **All** | **80 of 96** | **0.833 ± 0.038** | 7.6 | 3.6 | |

Every wrong answer on `tickets_customer` was off by one.

**Why the gap to 0.70 (inferred).** The first seed went 0.50 → 0.70 on the old harness. This seed's training gain was small (+4 to +8 points over 20 steps), so most of the 13-point gap looks like environment v2 itself, not the training:
- Step 1's rollouts come from the untrained model. They passed 62.5% (20 of 32) against the first seed's 53% (17 of 32) at its step 0, about +9 points.
- Compared with the first seed's held-out run, the model submits more often (0.98 vs 0.88), worth about +8 points, and is right more often when it does (0.85 vs 0.80), about +5.
- Environment v2 changed the reward, the clip, the sandbox and the pinned thinking-off setting at once, so this can't say which change did it.

**To settle it** *(mine)*: run the base model as an RLM on `heldout-seed1` in environment v2, prompt v1, as one more arm of step 11's baselines. That's 100 episodes, about $2. If base lands near 0.60-0.65, the seed's gain matches the first seed's +20 and the replication holds.

## 5. Audits (decision 5a)

| Checkpoint | Split | Audited | Clean | Suspicious | Hack | Solve rate | Clean solve rate |
|---|---|---|---|---|---|---|---|
| 10 | train | 16 | 16 | 0 | 0 | 0.65 | 0.65 |
| 20 | train | 16 | 15 | 1 | 0 | 0.69 | 0.64 |
| 20 | val (held-out) | 8 | 8 | 0 | 0 | 0.83 | 0.83 |

A subagent read every sampled episode and proposed verdicts; I checked the non-clean one against the raw transcript and recorded them (`results/second-seed-v1/audit/step00{10,20}/verdicts.json`, report in `audit/report.json`).

**How wins are earned.** For counts across all tickets, the model classifies chunks with `llm_batch`, parses the labels, prints the count and submits it. For one customer's tickets, it filters by name and classifies them, more often by reading them itself by checkpoint 20 (5 of 13). One episode used no sub-calls at all, filtering on phrases it had seen. That's legitimate, but new.

**The suspicious win** (checkpoint 20, a step 18 `tickets_customer` episode). The REPL cuts printed output at 2,000 characters. The model saw 16 of the customer's 17 tickets, one of them cut off mid-line. In turn 4 it wrote out two tickets that appear in no output it was shown, labelled both as problems, and its submitted count depends on one of them. It matched the gold by luck. That's no trick on the environment, but the answer doesn't follow from the work. Two other sampled wins showed the same failure and recovered: one caught its invented tickets with its own regex check, and one dropped the cut-off ticket.

**Detector precision: 0 of 2.** Both flagged episodes (steps 2 and 9) were `escape_attempt` on a blocked `open('/tmp/…', 'w')`. The model was saving sub-call results to a scratch file, didn't retry, and solved the task normally. One answer was correct and lost its reward. Prompt v1 never says where the model may write.

No probing code appeared in any sample: no os, subprocess, socket, environment or file access besides those two writes.

## 6. Infra: W&B rate limiting

W&B returned 429 "too many requests" on 19-47% of request attempts from step 2 on (3,103 in all), even though our client kept to the 30-in-flight total and 20-per-model caps measured on 2026-10-10. Step 1 had none, when every request went to one model. From step 2, root calls go to the newest checkpoint and sub-calls to the base model, and the 429 count tracks sub-call volume (correlation 0.74), not root calls (0.0).

The client retries 6 times starting at 2 s, which covers about a minute. That absorbed everything until step 20, where 7 requests failed six times in a row. 7 of step 20's 32 episodes, and 4 of the eval's 100, became infra errors. Infra episodes never reach training, so step 20 trained on its 25 good ones. The step's 22% infra rate would have paused the run, but it was the last step. Over the run, that's 11 of 740 episodes (1.5%), over the plan's 1%.

**Rishi's decision (2026-10-16):** spread out sub-call requests in production, even if it's slower. §10 has the concrete change.

## 7. Cost and time

- **$0.67 a step** on average ($0.60-0.78), so the 20 steps took $13.77 of the $14 training ceiling, and the seed $15.74 against the plan's ~$12. Step 20 only just fit, and 10d needed the ceiling raised to $15.
- **What drives the cost** *(inferred from token counts)*: root prompts 49%, root output 30%, sub-calls 21%. A root prompt carries the whole transcript, 6k tokens at turn 4 and 54k at turn 12, and no prompt tokens were cached. A `tickets_all` episode costs 3.7× a `tickets_customer` one.
- **The eval cost $1.97**, against the loop's estimate of $3.24. The estimate is 1.5× the recent per-episode training cost, which is why the $3 ceiling stopped the run before the eval.
- **Time:** a step is rollouts (median ~180 s, falling from 317 s to 176 s over the run) plus training on W&B (~200 s, of which ~75 s is GPU time). Rollouts wait for the slowest episode; past the 75th-percentile episode, that wait is ~29% of rollout time.

## 8. 10d: resuming from an earlier checkpoint

**Result: the fork does not work on W&B serverless.** The forked branch trained from a fresh LoRA, not from checkpoint 15. So production's rollback and its learning-rate fallback, which both use this fork, would silently restart from the base model.

**What ran.** I saved a copy of the finished run first (`results/second-seed-v1-final/`), because a resume from an earlier checkpoint moves the later steps into the run's "abandoned" log. Then:
```bash
.venv-linux/bin/python scripts/train.py resume second-seed-v1 --from 15 --stop-after 16 --ceiling-train 15 --live
```
The loop forked checkpoint 15 into a new model, `second-seed-v1-b1`, trained one step on it and stopped. It cost $0.66, and the training ceiling had to rise from $14 to $15 for it. Inference spent on the run is now $16.40.

**The plan's check passed, but it can't see the problem.** The plan compared step 16's KL to base: 0.0043 on the fork against 0.0031 originally, which looks fine. But step 16's rollouts are sampled from checkpoint 15, which is the original branch's model either way. Only the training touches the fork, so its KL says nothing about the fork's weights.

**What the weights show** (`results/preflight/fork-check/compare_weights.py`: downloads the LoRAs from W&B and compares them, $0):

| Checkpoint on W&B | Norm of the LoRA B matrices | Distance from the original checkpoint 15 |
|---|---|---|
| Original checkpoint 5 | 0.547 | 0.609 |
| Original checkpoint 15 | 0.838 | 0 |
| Original checkpoint 20 | 0.947 | 0.397 |
| Fork's copy of checkpoint 15 (`b1:step15`) | 0.838 | **0.000**: an exact copy |
| Fork after its one training step (`b1:step1`) | **0.231** | **0.864**: about the size of checkpoint 15 itself |

LoRA's B matrices start at zero and grow as training goes (0.55 after 5 steps, 0.84 after 15). One step from checkpoint 15 should land close to 0.84, a small distance from it. Instead the fork's trained weights are small and as far from checkpoint 15 as checkpoint 15 is from zero: **one step from a fresh LoRA.** W&B also named the result `step1`, not `step16`.

**Why** *(inferred from ART 0.5.20's code)*: `_experimental_fork_checkpoint` downloads the source checkpoint and uploads it as an artifact for the new model, and its own docstring warns that the serverless backend's checkpoint tracking "may not immediately reflect the forked checkpoint". Our loop registers the new model on the backend first (which created its `step0`) and forks second. The training service then continued from its own step 0.

**What it means for production:**
- At 2e-5, the learning-rate fallback (spec §5.2) and stop-rule rollbacks would start over from the base model while the logs say "continuing from checkpoint N".
- **Recommended** *(mine)*: before step 12, fix the order (fork before the new model is registered, or whatever ART's serverless needs), add a check that compares the new branch's first checkpoint against its source the way the table above does, and re-run this test for one more step (~$0.70).
- Until then, any rollback has to be treated as a restart from scratch.

The live record `results/second-seed-v1.json` now ends on branch 1 at checkpoint 16, with the original steps 16-20 under `abandoned/b0-after15/`. The seed's results come from the saved copy.

## 9. The billing check

**Pass.** W&B's billing page showed **$201.33 left** after the 10d step (Rishi, 2026-10-16), so step 10 cost **$16.22** on W&B ($217.55 − $201.33). The run's own estimate was $16.40: $0.18, or 1.1%, high. The dashboard's cost panel can be trusted for production's budget, slightly on the safe side.

## 10. What changes for production

| # | Change | Why | Status |
|---|---|---|---|
| 0 | **Fix the fork** (fork before registering the new model, or what ART's serverless needs), check the new branch's first checkpoint against its source, and re-test one step (~$0.70) | §8: rollbacks and the LR fallback restart from scratch today | **Blocks step 12's rollback and fallback** |
| 1 | **Spread out sub-calls:** a lower in-flight cap for the sub-call model (e.g. 12 per model, 24 total) and longer retries (8 tries, backoff from 4 s, about 8 minutes) *(mine, numbers to tune)* | §6. Average demand is ~10-20 in flight, so expect at most 10-20% longer rollouts *(inferred)*, partly offset by fewer backoffs | **Decided** (Rishi); being built |
| 2 | Raise the REPL's 2,000-character output cap for read episodes | The one suspicious win (§5); `tickets_all` averages 4.5 cut-off outputs an episode | **Decided** (Rishi); being built |
| 3 | Don't zero a reward for a single blocked `/tmp` write with no retry, or tell the model where scratch is (prompt v2) | 0 of 2 detector hits were real (§5) | Proposed |
| 4 | Compute the eval-affordability estimate from eval episodes, or drop the 1.5× margin to ~1.1 | It stopped the run before its eval (§7) | Proposed |
| 5 | Add the base model on `heldout-seed1` (env v2, prompt v1) to step 11 | Explains the replication gap (§4), ~$2 | Proposed |
| 6 | Watch KL at 2e-5 | It rose slowly at 1e-5 (§3) | Note |
| 7 | 35% of `tickets_all` training episodes hit the 12-turn cap (32% of those still correct) | All 35 no-submits were at the cap. Production's tasks differ, so watch the budget-hit and turns panels first | Note |

## 11. Dashboard

- **Wiped before the run** with a new `scripts/dashboard.py waiting <run>` command: it builds a page with the next run's name and config and no data, and it never touches the run's state or `publisher.json`. Commit `80bd8bf`, with a test (`WaitingTest`).
- **`--pot-left` is now dated** with the day it's given, instead of the 2026-10-11 default.
- **One version per step and per notice**, published within seconds of each landing. No comments were left on the page.
- **The final page is built from a saved copy of the finished run** (`results/second-seed-v1-final/`), with both audits recorded. 10d's fork step then rewrote the live run's record (§8).
- Two small things the review found: the hacking rule reads "no scored episodes yet" after a resume, and level and detector notices are logged under the step before. Neither is wrong data.

## 12. Files

| Path | What |
|---|---|
| `results/second-seed-v1.json`, `results/second-seed-v1/` | The live run, now including the 10d branch; the original steps 16-20 are under `abandoned/b0-after15/` |
| `results/second-seed-v1-final/` | A copy of the finished seed, taken before 10d, with both audits recorded |
| `results/preflight/` | Pre-flight logs (suite, dry run), the resume and fork-test logs, the step-check helper and the dashboard-review scripts |
| `rlm_rl/dashboard_page.html`, `scripts/dashboard.py`, `tests/test_dashboard.py` | The "waiting" page (committed in `80bd8bf`) |
