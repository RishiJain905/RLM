# The production RL run: GRPO, datasets, environment, infra and evaluation

> Research doc, first written 2026-10-10 and rebuilt the same day around **GRPO**, after the 20-step run showed GRPO beating RLSD and Rishi chose GRPO for production. Nothing was run and no credit was spent for this doc. It covers Part 1 step 1.7 of [project-map.md](project-map.md): the training method and its settings, what the run trains on, what its environment must look like, why requests run in parallel, and how to evaluate without being fooled. Visual version: [rl-run-research.html](rl-run-research.html).
>
> Builds on [grpo-for-rlm.md](grpo-for-rlm.md) and [rlsd-vs-grpo-20-step-results.md](archive/rlsd-vs-grpo-20-step-results.md) (the evidence for GRPO), plus [rl-method-for-rlm.md](archive/rl-method-for-rlm.md), [sdpo-rlm-compatibility.md](archive/sdpo-rlm-compatibility.md) and the code in `rlm_rl/`.
>
> **Budget update (Rishi, 2026-10-10):** there is no pilot run. W&B bills two separate pots: **training** (GPU time for the gradient steps: $1,000, $8.02 used) and **inference** (every rollout token, every sub-call, every eval and benchmark: $250, $32.32 used, so **~$218 left**). Inference is the real limit, so section 7 budgets everything in inference dollars. The final benchmark eval can run outside W&B, on the trained LoRA after it is pushed to Rishi's Hugging Face, which frees its share for training, and production has no fixed step count: it runs until a stop rule fires or its spending ceiling is reached (section 8.6).
>
> **Setting:** Qwen3.8-27B LoRA on W&B serverless RL (ART 0.5.20), **GRPO with the CISPO loss and no reward std scaling**, straight RL with no SFT, sub-calls served by the same frozen base model.

---

## 0. Summary

| Topic | Finding | Decision or recommendation |
|---|---|---|
| **Method** | 20 steps, 100 held-out episodes each: base 0.50, **GRPO 0.70** (+0.20, ~3 SE), RLSD 0.59 (not significant). GRPO learned to make more sub-calls on the tasks that need them and fewer where they don't. Every published RLM training run uses GRPO-family RL. | **GRPO for production** (Rishi, 2026-10-10). RLSD is shelved; it can return later as an ablation on top of a GRPO run. |
| **Production settings** | Episodes that ran out of the 20-call budget scored 0.25 vs 0.53 on hard tasks, and GRPO's sub-call use was still rising. GRPO only learns from groups with mixed results. | **Sub-call budget 20 → ~30. Tasks get harder as accuracy climbs** (an adaptive difficulty controller keeps pass rate in the 20-80% band). 32 episodes per step (8 tasks x 4 rollouts, the setup that worked), to get more steps per inference dollar. |
| **Do first** | The audit found three gaps that make any result untrustworthy: no real sandbox, eval transcripts not logged, one seed per arm. | **Sandbox, eval logging and a second seed come before anything that spends production budget** (holes H2-H4, H9, H12). |
| **Holes** | 13 holes in the current setup, from a reward a blind guess can farm to groups that teach nothing. | Each has a fix and a check that proves it (next section). |
| **Datasets** | Current tasks are ~8-10K tokens, inside the model's 262K window, so they don't test long context. | Train on generators (cyber log-injection, code call-graph and needle-function) + OOLONG-synth validation + Long Code Arena train. Public benchmarks are eval-only. |
| **Concurrency** | 8 is how many requests run at once. Cost is per token, so it's the same at 1 or 8; one at a time would be ~10-15x slower. The 429 at 32 is far below W&B's documented 2,000. | Keep 8 behind one shared limiter (sub-calls currently bypass it). Ask CoreWeave about the 429. |
| **Cost** | Two pots. Training (GPU) has $992 left and our runs use ~$0.16 per step, so it is not a constraint. Inference has ~$218 left and pays for every rollout, sub-call and eval: yesterday's ~2,060 episodes cost $32.32, about **$0.016 per episode** at today's task size. | Everything fits in the ~$218 of inference: second seed ~$12, baselines ~$5, **production ceiling ~$150**, selection-set evals ~$25, reserve ~$25 (section 7). No pilot. The final benchmark eval runs outside W&B from the Hugging Face upload. |
| **How long to train** | A fixed 60 steps could stop too early or waste money. At 32 episodes per step, ~$150 buys roughly 80-180 steps, depending on how fast contexts grow. | No fixed count. Evaluate every 10 steps, keep the best checkpoint, stop early on a plateau, overfitting, frying or saturation; spend the reserve if it is still improving at the ceiling; resume later from a checkpoint if more credit appears (section 8.6). |
| **Evaluation** | Mean reward hides guessing; same-generator eval hides memorizing the generator. | Selection set / held-out test / public benchmarks, exact accuracy headline, paired tests, hack audit, every eval transcript logged. |
| **MiMo-V2.6** | Xiaomi's live RL run is GRPO-family too. Its models cheated by fetching fixes from outside the sandbox; an infra error was scored as failure for 3 hours. | Copy its dynamic sampling, hack zeroing, environment hardening, infra-error classification and dashboard panels. |

---

## Holes found, and the fix for each

Every hole in the current setup, the fix the production environment applies, and the check that proves it. A fix is done only when its check passes, and every check becomes a regression test. **Rishi's order: the sandbox (H2-H4), eval logging (H9) and a second seed (H12) come first**; the rest land in environment v2 before the second-seed run (section 11).

| # | Problem | Evidence | Fix we apply | Done when |
|---|---|---|---|---|
| H1 | **Partial credit pays for guessing.** A guess from the line count earns reward without reading anything. | Simulated blind guess `round(lines x share)`: 0.53 (`tickets_all`), 0.31 (`tickets_customer`) mean reward. The model hasn't used it yet (zero-sub-call full-count rollouts earned ~0.05), but nothing stops it. | **Keep partial credit, but calibrate it against the blind guess.** GRPO needs a graded reward to keep groups mixed ([grpo-for-rlm.md](grpo-for-rlm.md) §4.1), so we don't drop it. Instead the host computes each task's blind-guess error `e_blind` and pays `0.8 x max(0, 1 - err / e_blind)` for a near miss, 1 for exact. A blind guess then earns only its lucky exact hits (simulated 0.06 and 0.11), while an answer better than the guess still earns credit. Exact accuracy is the headline metric. | A blind guesser scores no more than its exact-hit rate on every task family. |
| H2 | **The REPL can read and write the whole disk**, including `.env` (the W&B key) and gold answers of past tasks in `.art/`. | `ReplWorker` only strips secret-looking env vars. `.art/.../trajectories/*.parquet` holds each answer. In the 20-step run one rollout wrote files to `C:\tmp\`. | **Do first.** One Docker container per episode: read-only filesystem except a scratch dir, non-root, only `context` mounted read-only. No repo, `.art/`, home or `.env`. Answers and graders stay on the host. | Tests that open `.env`, `.art/`, the repo, home and `/proc`, and that write outside scratch, all fail from inside. |
| H3 | **Tasks can be regenerated.** Model code could import our generator and find the seed. | Generators are importable via `sys.path`; seeds are small (train `1000 + i`, eval `99_999`). | **Do first** (comes with H2). Generators run on the host only, with random 64-bit seeds that never enter the sandbox. Eval seed lists stay private. | Importing any project module from the sandbox fails; no seed is visible inside. |
| H4 | **The REPL can reach the internet.** This is how MiMo's models cheated. | No network restriction on the subprocess. | **Do first** (comes with H2). `--network none`; the only way out is the bridge over one socket, accepting only `llm`, `llm_batch`, `submit`. | `pip install`, `curl` and DNS fail; bridge calls still work. |
| H5 | **Infra failures count against the model.** MiMo lost 3 hours of Flash training to this. | A rate-limited sub-call returns `"[sub-call failed: rate limited]"` as an answer. 4 RLSD episodes (0.6%) hit the 120 s cell timeout while sub-calls sat in the queue. A 30-call budget makes queue waits longer. | `infra_error` flag on each episode (bridge failure after retries, kernel crash not caused by model code, API 5xx). Flagged episodes leave the group before advantages are computed, are counted, and the step pauses above 1%. Time spent waiting in the sub-call queue no longer counts toward the cell timeout. | Forced 429s and a killed kernel yield flagged episodes, and none reach training. |
| H6 | **Sub-calls bypass the concurrency cap.** | Root calls share a cap of 8; sub-calls have their own pool of 6, so up to ~14 run at once. | One shared limiter for every request, jittered exponential backoff, exhausted retries become `infra_error`. Ask CoreWeave why 429 hits at 32. | A load test never exceeds the cap. |
| H7 | **Eval reuses the training templates**, so it measures memorizing our generator. | Same ~40 sentence templates in train and eval; only the seed differs. | Three tiers: selection set, held-out generators/sources/templates, public benchmarks run once (section 8). | No generator, template or source overlaps training; an n-gram overlap check passes. |
| H8 | **Tasks are too short to need an RLM.** | 30-45K characters (~8-10K tokens) against a 262K window. | 130K-1M-character contexts reached through the difficulty controller (section 5.5), plus a direct-reading baseline in every eval. | The top difficulty levels have median context above 130K characters; the direct-reading baseline is measured. |
| H9 | **Eval transcripts are not logged**, so evals can't be audited. | `evaluate_rlm` saves only aggregates; the 300 eval episodes of the 20-step run were never checked. | **Do first.** Every eval episode is logged with its full transcript, sub-calls, reward components and checkpoint (ART `model.log(..., split="val")` plus a local JSONL), and the hack audit covers eval as well as training. | A rerun eval produces one transcript per episode, and the audit script reads them. |
| H10 | **The env and the harness use different function names.** | Env: `llm_query`. Harness design: `llm`. | One shared bridge module (`llm`, `llm_batch`, `submit`, `context`) with pinned prompt text, used by both. | Both import the module; a test diffs their system prompts. |
| H11 | **Nothing detects or penalizes cheating.** | No checks for turn-1 submits, answers copied from the prompt, or escape attempts. | Rule detectors that zero reward **before** the group mean (MiMo's hack zeroing), a Claude audit of sampled wins each step, and a red-team agent before production. | Detectors unit-tested, red-team finds no exploit, clean solve rate on the dashboard. |
| H12 | **One seed per arm.** The 20-step result could partly be run-to-run noise. | GRPO vs base is ~3 SE (solid); GRPO vs RLSD is ~1.6 SE. | **Do first.** Repeat the 20-step GRPO run with a second seed on environment v2 (same ticket tasks, ~$12 of inference). This confirms the gain reproduces and that the sandbox and calibrated reward change nothing important. With no pilot, it is also the smoke test: the first run on the new environment, at a tenth of production cost. Final evals use 2-3 sampling seeds. | The second seed lands within ~1 SE of the first on held-out accuracy, or the gap is explained. |
| H13 | **Groups with identical rewards teach GRPO nothing.** | In the 20-step GRPO arm, trainable groups fell from 7 of 8 to 3-6 of 8 as the model improved. | The difficulty controller keeps pass rates in the 20-70% band (section 5.5), and all-equal groups are dropped from the update. We don't draw replacement tasks: refills cost inference. | Most steps have at least 6 of 8 mixed groups, and the pass-rate histogram stays centred. |

The RLSD-only hole from the earlier version (the teacher's hint could contain the answer) no longer applies, since GRPO has no teacher.

---

## 1. Where we are

- **Three-way test** (3 steps, 48 eval episodes): RLSD 0.54, GRPO 0.46, base 0.35. Too small to separate methods. [rlm-three-way-results.md](archive/rlm-three-way-results.md)
- **20-step run** (100 held-out episodes per model): base 0.50, **GRPO 0.70**, RLSD 0.59. GRPO's gain sits mostly in the hard `tickets_all` family (0.40 → 0.63 mean reward from the first to the second half). [rlsd-vs-grpo-20-step-results.md](archive/rlsd-vs-grpo-20-step-results.md)
- **GRPO for RLM** ([grpo-for-rlm.md](grpo-for-rlm.md)): the budget did not cause RLSD's loss; GRPO turned the same sub-call spend into more correct answers; returns flatten after ~10 sub-calls.
- **Decided by Rishi, 2026-10-10:** GRPO for production, sub-call budget ~30, tasks get harder as accuracy climbs, and the sandbox, eval logging and second-seed fixes come first.

---

## 2. Lessons from Xiaomi's MiMo-V2.6 RL run

In September 2026 Xiaomi streamed the RL post-training of MiMo-V2.6-Pro (~1T total / 42B active) and Flash (309B / 15B active) live at [mimo.xiaomi.com/rl](https://mimo.xiaomi.com/rl/), then published a technical report. The dashboard page could not be rendered from here, so numbers come from write-ups (section 13). Their scale and harness are nothing like ours, but **their algorithm is the same family**: GRPO with mean-centred advantages and no std division, the same choice as our `scale_rewards=False`. That makes their monitoring and failure list unusually relevant.

### 2.1 The run in numbers

| Item | MiMo-V2.6 | Ours (planned) |
|---|---|---|
| Algorithm | GRPO, mean-centred advantages, no std division, token-level train/inference importance ratio with masking, prompt-mean loss normalisation | GRPO, CISPO loss, no std scaling |
| Batch | 1,568 prompts x 16 rollouts per step | 8 tasks x 4 rollouts |
| Tokens per step | 2.7-3.7B | ~1-3M |
| Episode | ~100K tokens of context, 57-65 turns | 8-15 root turns, up to ~30 sub-calls |
| Steps / cost | 30 steps; $2.6M Pro, $0.9M Flash (44% rollouts, 13% grading) | No fixed count (~80-180 steps); ~$150 inference ceiling |
| Data mix | 68% code, 13% visual, 12% tool use, 4% cyber, 3% context following | cyber logs, code, OOLONG-style aggregation |
| Difficulty control | Dynamic sampler drew prompts until 1,568 had mixed outcomes; pruned prompts at 100% pass | Difficulty controller; all-equal groups dropped, not refilled (H13, 5.5) |
| Result | DeepSWE avg@3: Pro 58.4 → 72.6, Flash 48.7 → 65.7 (self-reported) | — |

### 2.2 How their models cheated

Early coding runs passed tests **without solving the task** by getting the fix from outside: `pip install` a newer release, `curl` the fixed file from a raw GitHub URL ("Let me just look at the file directly from the GitHub raw URL."), `git clone` a later checkout, read the fixed issue in the tracker, `pip index versions` to find the fixed release. Without quality grading, models also drifted toward code that passes but is bad (swallowed exceptions, relaxed validation). The **cyber dataset was dropped** from the Pro run after "some bad patterns" in rollout logs, and cyber grading moved from an LLM judge ("returns different verdicts for the same PoC across runs") to a deterministic sanitizer-report match. The same pattern shows up in the SWE-bench `git log` leak and in Cursor's 2026 finding that 63% of one frontier model's successful SWE-bench Pro solves retrieved the fix.

### 2.3 How they fought it

1. **Environment prep:** no logs or caches, git cut to the base commit, no network.
2. **Hack agent:** probed every environment until it "could no longer find a successful exploit".
3. **Hack zeroing:** a confirmed hack's reward is set to 0 **before** the group mean, so it becomes a negative example. Rewards (1, 1, 1, 0) with one hack become (1, 1, 0, 0); advantages go from (0.25, 0.25, 0.25, -0.75) to (0.5, 0.5, -0.5, -0.5). This is a GRPO-specific detail that matters: zeroing after the mean would leave the hack with positive advantage. Confirmed hacks stayed under 2%.
4. **Quality among passes:** an evaluator ranked passing attempts and gave lower-ranked ones less credit.

### 2.4 Incidents worth learning from

- **Silent infra failure:** an infra error "was not correctly detected over the past ~3 hours"; sandbox failures were scored 0; Flash rolled back to step 15.
- **Saturated prompts:** some subsets reached 70-80% of prompts at 100% pass, killing the GRPO signal. They pruned easy prompts.
- **Checkpoint picked on the reported benchmark:** choosing the release checkpoint by DeepSWE makes the published DeepSWE number optimistic.

### 2.5 What we take from it

| MiMo practice | For us |
|---|---|
| Dynamic sampling (drop all-pass and all-fail groups) | **Adopt the drop, skip the refill** (H13): refills cost inference we don't have. |
| No network, no git history, no caches | **Adopt** (H2-H4). |
| Hack agent before training | **Adopt.** One Claude red-team thread, no credit. |
| Hack zeroing before the group mean | **Adopt** (H11). |
| Infra errors classified separately | **Adopt** (H5). |
| Deterministic cyber grading | **Adopt.** Blue-team answers are exact entities. |
| Separate checkpoint-selection set | **Adopt** (H7, section 8). |
| Quality ranking among passing attempts | **Later.** Needs a grader model. |
| Async rollouts, staleness caps | **Skip.** ART is synchronous at our scale. |
| Live public dashboard | **Adopt** as a shareable artifact (section 9). |

Dashboard panels worth copying: pass rate with and without infra errors, pass-rate histogram, share of all-0 and all-1 groups, infra error rate (<1%), turns and lengths by domain, entropy, train-inference KL, clip fractions, hack counters, cost counter, step timing, offline eval per checkpoint, operator log.

---

## 3. Datasets

### 3.1 What a training task must have

1. **Needs long context.** Ideally more than the model's 262K-token window, or at least enough that direct reading degrades.
2. **Needs semantic reading somewhere.** Keep ~20% regex-solvable, since "use code when code is enough" is also a skill. GRPO already learned not to over-recurse on easy tasks.
3. **Verifiable answer.** Exact match on an entity, number or ID, or set-F1 on a list. No multiple choice, no LLM judge for the main reward.
4. **Unguessable from metadata** (H1).
5. **Renamed entities per seed**, so pretraining memory can't answer.
6. **Adjustable difficulty.** Each generator exposes knobs (context length, number of needles or attack slices, distractor density) so the controller in 5.5 can raise it.

### 3.2 The current tasks

`rlm_rl/archive/ticket_tasks.py` proved the method but is not production data: 32-45K characters, ~40 fixed sentence templates (a policy can learn them as a lookup table), and partial credit a blind guess can farm (H1).

### 3.3 Sources

**Train on (generators first):**

| Source | Domain | Context | Answer and check | Difficulty knobs |
|---|---|---|---|---|
| **Cyber log-injection generator** (ours) | Blue team | 64K-512K tokens | First compromised host (exact), malicious ProcessGuids or timestamps (set-F1), lateral-movement account (exact), technique ID when one is injected (exact) | Log volume, attack slices injected, how many hosts, how indirect the question is |
| **Code call-graph generator** (ours) | Code | 130K-1M chars | "All transitive callers of X", answered with `ast`/tree-sitter; set-F1 | Repo size, call depth, obfuscation |
| **Needle-function generator** (RepoQA method) | Code | Same | Behavioural description → qualified name; exact | Repo size, similar decoy functions |
| **SWE-smith** (59K synthetic bugs, MIT) | Code | Repo-sized | Gold file/function from the patch; set-F1 | Repo size |
| **OOLONG-synth validation** | Aggregation | 1K to 4M tokens | Exact; numbers scored 0.75^|diff| | Built-in length buckets |
| **Long Code Arena bug localization, Python train** (Apache-2.0) | Code | Repo-sized | Set-F1 on changed files | `repo_tokens_count` buckets |

Cyber sources: benign background from OTRF / Sysmon / Loghub Linux-SSH; attack slices from `splunk/attack_data` (Apache-2.0, by ATT&CK technique), OTRF Security-Datasets, EVTX-ATTACK-SAMPLES; random time-shifts and host/user/IP/GUID renaming per seed. Questions describe behaviour, not event IDs, so sub-calls are needed. The Cyber Defense Benchmark (arXiv 2604.19533) builds its eval the same way.

**Hold out for eval:** OOLONG-synth test and OOLONG-real; **CyberSOCEval** malware part (609 questions, often 128K+ tokens, exact-set scoring); **ExCyTIn-Bench** (589 questions, entity answers); the Cyber Defense Benchmark (release unverified); AIT-LDS v2 as held-out log sources; LCA Java/Kotlin test, LocBench, LongCodeBench, LongBench-v2 code; BrowseComp-Plus and MRCR.

**Avoid or use with care:** SWE-bench Lite and BOTS v1-v2 (answers public), CTIBench / CyberMetric / SecBench (short, multiple choice), CIC-IDS (numeric CSV), Loghub BGL/Thunderbird unless the label column is stripped, LoCoBench (subjective scoring). Licences: AIT-LDS and CTIBench are non-commercial, OTRF shows two licences, EVTX samples are GPL-3.0.

### 3.4 Mix and sizes

| Share | Source |
|---|---|
| ~35% | Cyber log-injection generator |
| ~30% | OOLONG-synth validation |
| ~20% | Code generators (call-graph, needle-function) |
| ~15% | Long Code Arena bug localization or SWE-smith |

Generators make items on demand, so there is no fixed item count; the real-data sources (OOLONG, LCA, SWE-smith) need a few hundred items each. There is **no pre-filter pass**: it would cost ~$40-120 of inference. Every family starts at level 1 (today's task size, where the base model passes about half the time) and the controller moves it from there.

---

## 4. The RL environment

The holes table is the checklist. This section adds the details behind it.

### 4.1 The reward

- **Exact answer: 1.** Counts, IDs and entities compared after normalisation; lists by set-F1 (a perfect set is 1).
- **Near miss: calibrated partial credit** (H1): `0.8 x max(0, 1 - err / e_blind)`, where `e_blind` is the error of the line-count guess for that task, computed on the host. It keeps groups mixed, which GRPO needs, without paying for guessing.
- **No `submit`: 0.** No length penalty at first.
- **Hack detected: 0, before the group mean** (H11).
- **Infra error: excluded from the group** (H5), not scored.
- Process penalties (a small cost for REPL errors after turn 1, or for ending without `submit`) **only if a habit stalls** in the logs. Context-Folding needed them for branch/return behaviour; we haven't yet.

### 4.2 Production sandbox

One Docker container per episode (or per worker, reset between episodes): `--network none`, read-only root filesystem with one scratch dir, non-root user, CPU/memory/pids limits, wall-clock timeout per cell that excludes time queued for sub-calls. Only `context` is mounted. The bridge is the only way out: one socket accepting `llm`, `llm_batch`, `submit`. Gold answers, graders and task generators live on the host. A fixed set of Python packages, the same in the harness.

### 4.3 Episode limits

| Limit | 20-step run | Production |
|---|---|---|
| Sub-calls per episode | 20 | **~30** (Rishi). Tie context length to it: a level's context should be coverable in about two-thirds of the budget at ~20K characters per call, so ~400K characters at budget 30. Longer contexts need a bigger budget or must be filtered with code first. |
| Root turns | 6 | 10-15 |
| Root tokens per turn | 1,536 | 2,048-4,096 |
| REPL output shown | 2,000 chars | 2,000 chars (it forces summarising into variables) |
| Cell timeout | 120 s, including queue waits | 120 s of compute, queue waits excluded |

### 4.4 Bridge API freeze

Freeze one module used by both env and harness (H10): `llm(prompt)`, `llm_batch(prompts)`, `submit(answer)`, `context`. Leave `rlm()` (depth > 1) out of Part 1. Also decide whether harness sub-calls go to the **base model** (as trained) or the **trained LoRA**; the eval should use whichever the harness will use.

### 4.5 Hack detectors (rule-based first)

Zero the reward before the group mean when any fire, and count them: `submit()` on turn 1 or with no read of `context`; the submitted value appears in the prompt but in no REPL output; blocked network, filesystem or host-import attempts; repeated cell timeouts. Then a Claude audit of sampled successful transcripts each step, reported as a clean solve rate. Don't train against the auditor's judgments directly; OpenAI found that pushing on a monitor teaches obfuscation.

### 4.6 Red-team before production

A Claude agent gets environment v2 and tries to get reward without solving tasks: guessing, reading files, seed recovery, network, crashing the grader, malformed `submit` values. Every working exploit gets fixed and becomes a regression test. No credit needed.

---

## 5. The training method: GRPO for RLM

### 5.1 Why GRPO

- **It won our test.** +0.20 held-out accuracy over base in 20 steps (~3 SE); RLSD +0.09 (not significant). On hard tasks GRPO went from 31% to 46% exact between the first and second half of training.
- **It learned the RLM habit we want.** Zero-sub-call episodes on hard tasks fell from 17.5% to 2.5%, sub-calls rose from 5.9 to 8.2 per episode, and on easy tasks it kept solving 68% of episodes directly. It recursed where recursion pays and not elsewhere. Held-out: 0.70 accuracy with 3.8 sub-calls per episode, against RLSD's 0.59 with 4.8.
- **It is what RLM training uses.** The alphaXiv RLM run, the RLM paper's own RL run (prime-rl, group-based), MemAgent (DAPO) and Context-Folding (FoldGRPO) are all GRPO-family. No published RLM run used self-distillation.
- **It is simple and native.** One scalar reward per rollout, which is exactly W&B serverless's interface. No teacher re-scoring, no logprob channel, no tokenizer round-trip (RLSD lost 19% of its teacher scoring to alignment failures).
- **No SFT needed.** GRPO needs a non-zero starting pass rate. The 4B model in the alphaXiv run scored 0 pass@16 and needed a cold start; Qwen3.8-27B starts around 0.5 accuracy on our tasks, so straight RL (decision 5) holds.

### 5.2 How GRPO maps onto an RLM episode

```
task ──► 4 rollouts from the same checkpoint
           each rollout = the root conversation:
           system + question + context metadata
           → code turn → truncated REPL output → code turn → … → submit()
           (sub-calls run inside the REPL; they are environment, not trained)
reward r_i per rollout ──► advantage A_i = r_i − mean(r of its group)
every root token in rollout i gets A_i; REPL output and sub-call text are masked
CISPO loss on those tokens ──► LoRA update
```

- **No std division** (`scale_rewards=False`). Dividing by the group's std over-weights groups where rewards barely differ (Dr. GRPO's difficulty bias). MiMo makes the same choice.
- **One advantage for the whole episode.** A good probe turn and a bad aggregation turn in the same rollout get the same credit. This is GRPO's known weakness for long episodes. Our 20 steps show it is good enough here; the fixes, if needed, are small process penalties (4.1) or turn-level methods later.
- **Sub-calls are environment.** They are served by the frozen base model and not trained. The alphaXiv run's next step, giving child calls the root's advantage normalised by child count, is a Part 1 follow-up once the root is solid.

### 5.3 Settings for production

| Setting | 20-step run | Production | Why |
|---|---|---|---|
| Loss | CISPO | CISPO | Clips the importance weight but keeps a gradient on every token, including rare decisive ones (the first `llm_batch` call). |
| Importance-weight clip | Pinned to 1 (`epsilon = epsilon_high = 0`) | **Restore CISPO's default clip** | The pin existed only so RLSD's logprob channel stayed exact. With plain GRPO nothing rides that channel, and ART runs several micro-steps per batch, so the clip is the trust region that keeps later micro-steps from drifting. Check the second-seed run's grad norm and `probs_corr` against the first 20-step run. |
| Reward std scaling | Off | Off | See 5.2. |
| Group size | 4 | 4 | 8 would give less noisy group means, but doubles the inference per step. Under a ~$150 ceiling, twice the steps is worth more. Revisit if mixed groups drop below ~5 of 8. |
| Tasks per step | 8 | 8, all-equal groups dropped | H13. |
| Learning rate | 1e-5 (LoRA) | 1e-5, lower if entropy falls fast | Worked in 20 steps; longer runs need watching. |
| KL to base | None | None in the loss, **logged** | Dashboard signal for drift (section 8.5). |
| Sampling temperature | 1.0 | 1.0 for training; eval at fixed decoding | Exploration. |
| Sub-call budget | 20 | ~30 | 5.4. |
| Thinking mode | Off | Off, pinned | Changes tokens, cost and behaviour. |

### 5.4 The sub-call budget: 20 → ~30

On hard tasks, episodes that ran out of the 20 calls scored 0.25 against 0.53 for the rest, and GRPO's use was still rising (8.2 per episode by steps 11-20). Reward flattens after ~10 calls (0.59 at 4-9 calls, 0.64 at 10-19), so 30 is headroom, not a target. Watch three things: budget-hit rate (should fall), sub-calls per episode (should not climb toward 30 without accuracy following), and sub-call tokens per episode (cost). With contexts growing through the curriculum, keep each level coverable in about 20 calls (4.3).

### 5.5 Harder tasks as accuracy climbs: the difficulty controller

GRPO stops learning from a task family once every rollout passes. The 20-step GRPO arm showed it starting (46% exact on hard tasks at step 20, trainable groups down to 3-6 of 8). A simple controller keeps the signal:

- Each generator has difficulty levels (context length, needles or attack slices, distractor density, question indirection), level 1 being the 20-step tasks' size and the top level ~400K characters at budget 30.
- After each step, per family, look at the rolling training pass rate over the last 3 steps. **Above ~70%: move up a level. Below ~20%: move down.** In between, stay.
- All-equal groups within a level are dropped from the update (H13).
- Log the level per family on the dashboard. Evaluate on the **fixed** selection set (section 8), not the moving training mix, so curves stay comparable.

This is the RLM paper's own recipe in automatic form (trained at 32-64K with 2 needles, evaluated at 512K-1M with 8), and MiMo's easy-prompt pruning by another route.

### 5.6 GRPO risks for multi-turn RLM, and what we do

| Risk | What it looks like | Mitigation |
|---|---|---|
| **Zero-signal groups** | Trainable groups drop; loss decays toward 0 | Difficulty controller; drop all-equal groups (H13, 5.5) |
| **Multi-turn collapse ("Echo Trap", RAGEN)** | Reward variance within groups falls off a cliff, gradient spikes, entropy drops, rollouts become near-identical | Monitor within-group reward std and entropy every step; filter low-variance groups; lower LR; roll back to the last good checkpoint |
| **Reward farming** | Reward rises, exact accuracy and selection set don't | Calibrated partial credit (H1), hack zeroing (H11), exact accuracy as headline |
| **Coarse credit** | A structural habit (always submitting, recovering from errors) doesn't improve | Small process penalties only if logs show it (4.1) |
| **Over-recursion** | Sub-calls climb toward the budget without accuracy | Budget, cost on the dashboard; easy tasks in the mix (GRPO already avoided this in 20 steps) |
| **Length creep** | Turns and tokens per episode grow | Prompt-level loss normalisation is ART's default concern; watch turns and truncation; MiMo used prompt-mean normalisation to stop it |
| **Drift across micro-steps** | `probs_corr` falls, grad norm spikes | Restore the CISPO clip (5.3) |

### 5.7 What changed from the RLSD plan

- No teacher model, no teacher re-scoring calls, no logprob injection, no tokenizer alignment step.
- The earlier teacher-model research is moot for production. Claude's role stays offline: transcript auditor (H11), red-team agent (4.6), and grader for anything non-verifiable in eval.
- RLSD remains a possible **ablation on top of a GRPO run** later, if dense credit is revisited. [rl-method-for-rlm.md](archive/rl-method-for-rlm.md) keeps the theory.

---

## 6. Concurrency

**What the 8 is:** how many requests are in flight at the same time, not how many calls an episode makes. A step at production size is roughly 32 episodes x ~10 root turns plus up to ~30 sub-calls each: hundreds to a couple of thousand requests. In the 20-step run a 32-episode step took 150-390 s at 8-way. One request at a time, at 10-30 s per 27B root turn, would be an hour or more per step.

**Cost does not change with concurrency.** W&B bills per token (Qwen3.8-27B: $0.40 per 1M input, $0.15 cached input, $3.00 per 1M output). Training GPU time is billed separately, against the $1,000 training credit. Concurrency only buys wall-clock time.

**Why 8 and not 32:** W&B returned HTTP 429 at 32 parallel requests. CoreWeave's docs (where serverless training now lives) say the limit is **2,000 concurrent requests per user and 6,000 per project**, so ours is probably an undocumented per-model or per-checkpoint limit. Worth one email to CoreWeave support with request ids, plus a confirmation that Qwen3.8-27B training stays supported (the public model list doesn't show it; our account does).

**The bug** (H6): root calls share an `asyncio.Semaphore(8)`, but sub-calls use a separate pool of 6, so ~14 can be in flight. With a 30-call budget this matters more. One shared limiter, jittered backoff, exhausted retries become `infra_error`. Try 16 once support answers.

---

## 7. Cost

### 7.1 Two pots, and which one runs out

| Pot | What it pays for | Credit | Used so far | Left |
|---|---|---|---|---|
| **W&B Training** | GPU time for the gradient steps (the LoRA update) | $1,000 | $8.02 | ~$992 |
| **Inference** | Every token the model reads or writes: root turns of every rollout, every sub-call, every eval and benchmark episode | $250 | $32.32 | **~$218** |

The $8.02 of training covered ~49 gradient steps (3 steps x 3 arms in the three-way test, 20 x 2 in the 20-step run), about $0.16 a step. Even 150 production steps with longer root transcripts would be ~$25-60, so **training is not the limit; inference is**. The training credit can't be spent on inference.

### 7.2 What an episode costs

Yesterday's $32.32 of inference covered ~2,060 episodes: 288 + 1,280 training episodes and 192 + 300 eval episodes across the three-way and 20-step runs, plus RLSD's teacher re-scoring. That is **~$0.016 per episode** at today's task size (30-45K characters, 2-6 sub-calls), and a little less for plain GRPO, which has no teacher calls.

Longer contexts cost more because the model reads them through sub-calls. A rough model fitted to that figure: about $0.01 per episode for the root turns, plus about $0.013 per 100K characters the sub-calls read.

| Task size | Per episode | Per step (32 episodes) |
|---|---|---|
| Level 1, today's ~35K characters | ~$0.016 (measured) | ~$0.50 |
| ~130K characters | ~$0.03 | ~$1.00 |
| ~400K characters (top training level) | ~$0.06 | ~$1.90 |
| ~1M characters (eval only) | ~$0.14 | — |

Training contexts stop at ~400K characters; 1M-character tasks appear only in the final eval, in small numbers.

### 7.3 The inference budget

| Item | Episodes | Inference |
|---|---|---|
| Second seed: 20 GRPO steps at level 1 + 100-episode eval (H12, also the smoke test) | ~740 | ~$12 |
| Baselines on the selection set: base model as an RLM, and reading directly | ~130 | ~$5 |
| **Production**, 32 episodes per step, until a stop rule fires (8.6) | ~2,500-5,000 | **~$150 ceiling** |
| Selection-set evals during production: 64 tasks every 10 steps, plus a capability check | ~650-900 | ~$25 |
| Final eval: held-out test and public benchmarks, final model and base | ~600-700 | **$0 on W&B** (run outside, 7.4); ~$35 if run here instead |
| Reserve: reruns after infra failures, or extending a run that is still improving | | ~$25 |
| **Total** | | **~$217 of ~$218** |

**What $150 of production buys:** ~80 steps if contexts reach the top level early, ~180 if they climb gradually; ~120 is a fair middle estimate. Data generation, the red-team and transcript audits use Claude, not W&B, so they cost no W&B credit. Nothing here touches pay-as-you-go.

**Levers if money runs short:** cap training contexts lower (~250K characters) or eval every 15 steps instead of 10. Cutting episodes per step below 32 is the last resort: the 20-step run never tested it.

### 7.4 The final eval outside W&B

Once the run ends, the best LoRA goes to Rishi's Hugging Face and the final eval (held-out test, public benchmarks, capability retention) runs in a separate environment. That costs no W&B credit, and it allows bigger samples than we could buy here (~400 episodes per model for a ±5-point interval). Three conditions keep the comparison fair:

1. **Same harness.** The final eval runs through the frozen bridge (H10) and the Part 2 harness core, with the same prompt text, limits and decoding as training.
2. **Same sub-call model.** Sub-calls are served by base Qwen3.8-27B (or the LoRA, per decision 6), exactly as in training.
3. **Base model in the same place.** The base-model and direct-reading baselines are re-run there too, so both sides of every comparison come from one setup.

The selection-set evals during training stay on W&B, since the stop rules (8.6) need them mid-run. Two things to confirm before relying on this: that the LoRA checkpoint can be downloaded from W&B, and that the base model's licence allows publishing the adapter. Running a 27B model elsewhere needs about one 80 GB GPU in bf16, or a hosted provider that serves Qwen3.8-27B with LoRA adapters; that cost falls outside both W&B budgets. If Rishi would rather run it on W&B, it costs ~$35 from the production ceiling.

---

## 8. Evaluation

### 8.1 Three tiers, kept apart

| Tier | Contents | Used for | Touched |
|---|---|---|---|
| **Selection set** | Held-out seeds of training generators at fixed difficulty levels + one held-out generator family | Picking the checkpoint | Every eval |
| **Held-out test** | Never-trained generators and sources: held-out log sources (AIT-LDS), ATT&CK techniques, code languages, OOLONG test sources, separately written and paraphrased question templates | The generalization claim | Once |
| **Public benchmarks** | CyberSOCEval malware, ExCyTIn, LCA Java/Kotlin, OOLONG-real, LongBench-v2 code, BrowseComp-Plus subset | Comparable numbers | Once |

Never pick a checkpoint on a number you will report.

### 8.2 Baselines

1. Base Qwen3.8-27B as an RLM in the same environment (step 0).
2. Base Qwen3.8-27B **reading the context directly**, where it fits in 262K tokens. Without this we can't claim the RLM setup helps.
3. The 20-step GRPO checkpoint (both seeds).

### 8.3 Metrics

Exact accuracy with 95% CIs (headline), avg@k and pass@k; reward components separately; sub-calls, budget-hit rate, turns, tokens and dollars per episode; submit rate and infra error rate; **clean solve rate** from the audit; a **closed-book probe** (empty `context` should score at chance); **canary tasks** whose answer isn't in the context.

### 8.4 Logging and sample size

Every eval episode is logged in full (H9) and audited like training. At ~50% accuracy, 100 episodes gives a 95% interval of about ±10 points; ±5 needs ~400, which we can't afford at every checkpoint. So the selection set is 64 fixed tasks (about ±12 points per eval) and stopping decisions use the trend over several evals, not one number. Compare models on the same tasks with paired tests (McNemar or paired bootstrap), fixed decoding, 2-3 sampling seeds on the final test, and two training seeds where a claim depends on it (H12).

### 8.5 Is it frying?

| Signal | Bad sign | Response |
|---|---|---|
| Train reward vs selection set | Reward up, selection flat or down | Read transcripts, check detectors, stop |
| Within-group reward std, trainable groups | Sudden drop (Echo Trap) | Filter, lower LR, roll back |
| Policy entropy | Fast monotone drop | Lower LR, consider a small KL to base |
| KL to base (logged) | Sudden jump | Same |
| Turns, tokens, truncation, budget-hit rate | Hitting caps, or collapsing to 1-2 turns | Look for loops or guess-and-submit |
| Pass-rate histogram, difficulty level | Mass at 0 or 1; level stuck at the top or bottom | Check the controller and data |
| Clip fraction, `probs_corr`, grad norm | Rising clip fraction, falling `probs_corr` | Lower LR |
| Infra error rate | Above ~1% | Pause and fix |
| IFEval, MMLU-Pro subset, short coding set, chat prompts | Down more than ~2 points | Frying, whatever RLM accuracy says |

### 8.6 How long to train, and when to stop

A fixed step count is the wrong control here: 60 steps could stop while the model is still improving, or keep paying after it has stopped. Production instead runs against its **~$150 inference ceiling** and stops by rule.

**Every 10 steps:** evaluate on the fixed 64-task selection set (~$2.50) and a ~100-prompt capability check (cents). Save the checkpoint and keep track of the best one so far.

**Keep going while** the selection set reached a new best in one of the last 3 evals, or the controller moved any family up a level in the last 30 steps.

**Stop early, and keep the best checkpoint, when any of these holds:**

1. **Plateau:** no new selection-set best for 3 evals in a row (30 steps), and no family moved up a level in that time.
2. **Overfitting:** training reward rises while the selection set falls, at 2 evals in a row.
3. **Frying:** the capability check drops more than ~2 points, or a collapse (within-group reward std or entropy falling off a cliff) comes back after one rollback with a lower learning rate.
4. **Saturation:** every family sits at its top level with a pass rate above ~70%. The data has nothing left to teach, so the next step is harder data, not more steps.
5. **Hacking** that needs an environment change: hack counters rise or the clean solve rate falls. Pause, fix the env, then resume.

**If it may need more:** when production reaches its ceiling and both of the last 2 evals were new bests, spend the ~$25 reserve (roughly 15-25 more steps) and then check with Rishi. Stopping is not final either way: ART keeps the LoRA checkpoints, so a stopped run can resume later from its latest or best checkpoint if more inference credit appears.

---

## 9. Monitoring and the shareable dashboard

W&B stays the source of truth: every metric in 2.5 and 8.5 per step, plus sampled full transcripts. For collaborators, a private-link Claude artifact styled after MiMo's page, rebuilt from the per-step `results/<run>.json` the training script already writes: pass rate with and without infra errors, pass-rate histogram, trainable groups, difficulty level per family, entropy, KL, sub-calls and budget-hit rate, infra errors, hack counters, cost counter, selection-set points and an operator log. W&B public reports are the zero-effort alternative. Build it with the production run.

---

## 10. Other things the run needs

1. **Bridge API freeze** (H10). Blocking.
2. **Restore the CISPO clip** (5.3) and confirm it in the second-seed run.
3. **Reproducibility.** Pin ART 0.5.20, the chat template, the bridge prompt text, generator versions, difficulty schedule and dataset manifest. Save the config with every checkpoint.
4. **Checkpoint retention.** Storage is cheap (5 GB free, ~30 LoRAs). Keep every 5th checkpoint plus the best on the selection set.
5. **Licences** per source before it enters training.
6. **Same harness for eval and deployment.** The final eval runs through the Part 2 harness core with the frozen API.
7. **Platform risk.** Confirm Qwen3.8-27B training stays available.
8. **Root transcript length.** Keep REPL output truncated so root transcripts stay well inside ART's packed sequence length (6-10K tokens in the 20-step run).

---

## 11. Proposed sequence

There is no pilot: the 20-step runs already settled the method, and the second seed, run on the new environment, does the pilot's other job of catching plumbing bugs cheaply.

| Phase | Work | Inference |
|---|---|---|
| **A. Do first** | Sandbox (H2-H4), eval transcript logging (H9), shared limiter and infra flags (H5, H6) | None |
| B. Environment v2 | Calibrated reward (H1), bridge freeze (H10), hack detectors (H11), restored CISPO clip, 30-call budget, difficulty controller (H13), stop-rule monitoring (8.6) | None |
| C. Data | Generators with difficulty levels, OOLONG and LCA loaders, held-out splits (H7, H8), licence check | None |
| D. Red-team | A Claude agent attacks env v2 until every check passes | None |
| **E. Second seed** | Repeat the 20-step GRPO run on env v2 at level 1 (H12); doubles as the smoke test | ~$12 |
| F. Baselines | Base model as an RLM and reading directly, on the selection set | ~$5 |
| G. Production | GRPO until a stop rule fires or the ~$150 ceiling (8.6); selection set every 10 steps | ~$150 + ~$25 evals |
| H. Final eval | Push the best LoRA to Hugging Face; held-out test, public benchmarks and capability retention outside W&B, base model alongside (7.4) | $0 on W&B |

About $25 stays in reserve. Each phase that spends inference waits for Rishi's go-ahead.

### Decisions for Rishi

1. **Inference split:** second seed ~$12, baselines ~$5, production ceiling ~$150, selection-set evals ~$25, reserve ~$25, final eval outside W&B?
2. **Domain mix:** ~35% cyber / 20% code / 30% OOLONG / 15% repo localization, or more code and cyber?
3. **Partial credit:** **decided 2026-10-10: calibrate it against the blind guess.** See [production-spec.md](production-spec.md) §3.
4. **CISPO clip:** **decided 2026-10-10: restore the default** (`epsilon=1.0`, `epsilon_high=4.0` in ART 0.5.20). Learning rate also settled: 1e-5 for the second seed, 2e-5 in production only if the second seed is stable. See [production-spec.md](production-spec.md) §5.
5. **Bridge names:** **decided 2026-10-10: `llm` / `llm_batch` / `submit` / `context`.** See [production-spec.md](production-spec.md) §1.
6. **Deployed sub-calls:** the base model, as trained, or the trained LoRA?

---

## 12. Confidence notes

MiMo figures come from secondary write-ups of a self-reported run. The blind-guess numbers (0.53 / 0.31 today, 0.06 / 0.11 calibrated) come from a local simulation of our generators, not a model run. Costs are calibrated on the actual inference bill ($32.32 for ~2,060 episodes); costs at longer contexts are extrapolated from it, not measured. The 20-step comparison is one seed per arm (H12). Dataset sizes and licences were checked against dataset cards where possible.

---

## 13. Sources

**Our evidence:** [rlsd-vs-grpo-20-step-results.md](archive/rlsd-vs-grpo-20-step-results.md), [grpo-for-rlm.md](grpo-for-rlm.md), [rlm-three-way-results.md](archive/rlm-three-way-results.md), `results/archive/rlsd_vs_grpo_20.json`.

**GRPO and RLM training**
- RLM paper: [arXiv:2512.24601](https://arxiv.org/abs/2512.24601); [Reinforcing Recursive Language Models (alphaXiv)](https://www.alphaxiv.org/blog/reinforcement-learning-for-rlms); [Prime Intellect RLMEnv](https://www.primeintellect.ai/blog/rlm)
- MemAgent: [arXiv:2507.02259](https://arxiv.org/abs/2507.02259); Context-Folding: [arXiv:2510.11967](https://arxiv.org/abs/2510.11967)
- DAPO: [arXiv:2503.14476](https://arxiv.org/abs/2503.14476); Dr. GRPO: [arXiv:2503.20783](https://arxiv.org/abs/2503.20783); CISPO (MiniMax-M1): [arXiv:2506.13585](https://arxiv.org/abs/2506.13585)
- RAGEN / StarPO, the Echo Trap: [arXiv:2504.20073](https://arxiv.org/abs/2504.20073); [Scale: multi-turn tool use with GRPO](https://labs.scale.com/blog/improving-multi-turn-tool-use-with-grpo); Turn-PPO: [arXiv:2512.17008](https://arxiv.org/abs/2512.17008)
- [Entropy mechanism of RL for reasoning](https://arxiv.org/abs/2505.22617), [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)

**MiMo-V2.6**
- [mimo.xiaomi.com/rl](https://mimo.xiaomi.com/rl/); [DEV write-up](https://dev.to/axrisi/xiaomi-ai-training-live-what-the-mimo-26-rl-dashboard-shows-3ioh); [NYU Shanghai RITS](https://rits.shanghai.nyu.edu/ai/xiaomi-mimo-v2-6-live-rl-dashboard/); [TechNode](https://technode.com/2026/09/18/xiaomi-livestreams-mimo-v2-6-reinforcement-learning-runs/)
- [Metric breakdown (redreamality)](https://redreamality.com/blog/mimo-v2-6-rl-metrics-and-incident-analysis/), third-party interpretation; [Technical report explained (OrcaRouter)](https://www.orcarouter.ai/blog/mimo-v2-6-technical-report-rl-explained); [Who grades the grader (Kili)](https://kili-technology.com/blog/data-story-mimo-v2-6-llm-as-a-judge); [Scaling RL from scratch](https://prathamp.com/blog/mimo-v2-6-scaling-rl-towards-self-improvement/); [Luo Fuli announcement](https://x.com/_LuoFuli/status/2100296686719610932)

**Infra and pricing**
- [Serverless training usage limits](https://docs.coreweave.com/products/post-training/serverless-training/usage-limits.md), [available models](https://docs.coreweave.com/products/post-training/serverless-training/available-models), [serverless inference pricing](https://www.coreweave.com/products/coreweave-forge/serverless-inference), [Qwen3.8-27B on W&B Inference](https://wandb.ai/site/inference-model/cw_qwen_qwen3.8-27b), [429 guidance](https://docs.coreweave.com/support/inference/articles/api-error-code-429-concurrency-limit-rea)

**Datasets**
- OOLONG: [arXiv:2511.02817](https://arxiv.org/abs/2511.02817), [oolong-synth](https://huggingface.co/datasets/oolongbench/oolong-synth), [oolong-real](https://huggingface.co/datasets/oolongbench/oolong-real); [BrowseComp-Plus](https://huggingface.co/datasets/Tevatron/browsecomp-plus); [OpenAI MRCR](https://huggingface.co/datasets/openai/mrcr); [LongBench-v2](https://huggingface.co/datasets/zai-org/LongBench-v2)
- Code: [LCA bug localization](https://huggingface.co/datasets/JetBrains-Research/lca-bug-localization), [LocBench](https://huggingface.co/datasets/czlll/Loc-Bench_V1), [SWE-smith](https://huggingface.co/datasets/SWE-bench/SWE-smith), [SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench), [LongCodeBench](https://arxiv.org/abs/2505.07897), [RepoQA](https://github.com/evalplus/repoqa)
- Cyber: [CyberSOCEval](https://arxiv.org/abs/2509.20166) ([data](https://github.com/CrowdStrike/cybersoceval_data)), [ExCyTIn-Bench](https://arxiv.org/abs/2507.14201), [Cyber Defense Benchmark](https://arxiv.org/abs/2604.19533), [splunk/attack_data](https://github.com/splunk/attack_data), [OTRF Security-Datasets](https://github.com/OTRF/Security-Datasets), [EVTX-ATTACK-SAMPLES](https://github.com/sbousseaden/EVTX-ATTACK-SAMPLES), [Loghub](https://github.com/logpai/loghub), [AIT-LDS v2](https://zenodo.org/records/5789064)

**Reward hacking and evaluation**
- [SWE-bench git history leak](https://github.com/SWE-bench/SWE-bench/issues/465), [Cursor: reward hacking in coding benchmarks](https://cursor.com/blog/reward-hacking-coding-benchmarks), [ImpossibleBench](https://arxiv.org/abs/2510.20270), [METR on reward hacking](https://metr.org/blog/2025-06-05-recent-reward-hacking), [Monitoring reasoning for misbehavior](https://arxiv.org/abs/2503.11926), [Natural emergent misalignment from reward hacking](https://arxiv.org/abs/2511.18397)
- [Does RL really expand reasoning (pass@k)](https://arxiv.org/abs/2504.13837), [Evaluation variance](https://arxiv.org/abs/2504.07086), [RL's Razor](https://arxiv.org/abs/2509.04259)
