# Production RL build plan

> **Status:** draft for discussion, 2026-10-10. Step 9, the live dashboard, was added at your request. **Updated 2026-10-11:** steps 1-7 are built, and decision 7c dropped cyber from the run (step 7). **Updated 2026-10-12:** code editing joins the run: build steps E1-E6 from [code-editing-plan.md](code-editing-plan.md) §13, spec v2 in [production-spec.md](production-spec.md) §12, and the split below (no E7 probe, reserve $10). No W&B credit has been spent.
>
> **Updated 2026-10-16:** steps 1-9 and the code-editing build (E1-E6, "step 9.5") are complete, and the three findings of the edit red team (E6) were fixed on 2026-10-15. Steps 10-12 are brought up to date below: the decisions already taken, the edit data as actually built, what each step takes on the PC, and what is still open (10d, 11a, 11b, 12b; 13a and 13b later). W&B inference left: ~$217.55 of $250. The $32.45 spent is the earlier experiments plus $0.13 of step 2's checks, and nothing since. Every paid step still waits for your go-ahead.
>
> **Source of truth:** [rl-run-research.md](rl-run-research.md). Its html version says the same thing. Section numbers (§) below point into it. Anything marked *(mine)* is my recommendation, not the research's. Anything marked *(inferred)* hasn't been checked yet.
>
> **How to answer:** each step ends with its own open decisions, numbered by step (1a, 1b, …). Reply with the number and your pick, for example "1a yes, 1b calibrate, 3b WSL2". Where a step lists defaults, I'll use them unless you say otherwise.

## The goal

The goal is to train a Qwen3.8-27B LoRA as an RLM root with GRPO on W&B serverless RL, using long-context aggregation and code tasks. The environment is sandboxed so it can't be gamed. Everything has to fit in the ~$218 of W&B inference credit left. The trained model then has to be shown to beat the base model on held-out tasks.

## Where we start

- **Method:** GRPO, decided by you. In the 20-step run, held-out accuracy was 0.50 for the base model, 0.70 for GRPO and 0.59 for RLSD, over 100 episodes each.
- **Budget:** two separate W&B pots.
  - Inference has ~$218 of $250 left. It pays for every rollout, sub-call and eval, and it is the limit.
  - Training (GPU) has ~$992 of $1,000 left. At ~$0.16 a step, it is not a constraint.
  - There is no pilot run.
- **Code:** everything from testing sits in `archive/` folders and won't run in place, because its imports expect `rlm_rl.<module>`. Production code goes at the top level of `rlm_rl/`, `scripts/` and `notebooks/`. Pieces worth porting:
  - `rlm_rl/archive/rlm_env.py`: the episode loop, `SubcallBridge`, `ReplWorker`, scoring and eval.
  - `rlm_rl/archive/repl_worker.py`: the REPL subprocess. It already talks to the host over an HTTP bridge and strips API keys from its environment.
  - `rlm_rl/archive/ticket_tasks.py`: the ticket tasks the second seed reuses.
  - `rlm_rl/archive/rollout.py`: request retries, plus rebuilding token ids from prompt logprobs (W&B's gateway strips the ids ART's trainer needs).
  - `scripts/archive/run_rlsd_vs_grpo.py`: a resumable runner that writes results JSON after every step.
  - Not ported: `sdpo.py` and `shaping.py` (RLSD/SDPO only) and `tasks.py` (toy smoke test).

## At a glance

**Types:**
- *Design*: doc work, no code.
- *PC*: code that runs on your PC through a Remote Control thread.
- *Spend*: uses W&B inference credit, and waits for your go-ahead.

| # | Step | Type | Covers | W&B inference | Decisions | Status (2026-10-16) |
|---|---|---|---|---|---|---|
| 1 | Freeze the design spec | Design | H1, H10 (spec) | $0 | 1a, 1b, 1c | Done |
| 2 | Prerequisite checks | You + PC | §6, §7.4, §10.7 | $0 ($0.13 of checks) | none (one action for you) | Done |
| 3 | Sandbox and bridge (do first) | PC | H2, H3, H4, H10 | $0 | 3a, 3b, 3c | Done |
| 4 | Limiter, infra flags, eval logging (do first) | PC | H5, H6, H9 | $0 | none | Done |
| 5 | Environment v2 | PC | H1, H11, H13 | $0 | 5a | Done |
| 6 | Training loop and monitoring | PC | §5.3, §8.5, §8.6 | $0 | none | Done |
| 7 | Data | PC + Design | H7, H8 | $0 | 7a, 7b, 7c | Done |
| 8 | Red-team | Claude agent | H11, re-checks H1-H4 | $0 | none | Done |
| 9 | Live dashboard | PC + Claude | §2.5, §8.5, §9 | $0 | 9a, 9b, 9c | Done; the billing check waits for step 10 |
| E1-E6 | Code editing build, "step 9.5" ([code-editing-plan.md](code-editing-plan.md) §13) | Design + PC | spec §12 | $0 | taken (E-a to E-g) | Done 2026-10-13; red-team fixes 2026-10-15 |
| 10 | Second seed (also the smoke test) | Spend | H12 | ~$12 | taken (E-d, E-g); 10d new | **Next:** ready, waits for your go-ahead |
| 11 | Baselines | Spend | §8.2 | ~$9 | 11a, 11b taken | Done 2026-10-17 ([results](step-11-results.md)); three calls before step 12 |
| 12 | Production run | Spend | §8.6 | up to ~$157 + ~$30 evals | 12a taken; 12b new | After step 11; 12b first |
| 13 | Final eval off W&B | PC or GPU host | §7.4 | $0 on W&B | 13a, 13b | After step 12 |

Inference adds up to ~$218 of the ~$218 left, counting a ~$10 reserve.

---

## Step 1. Freeze the design spec

**Type:** Design · **Cost:** $0 · **Research:** §4.1, §4.3, §4.4, §5.3, §5.5, §10.3

### What we make

`documentation/production-spec.md` is one short spec that every later step is coded against. It covers:

- **Bridge API:** `llm(prompt)`, `llm_batch(prompts)`, `submit(answer)` and `context`, with the exact system-prompt text pinned. There is no `rlm()` (depth > 1) in Part 1.
- **Reward:**
  - An exact answer scores 1. Lists are scored by set-F1.
  - A near miss gets calibrated partial credit (per 1b).
  - No submit scores 0.
  - A detected hack scores 0, applied before the group mean.
  - An infra error is excluded from the group.
- **Episode limits and GRPO settings:** the defaults below.
- **Difficulty controller rule:** for each family, take the pass rate over the last 3 steps. Above ~70% the family moves up a level, and below ~20% it moves down. The top level is ~400K characters, which can be covered in ~20 sub-calls.
- **Code layout:** production modules at the top level of `rlm_rl/` (bridge, sandbox, env, reward, detectors, controller, tasks, evals) and `scripts/` (train, eval). The README Layout section gets updated to match.
- **What is pinned and saved with every checkpoint:** ART 0.5.20, the chat template, the bridge prompt text, generator versions, the difficulty schedule, the dataset manifest and the run config.

### Why it comes first

The sandbox, environment and training loop are all coded against these choices. The model learns whatever function names and prompt it is trained on, so changing them after training is expensive.

### Starts from

It starts from the 20-step run's settings in `scripts/archive/run_rlsd_vs_grpo.py` and `rlm_rl/archive/rlm_env.py`.

### Done when

The spec is written and you have answered 1a-1c.

**Status: complete (approved by Rishi, 2026-10-10).** 1a-1c are answered, and the spec is [production-spec.md](production-spec.md). Step 2 is next.

### Defaults I'll pin unless you change them

| Setting | 20-step run | Production default |
|---|---|---|
| Sub-calls per episode | 20 | 30 |
| Root turns | 6 | 12 (research: 10-15) |
| Root tokens per turn | 1,536 | 3,072 (research: 2,048-4,096) |
| REPL output shown | 2,000 chars | 2,000 chars |
| Cell timeout | 120 s, queue waits included | 120 s of compute, queue waits excluded |
| Loss | CISPO | CISPO |
| Reward std scaling | off | off |
| Episodes per step | 8 tasks × 4 rollouts | 8 tasks × 4 rollouts, all-equal groups dropped |
| Learning rate | 1e-5 | 1e-5 for the second seed; 2e-5 in production only if the second seed is stable, falling back to 1e-5 (decided by you) |
| Temperature | 1.0 | 1.0 for training, fixed decoding for eval |
| KL to base | none | logged, not in the loss |
| Thinking mode | off | off, pinned |

Turn and token limits are caps, not targets.

### Decisions for this step

**1a. Bridge function names: decided 2026-10-10.** `llm`, `llm_batch`, `submit`, `context`. The environment and the harness import the same module (spec §1).

**1b. Partial credit: decided 2026-10-10.** Calibrate it against the blind guess (spec §3).

**1c. CISPO importance-weight clip: decided 2026-10-10.** Restore ART's default, `epsilon=1.0` and `epsilon_high=4.0`, passed explicitly (spec §5.1).

**Learning rate: decided 2026-10-10.** 1e-5 for the second seed. Production goes to 2e-5 only if the second seed's stability checks pass, and falls back to 1e-5 on instability (spec §5.2).

---

## Step 2. Prerequisite checks

**Type:** You + PC · **Cost:** $0 · **Research:** §6, §7.4, §10.7

### What we do

- **CoreWeave support email** (you send it, I'll draft it). It asks why we get HTTP 429 at 32 parallel requests when their docs allow 2,000 per user, with request ids if the logs have them. It also asks them to confirm Qwen3.8-27B training stays supported, since the model isn't on the public list.
- **Docker and WSL2 on your PC.** The sandbox needs Docker, and step 3 works best if the environment runs inside WSL2 (see 3b). Nothing so far confirms that either is installed *(inferred)*.
- **LoRA download.** Pull one checkpoint from the 20-step GRPO run off W&B, to confirm the final eval can run elsewhere.
- **Licence.** Confirm the Qwen3.8-27B licence allows publishing a LoRA adapter on Hugging Face.

### Why it comes here

Each check can block a later step (3, 4 or 13). None of them costs W&B credit, and CoreWeave's reply may take a while.

### Done when

- Docker runs a `--network none` container on your PC.
- A checkpoint is on disk.
- The licence answer is written into the spec.
- The email is sent.

### Decisions for this step

None. The one action for you is to send the CoreWeave email.

---

## Step 3. Sandbox and bridge (do first)

**Type:** PC · **Cost:** $0 · **Holes:** H2, H3, H4, H10 · **Research:** §4.2, §4.4

### What we build

- **A Docker image** with Python and the fixed package set (3c), running as a non-root user.
- **A container for each episode** (3a):
  - `--network none` and a read-only root filesystem with one writable scratch dir.
  - CPU, memory and pids limits.
  - The task's `context` is mounted read-only.
  - The repo, `.art/`, the home directory and `.env` are not inside.
- **The bridge module from step 1**, which is the container's only way out: one socket that accepts `llm`, `llm_batch` and `submit` and nothing else. Decision 3b covers how the socket reaches the host.
- **Host-only task generation.** Generators, gold answers and graders stay on the host. Seeds are random 64-bit numbers and are never passed in. Eval seed lists stay private.
- **A test suite** that becomes the regression tests for every later change.

### Why it comes here

It is your first priority. Until it exists, the REPL can read the W&B key in `.env` and past gold answers in `.art/`. It can also write anywhere (one rollout wrote to `C:\tmp`) and reach the internet, which is exactly how MiMo's models cheated. The bridge is the sandbox's only door, so it is built here. The research lists it under environment v2.

### Starts from

It starts from `repl_worker.py` (the REPL subprocess, HTTP bridge and key stripping) and `SubcallBridge` / `ReplWorker` in `rlm_env.py`.

### Done when

- From inside the sandbox, all of these fail:
  - opening `.env`, `.art/`, the repo, home and `/proc`
  - writing outside scratch
  - `pip install`, `curl` and DNS
  - importing any project module
- Bridge calls work, and no seed is visible inside.
- The environment and the harness import the same bridge module, and a test diffs their system prompts.

### Decisions for this step

**3a. Container lifetime**
- **A fresh container for each episode: recommended (mine).** This is the simplest option, and nothing can leak from one episode to the next. Start-up takes a second or two, which is small next to the 150-390 s steps of the 20-step run *(inferred)*.
- **One container per worker, reset between episodes.** This is faster, but the reset has to be perfect or one episode can leave files behind for the next.

The research allows either.

**3b. How the sandbox reaches the bridge on Windows**
- **Run the environment inside WSL2 and mount a Unix socket into a `--network none` container: recommended (mine).** The container has no network at all, and the socket is its only way out.
- **A Docker internal network, with a small relay container as the only host it can reach.** This works on plain Docker Desktop but adds one more moving part.
- **Plain HTTP to `host.docker.internal`.** This is simplest, but the container then has a network route, which weakens H4.

This needs deciding because Docker on Windows runs containers in a Linux VM, so a Windows-side socket can't simply be mounted in *(inferred from how Docker Desktop works; step 2 checks the setup)*.

**3c. Python packages inside the sandbox**
- **Standard library only: recommended (mine).** `re`, `json`, `collections`, `itertools`, `statistics` and `ast` cover parsing logs and code. The image stays small and the harness can match it easily.
- **Add pandas and numpy.** They're handy for log tables, but the image gets bigger and the harness must ship them too.

The harness has to use the same set (§4.2).

---

## Step 4. Limiter, infra flags, eval logging (do first)

**Type:** PC · **Cost:** $0 · **Holes:** H5, H6, H9 · **Research:** §6, §8.4

### What we build

- **One shared limiter** for every request, root turns and sub-calls alike. It caps requests in flight at 8 and uses jittered exponential backoff. When retries run out it raises `infra_error`, instead of returning a fake answer like today's `"[sub-call failed: rate limited]"`.
- **An `infra_error` flag on each episode.** It is set by a bridge failure after retries, a kernel crash not caused by model code, or an API 5xx. Flagged episodes leave their group before advantages are computed and are counted. The step pauses if more than 1% are flagged.
- **Sub-call queue time no longer counts toward the 120 s cell timeout.**
- **Eval logging.** Every eval episode is saved with its full transcript, sub-calls, reward components and checkpoint, through ART `model.log(..., split="val")` plus a local JSONL. The hack audit reads eval logs as well as training logs.

### Why it comes here

It finishes the do-first phase. Every run from here on depends on it, starting with the second seed. With a 30-call budget, sub-calls queue for longer, so the timeout bug would bite more than it did in testing.

### Starts from

- `chat_with_retry` in `rollout.py`.
- The `Semaphore(8)` and the separate sub-call pool of 6 in `rlm_env.py`, which together let ~14 requests run at once.
- `evaluate_rlm`, which saves only aggregates today.

### Done when

- A load test never has more than 8 requests in flight.
- Forced 429s and a killed kernel produce flagged episodes, and none of them reach training.
- A rerun eval writes one transcript per episode, and the audit script reads them.

### Defaults

- The cap stays at 8 until CoreWeave answers, then we try 16 once.
- The pause threshold is 1%.

### Decisions for this step

None open.

---

## Step 5. Environment v2

**Type:** PC · **Cost:** $0 · **Holes:** H1, H11, H13 · **Research:** §4.1, §4.3, §4.5, §5.5

### What we build

- **Calibrated reward** (per 1b). The host computes `e_blind` for each task from the line-count guess. Exact accuracy is logged as the headline metric.
- **Rule-based hack detectors** that zero the reward before the group mean, each counted on the dashboard. They fire on:
  - `submit()` on turn 1, or without reading `context`
  - a submitted value that appears in the prompt but in no REPL output
  - blocked network, file or host-import attempts
  - repeated cell timeouts
- **A Claude audit** of sampled wins, reported as a clean solve rate (5a decides how it runs). It is never trained against directly, because pushing on a monitor teaches the model to hide what it's doing.
- **The difficulty controller** for each family (rule from step 1). All-equal groups are dropped from the update and not refilled, because refills cost inference.
- **The episode limits from step 1.** The ticket tasks get ported in with levels, so the second seed can run on this environment.

### Why it comes here

This is the environment the second seed tests and production uses. It needs the sandbox and plumbing underneath it.

### Starts from

It starts from `score_answer` and `run_episode` in `rlm_env.py`, and from `ticket_tasks.py`.

### Done when

- A blind guesser scores no more than its exact-hit rate on every family.
- Each detector has a unit test.
- The controller moves levels correctly on synthetic pass rates.

### Note

The research's summary table gives a 20-80% pass-rate band, but H13 and §5.5 both use 20-70%. This plan uses 20-70%, so a family moves up a level above ~70%.

### Decisions for this step

**5a. How the Claude audit runs**
- **A Claude thread audits sampled wins from the logged transcripts at each 10-step checkpoint, or in batches when you check in: recommended (mine).** It needs no API key and costs nothing outside the pots. It runs at the same rhythm as the evals, which is what the stop rule on hacking needs.
- **An Anthropic API call inside the training loop on every step.** This is live, but it needs an API key on your PC and costs money outside both W&B pots.

---

## Step 6. Training loop and monitoring

**Type:** PC · **Cost:** $0 to build · **Research:** §5.3, §8.5, §8.6, §10.3, §10.4

### What we build

`scripts/train.py` is a resumable GRPO script on ART 0.5.20. It has:

- **The settings from step 1:** CISPO with the clip per 1c, no reward std scaling, 8 tasks × 4 rollouts, LR 1e-5, temperature 1.0, and KL to base logged but not in the loss.
- **An eval every 10 steps** on the 64-task selection set, plus a ~100-prompt capability check. It saves the checkpoint and tracks the best one so far.
- **Automatic stop rules** from §8.6. When one fires, the script stops and reports which rule it was.
- **Checkpoints:** every 5th is kept plus the best. The run can resume from the latest or the best, and the config is saved with each checkpoint.
- **An inference spend counter with a hard ceiling,** so the run can't go past its budget.
- **Per-step metrics** to W&B and `results/<run>.json`:
  - pass rate with and without infra errors, the pass-rate histogram and trainable groups
  - level per family
  - within-group reward std, entropy, KL, clip fraction, `probs_corr` and grad norm
  - sub-calls and budget-hit rate, turns, lengths and truncation
  - infra error rate, hack counters, cost and step timing
- **Operator controls:** a "stop after this step" switch, and resume from any kept checkpoint, so the run can be stopped, fixed and restarted from the last good checkpoint (step 9).
- **Truncated REPL output,** so root transcripts stay inside ART's packed sequence length.

### Why it comes here

The second seed already runs through this script, so it has to exist before any spend.

### Starts from

It starts from `scripts/archive/run_rlsd_vs_grpo.py`, which is already resumable and writes results JSON every step, minus the RLSD and teacher code.

### Done when

A dry run with a fake model client, making no W&B calls, goes through steps, an eval, a stop rule, a checkpoint save and a resume.

### Defaults

- It is a script, not a notebook.
- Production will run for many hours (~120 steps at several minutes each, *inferred*), so your PC has to stay on. Resume covers interruptions.

### Decisions for this step

None open. The clip is decided in 1c.

---

## Step 7. Data

**Type:** PC + Design · **Cost:** $0 · **Holes:** H7, H8 · **Research:** §3, §8.1

**Status: complete (2026-10-11).** Results: [step-7-results.md](step-7-results.md). Decisions 7a, 7b and 7c are below. Decision 7c dropped cyber, so production trains on aggregation and code only.

### What we build

**Training sources.** Each one gets difficulty levels, from today's ~35K characters up to ~400K. The production mix is set by decision 7c:

| Family | Share | Source | Answer and check | Difficulty knobs |
|---|---|---|---|---|
| OOLONG-style aggregation generator (ours) | 45% | Labelled texts from CLINC150, MASSIVE, LEDGAR contract provisions and arXiv titles (all CC-BY or CC0) | Counts (int), topic or user lists (set-F1), the most active user (exact) | Context length, number of labels and users, question kind |
| Code call-graph generator (ours) | 20% | Generated Python packages | All transitive callers of X (set-F1) | Package size, call depth, hidden call styles |
| Needle-function generator (RepoQA method) | 20% | Generated Python packages | Behavioural description → qualified name (exact) | Package size, look-alike decoys |
| Long Code Arena bug localization, Python train | 15% | Real GitHub issues and repositories | Set-F1 on changed files | Context size, filler drawn from the changed files' folders |

OOLONG-synth itself fails the licence bar (7b), so our generator rebuilds its method on permissive data. SWE-smith isn't used, because LCA covers the repo share.

**Every training task must** (§3.1):
- need long context
- need semantic reading somewhere, with ~20% kept regex-solvable
- have a verifiable answer (no multiple choice, no LLM judge)
- be unguessable from metadata
- rename its entities per seed
- expose difficulty knobs

**Eval tiers (H7):**
- **Selection set (`selection-v1`).** 64 fixed tasks: held-out seeds of the training generators at fixed levels, three LCA repositories training never sees, and the ticket tasks as the held-out generator family. It picks the checkpoint and runs at every eval.
- **Held-out test (`heldout-test-v1`).** 170 tasks, touched once:
  - a held-out aggregation source (HuffPost headlines)
  - held-out behaviours and call styles in the code generators
  - LCA's Python test split
  - a sample of OOLONG-synth's test split
  - separately written question templates
  
  An n-gram overlap check runs against training.
- **Public benchmarks.** These are touched once, in step 13:
  - LCA Java/Kotlin test
  - OOLONG-real
  - LongBench-v2 code
  - a BrowseComp-Plus subset
  - LocBench, LongCodeBench and MRCR
- **Capability check.** 100 prompts: IFEval, an MMLU-Pro subset, CRUXEval as the short coding set, and chat prompts. It catches frying.

**Avoid:**
- SWE-bench Lite: its answers are public.
- LoCoBench: subjective scoring.

**No pre-filter pass.** It would cost ~$40-120 of inference. Every family starts at level 1, where the base model passes about half the time, and the controller moves it from there.

### Why it comes here

Production and the baselines need this data. The second seed doesn't, because it runs on the ticket tasks.

### Done when

- The top levels have a median context above 130K characters.
- No generator, template or source overlaps training.
- Every training source passes the licence bar (7b).
- A dataset manifest pins every version.

All four pass (step-7-results.md §4-§7).

### Decisions for this step

**7a. Domain mix: decided 2026-10-11.** The research mix: 35% cyber, 30% OOLONG-style aggregation, 20% code generators, 15% repo localization. Decision 7c later replaced it.

**7b. Licence bar for training data: decided 2026-10-11.** Permissive licences only (Apache-2.0, MIT, CC-BY, CC0 and similar) in training. Anything else is eval-only or dropped. `tasks.check_mix()` enforces it.

**7c. The cyber share: decided 2026-10-11.** Drop cyber as a whole, and split its 35% between better coding and one more thing we already have. That gives:
- code generators 40% (call graph 20%, needle 20%)
- aggregation 45%
- repo localization 15%

Aggregation gained two sources (LEDGAR and arXiv titles) for its bigger share.

---

## Step 8. Red-team

**Type:** Claude agent · **Cost:** $0 · **Holes:** H11, re-checks H1-H4 · **Research:** §4.6

### What we do

A Claude agent gets environment v2 with every task family and tries to earn reward without solving the tasks. It tries guessing, reading files, recovering seeds, using the network, crashing the grader and sending malformed `submit` values. Each exploit that works gets fixed and added as a regression test, then the agent tries again.

### Why it comes here

It is the last gate before any credit is spent. Once runs get long, RL finds whatever is reachable.

### Starts from

It starts from the test suite from step 3 and the detectors from step 5.

### Done when

The agent can no longer find an exploit.

### Defaults

A Claude Code thread on your PC runs it. It uses no W&B credit.

### Decisions for this step

None open.

---

## Step 9. Live dashboard

**Type:** PC + Claude · **Cost:** $0 on W&B (the Claude session that updates it uses your Claude plan) · **Research:** §2.5, §8.5, §9 · **Added by:** you, 2026-10-10

### What we build

This is a live, shareable dashboard published as a Claude artifact, modelled on MiMo-V2.6's public RL page. You can send the link to your researcher friends so they can watch the run step by step and flag anything we miss. W&B stays the source of truth, and the dashboard mirrors it.

**Panels.** These follow MiMo's layout, which had a metrics view, a benchmarks view and a notices log:
- **Run header:** status (running, paused, or stopped and why), the current step, the difficulty level per family, and the last update time.
- **Cost:**
  - inference spent, against the $150 ceiling and the ~$218 pot
  - cost per step, and projected steps left at the current rate
  - reserve status and training GPU spend
- **Pass rates** (MiMo's "dynsam" panels), per family:
  - pass rate with and without infra errors
  - share of all-0 and all-1 groups, and trainable groups per step
  - the pass-rate histogram
- **Policy health** (MiMo's "actor" panels): entropy, KL to base, clip fraction, `probs_corr`, grad norm, learning rate, and within-group reward std (the Echo Trap signal).
- **RLM behaviour:** sub-calls per episode, budget-hit rate, root turns, context and response lengths per family, and truncation rate.
- **Reward and cheating:** exact accuracy, reward components, hack-detector hits by rule, and the clean solve rate from the Claude audit.
- **Infra and timing:**
  - infra error rate, against the 1% pause line
  - step time, split into rollouts and training
  - tokens per step
- **Evals:**
  - selection-set accuracy every 10 steps, with the best checkpoint marked
  - the baselines for comparison: the base model as an RLM, direct reading, and the 20-step seeds
  - the capability check
- **Stop rules:** where each of the five rules stands, for example "plateau: 1 of 3 evals without a new best".
- **Notices:** an operator log of incidents and interventions (pauses, fixes, rollbacks, resumes), like MiMo's Notices tab.
- **Sampled transcripts** (per 9c): a few wins and failures from each eval, with their detector flags.

**How it stays live.** An artifact page can't fetch from W&B or from your PC, because its sandbox blocks outside requests. The training script already writes `results/<run>.json` every step. So a Claude session on your PC watches that file and pushes each new step to the dashboard (9a decides the route).

**Operator controls** live in the training script (added to step 6): a "stop after this step" switch, and resume from any kept checkpoint. When a friend spots something, you decide what to do. Then I stop the run, fix the problem, log a notice and resume from the last good checkpoint.

**Nothing secret goes on it.** That means no keys, no task seeds or eval seed lists (H3), and no public-benchmark items.

### Why it comes here

It has to exist before production, and putting it before the second seed means the second seed is its first real run. Your friends can then check it on a cheap run before the expensive one.

### Starts from

It starts from the per-step metrics built in step 6 and the panel lists in research §2.5 and §8.5. MiMo's live page is down, so the layout follows write-ups of it ([intelligentliving.co](https://www.intelligentliving.co/mimo-v2-6-rl-training/), [redreamality](https://redreamality.com/blog/mimo-v2-6-rl-metrics-and-incident-analysis/)).

### Done when

- A dry run of the training loop (step 6's fake model) updates every panel each step.
- A second browser with the page open sees new steps arrive without reloading.
- After the second seed, the cost panel matches W&B's billing page.
- A check confirms that no key, seed or benchmark item appears in the page or its data.

### Decisions for this step

**9a. How the dashboard gets each step's data**
- **A Claude session on your PC republishes the dashboard with the newest results after each step: recommended (mine).** Everyone you share it with sees each update arrive in the open page. It is the simplest route. I believe it also works for viewers who aren't signed in, but that's *inferred*.
- **The page keeps its data in the artifact's own database and listens for changes.** Each new step appears in place without a new page version, but every viewer has to be signed in to claude.ai to see any data.

Either way, the training script can't push to the artifact on its own, so a Claude session runs alongside the second seed and production. I checked this against the artifact docs.

**9b. Who can see it**
- **Invite your friends from the page's Share menu: recommended (mine).** Invited viewers can leave comments on the page, and I can read those and bring them to you *(inferred)*.
- **A public link.** Anyone with the link can watch with nothing to set up, but they can only view *(inferred)*.

You change sharing yourself from the page's Share menu; I can't do it for you. Stop and resume calls stay with you: a friend's comment is a flag to look at, not an order to act on.

**9c. Sampled transcripts on the dashboard**
- **Show a few wins and failures from each eval, with detector flags: recommended (mine).** Transcripts are where cheating shows up, which is what you want extra eyes for.
- **Charts only.** The page stays smaller and there is less for your friends to read.

---

## Step 10. Second seed, which is also the smoke test

**Type:** Spend · **Cost:** ~$12 · **Hole:** H12 · **Research:** §7.3, §11 · **Status (2026-10-16):** next. Everything it needs is built and tested; it waits only for your go-ahead, because it spends credit.

### What we run

We repeat the 20-step GRPO run at level 1 on the ticket tasks, on environment v2, plus a 100-episode held-out eval. That is about 740 episodes. `config.second_seed` pins it:
- learning rate 1e-5, fixed, with the difficulty controller frozen;
- 20 steps, then the held-out eval on the first seed's tasks (`heldout-seed1`);
- prompt v1, so it repeats the setup the 20-step run used (decision E-g; production and the baselines use v2);
- thinking off on root calls and sub-calls, which a replication needs (step 2 results §6).

It uses read tasks only: no edit sandboxes, and almost no load on the data drive. Read-only dry runs took a median 290 s a step, so expect roughly 2-3 hours with the eval *(inferred: live inference adds latency the dry runs don't have)*.

**How it runs** ([step-9-results.md](step-9-results.md) §7):
1. In WSL: `.venv-linux/bin/python scripts/train.py start --kind second_seed --run second-seed-v1 --live`.
2. Beside it, a Claude session on the PC runs the dashboard loop: `scripts/dashboard.py wait second-seed-v1 --pot-left <W&B balance>`. It publishes each step to the same dashboard URL your friends already have.

**Before it starts** (all $0):
- your go-ahead;
- the inference balance read off W&B's billing page, for `--pot-left` and for the billing check afterwards;
- Docker and WSL up, no leftover `rlm-` containers, and the sandbox folder owned by your user;
- the full test suite on the final code (`scripts/sandbox_ctl.py check`): after the 2026-10-15 red-team fixes only the changed modules were rerun (128 of 128), not the whole suite;
- one fake-model dry run on the final code (`--dry-run`) that comes out clean.

### Why it comes here

There is no pilot, so this is the first run on the new environment. It catches plumbing bugs at about a tenth of production cost. It also shows whether the gain reproduces with the sandbox, calibrated reward and restored clip in place. Since decision E-g the code-editing build no longer gates it: it never touches edit tasks. Running it before the baselines also gives the dashboard its first live run.

### Done when

- Held-out accuracy lands within ~1 SE of the first seed's 0.70, or the gap is explained.
- Grad norm and `probs_corr` look like the first run's.
- Its stability curves decide production's learning rate: 2e-5 if they pass the checks in spec §5.2, otherwise 1e-5.
- Infra errors stay under 1%.
- No detector fires unexpectedly.
- The dashboard's cost panel matches W&B's billing page. That is step 9's last open check.

### Decisions for this step (10a-10c taken, 10d new)

10a, 10b and 10c were settled with the code-editing plan on 2026-10-12 (decisions E-d and E-g): the split below, run the second seed, and run it as soon as it's ready, independently of the edit build. They are kept here for the record.

**10a. The inference split**

| Item | Inference |
|---|---|
| Second seed (this step) | ~$12 |
| Baselines (step 11) | ~$9 |
| Production ceiling (step 12) | ~$157 |
| Selection-set evals during production | ~$30 |
| Reserve | ~$10 |
| Final eval (step 13, off W&B) | $0 on W&B |
| **Total** | **~$218 of ~$218** |

Updated 2026-10-12 for code editing (Rishi: no E7 probe, reserve $25 to $10). Baselines add the 16 edit tasks, run twice; each selection eval costs ~$3.50 with edit tasks. Spec §12.11 holds the same table.

- **Taken: approved as is** (decision E-d, 2026-10-12; spec §12.11).

**10b. Run the second seed at all**
- **Taken: run it** (decision E-g). It replaces the pilot, and it checks that 0.70 wasn't a lucky seed.
- *Not taken:* skip it and save ~$12, making production's first steps the smoke test.

**10c. When to run it**
- **Taken: as soon as it's ready, without waiting for the edit work** (decision E-g). It's ready now.
- *Not taken:* after the code-editing build, so that every paid run uses prompt v2.

**10d. Test resuming from an earlier checkpoint on W&B** (new, 2026-10-16; from [step-6-results.md](step-6-results.md) §8)

Rolling back and the learning-rate fallback both fork an earlier checkpoint through ART's experimental fork call. ART documents it for serverless, but no run has used it yet *(inferred that it works)*.
- **Test it once the second seed finishes: recommended (mine).** `train.py resume second-seed-v1 --from 15 --stop-after 16 --live` trains one extra step, about $0.50, inside the second seed's $14 ceiling. If training continued from the fork, the new step 16's KL to base matches the original's; if it silently started from the base model, KL drops to about 0.
- **Skip it.** It saves $0.50, but production's first rollback becomes the first test, and a broken fork would cost production steps.

---

## Step 11. Baselines

**Type:** Spend · **Cost:** ~$9, plus ~$2-3 if 11b adds the thinking arm · **Research:** §8.2 · **Status (2026-10-17):** done, $5.89 measured (up to $8.04 counting failed tries W&B may bill). Results in [step-11-results.md](step-11-results.md): base RLM 42.5%, direct reading 59.1% (read tasks), medium thinking 59.4% (+17 points over off). Prompt v2 made the base model answer in tool-call syntax on many turns. **Rishi's decisions (2026-10-17):** prompt v3 runs that code and says the tags don't run; production's root thinks at `medium`; edit turns go to 30; the selection set becomes `selection-v3` (the ticket tasks replaced). The base model was rerun in that set-up (results §14). The baseline script is `scripts/baselines.py`.

### What we run

On the 64-task selection set (`selection-v2`, spec §12.10), about 150 episodes. As built, it holds 48 read tasks (aggregation, call graph, needle, repo localization and tickets) and 16 edit tasks: 10 `edit_real` (Python 3, Go 4, JavaScript 3) and 6 `edit_synth`.
- **Base Qwen3.8-27B as an RLM** in environment v2, prompt v2, thinking off. This is the step-0 number. The 16 edit tasks run twice. That also measures, before production:
  - the cost per edit episode, against code-editing-plan.md §10's guesses: ~$0.06 for `edit_synth` and ~$0.15 for `edit_real` *(inferred)*;
  - turn-cap hits;
  - the base pass rate per tier, the dropped E7 probe's job.
- **The base model reading the context directly,** where it fits in 262K tokens (read tasks only).
- **If 11b says so,** the base model as an RLM with root thinking at `medium` on the same 64 tasks. Sub-calls stay off.

The 20-step checkpoints (both seeds) are the research's third baseline. They are optional here, and not included in the ~$9.

The edit episodes run in Docker on your PC, on the D: data drive. A Go task's tests take up to about an hour under load ([edit-build-results.md](edit-build-results.md) §5a), so expect a few hours for the whole step *(inferred)*.

**Before it runs** ($0, on the PC): a small baseline script. `scripts/train.py` only has `start`, `resume`, `stop` and `status`, and nothing in `rlm_rl/` reads a context without the REPL (checked 2026-10-16). Production's checkpoint-0 eval covers the base model as an RLM once, but not the second pass over the edit tasks, direct reading or the thinking arm. The script reuses `evals.run_eval` and the episode loop, and gets a fake-model dry run first.

### Why it comes here

The stop rules and every claim compare against these numbers. Without the direct-reading number we can't claim the RLM setup helps. It is also the first paid run of edit tasks. Its measured cost per edit episode checks production's ~$2.10 a step before the $157 is committed (code-editing-plan.md §16).

### Done when

- Both baselines are logged with full transcripts, and so is the thinking arm if it runs.
- The cost per edit episode is measured, and production's step count is re-estimated from it. If edit episodes cost much more than estimated, production drops toward ~50 steps or the edit share drops; that is your call before step 12.
- Two settings are checked against code-editing-plan.md §9 and §11: if more than ~20% of the base model's edit episodes hit the 20-turn cap with a test failing, edit turns go to 30; if the base model passes under ~15% of `edit_real` level 1, `edit_real` starts at a lower share for its first 20 steps.
- The 11b comparison is settled, if the arm ran.

### Decisions for this step

**11a. Which model serves sub-calls once deployed**
- **The base model, as in training: recommended (mine; the research makes no call).** The root was trained against base-model sub-calls, so evals stay comparable and nothing shifts at deployment.
- **The trained LoRA.** That means one model to serve, but sub-calls then behave differently from training and need their own eval.

This is needed here because every eval from now on should use whichever model the harness will use. Training already uses the frozen base model for sub-calls (spec §0); this decision is about deployment and the evals.

**11b. Thinking for the production root** (from [step-2-prerequisite-results.md](step-2-prerequisite-results.md) §6; not in this plan until now)
- **Add one base-model arm with root thinking at `medium` to the baselines: recommended (step 2, mine).** It costs ~$2 for the 64 tasks, more with edit tasks (~$2-3 *(inferred)*), which is about one production step. Production uses `medium` only if it beats thinking off by a clear margin, about 12 points or more (~2 SE at n = 64). Otherwise production stays off. Sub-calls stay off either way, and `xhigh` is never used. Thinking at `medium` costs ~1.6× per read episode, so production would buy fewer steps. Editing may gain more from thinking than reading does (code-editing-plan.md §16).
- **Skip it.** Thinking stays off everywhere, as in the 20-step run, and the question stays untested.

---

## Step 12. Production run

**Type:** Spend · **Cost:** up to ~$157 + ~$30 evals; training GPU ~$25-60 of the ~$992 left · **Research:** §5.6, §8.5, §8.6, §9 · **Status (2026-10-16):** after step 11. Code, data and dry runs are ready. The edit red team's gate for edit spend is open now that F1-F3 are fixed ([step-9.5-red-team-results.md](step-9.5-red-team-results.md), Fixes). 12b comes first.

### What we run

GRPO at 32 episodes per step (8 tasks × 4 rollouts), with every family starting at level 1. It runs until a stop rule fires or it reaches the $157 ceiling. The live dashboard (step 9) updates every step at the same URL, so you and your friends can watch it.
- **Learning rate:** whatever the second seed decided. At 2e-5, entropy or reward-std collapse rolls the run back to the last good checkpoint and continues at 1e-5 (spec §5.2).
- **Mix** (spec §12.7):
  - aggregation 25%;
  - code reading 20% (`callgraph` 10%, `needle` 10%);
  - repo localization 15%;
  - editing 40% (`edit_real` 25%, `edit_synth` 15%).
- **Cost:** with 40% edit tasks (code-editing-plan.md §10) a step costs ~$2.10, so that is ~75 steps (range ~50-90). Step 11 re-checks this. **Step 11 measured ~$0.95 a step with thinking off and ~$1.19 at medium, so ~130-165 steps for $157** *(inferred: training lengthens episodes; a working range is ~$1-1.5 a step)*; step-11-results.md §7.
- **Evals:** the selection set and capability check run every 10 steps, at ~$3.50 each.
- **Thinking:** root at `medium` (11b's arm earned it: +17 points; Rishi, 2026-10-17). Sub-calls stay off. Training on thinking tokens needs the loop wired for it first (below).
- **Prompt v3 and `selection-v3`** (Rishi, 2026-10-17, step-11-results.md §14): tool-call code runs, edit episodes get 30 turns, and the selection set's ticket tasks are replaced.

The command is `train.py start --kind production --run <name> --lr <2e-5 or 1e-5> --live`, with the same dashboard loop as step 10.

**The edit data it trains on** (round 1, frozen 2026-10-13; [edit-build-results.md](edit-build-results.md) §7):
- **`edit_real` training: 107 tasks.** Python 86, Go 16 (2 of them real pull requests), JavaScript 5. Levels: 46 at level 1, 48 at level 2, 13 at level 3, none at 4-5 (a draw above level 3 takes the nearest level).
- **Eval splits:** selection 10 and test 40 (Python 16, Go 15, JavaScript 9; 8 of them feature tasks, where spec §12.10 asks for 15).
- **Against the plan:** the spec planned ~314 `edit_real` tasks in five language groups. Rust, TypeScript, Java and the Go top-up have their candidates, images and fixes on D: for a later round.
- **`edit_synth`** generates its tasks per seed, so it has no pool to run out of.

**How long it takes on the PC.** Edit episodes and their graders run in Docker on the D: data drive, a USB hard disk. Measured under load, a step averages ~30-35 minutes: ~20-25 when its `edit_real` tasks are Python or JavaScript, about an hour when one is Go (edit-build-results.md §5a). 75 steps is ~40 hours. Add about an hour for each of the ~7-8 selection evals, and the run takes ~2-2.5 days of the PC running without a break *(inferred)*. A slow drive costs time, never reward: every limit was set from measurements, and an infra error is never scored as a 0.

**Before it starts** (all $0, on the PC):
0. **Training with root thinking at `medium`:** done 2026-10-17 ($0). `config.production` sets it, the loop's client and pinned config carry it, and the KL monitor re-scores thinking turns one at a time with their thinking (spec §5.3). A 3-step fake-model dry run of production with thinking on came out clean (every re-scored turn lined up). Production's first step is the first live check of the KL alignment (`kl_misaligned` on the dashboard).
1. **What 12b chooses,** built and tested.
2. **The training run clears leftover containers at start.** The data build does this since 2026-10-15, but a training run doesn't yet, so a crashed run's containers would pile up. It's a few lines and a test.
3. **Optional drive settings**, yours to make (edit-build-results.md §5a):
   - Defender exclusions for `D:\Docker-Image`, `D:\WSL` and `D:\RLM-data` (as administrator);
   - other WSL distros stopped during the run (e.g. `EXL3-Runtime`);
   - no other jobs on D:.
4. **Edit wins in the Claude audit** (decision 5a). `scripts/audit.py`'s sample shows no diff or grade, and audit-procedure.md has no edit checks (checked 2026-10-16). Since the audit is the backstop for the canary gap, each sampled edit win should show its diff, and the procedure should say what to look for (test or runner files touched, code that detects a test run, hard-coded expected values).
5. **A last fake-model dry run** with `--real-edit` on the final code.

**It keeps going while** the selection set has reached a new best in one of the last 3 evals, or a family has moved up a level in the last 30 steps.

**It stops early, keeping the best checkpoint, on any of these:**
1. **Plateau:** no new best for 3 evals (30 steps), and no level-up.
2. **Overfitting:** training reward rises while the selection set falls, two evals in a row.
3. **Frying:** the capability check drops more than ~2 points, or a collapse comes back after one rollback with a lower LR.
4. **Saturation:** every family is at its top level with a pass rate above ~70%. The next step is then harder data, not more steps.
5. **Hacking that needs an environment change:** pause, fix, resume.

**Watch for** (§5.6, §8.5):
- Echo Trap collapse, where within-group reward std falls off a cliff
- entropy dropping fast, or KL jumping
- sub-calls climbing toward 30 without accuracy following
- turns or tokens hitting their caps
- infra errors above 1%; on edit steps this includes grader containers that never report (InfraPause: 4 of the last 320 episodes)
- the edit detectors (`test_tamper`, `grader_tamper`, `test_aware`) firing, and the Claude audit of sampled edit wins. Recognizable canaries stay a known gap, and the audit is its backstop (step-9.5-red-team-results.md)
- all-0 edit groups piling up, a language stuck near 0 on the dashboard's per-language table, and the `edit_real` pool shrinking through retirement (12b)

**If money runs short:** cap training contexts at ~250K characters, or eval every 15 steps instead of 10. Cutting episodes per step is the last resort, because the 20-step run never tested it.

### Done when

A stop rule fires or the run reaches the ceiling, and the best checkpoint on the selection set is identified. A stopped run can resume later from its checkpoint if more credit appears.

### Decisions for this step

**12a. The $10 reserve**
- **Taken: I ask you before touching it** (2026-10-12; spec §12.11). The run stops at its ceiling and says whether the research's rule would spend the reserve; resuming with a higher ceiling spends it.
- *Not taken:* spend it automatically when the last 2 evals were both new bests, then ask (the research's rule).

**12b. When the `edit_real` pool runs out** (new, 2026-10-16)

About 75 steps × 2 `edit_real` tasks is ~150 draws from 107 training tasks. Retirement takes a task out for good once its group comes back all-0 twice or all-1 once. If every task retires, `editreal.make_task` raises "no edit_real task left to draw", nothing catches it, and the run stops mid-step. It can resume from its last checkpoint, but someone has to be there.
- **Add a fallback before production: recommended (mine).** When the pool is empty, draw again from the tasks retired longest ago, or move `edit_real`'s 25% to `edit_synth` for the rest of the run. It takes a few hours on the PC with a test, costs $0, and the run never stops for want of tasks. Round 2 can still be added later.
- **Build round 2 first.** Validate Rust, TypeScript, Java and the Go top-up toward the ~314 planned tasks: more languages and less repetition. Round 1's validation took about two days on the data drive, so production would start days later *(inferred)*.
- **Start as is.** It costs nothing up front, but risks a stop partway through the run; I'd then add one of the above and resume.

---

## Step 13. Final eval off W&B

**Type:** PC or GPU host · **Cost:** $0 on W&B · **Research:** §7.4, §8

### What we run

We push the best LoRA to your Hugging Face. Then we run the held-out test, the public benchmarks and the capability-retention check outside W&B:
- ~400 episodes per model, for a ±5-point interval
- 2-3 sampling seeds
- paired tests (McNemar or paired bootstrap)
- the base model and direct reading re-run in the same place

Code editing adds (code-editing-plan.md §14):
- the edit tiers: the 40-task `edit_real` test split as built, with 8 feature tasks rather than 15;
- SWE-bench Pro, DeepSWE v1.1 and the normal-coding-agent comparison (decision E-e), run through Harbor.

That needs Docker on the eval host for the repository images.

**The comparison is fair only if** it uses the same harness (frozen bridge, prompts, limits and decoding), the same sub-call model (11a), and runs the base model in the same setup.

### Why it comes last

The held-out tiers are touched once, so nothing gets tuned on them, and we never pick a checkpoint on a number we report.

### Done when

The trained model and the base model both have final numbers with confidence intervals, and every transcript is logged and audited.

### Decisions for this step

**13a. Where the final eval runs**
- **Rent one 80 GB GPU (bf16) and serve the base model and LoRA there.** You get full control. The cost sits outside both W&B pots and isn't priced yet.
- **A hosted provider that serves Qwen3.8-27B with LoRA adapters.** This is the least setup, but I haven't checked which providers do this yet.
- **On W&B.** There's no new setup, but it takes ~$35 from production (fewer steps) and the samples are smaller.

No recommendation yet. I'll price the first two before you pick. This isn't needed until production ends.

**13b. Which harness runs it**
- **The Part 2 harness core, if it's built by then** (research).
- **The training environment's episode loop with the frozen bridge: recommended as the fallback (mine).** It is still fair, because both models run on the same loop. The harness eval follows once Part 2 exists.

Nothing in this plan builds the Part 2 harness core, so this decision matters if Part 2 hasn't started by the time production ends.

---

## Sources

All numbers and choices come from [rl-run-research.md](rl-run-research.md) unless marked *(mine)* or *(inferred)*. The evidence behind them is in [grpo-for-rlm.md](grpo-for-rlm.md) and [rlsd-vs-grpo-20-step-results.md](archive/rlsd-vs-grpo-20-step-results.md).
