# Production RL build plan

> **Status:** draft for discussion, 2026-10-10. Step 9, the live dashboard, was added at your request. Nothing has been built, run, committed or pushed, and no W&B credit has been spent.
>
> **Source of truth:** [rl-run-research.md](rl-run-research.md). Its html version says the same thing. Section numbers (§) below point into it. Anything marked *(mine)* is my recommendation, not the research's. Anything marked *(inferred)* hasn't been checked yet.
>
> **How to answer:** each step ends with its own open decisions, numbered by step (1a, 1b, …). Reply with the number and your pick, for example "1a yes, 1b calibrate, 3b WSL2". Where a step lists defaults, I'll use them unless you say otherwise.

## The goal

The goal is to train a Qwen3.8-27B LoRA as an RLM root with GRPO on W&B serverless RL, using long-context blue-team cyber and code tasks. The environment is sandboxed so it can't be gamed. Everything has to fit in the ~$218 of W&B inference credit left. The trained model then has to be shown to beat the base model on held-out tasks.

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

| # | Step | Type | Covers | W&B inference | Decisions |
|---|---|---|---|---|---|
| 1 | Freeze the design spec | Design | H1, H10 (spec) | $0 | 1a, 1b, 1c |
| 2 | Prerequisite checks | You + PC | §6, §7.4, §10.7 | $0 | none (one action for you) |
| 3 | Sandbox and bridge (do first) | PC | H2, H3, H4, H10 | $0 | 3a, 3b, 3c |
| 4 | Limiter, infra flags, eval logging (do first) | PC | H5, H6, H9 | $0 | none |
| 5 | Environment v2 | PC | H1, H11, H13 | $0 | 5a |
| 6 | Training loop and monitoring | PC | §5.3, §8.5, §8.6 | $0 | none |
| 7 | Data | PC + Design | H7, H8 | $0 | 7a, 7b |
| 8 | Red-team | Claude agent | H11, re-checks H1-H4 | $0 | none |
| 9 | Live dashboard | PC + Claude | §2.5, §8.5, §9 | $0 | 9a, 9b, 9c |
| 10 | Second seed (also the smoke test) | Spend | H12 | ~$12 | 10a, 10b, 10c |
| 11 | Baselines | Spend | §8.2 | ~$5 | 11a |
| 12 | Production run | Spend | §8.6 | up to ~$150 + ~$25 evals | 12a |
| 13 | Final eval off W&B | PC or GPU host | §7.4 | $0 on W&B | 13a, 13b |

Inference adds up to ~$217 of the ~$218 left, counting a ~$25 reserve.

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

### What we build

**Training sources.** Each one gets difficulty levels, from today's ~35K characters up to ~400K:

| Source | Domain | Answer and check | Difficulty knobs |
|---|---|---|---|
| Cyber log-injection generator (ours) | Blue team | First compromised host (exact), malicious ProcessGuids or timestamps (set-F1), lateral-movement account (exact), technique ID when one is injected (exact) | Log volume, attack slices, number of hosts, how indirect the question is |
| Code call-graph generator (ours) | Code | All transitive callers of X, from `ast` / tree-sitter (set-F1) | Repo size, call depth, obfuscation |
| Needle-function generator (RepoQA method) | Code | Behavioural description → qualified name (exact) | Repo size, similar decoy functions |
| OOLONG-synth validation | Aggregation | Exact; numbers scored 0.75^\|diff\| | Built-in length buckets |
| Long Code Arena bug localization, Python train | Code | Set-F1 on changed files | Repo token buckets |
| SWE-smith | Code | Gold file or function from the patch (set-F1) | Repo size |

**How the cyber generator works:**
- Benign background logs come from OTRF, Sysmon and Loghub Linux-SSH.
- Attack slices come from `splunk/attack_data`, OTRF Security-Datasets and EVTX-ATTACK-SAMPLES.
- Each seed gets random time shifts and renamed hosts, users, IPs and GUIDs.
- Questions describe behaviour rather than event IDs, so the model has to read.

**Every training task must** (§3.1):
- need long context
- need semantic reading somewhere, with ~20% kept regex-solvable
- have a verifiable answer (no multiple choice, no LLM judge)
- be unguessable from metadata
- rename its entities per seed
- expose difficulty knobs

**Eval tiers (H7):**
- **Selection set.** 64 fixed tasks: held-out seeds at fixed levels, plus one held-out generator family. It picks the checkpoint and runs at every eval.
- **Held-out test.** AIT-LDS v2 log sources, unseen ATT&CK techniques and code languages, OOLONG test sources, and separately written and paraphrased question templates. An n-gram overlap check runs against training. It is touched once.
- **Public benchmarks.**
  - CyberSOCEval malware (609 questions, often 128K+ tokens)
  - ExCyTIn-Bench (589 questions)
  - LCA Java/Kotlin test
  - OOLONG-real
  - LongBench-v2 code
  - a BrowseComp-Plus subset
  - LocBench, LongCodeBench and MRCR
  - the Cyber Defense Benchmark, if it is released

  These are touched once.
- **Capability check.** About 100 prompts: IFEval, an MMLU-Pro subset, a short coding set and chat prompts. It catches frying.

**Avoid:**
- SWE-bench Lite and BOTS v1-v2: their answers are public.
- CTIBench, CyberMetric and SecBench: short multiple choice.
- CIC-IDS: numeric CSV.
- Loghub BGL/Thunderbird, unless the label column is stripped.
- LoCoBench: subjective scoring.

**No pre-filter pass.** It would cost ~$40-120 of inference. Every family starts at level 1, where the base model passes about half the time, and the controller moves it from there.

### Why it comes here

Production and the baselines need this data. The second seed doesn't, because it runs on the ticket tasks.

### Done when

- The top levels have a median context above 130K characters.
- No generator, template or source overlaps training.
- Every training source passes the licence bar (7b).
- A dataset manifest pins every version.

### Decisions for this step

**7a. Domain mix**
- **The research mix: ~35% cyber, ~30% OOLONG, ~20% code generators, ~15% repo localization. Recommended (research).** OOLONG is ready-made with length buckets, so there is less to build.
- **Heavier on your two use cases,** for example 45% cyber, 30% code generators, 15% repo localization and 10% OOLONG. This is closer to blue team and coding, but it means more generator work and less general aggregation practice.
- **Something else.**

**7b. Licence bar for training data**
- **Permissive licences only (Apache-2.0, MIT, CC-BY and similar) in training; anything else is eval-only or dropped. Recommended (mine).** The LoRA is going on your Hugging Face, so this keeps it clean. It rules out EVTX-ATTACK-SAMPLES (GPL-3.0) for training and means checking which of OTRF's two licences applies.
- **Decide case by case for each source.**

AIT-LDS and CTIBench are non-commercial. AIT-LDS is already eval-only, and CTIBench is on the avoid list.

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

**Type:** Spend · **Cost:** ~$12 · **Hole:** H12 · **Research:** §7.3, §11

### What we run

We repeat the 20-step GRPO run at level 1 on the ticket tasks, on environment v2, plus a 100-episode held-out eval. That is about 740 episodes.

### Why it comes here

There is no pilot, so this is the first run on the new environment. It catches plumbing bugs at about a tenth of production cost. It also shows whether the gain reproduces with the sandbox, calibrated reward and restored clip in place.

### Done when

- Held-out accuracy lands within ~1 SE of the first seed's 0.70, or the gap is explained.
- Grad norm and `probs_corr` look like the first run's.
- Its stability curves decide production's learning rate: 2e-5 if they pass the checks in spec §5.2, otherwise 1e-5.
- Infra errors stay under 1%.
- No detector fires unexpectedly.

### Decisions for this step

**10a. The inference split**

| Item | Inference |
|---|---|
| Second seed (this step) | ~$12 |
| Baselines (step 11) | ~$5 |
| Production ceiling (step 12) | ~$150 |
| Selection-set evals during production | ~$25 |
| Reserve | ~$25 |
| Final eval (step 13, off W&B) | $0 on W&B |
| **Total** | **~$217 of ~$218** |

- **Approve as is: recommended (research).**
- **Change it,** and say which line.

**10b. Run the second seed at all**
- **Run it: recommended (research).** It replaces the pilot, and it checks that 0.70 wasn't a lucky seed.
- **Skip it and save ~$12.** Production's first steps then become the smoke test, so a plumbing bug costs production budget, and the 20-step result stays a single seed.

**10c. When to run it**
- **After data and the red-team (steps 7-8),** as the research orders it.
- **Straight after step 6 and the dashboard (step 9), once a red-team pass on the ticket environment comes back clean: recommended (mine).** Data gets built while it runs, and the new task families get their own red-team before production. It costs the same and warns us earlier about plumbing bugs.

---

## Step 11. Baselines

**Type:** Spend · **Cost:** ~$5 · **Research:** §8.2

### What we run

On the 64-task selection set, about 130 episodes:
- **Base Qwen3.8-27B as an RLM** in environment v2. This is the step-0 number.
- **The base model reading the context directly,** where it fits in 262K tokens.

The 20-step checkpoints (both seeds) are the research's third baseline. They are optional here, and not included in the ~$5.

### Why it comes here

The stop rules and every claim compare against these numbers. Without the direct-reading number we can't claim the RLM setup helps.

### Done when

Both baselines are logged with full transcripts.

### Decisions for this step

**11a. Which model serves sub-calls once deployed**
- **The base model, as in training: recommended (mine; the research makes no call).** The root was trained against base-model sub-calls, so evals stay comparable and nothing shifts at deployment.
- **The trained LoRA.** That means one model to serve, but sub-calls then behave differently from training and need their own eval.

This is needed here because every eval from now on should use whichever model the harness will use.

---

## Step 12. Production run

**Type:** Spend · **Cost:** up to ~$150 + ~$25 evals; training GPU ~$25-60 of the ~$992 left · **Research:** §5.6, §8.5, §8.6, §9

### What we run

GRPO at 32 episodes per step, with every family starting at level 1. It runs until a stop rule fires or it reaches the $150 ceiling. The live dashboard (step 9) updates every step, so you and your friends can watch it. The learning rate is whatever the second seed decided. At 2e-5, entropy or reward-std collapse rolls the run back to the last good checkpoint and continues at 1e-5 (spec §5.2). That is ~80-180 steps, with ~120 a fair middle estimate. The selection set and capability check run every 10 steps, at ~$2.50 each.

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
- infra errors above 1%

**If money runs short:** cap training contexts at ~250K characters, or eval every 15 steps instead of 10. Cutting episodes per step is the last resort, because the 20-step run never tested it.

### Done when

A stop rule fires or the run reaches the ceiling, and the best checkpoint on the selection set is identified. A stopped run can resume later from its checkpoint if more credit appears.

### Decisions for this step

**12a. The $25 reserve**
- **I ask you before touching it: recommended (mine).** This follows your rule to ask before spending credit.
- **Spend it automatically when the last 2 evals were both new bests, then ask** (the research's rule). This saves a round-trip if you're away when the run hits the ceiling.

---

## Step 13. Final eval off W&B

**Type:** PC or GPU host · **Cost:** $0 on W&B · **Research:** §7.4, §8

### What we run

We push the best LoRA to your Hugging Face. Then we run the held-out test, the public benchmarks and the capability-retention check outside W&B:
- ~400 episodes per model, for a ±5-point interval
- 2-3 sampling seeds
- paired tests (McNemar or paired bootstrap)
- the base model and direct reading re-run in the same place

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
