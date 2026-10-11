# Step 6 results: training loop and monitoring

> **Step 6 of [production-build-plan.md](production-build-plan.md)** (research §5.3, §8.5, §8.6, §10.3, §10.4). Built and tested 2026-10-11 on Rishi's PC, on Windows and inside WSL2 with Docker. **No step 6 decisions were open.** Where the plan left a fork, I picked a default; §6 lists them.
>
> **Status: complete (confirmed by Rishi, 2026-10-11).** Committed and pushed to `claude/project-thread-2apfvb`.
>
> **No W&B credit was used.** Every test and dry run serves the model from `tests/fake_model.py` and trains on `tests/fakes.FakeBackend`, an in-memory stand-in for W&B serverless. The `.env` file was never opened. Nothing was installed. The only new process was Docker Desktop, started for the WSL test suite as in steps 3-5.
>
> Anything marked *(mine)* is my choice, not the spec's. Anything marked *(inferred)* hasn't been checked against the live system.

## Summary

**Step 6 is built, and the done-when check passes.** A dry run with a fake model client goes through steps, evals, a stop rule, checkpoint saves and resumes, with no W&B call. The regression suite is now **214 tests**. All of them pass in WSL with Docker, and one skips there, the tokenizer check from step 4, as before.

| Done-when check | Result | Evidence |
|---|---|---|
| A dry run with a fake model client, making no W&B calls, goes through steps, an eval, a stop rule, a checkpoint save and a resume | **Pass** | `tests/test_training.py`, 17 tests. The second seed runs 20 steps, then its 100-episode held-out eval. Production runs until the plateau rule stops it at checkpoint 30. Resumes are tested after the stop switch, after a crash during training, from an earlier checkpoint, and after the automatic learning-rate fallback. One test runs steps and a resume through real Docker sandboxes. |

A full production dry run (`scripts/train.py start --dry-run`) took **34 seconds for 42 steps and 5 evals**. Both families climbed from level 1 to level 5, and the run stopped on rule 4, saturation, keeping checkpoint 40 as its best (§4).

Everything the plan lists for step 6 is in:

| Plan item | Where |
|---|---|
| Step 1's settings: CISPO with the 1c clip, no reward std scaling, 8 × 4 episodes, LR 1e-5, temperature 1.0, KL to base logged but not in the loss | `config.TRAIN_ARGS`. Each train call passes `loss_fn="cispo"`, `epsilon=1.0`, `epsilon_high=4.0`, `scale_rewards=False`, `kl_penalty_coef=0.0`. A test checks every call. |
| An eval every 10 steps on the 64-task selection set, plus the capability check; the best checkpoint is saved and tracked | `TrainingRun._run_due_evals`. The capability runner is `rlm_rl/capability.py`. Its ~100 prompts are plan step 7's data, so until then the check is skipped with a notice. |
| Automatic stop rules from research §8.6; the script stops and says which rule fired | `rlm_rl/stoprules.py`, 16 tests on synthetic curves. The rule and its reason go in the results file, the notices and the console. |
| Every 5th checkpoint kept, plus the best; resume from the latest or the best; the config saved with each checkpoint | Snapshots go in `results/<run>/checkpoints/`, and `config.json` is rewritten at each kept checkpoint. Other checkpoints are deleted from the backend. Resume works from any kept checkpoint. |
| An inference spend counter with a hard ceiling | The client counts every response it pays for. The loop never starts a step or an eval that would cross its ceiling (§2.5). |
| Per-step metrics to W&B and `results/<run>.json` | `rlm_rl/monitor.py` covers every item in spec §9 and the plan's list, including entropy, KL to base, clip fraction, grad norm and `probs_corr` (§2.3). |
| Operator controls: a stop-after-this-step switch, and resume from any kept checkpoint | `scripts/train.py stop`, Ctrl-C, `--stop-after`, and `resume --from latest|best|<checkpoint>` (§3). |
| Truncated REPL output, so root transcripts fit ART's packed sequence length | Already pinned at 2,000 characters since step 3. The loop now logs each step's root transcript length (mean, p95, max), so the second seed can set `packed_sequence_length` for production (§7). |

The carry-overs in the brief are in too:
- **Limiter caps:** 30 in flight in total and 20 per model, unchanged from step 4.
- **Infra pause:** counted within each step for the second seed, and over a 10-step window in production.
- **Learning-rate rule:** the second seed runs at 1e-5. Production starts at 2e-5 only when you pass `--lr 2e-5`, after reading the second seed's checks, which the loop computes when it finishes. The fallback to 1e-5 is automatic and logged as a notice.
- **Claude audit:** the loop writes `audit.py`'s sample at every 10th checkpoint and at the last one, and reads the verdicts every step.

---

## 1. What was built

| Path | What it is |
|---|---|
| `rlm_rl/training.py` | `TrainingRun`: the resumable loop, with evals, audit samples, checkpoint snapshots and pruning, the stop switch, resume, the LR fallback and the spend ceiling |
| `rlm_rl/monitor.py` | Per-step metrics (spec §9), entropy, and KL to base |
| `rlm_rl/stoprules.py` | The five stop rules, the collapse check, and the second seed's learning-rate decision |
| `rlm_rl/capability.py` | The capability check: a JSONL prompt format, 10 kinds of machine check, and the runner |
| `rlm_rl/config.py` | `RunConfig`, the second-seed and production presets, and `pinned()`, everything spec §8 saves with a checkpoint |
| `rlm_rl/backend.py` | `ArtBackend`: train, list, delete and fork checkpoints, and log, through ART 0.5.20's serverless backend |
| `scripts/train.py` | `start`, `resume`, `stop` and `status`, live (`--live`) or against fakes (`--dry-run`) |
| `rlm_rl/client.py` | Gains a spend counter on every response, `chat()` for the capability check, and `prompt_logprobs()` for KL to base |
| `rlm_rl/evals.py`, `rlm_rl/tasks/__init__.py` | Gain the eval-set registry: `heldout-seed1` and the stand-in `selection-tickets-v1`, with its seeds kept in `private/` |
| `rlm_rl/transcripts.py` | Gains `extend()`, which writes a step's 32 lines with one flush to disk instead of 32 (each costs ~0.1 s in WSL2) |
| `tests/test_training.py` | 17 tests: the loop end to end, including through Docker |
| `tests/test_stoprules.py`, `test_monitor.py`, `test_capability.py` | 16, 7 and 4 tests |
| `tests/dryrun.py` | The dry-run harness: an oracle policy that improves with the checkpoint, the fake server and the fake backend |
| `tests/fakes.py`, `tests/fake_model.py` | `FakeBackend`. The fake server now renders prompts the way Qwen3.8's template does, and can vary replies and logprobs by checkpoint. |
| `.gitignore`, `README.md` | `results/*/pending/`, `capability/`, `abandoned/` and `results/dry/` are ignored. The Layout section lists the new files. |

## 2. How the loop works

### 2.1 One step

Step *n* samples from checkpoint *n*−1 and trains checkpoint *n*. Checkpoint 0 is the base model.

1. **Draw and roll out.** The loop draws 8 tasks at the controller's levels and runs 4 rollouts of each. Every episode goes to `results/<run>/transcripts/train/step<n>.jsonl`.
2. **Filter.** `filter_step` removes infra episodes, then small and all-equal groups (spec §5.4). The controller records the step.
3. **Measure.** KL to base comes from 4 rollouts re-scored under the base model. The step's metrics are computed.
4. **Commit the spend, then train.** The batch is saved to `pending/` first, so a crash during training can be resumed (§3.3). With fewer than 2 trainable groups the train call is skipped, and checkpoint *n* is checkpoint *n*−1.
5. **Log.** The step's row goes to `results/<run>.json`, and its trajectories and metrics go to W&B through ART's `model.log(split="train")`. A W&B logging failure becomes a notice and never stops the run.

### 2.2 Between steps

At the top of every loop pass, at the latest checkpoint *c*:

1. **Evals that are due and not yet done.**
   - Production: the selection set and the capability check at checkpoint 0 and every 10th. Checkpoint 0's results are step 11's first baseline and the frying rule's reference.
   - Second seed: the 100-episode held-out eval at checkpoint 20.
   - Only the selection set (or the second seed's final eval) logs trajectories through ART. The capability check logs metrics only, because ART keeps one val parquet per step (step 4 §5).
2. **The Claude audit's sample** at every 10th checkpoint and the last one (`audit.py checkpoint`). The loop doesn't wait for verdicts. It reads them on every pass.
3. **A snapshot** of every kept checkpoint, and a rewrite of `config.json` (§3.2).
4. **Pruning** of checkpoints that aren't kept.
5. **The checks**, in this order: the infra pause, the collapse check, the five stop rules, `max_steps`, the stop switch, and whether the next step fits under the ceiling.

### 2.3 What gets logged each step

Every row of `results/<run>.json`, and W&B as `rlm/<section>/<name>`:

| Section | Holds |
|---|---|
| `pass` | Pass rate with and without infra errors, the pass-rate histogram in quarters, groups, trainable groups, all-0, all-1 and other all-equal groups |
| `families` | Per family: level, levels drawn, episodes, pass rate, mean reward, the controller's window pass rate, and whether it's saturated. Plus `moves` |
| `reward` | Mean reward, exact accuracy, partial credit, the `near_blind` rate, submits, hacked episodes and hits by rule |
| `policy` | Entropy, KL to base (and k3), within-group reward std, grad norm, `probs_corr`, clip fraction, importance-ratio mean, p95 and p99, and the loss |
| `shape` | Sub-calls, budget-hit rate, turns, first-turn submits, turns with no code, REPL errors, timeouts, root and sub-call tokens, truncated turns, truncated REPL outputs, root transcript tokens (mean, p95, max), context size |
| `infra` | Infra rate and kinds, plus request stats: 429s, 5xx, retries and limiter peaks |
| `cost`, `time`, `tokens` | Step cost, KL cost, train spend so far; rollout, KL, train and total seconds; token totals |

The top of the file holds the run's status, stop reason, best checkpoint, spend and ceilings, learning-rate state, branches, kept checkpoints, evals, audits, each stop rule's status ("1 of 3 evals without a new best") and the notices (the operator log). A `live` heartbeat, refreshed every 15 seconds, says which step and phase are running. At ~3 KB a step, a 120-step run's file stays well under 1 MB. Plan step 9's dashboard reads this file.

**Entropy** (spec §5.2). ART's serverless metrics don't report it, so it is the mean of −logprob over every sampled root token of the step's scored episodes. The root calls already return those logprobs, so it's free.

**KL to base** (spec §5.3). Each step, 4 scored rollouts are re-scored under the base model with `prompt_logprobs`, one request per rollout carrying its whole conversation.
- **Why the tokens line up.** I checked Qwen3.8's chat template in the local Hugging Face cache. With `preserve_thinking` unset, it renders every earlier assistant turn with the same empty think block the policy generated after. So each turn's prompt plus its sampled tokens is an exact prefix of the whole conversation, and the policy's and the base model's logprobs match token for token.
- **What's logged.** The KL is the mean of (policy logprob − base logprob) over those tokens, as the spec says. The k3 estimator is logged beside it *(mine)*, because it is never negative and has lower variance.
- **When tokens don't line up.** A turn that doesn't match is skipped and counted.
- **Cost.** About 4 transcripts' worth of input tokens a step, roughly $0.01 at production sizes *(inferred)*.
- **Failures.** A failed re-score is counted and never stops the run.

### 2.4 Stop rules, the collapse check and the LR rule

| Rule | Fires when | Then |
|---|---|---|
| 1. Plateau | The last 3 selection evals were not new bests, and no family moved up a level in the last 30 steps | Stop, keep the best |
| 2. Overfitting | Two evals in a row where the mean training reward over the 10 steps before the eval rose and the selection set fell | Stop |
| 3. Frying | The capability check is more than 2 points below checkpoint 0's (only checks at least 90% complete are compared *(mine)*), or a collapse comes back after the LR fallback | Stop |
| 4. Saturation | Every family is at its top level with a pass rate above 70% | Stop |
| 5. Hacking | The Claude audit marked a win `hack`, or detectors fired on at least 5% of episodes over the last 10 steps and the rate is rising *(5% mine)* | **Pause**: you make the resume call |

**Collapse** (spec §5.2 step 3, the proposed thresholds). It fires on either signal:
- entropy below 70% of its highest value over the previous 10 steps, with at least 3 steps of history
- within-group reward std below half its previous-10-step average for 3 steps in a row, with at least 5 steps of history

What happens next depends on the run:

| Run | On a collapse |
|---|---|
| Production at 2e-5, first time | Roll back to the last good kept checkpoint and continue at 1e-5, logged as a notice |
| Production after that fallback | Stop rule 3 |
| Production started at 1e-5 | Pause, because the agreed rule has no lower rate *(mine)* |
| Second seed | A notice only, because its curves are what decide production's rate *(mine)* |

**The last good checkpoint.** Rollouts at step *s* come from checkpoint *s*−1. So the last checkpoint known to be good is the first bad step minus 2. The loop rolls back to the newest kept checkpoint at or before it.

**The second seed's decision** (spec §5.2 step 2). When the second seed finishes, the loop computes `lr_decision`:
- entropy over steps 16-20 is at least 80% of steps 1-5
- grad norm is never above 3× its median
- `probs_corr` stays above 0.98

If all three hold, production may start at 2e-5. Otherwise, or if data is missing, it stays at 1e-5. You still make the call, by passing `--lr`.

**Order.** When several things fire at once, a pause wins over a stop, and a stop wins over finishing. The others are logged as notices.

### 2.5 Spend

- **The counter.** The client adds the estimated cost of every response it receives, at the research's prices. The loop commits it to the results file after each step's rollouts and KL, and after each eval.
- **Interruptions.** The heartbeat records spend not yet committed, so a crash or Ctrl-C still counts it (`interrupted_usd`). A test checks this.
- **Ceilings.** These are hard stops, by split:

  | Run | Steps (rollouts and KL) | Evals |
  |---|---|---|
  | Second seed | $14 | $3 |
  | Production | $150 | $25 |

  The plan puts the second seed at ~$12 (10a). The $25 reserve isn't in production's ceilings.
- **The check.** Before each step, it compares the spend plus the largest of the last 3 steps' costs against the ceiling ($1 before any step is measured). Before each eval, it uses that eval's last cost, or 1.5× the recent cost per training episode.
- **At the ceiling.** If both of the last 2 evals were new bests, the stop message says the research would spend the reserve and that it's your call (12a). Resuming with `--ceiling-train 175` spends it.
- **Training GPU spend** is an estimate at $0.16 a step (research §7.1). W&B's billing page is the truth for both pots.

## 3. Operator controls, checkpoints and resume

### 3.1 Stopping

- `scripts/train.py stop <run> --reason "..."` writes `results/<run>/STOP`. The run finishes the step it's on, and that step's eval if one is due, then stops with rule `operator`. The switch is consumed when it takes effect.
- Ctrl-C does the same. A second Ctrl-C stops at once. Resume then redoes the unfinished step, and its spend stays counted.
- `--stop-after N` stops once step N is done.
- Every stop is resumable, including a stop rule. That's how a stopped run continues later if more credit appears (research §8.6).

### 3.2 Checkpoints

- **Kept:** checkpoint 0, every 5th, every evaluated one, the best and the latest. Each kept checkpoint gets a snapshot in `results/<run>/checkpoints/step<NNNN>.json`: the backend checkpoint it maps to, the learning-rate state, the controller state, and the pinned config.
- **`config.json`** is rewritten at the same time (spec §8). It holds:
  - the ART version and the base model
  - the chat template's hash (from the local Hugging Face cache, when it's there) and a fingerprint of how the server rendered the system prompt *(mine)*. The fingerprint comes free from the rollouts, and the loop posts a notice if it ever changes.
  - the bridge versions and prompt hash, the episode limits, the train call and the learning rate
  - the reward, detector and controller settings, with the controller's state
  - generator versions, a stand-in dataset manifest, the limiter and retry policy, the git commit, and spend
- **Pruned:** every other checkpoint of the current branch is deleted from the backend after the next one exists. Pruning never touches the latest checkpoint, a checkpoint 0, or any backend checkpoint the run didn't record. `--no-prune` turns it off. **Deleting on W&B can't be undone**, so this is the one place step 6 deletes anything, following spec §8's retention rule.

### 3.3 Resuming

- **From the latest checkpoint** (the default). The loop first settles two things:
  - **Spend lost to an interruption** is added to the totals.
  - **A step that was sent for training.**
    - If the backend has a new checkpoint, its training finished while the script was down. The loop records it and notes that its training metrics weren't captured.
    - If it doesn't, the loop refuses to guess, because the training job might still be running. `--retrain-pending` trains the saved batch. `--discard-pending` redoes the step from new rollouts.
- **From an earlier kept checkpoint** (`--from best` or `--from 40`).
  - The checkpoint is **forked into a new branch**, a new model named `<run>-b1`, so nothing is deleted.
  - The later steps' rows and logs move to `abandoned/`. The controller and learning rate come back from the checkpoint's snapshot.
  - Spend never rolls back.
  - The logical step numbers continue (41, 42, …) whatever numbers the backend gives the new branch's checkpoints. A test covers a backend that restarts its numbering after a fork.
- **Acknowledging the pause.** Resuming acknowledges whatever paused the run: audit findings so far, and the infra and detector windows.
- **The LR fallback** uses the same fork.

## 4. Dry runs

```bash
.venv-linux/bin/python scripts/train.py start --kind second_seed --run ss-dry --dry-run
.venv-linux/bin/python scripts/train.py start --kind production --run prod-dry --lr 2e-5 --dry-run --stop-after 12
.venv-linux/bin/python scripts/train.py resume prod-dry --dry-run --from best
.venv-linux/bin/python scripts/train.py status prod-dry --dry-run
```

- **The fake policy.** It knows each task's answer and submits it with a probability that rises with the checkpoint and falls with the level. So pass rates climb, families move up, entropy drifts down, and every panel has something to show.
- **Where it writes.** To `results/dry/`, which is git-ignored. The fake backend's checkpoints persist in a file there, so a dry run can be stopped and resumed across processes.
- **Speed.** A production dry run took 34 seconds for 42 steps and 5 evals. It stopped on saturation with checkpoint 40 best.
- **Not cost estimates.** Its dollar figures are tiny, because the fake model's replies are a few dozen tokens.

## 5. How it was tested

Run inside Ubuntu from the repo root:

```bash
.venv-linux/bin/python scripts/sandbox_ctl.py check
```

| File | Tests | What they cover |
|---|---|---|
| `tests/test_training.py` | 17 | **The second seed:** 20 steps, then its held-out eval (100 episodes), with every train call's settings, the frozen controller, pruning to 5/10/15/20, the snapshots and `config.json`, audit samples at 10 and 20, the LR decision, W&B logs and spend. **Production:** plateau stops the run at checkpoint 30 and keeps checkpoint 0. **Resumes:** after the stop switch; after a failed train call (blocked, then the saved batch is retrained); after training that finished while the script was down; after a discarded step; after an interruption, whose spend stays counted. **Rollback:** resume from checkpoint 5 forks a branch under a backend that restarts its numbering. **LR fallback:** entropy collapses at 2e-5, the run rolls back to checkpoint 10 at 1e-5, the collapse comes back, and rule 3 stops it. **Pauses:** an infra error in the second seed, an audit hack (and its acknowledgement on resume), and the spend ceiling. **Secrets:** no eval seed and no API key appears in any results file. The CLI end to end. One run through Docker. |
| `tests/test_stoprules.py` | 16 | Each rule's firing and non-firing cases, the collapse signals and their history needs, the last good checkpoint, and the LR decision passing, failing each check, and with missing data |
| `tests/test_monitor.py` | 7 | Entropy, KL alignment (and what a mismatch does), KL failures, reward std and the histogram, every spec §9 field, the spend counter |
| `tests/test_capability.py` | 4 | Every check kind, loading, and the runner with an infra failure |

- **In WSL with Docker:** 214 tests, 213 pass and 1 skips (the tokenizer check, as before), in about 8 minutes. The new loop tests are most of that; the suite took 2.5 minutes before step 6. No container was left behind.
- **On Windows:** 178 tests run, 162 pass and 16 skip because they need Docker or a Unix socket.
- **Existing tests:** none changed. The fake server's new rendering is a drop-in.

**Two fixes from the WSL runs:**
- **A test with too thin a margin.** The plateau test once stopped at checkpoint 40 instead of 30, because one eval beat checkpoint 0 by noise. Its fake policy now gets worse too fast for that to happen.
- **Slow loop tests.** They took 13 minutes in WSL. Each transcript line's flush to disk costs ~0.1 s there, and each audit sample started a new Python process, which takes seconds of imports from the Windows drive. A step's lines now share one flush, and the audit runs in the same process. Eval transcripts still flush each episode as it finishes, as step 4 designed, so a crash mid-eval loses nothing.

## 6. Defaults I picked

| Fork | Default |
|---|---|
| Where the loop lives | `rlm_rl/training.py`, with `scripts/train.py` as a thin command line. Spec §10 put the loop in the script, but tests can't import a script. |
| Step numbering | Step *n* samples from checkpoint *n*−1 and trains *n*, so "training steps 1-10" feed checkpoint 10, as `audit.py` already assumes |
| A step with fewer than 2 trainable groups | No train call; checkpoint *n* is checkpoint *n*−1 |
| An infra pause | The step still trains its clean groups, then the run pauses |
| The capability check's decoding | Temperature 0 *(mine)*, so a 2-point drop isn't sampling noise. It has no earlier numbers to stay comparable with. |
| The capability set | A JSONL file with machine checks. Step 7 writes it. Until then the check is skipped with a notice. |
| The production selection set | `selection-tickets-v1`, 64 ticket tasks (32 per family across levels 1-5) from private seeds. It stands in until step 7's set, which adds a held-out family. |
| Evaluating checkpoint 0 | Production does, which gives step 11's first baseline and rule 3's reference |
| Resume from an earlier checkpoint | A fork into a new branch, never deleting the later checkpoints |
| Pruning | On, per spec §8, only on the current branch. Abandoned branches keep their checkpoints. |
| A training job that may still be running after a crash | The resume refuses and says what to do |
| A collapse at 1e-5 in production | Pause for you |
| A collapse in the second seed | A notice only |
| Hack-rate threshold for rule 5 | 5% of scored episodes over 10 steps, and rising |
| Ceilings | Second seed $14 + $3; production $150 + $25 |
| `packed_sequence_length` | Not passed, as in the 20-step run, so the trainer fits the batch *(inferred)*. Lengths are logged for the second seed to pin it. |
| Live runs | Need `--live`. Without it, or `--dry-run`, the script refuses to start. |

## 7. What changes the spec or plan

For the planning thread, which owns both files.

1. **Spec §10, code layout:**
   - The loop is `rlm_rl/training.py`, and `scripts/train.py` is its command line.
   - Add `rlm_rl/monitor.py`, `stoprules.py`, `capability.py`, `backend.py`, `config.py` (now built), `tests/dryrun.py`, and `FakeBackend` in `tests/fakes.py`.
   - `scripts/eval.py` is still unbuilt; it belongs to step 11.
2. **Spec §5.1:** `packed_sequence_length` stays unset for the second seed. Production's value comes from the second seed's measured `root_transcript_tokens` (max and p95).
3. **Spec §5.2:**
   - Entropy is defined as in §2.3 above.
   - The collapse check gains its details: the peak over the previous 10 steps with at least 3 points of history, at least 5 points for the reward-std rule, and the last-good-checkpoint rule.
   - What happens at 1e-5 in production (pause), and in the second seed (a notice).
4. **Spec §5.3:** KL to base is one request per rollout carrying the whole conversation, and it relies on the template keeping empty think blocks in history. Add the k3 estimator and the misalignment count. Add a *(verify)* row in §11 for W&B accepting such a request (§8 below).
5. **Spec §8:**
   - The checkpoint snapshot's contents, and that pruning only touches the current branch.
   - Resuming from an earlier checkpoint forks a new model; nothing is deleted.
   - The template fingerprint.
6. **Spec §9:** the field names in §2.3 above, plus root transcript tokens, truncated REPL outputs, the KL's k3 and misaligned turns, and the `live` heartbeat.
7. **Research §8.6 and plan step 12, the stop rules:** the definitions in §2.4 above.
   - Rule 2 compares the mean training reward over the 10 steps before each eval.
   - Rule 3 compares only capability checks that are at least 90% complete.
   - Rule 5 pauses rather than stops, at 5% and rising, or on any audited hack.
   - On the reserve (12a): the run stops at its ceiling and says when the research's rule would spend the reserve, and resuming with a higher ceiling spends it.
8. **Plan step 10, the second seed:**
   - The command is `scripts/train.py start --kind second_seed --run second-seed-v1 --live`.
   - When it finishes, the results file holds `lr_decision`.
   - It's also the first live test of the KL request and the thinking checks under the loop.
9. **Plan step 9, the dashboard:**
   - Its data is `results/<run>.json` (§2.3), and `scripts/train.py --dry-run` produces realistic files to build it against.
   - The `live` heartbeat gives the run header's "last update".
   - `stop_rules` gives each rule's status line, and `notices` is the operator log.
10. **Plan step 7:**
    - The capability set's format is in `rlm_rl/capability.py`.
    - The production selection set replaces `selection-tickets-v1` by name, in `evals.eval_set` and `config.production`.
    - New families need nothing else from the loop.

## 8. Known limits, and what a live run should check first

| Item | Why it's open | How to check |
|---|---|---|
| **Serverless training continues from a forked checkpoint** *(inferred)* | Resume from an earlier checkpoint and the LR fallback depend on ART's `_experimental_fork_checkpoint`. ART documents it for serverless, but no run has used it. | After the second seed finishes: `resume --from 15 --stop-after 16`. That's one extra step, about $0.50. If training continued from the fork, step 16's KL to base matches the original step 16's. If it started from the base model, KL drops to ~0. **I'd suggest adding this to step 10. It spends credit, so it's your call then.** |
| **W&B accepts the KL request** *(inferred)* | It sends a conversation that ends with an assistant message, with `prompt_logprobs` and `max_tokens=1`. vLLM accepts this, but W&B's gateway hasn't been tried. | The second seed's first step. A rejection shows as `kl.failed`, and the run continues. |
| **Checkpoint list and delete** | They use ART's internal serverless client (`backend._client.models.checkpoints`), the calls `delete_checkpoints()` itself uses. That's fine with ART pinned at 0.5.20. | The second seed's pruning, visible in W&B's artifact list |
| **The capability check has no prompts yet** | Step 7 | Until then, rule 3 only watches for a repeat collapse |
| **The selection set is a ticket stand-in** | Step 7 | Replaced by name |
| **Spend is an estimate** at the research's prices | W&B's billing page is the truth | Plan step 9's done-when compares them after the second seed |
| **Abandoned branches keep their checkpoints** | Pruning only touches the current branch | Delete them by hand if storage gets tight |

## 9. Commands

```bash
.venv-linux/bin/python scripts/sandbox_ctl.py check                                       # all 214 tests, inside Ubuntu
.venv-linux/bin/python scripts/train.py start --kind second_seed --run second-seed-v1 --live   # plan step 10 (spends ~$12)
.venv-linux/bin/python scripts/train.py stop second-seed-v1 --reason "..."                # stop after the current step
.venv-linux/bin/python scripts/train.py resume second-seed-v1 --live [--from best|N]      # continue
.venv-linux/bin/python scripts/train.py status second-seed-v1                             # where it stands
.venv-linux/bin/python scripts/train.py start --kind production --run prod-dry --lr 2e-5 --dry-run   # no W&B, no credit
```
