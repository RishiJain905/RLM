# Step 9 results: live dashboard

> **Step 9 of [production-build-plan.md](production-build-plan.md)** (research §2.5, §8.5, §9). Built and tested 2026-10-11 on Rishi's PC, on Windows, against step 6's dry-run harness.
>
> **Decisions applied (Rishi, 2026-10-11):**
> - **9a:** a Claude session on the PC republishes the dashboard after each step.
> - **9b:** Rishi invites friends from the page's Share menu.
> - **9c:** the page shows sampled transcripts.
>
> Rishi also asked for updates every step, or every 5 steps, rather than every 10, at whatever rate W&B's data actually lands. §2 says what lands when and what cadence I reached.
>
> **The dashboard:** [claude.ai/artifact/Jrm5PZd6VQNJT7cpkM9ou3](https://claude.ai/artifact/Jrm5PZd6VQNJT7cpkM9ou3). It is private until Rishi shares it. It currently shows the paced dry run from §5, marked as a dry run.
>
> **No W&B credit was used.** Every run here used `tests/fake_model.py` and the fake backend. The `.env` file was never opened. Nothing was installed.
>
> **Status: complete, committed and pushed to `claude/project-thread-9nnh3n` with Rishi's approval (2026-10-11),** except the billing comparison, which needs the second seed (step 10).
>
> Anything marked *(mine)* is my choice, not the plan's. Anything marked *(inferred)* hasn't been checked against the live system.

## Summary

**The dashboard is built, published and tested, and it updates after every training step.** A paced 14-step production dry run got its own new page version after each of its 14 steps. Each version went up 7-26 seconds after the step's results landed (9 s median). Rishi watched it on his phone, saw the new steps arrive, and could share it. Every build passes a scan for keys, seeds and eval items before it can be published.

**Cadence: every step.** The training loop writes each step's metrics once, when the step ends, and W&B gets the same numbers at the same moment. Nothing finer exists to show (§2). So the page updates once per step, plus once when an eval starts, when it ends, and whenever a notice or the run's status changes.

| Done-when check | Result | Evidence |
|---|---|---|
| A dry run of the training loop (step 6's fake model) updates every panel each step | **Pass** | Paced production dry run `dash-live1` (36 s median step): versions 2-21 of the artifact, one per step plus the evals and the stop (§5). Every panel is drawn from the same per-step data, so every panel moves with each version. |
| A second browser with the page open sees new steps arrive without reloading | **Pass** (Rishi's phone) | Rishi had the page open on his phone during the run and confirmed it showed him the new steps. The app's own browser pane isn't signed in to claude.ai, so I couldn't watch a second view myself (§5.2). |
| After the second seed, the cost panel matches W&B's billing page | **Not yet: step 10** | It needs a live run. The comparison is in the step 10 runbook (§7). |
| No key, seed or benchmark item appears in the page or its data | **Pass** | Every build runs `dashboard.check()` and is written only if it passes. The deep check on the final page covered 248 private seeds, 294 eval tasks (selection-v1, heldout-test-v1 and the run's own evals) and the 100 capability prompts: no findings (§4). |

The new test file, `tests/test_dashboard.py`, has 14 tests. They all pass on Windows, and so does the full Windows suite: 224 tests, 16 skipped for Docker as before (§6).

---

## 1. What was built

| Path | What it is |
|---|---|
| `rlm_rl/dashboard.py` | Reads `results/<run>.json` and builds the page's data from an allowlist. It samples training rollouts, embeds the data in the page, and holds the check that scans every build for secrets. Standard library only. |
| `rlm_rl/dashboard_page.html` | The page: one self-contained HTML file with its CSS, its JavaScript and an empty slot for the data. Charts are hand-drawn SVG, with no chart library. |
| `scripts/dashboard.py` | `build`, `check` and `wait`: the publish loop's command line (§3) |
| `scripts/train.py` | Gains `--latency` for dry runs. It slows the fake server to a live run's pace, which is how §5 was tested. |
| `tests/test_dashboard.py` | 14 tests (§6) |
| `.gitignore` | Adds `results/dashboard/`, which holds the built page, its sample cache and the publish state, and `.playwright-mcp/` |
| `README.md` | The Layout section lists the new files |

### 1.1 What's on the page

The panels follow the plan's list, which follows MiMo-V2.6's metrics, benchmarks and notices layout. Each panel's label is its W&B namespace (`rlm/pass`, `rlm/policy`, and so on), so a friend can find the same number in W&B.

| Panel | Shows |
|---|---|
| Run header | Run name and kind; a "dry run" badge when it is one; status (running, paused, stopped, finished, crashed) with the rule and reason; step and checkpoint; learning rate; what's running now ("step 43 training on W&B, due ~14:09"); when the page was built, ticking ("4 min ago") |
| Overview | Training pass rate with a sparkline; the best selection accuracy against the base model; spend this run; stop rules fired or worth watching; each family's level |
| Cost | Meters for steps against the $150 ceiling, evals against $25, the W&B inference pot (an estimate from the ~$218 left on 2026-10-11) with the reserve's status, and the training GPU pot. Also cost per step (rollouts and KL), spend so far, and steps left at the recent rate. |
| Pass rates and difficulty | Pass rate with and without infra; each family's 3-step window pass rate against the 70% and 20% lines; each family's level over time; trainable, all-0, all-1, other-equal and too-small groups per step; the correct-per-group histogram over the last 10 steps; a heat map of pass rate by family and level over the whole run |
| Policy health | Entropy with its collapse line (70% of the 10-step peak); within-group reward std with half its 10-step mean; KL to base (k1 and k3); grad norm with 3× its median; `probs_corr` with the 0.98 floor; clip fraction; importance ratio (mean, p95, p99); learning rate; loss |
| RLM behaviour | Sub-calls against the budget of 30; budget-hit rate; turns against the cap of 12; first-turn submits; REPL errors, timeouts, turns without code and cut-off outputs per episode; transcript length (mean, p95, max); output tokens; context size; turns cut at the token cap |
| Reward and cheating | Exact accuracy and mean reward; partial credit and near-blind answers; hack-detector hits by rule; submit rate; the Claude audit's clean share per checkpoint |
| Infra and timing | Infra rate against the 1% pause line; 429s, 5xx and retries; step time split into rollouts, KL and training; tokens per step; requests in flight against the cap of 30 |
| Evals | Accuracy by checkpoint with ±1 SE bars and the best checkpoint ringed; the 20-step run's baselines for the second seed (base 0.50, GRPO 0.70, RLSD 0.59); the capability check against the frying line; per-family accuracy at checkpoint 0 against the latest |
| Stop rules | Each of the five rules with its state and status line; the learning-rate rule and any fallback; the second seed's decision on production's learning rate; the infra pause |
| Notices | The operator log, newest first, filterable by interventions, warnings, evals and audits, and level moves |
| Sampled transcripts | Training rollouts (§1.2), expandable, filterable by win, failure and flagged |
| Table view | The last 15 steps' main numbers, and a button that copies every step as CSV |

**How it behaves.** Every chart shares one step axis, so hovering any chart puts a crosshair on that step in all of them, with a tooltip on the chart under the pointer. The arrow keys do the same from the keyboard. Rolled-back steps stay on the main charts as faint lines, with a rollback marker. It has light and dark themes and works at phone width. When a new version arrives in an open page, a note says which steps arrived, and open transcripts and filters stay as they were.

**Flags.** Each panel has a "Flag this" button that opens claude.ai's own comment box anchored to that panel. The page declares only the composer-only comments capability, so it never posts anything itself and never asks for consent. Where the viewer can't comment, the button stays hidden and the shell's comment mode still works.

### 1.2 Sampled transcripts (9c)

- **From training rollouts only** *(mine, per the brief)*. Eval transcripts are never read for the page. The selection set is reused at every eval, and the held-out sets are touched once, so their tasks stay off a page that goes to people outside the project. The plan's 9c wording said "from each eval"; this keeps its intent, wins and failures with their detector flags, without showing eval items.
- **What's sampled.** Each step gives one win, one failure, and up to two rollouts a detector flagged or that hit infra. The pick is a hash of the episode key, so it's stable and unbiased without a random generator.
- **What's kept.** Every sample from the newest 6 steps, the samples from step 1 and every eval checkpoint (every 10 steps) for the whole run, and the last 24 flagged rollouts. Samples from rolled-back steps are dropped.
- **Cut short.** Each sample keeps:
  - the question and the context's first 3 lines
  - the first 3 and last 3 turns, each with its plan, code and REPL output
  - the first 3 sub-calls
  - the submitted answer, the gold answer, the reward, its components and the detectors that fired

  Every text field is capped, and every one passes through the same scrubber, which removes URLs, local paths, checkpoint names and long tokens.

## 2. How often data lands, and the cadence reached

| Source | What it gives | How often |
|---|---|---|
| `results/<run>.json` | The step's whole row of metrics (spec §9), written when the step ends | **Once per step** |
| The same file's `live` heartbeat | Which step and phase is running (rollouts, KL, training, eval, capability) | Every 15 s, but metrics don't change with it |
| Evals and notices in the same file | Each eval's result when it ends; each notice when it happens | Every 10 steps (evals); whenever something happens (notices) |
| W&B, through ART's `model.log(split="train")` | The same numbers as the row, logged right after the row is written | Once per step |
| W&B's training job | A `gradient_step` event for each gradient step inside one train call | Several per step, but ART 0.5.20 averages them into one metrics dict per train call (`art/serverless/backend.py`), so the loop never sees them |

So **one step is the finest grain any metric has**, on W&B as much as here. Five-step batching would only make the page later. Mid-step progress exists only as the phase, and the page shows it: each build waits 3 seconds after a step ends, so it catches the next step starting ("step 43 rolling out, due ~14:09").

**What the loop publishes:**
- every completed step
- an eval starting and an eval ending
- every new notice, audit verdict or status change (paused, stopped, finished, crashed)
- a run whose loop stopped writing its state for 5 minutes, once, so the page can say it went quiet

The loop keeps at least 20 seconds between publishes *(mine)*. Steps that land faster than that share one version, and every step is still drawn.

**At production's pace.** The 20-step run's steps took a median of 290 s, and production's will likely be longer *(inferred)*. Each step then gets its own version, about 10 seconds after it lands, and an open page picks the version up within seconds. A publish costs the Claude session about two tool calls, and no W&B credit.

## 3. How it stays live (9a)

An artifact page can't fetch from W&B or from this PC, and the training script can't publish an artifact. So a Claude session on the PC runs a loop beside the training run:

```
python scripts/dashboard.py wait <run>          # blocks until there's something new, builds the page, exits
  -> the session publishes results/dashboard/page.html to the dashboard's URL
python scripts/dashboard.py wait <run> --ack    # records that publish, then blocks again
  -> ...
```

`wait` exits with one line that says what to do:

| Line | Means | The session |
|---|---|---|
| `PUBLISH ... check=passed` | A new page is built and passed the check | Publishes it |
| `STALE ... heartbeat_age=...` | The run says it's running but hasn't written its state for 5 minutes, so it may have died silently | Publishes it, so the page shows "not responding", and tells Rishi |
| `BLOCKED ... check=FAILED {...}` | The check found something. Nothing was written, and the last good page stands. The line names the categories and counts, never the match. | Doesn't publish, and tells Rishi |
| `TIMEOUT` | Nothing new within `--timeout` | Runs `wait` again |

**Why `wait` and not a monitor.** It runs as a background command that exits once per event, so the session wakes once per step. It has no 30-minute expiry and no event-rate limit. Rishi sees one version per step.

**When the session isn't running,** nothing republishes. The page itself then says "No new data for N minutes" once it is more than 15 minutes, or three typical steps, older than its build, so a friend can tell.

**Builds are fast.** A build takes about 0.3 s on Windows. It reads the state file and only the newest step's training transcripts, and it caches the samples per run in `results/dashboard/cache-<run>.json`.

## 4. What never reaches the page

The page carries an allowlist of the state, never a copy. These are left out:
- model and checkpoint names, which carry the W&B entity and project
- task ids, seeds and context hashes
- the prompt hash
- the audit sample's file paths
- per-model request peaks, which are keyed by checkpoint name
- the run config's W&B project and capability file path

Free text (notices, stop reasons, rule texts and transcripts) goes through the scrubber.

`dashboard.check()` scans every build, in the raw HTML and in the decoded strings of its embedded data, so JSON escaping can't hide a match. It reports counts only:

| Looks for | How |
|---|---|
| Keys and tokens | Any hex run of 32+ characters (W&B keys are 40), key variable names, common token prefixes (`sk-`, `hf_`, `ghp_`, `wandb_v1_` and others), bearer headers, private key blocks, and the value of any key-like variable already in the process's environment. The `.env` file is never read. |
| Seeds | Any integer of 13+ digits, and every seed in `private/seeds/` and the dry run's `results/dry/private/seeds/` |
| Checkpoint names and local paths | `wandb-artifact://`, `entity/project/model:stepN`, Windows and WSL paths |
| Eval tasks | Every eval task the run has logged, keyed by its question **and** its context's first lines. A question alone isn't enough, because questions are templated: two ticket tasks can ask the same thing of different logs. The first version of the check blocked the page on exactly that, so the test now pins it. |
| Deep check (`check --deep`) | Also builds selection-v1 and heldout-test-v1 from their private seeds, and loads the 100 capability prompts |

**The final page's deep check** read 248 seeds, 294 eval tasks and 100 prompts, with no findings.

## 5. The paced dry run

```bash
.venv/Scripts/python.exe scripts/train.py start --kind production --run dash-live1 --lr 2e-5 --dry-run --latency 6 --stop-after 14
```

This is step 6's loop on the production families, with the fake server at about 6 s a request. It gave a 36 s median step, with evals at checkpoints 0 and 10 and the audit sample at 10. It stopped on the operator rule after step 14. Every step of the loop in §3 ran for real, against the published artifact:

| Version | Published for | Seconds after the data landed |
|---|---|---|
| 1 | The finished 42-step dry run `dash-dry1`, the first publish | n/a |
| 2-4 | Run start, then the checkpoint 0 selection eval and capability check | n/a |
| 5-14 | Steps 1-10 | 21, 13, 9, 8, 8, 9, 10, 7, 8, 7 |
| 15-17 | The checkpoint 10 eval starting, its result, and the capability check | n/a |
| 18-21 | Steps 11-14 (21 also carries the stop) | 26, 19, 12, 8 |

**Latency.** The first step waited on the session's first round trip. Steps 11 and 12 waited out the 20-second gap after the capability check's publish. Every other step was up within 7-13 s.

### 5.1 What I checked on the page

The template's script passes a syntax check. Rendered locally in a headless browser, at desktop width in light and dark, the page had no script errors. Hovering one chart put the crosshair on the same step in all 35 step charts, and transcripts opened. One look at the charts led to these fixes:
- the pot meters now show money used against the pot, not money left, so a full bar always means spent
- a chart that mixed a rate and a count on one axis is now two charts
- the first series is drawn on top
- the dark theme's heat-map text is now dark on the light cells
- "1 sub-calls" now reads "1 sub-call"

### 5.2 The second viewer

The app's browser pane showed claude.ai's sign-in page for the artifact, and I don't sign in for anyone. Rishi watched the run on his phone instead, and confirmed it showed the new steps arriving and could be shared. An invited friend's view is still untested (§10).

## 6. How it was tested

```bash
.venv/Scripts/python.exe -m unittest tests.test_dashboard -v
```

All tests in `tests/test_dashboard.py` run on one 12-step production dry run (the ticket stand-in, with fakes):

| Test | Covers |
|---|---|
| `PageTest` (5) | Every step and panel has its data; a build round-trips and passes the check, with seeds and eval tasks actually read; no checkpoint name, path, hash or seed reaches the page; samples are training rollouts, cut short, bounded per step, and read once; rolled-back steps drop their samples |
| `CheckTest` (5) | A key in a notice is scrubbed, and a planted one is caught without being echoed; a planted private seed is caught; an eval task is caught, but a shared question template is not; text can't break out of the data's script element; a failed check writes nothing |
| `SignatureTest` (2) | A heartbeat alone isn't news, but a step, notice, status or eval phase is; the scrubber |
| `WaitTest` (2) | PUBLISH, then an ack and a TIMEOUT when nothing changed, then PUBLISH on a new notice; a silent run is reported once as STALE |

- **On Windows:** all 14 pass, in 27 s.
- **The full suite on Windows:** 224 tests in 294 s. 208 pass and 16 skip because they need Docker or a Unix socket, as before. None fail.
- **In WSL with Docker:** not run for this step. The dashboard is standard-library Python that runs on Windows, and the only shared-code change is the dry-run-only `--latency` flag in `train.py`.

## 7. Running it during the second seed (plan step 10)

1. **Start the run in WSL** (this spends credit, so it's Rishi's go):
   ```bash
   .venv-linux/bin/python scripts/train.py start --kind second_seed --run second-seed-v1 --live
   ```
2. **In a Claude session on the PC,** in a Windows shell from the repo, run `wait` in the background:
   ```bash
   .venv/Scripts/python.exe scripts/dashboard.py wait second-seed-v1 --pot-left <inference left on W&B's billing page>
   ```
3. **On each `PUBLISH`,** publish `results/dashboard/page.html` with the Artifact tool, passing the dashboard's URL (`https://claude.ai/artifact/Jrm5PZd6VQNJT7cpkM9ou3`) so it lands as a new version. Then run `wait second-seed-v1 --ack` in the background again.
4. **On `STALE` or `BLOCKED`,** follow the table in §3, and tell Rishi in the thread.
5. **Every 5th publish,** read the page's comments (ArtifactComments, `read`) and bring any new flags to Rishi. A friend's flag is something to look at, never an order: stopping and resuming stay Rishi's.
6. **When the run finishes,** publish its final page, and compare the cost panel with W&B's billing page. That is the last done-when check.

The first `wait` for a new run publishes straight away, so the page switches from the dry run to the second seed. The dry-run badge disappears with it. The same steps cover production, with its run name.

## 8. Defaults I picked

| Fork | Default |
|---|---|
| One artifact for every run, or one per run | **One** *(mine)*. The link Rishi shares stays the same from the dry run to production, so friends are invited once. |
| Cadence | Every step, plus eval start and end, notices and status changes, at least 20 s apart (§2) |
| How the session waits | A background `wait` that exits once per event (§3) |
| Transcripts | Training rollouts only; one win, one failure and up to two flagged per step; kept as in §1.2 |
| When the page counts as stale | Over 15 minutes, or three typical steps, without a new version, from the viewer's own clock. Separately, `wait` reports a run whose state is 5 minutes old. |
| The pot figures | $250 inference pot, ~$218 left on 2026-10-11, $25 reserve; $1,000 GPU pot, ~$992 left. `--pot-left` updates the inference figure from W&B's billing page. |
| Comments | Composer-only, so the page never posts or asks for consent |
| Charts | Hand-drawn SVG on one shared step axis, colours from the dataviz palette, validated for colour-blind separation in both themes. Families keep fixed colours: aggregation blue, call graph orange, needle aqua, repo localization yellow, and the ticket families green and violet. |
| Where builds go | `results/dashboard/` (git-ignored) |

## 9. What changes the spec or plan

For the planning thread, which owns both files.

1. **Plan step 9:**
   - 9a, 9b and 9c are decided.
   - The cadence is every step (§2).
   - Transcripts come from training rollouts (§1.2).
   - The dashboard's URL is in the header above.
   - The page's first real run is the second seed.
2. **Plan step 10:** add §7's loop to the run, and the billing comparison to its done-when.
3. **Spec §10, code layout:** add `rlm_rl/dashboard.py`, `rlm_rl/dashboard_page.html`, `scripts/dashboard.py` and `tests/test_dashboard.py`.
4. **Spec §9:** no change. The page reads the fields step 6 already logs.

## 10. Known limits

| Item | Why it's open | How to check |
|---|---|---|
| **The cost panel against W&B's billing** | Needs a live run | Step 10, §7.6 |
| **An invited friend's view** | Only Rishi's own phone has seen it. Whether claude.ai shows the Flag button to someone invited by email isn't documented *(inferred: it may not)*. Comment mode works for anyone who can comment. | Rishi invites one friend before step 10 |
| **Many versions** | Production makes ~150-250 versions (a step each, plus evals). I found no version limit in the artifact docs *(inferred)*. | The first production publishes |
| **The session has to stay up** | If the Claude session stops, so do updates. The page then shows its age and the "no new data" banner. | Resume the loop from §7 in any session on the PC |
| **Page size** | 244 KB at 42 steps. About 3.5 KB a step plus at most ~110 samples puts a 150-step run near 1-1.5 MB *(inferred)*, far under the 16 MB limit. | The build line prints the size |
| **Real numbers** | Every chart has only seen the fake policy's numbers | The second seed |

## 11. Commands

```bash
.venv/Scripts/python.exe scripts/dashboard.py build dash-live1 --dry-run                     # one build and check
.venv/Scripts/python.exe scripts/dashboard.py wait second-seed-v1 [--ack] [--pot-left 205]     # the publish loop (§3, §7)
.venv/Scripts/python.exe scripts/dashboard.py check results/dashboard/page.html --run second-seed-v1 --deep   # the deep check
.venv/Scripts/python.exe scripts/train.py start --kind production --run dash-live1 --lr 2e-5 --dry-run --latency 6 --stop-after 14   # a paced dry run
.venv/Scripts/python.exe -m unittest tests.test_dashboard -v                                  # the 14 tests
```
