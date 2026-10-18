# Step 12 runbook: the production run

> **Status (2026-10-17):** ready. Everything before the run is done at $0; Rishi starts the run. The plan's step 12 is in [production-build-plan.md](production-build-plan.md#step-12-production-run); this page is how to run and watch it.
>
> **Live dashboard:** https://claude.ai/artifact/HqtrsXwnHQcJKud7GnUHMm (its own page; the step 10 and step 11 pages are frozen, and copies are in this folder as [step-10-dashboard.html](step-10-dashboard.html) and [step-11-dashboard.html](step-11-dashboard.html)). It is private until it's shared from its Share menu.

## 1. Before you start

- **Docker Desktop is running and Ubuntu can reach it:** `docker ps` works in Ubuntu. If Ubuntu says docker "could not be found in this WSL 2 distro", or can't connect to the daemon, quit Docker Desktop, run `wsl --shutdown` in PowerShell, start Docker Desktop again and wait for it, then open Ubuntu. That is what fixed it on 2026-10-17: Docker Desktop had started while Ubuntu was already running, and its link into Ubuntu never came up.
- **The PC stays on for ~2-3 days:** plugged in, sleep off, nothing else heavy on D:. The optional drive settings are in the plan (step 12, item 3).
- **Budget:** $191.01 of W&B inference was left on 2026-10-17. The run's ceilings are $146 for steps and $34 for evals, both hard stops, which keeps the $10 reserve and ~$1 of margin untouched.

## 2. Start it

In Ubuntu (`wsl -d Ubuntu` from PowerShell), inside tmux so the run survives the window closing:

```bash
tmux new -s prod
```

```bash
cd /mnt/f/Personal/RLM/RLM
```

```bash
.venv-linux/bin/python scripts/train.py start --kind production --run prod-v1 --lr 2e-5 --live
```

Detach with Ctrl-b then d; come back with `tmux attach -t prod`.

The run uses prompt v3, root thinking at `medium` (sub-calls off), 30 turns for edit episodes, `selection-v3` as the selection set, and learning rate 2e-5 with one automatic fall back to 1e-5 after a collapse. It trains until a stop rule fires or a ceiling is reached.

## 3. What a healthy start looks like

1. **At once:** a "production run started at learning rate 2e-05, ceilings train $146, eval $34" line. A "removed N leftover container(s)" line may come first.
2. **Checkpoint 0's evals, about 40 minutes:** the selection set (64 episodes, ~$2.7) should land near **52%** (step 11's rerun of the base model in this set-up: 51.6% ± 6.2), then the capability check (100 prompts, a few cents).
3. **Step 1, about 15-35 minutes** (up to an hour when a Go task's tests are slow):
   - **cost** about $1.3-1.6 a step;
   - **pass rate** roughly 40-65% with every family at level 1, and 4-7 of the 8 groups trainable;
   - **KL** a small positive number (step 10's first step read 0.0016). The dashboard's KL chart says how many re-scored turns lined up: **"8 re-score requests, 0 misaligned turns, 0 failed" is healthy.** This is the first live check that W&B renders a continued thinking reply the way it was sampled. If every turn is misaligned, KL reads n/a; training is unaffected, since KL is in no loss and no stop rule, but tell me and we'll look;
   - **infra** 0 infra episodes. HTTP 500 retries are normal (about a quarter of tries in step 11), as long as they recover.

   **What prod-v1's first step showed (2026-10-18):** checkpoint 0 at 50.0% on `selection-v3` and 76% on the capability check ($2.63); step 1 at 56% with 4 of 8 groups trainable, $0.84, 11 minutes, 0 infra episodes, probs_corr 0.997. All 8 KL re-score turns came back misaligned. The cause: W&B's chat template doesn't parse `<think>` tags inside an assistant message's content, so it put an empty think block in front of them. Rishi had the run paused after step 2 (stop switch, nothing lost) to fix it. `monitor.thinking_turn_messages` now sends the thinking in the message's `reasoning_content`, and probes on the base model ($0.01 in all) lined up token for token on a first turn and on later turns with real history. The fake model server now renders thinking the way W&B's template does, so the test catches the old form. The run resumes from checkpoint 2 with its checkpoint 0 evals kept; steps 1 and 2 have no KL value.
4. **Every 10 steps:** the selection set and the capability check again (~40 minutes, ~$3), and a Claude audit sample.

**What pauses or stops it** (all in the plan): more than 1% infra episodes over 10 steps (pause), detectors firing on more than 5% of episodes and rising (pause), a hack verdict in the audit (pause), a collapse (one automatic rollback at 1e-5, then a stop), plateau, overfitting, frying, saturation, or a ceiling. Every stop keeps the best checkpoint and can be resumed.

## 4. Stopping and resuming

All from the repo folder in Ubuntu:

| To | Run |
|---|---|
| Stop after the current step | `.venv-linux/bin/python scripts/train.py stop prod-v1 --reason "why"` (or Ctrl-C once; twice stops at once) |
| See where it is | `.venv-linux/bin/python scripts/train.py status prod-v1` |
| Continue from the latest checkpoint | `.venv-linux/bin/python scripts/train.py resume prod-v1 --live` |
| Continue from the best, or a kept checkpoint | `... resume prod-v1 --from best --live` or `--from 40` |

- **After a crash or a power cut,** resume. If it says another process may still be running the run, check that none is and resume again after 2 minutes.
- **If a step was sent for training when the script died,** resume says so: wait a few minutes and resume, or pass `--retrain-pending` (train the saved batch) or `--discard-pending` (redo the step).
- **At a ceiling,** the run stops and says whether the research's rule would spend the reserve. Spending it is your call (12a): resume with a higher `--ceiling-train` or `--ceiling-eval`.

## 5. The dashboard loop (the Claude session on the PC)

From a Windows shell in the repo, in the background:

```bash
.venv/Scripts/python.exe scripts/dashboard.py wait prod-v1 --pot-left 191.01 --out-dir results/dashboard-step12
```

- **PUBLISH:** publish `results/dashboard-step12/page.html` to https://claude.ai/artifact/HqtrsXwnHQcJKud7GnUHMm (a new session passes that URL), then run the same `wait` with `--ack`.
- **STALE:** the run stopped writing its state without saying why; publish, check the run, tell Rishi.
- **BLOCKED:** the page scan found a key, seed or eval item; nothing was written or published; tell Rishi.

The step 10 and step 11 pages are never published to again.

## 6. The last dry run (2026-10-17, $0)

`train.py start --run prod-dry-final --dry-run --kind production --lr 2e-5 --real-edit --stop-after 2`, then `resume --stop-after 3`: the fake model, the fake trainer, and edit episodes in the real edit sandbox and grader (Docker), on the final code.

- The production set-up as pinned: prompt v3, root thinking `medium`, `selection-v3`, ceilings $146 and $34.
- Checkpoint 0's selection set (64 tasks, its edit tasks in real containers) and capability check ran; 3 trained steps of 32 episodes each with 0 infra episodes; the KL re-score lined up every turn (8 requests a step, 0 misaligned, 0 failed).
- The stop after step 2 and the resume ran cleanly. Step 3 retired one `edit_real` task, and the notice and the page showed "106 left to draw".
- No container was left afterwards (0 before, 0 after), and the dashboard built from the dry run passed its scan.

Tests, in WSL with Docker, all passing: 136 in the training loop, edit environment, grader, evals and audit, dashboard and data-family modules, then 137 in the grader, dashboard, detector, edit environment, edit sandbox and evals modules again after the last grader and dashboard changes.

## 7. Stop-rule changes after the first steps (2026-10-18, Rishi: "fix what would stop the run for nothing")

After step 5 the collapse rule was about to roll the whole run back to checkpoint 0 and halve the learning rate. Raw entropy had fallen 0.48, 0.40, 0.38, 0.40, 0.31 on the same family mix while KL to base stayed near 0.003: the model had barely moved, and the drop came from longer, code-heavier episodes. The run was stopped mid-training (Ctrl-C twice, Rishi's yes) with step 5's batch saved, and every rule was reviewed for false stops:

| Rule | Before | Now | Why |
|---|---|---|---|
| Collapse (LR fallback) | raw entropy under 70% of its 10-step peak, or reward std under half its mean for 3 steps | the trained model 30% less surprised than the base model by the same re-scored tokens, pooled over the last 3 steps (`stoprules.pooled_entropy_drop`, see below); raw entropy only without that reading, and then 3 steps in a row | raw entropy moves with what was sampled; comparing on the same tokens doesn't |
| Reward std (Echo Trap) | part of the collapse rule | a warning notice | it falls whenever a step draws easy or hard families |
| 3. Frying | capability more than 2 points below checkpoint 0 once | twice in a row, or more than 5 points once | 100 prompts move a few points by chance |
| 2. Overfitting | any selection-set fall, twice, with training reward rising | a fall of more than 5 points, twice | 64 tasks move about 6 points by chance |
| 5. Hacking (pause) | every detector hit | blocked escape attempts that are only file or process attempts are left out (`detectors.contained`); they still score 0 | step 4's three were scratch files to /tmp and a `find`; nothing to fix |
| Controller | a level move waited for 3 steps in its window | a unanimous window (every episode right, or every one wrong, on 8 or more) moves at once | edit synth went 20 of 20 over its first 4 steps, all groups that teach nothing |
| Train call | one failure crashed the run until someone resumed | waits up to 15 minutes for the checkpoint to land, accepts it if it did, else sends the batch again (2 more tries) | one W&B hiccup mustn't stop a 2-day run |

The dashboard's Policy health panel gained "Entropy against the base model (collapse signal)", and the chart guide explains each change.

**Pooling the matched signal (after step 8, 2026-10-18).** Step 8's own reading was 10.5% while KL barely moved: it re-scored 2,133 tokens, mostly code, which both models find predictable (base surprise 0.083 against ~0.24 on steps 6-7). One step's few thousand tokens swing with their content, so the rule now reads the readings of the last 3 steps pooled, token-weighted (`stoprules.pooled_entropy_drop`; steps 6-8 pool to under 4%). It needs 2 readings and 1,500 tokens; short of that it reaches back a step at a time to 6 steps, so short re-scores can't leave it blind. A real collapse persists, so it still crosses within a step or two. The rollback target is dated from the first pooled step whose own reading is past 30%, not the pool's start, so checkpoints sampled before the collapse aren't thrown away. The run was stopped after step 8 with the stop switch for this change.

## 7b. Overnight changes (2026-10-18, Claude, with Rishi's go-ahead to stop, fix and resume while he slept)

**Learning rate to 1e-5 after checkpoint 10.** Steps 9 and 10 failed spec §5.2's own gate for 2e-5: `probs_corr` 0.969 and 0.975 (needs above 0.98) and grad norm 3.06 (above 3x its median). W&B's trainer takes one optimizer step per ~14K-token packed sequence, so a production step makes 30-100 updates (the second seed made 7-21) and the policy drifts within a step. The run stopped after checkpoint 10's evals (selection 53.1%, capability 76%, audit 24 of 24 wins clean) and resumed at 1e-5 with no rollback. Steps 11-15 then read 0.992-0.997.

**Broken replies, and a format penalty, with a rollback to checkpoint 10.** From step 11 a growing share of replies ended on their own mid-code (`r1 = read_file("`), empty, or as a bare `<tool_call>`, with `finish_reason` stop and the token counts matching what was written, so the policy itself was emitting its end-of-turn early. By step 16 it was 125 of 494 turns (25%; at most about 6% a step through step 11), 21 of 32 episodes had at least one, and they snowballed inside an episode (one edit episode: 23 of 30 turns). The pass rate fell to 28% and half the edit episodes never submitted. Neither collapse signal saw it (pooled matched drop 3.7%). The cause on the reward side: the outcome reward never told a win with broken replies from a clean win, and in steps 6, 8, 9, 12, 13 and 14 the episodes with broken replies had a positive mean advantage, so wins kept reinforcing them. The fix, per research §4.1 ("small process penalties only once the logs show the need"):

| Change | Where | What it does |
|---|---|---|
| Format penalty | `reward.broken_replies`, `reward.format_penalty`, `env.run_episode(format_penalty=)` | 0.1 off the reward per broken reply (a turn that ended on its own with no runnable code; a reply cut at the token cap doesn't count), at most 0.3 an episode, before the detectors (a hack is still exactly 0). A wrong answer can go below 0. Training rollouts only; evals are unchanged |
| Config | `RunConfig.format_penalty`, `config.production` | New production runs get 0.1. prod-v1 started before it existed, so the resume turns it on |
| Resume switch | `train.py resume --format-penalty 0.1`, `TrainingRun.resume(format_penalty=)` | Sets it in the run's config with a notice |
| Dashboard | `monitor`, `dashboard.SHAPE_KEYS`, Trouble per episode chart | Broken replies per episode, and the penalty taken |

The run stopped during step 17's rollouts (Ctrl-C twice, nothing trained on them) and resumed from checkpoint 10 (its samples at step 11 had 4% broken replies) on branch 1 at 1e-5 with the penalty on:

```bash
.venv-linux/bin/python scripts/train.py resume prod-v1 --live --from 10 --lr 1e-5 --format-penalty 0.1
```

Steps 11-16 of branch 0 (~$8.40) are kept under `results/prod-v1/abandoned/`.

**The penalty made it worse; masking instead (second restart).** Branch 1's first step (from checkpoint 10, 129 optimizer updates, all 8 groups trainable, a fresh optimizer) took broken replies from 14% of turns in its own rollouts to 42% in the next step's, with 9 episodes broken from their first turn. Decoding the sampled token ids with the Qwen3.8 tokenizer confirmed the model itself ends turns early (``...specifically the `load_t<|im_end|>`` inside its thinking, or ``<tool_call> <python> ... </user><|im_end|>``): the server returns exactly what was sampled. A likely mechanism: a broken reply is mostly ordinary tokens, so a negative advantage pushes those down and lifts everything else, the end-of-turn token included; a rollback's fresh optimizer at full rate adds a large first update. The run stopped during step 12's training and changed course:

| Change | Where | What it does |
|---|---|---|
| Mask broken replies | `env.to_trajectory(mask_broken=)`, `env.trainable_turns`, `TrainingRun._trajectory_groups`, `RunConfig.mask_broken` | A broken reply stays in the trajectory as a plain message, so it is context for later turns but takes no gradient either way. An episode with no turn left to train on drops out of its group; a step left with fewer than 2 groups skips its train call. Rows record `masked` |
| Format penalty off | `RunConfig.format_penalty` | Production's default is 0 again; the code stays for later use |
| Warm-up after a rollback | `training.BRANCH_WARMUP` | A new branch's first two train calls run at 0.25x and 0.5x the learning rate |
| Resume switch | `train.py resume --mask-broken` | Turns masking on for a run started before it existed |

```bash
.venv-linux/bin/python scripts/train.py resume prod-v1 --live --from 10 --lr 1e-5 --format-penalty 0 --mask-broken --discard-pending
```

Branch 1's step 11 (~$1.85) is kept under `abandoned/b1-after10`.

**Drawing broken replies again (branch 2, from step 14).** Per-step broken-reply counts swing with a few episodes that snowball (one edit episode can contribute 25), so per-episode numbers read better: episodes with a run of 5 or more broken replies went 0-2 a step through step 9, then 5 at step 10 (sampled from checkpoint 9), 8 at branch 0's step 16, and 2-7 on branch 2's first steps. The base model's own broken replies (steps 1-3) were mostly native tool-call tags (`<tool_call>` then `<function=shell>`), also cut short; prompt v3 already tells it those don't run. Masking stops the training signal; this stops the snowball:

| Change | Where | What it does |
|---|---|---|
| Draw a broken reply again | `env.run_episode(resample_broken=)`, `config.RESAMPLE_BROKEN = 2`, `RunConfig.resample_broken` | In training rollouts, a broken reply is drawn again, up to twice a turn, before it enters the conversation: the model never sees it, so it can't copy it, and it never reaches training. The turn keeps what was dropped (`Turn.resampled`); if the third draw is broken too it stays, masked, as before. Evals keep one draw, so checkpoint scores stay comparable |
| Resume switch | `train.py resume --resample-broken 2` | Turns it on for a run started before it existed |
| Dashboard | Trouble per episode | "broken replies drawn again" per episode |

The run stopped after branch 2's step 13 (stop switch) and resumed on the same branch with `--resample-broken 2`; no rollback, since branch 2 never trained on a broken reply. Each extra draw is one more root call, so a step costs a little more while broken replies are common. If drawn-again replies keep climbing, the next lever is a lower learning rate.

**Two false pauses fixed (after branch 2's step 15).** (1) The infra pause cleared its 10-step window on every resume, so one W&B server error in the first step after a resume was 1 of 32 (3.1% > 1%) and paused the run; it paused after step 15 on 2 server errors in 64 episodes. A window that isn't full is now measured against a full window's episodes (`InfraPause.update`): 4 errors in 320 still pause, and a burst (half a step) still pauses at once. (2) The dashboard check blocked the page because a sampled rollout showed a repository's `api_key` variable; rollouts whose shown text names a key are no longer sampled (`dashboard.shows_key_text`), so the check stays strict everywhere else.

## 8. What changed for step 12 ($0, tested)

| Change | Where | What it does |
|---|---|---|
| `edit_real` fallback (decision 12b) | `rlm_rl/tasks/__init__.py`, `training.py` | Once retirement leaves too few `edit_real` tasks to draw, their slots go to `edit_synth` at its own level, so the run never stops for want of tasks. Each step records the moved slots and the tasks left; the Notices panel says when it starts, and the pass-rate panel shows the counts |
| Leftover containers cleared | `sandbox.clear_for_run`, `training.py`, `scripts/train.py` | `start` and `resume` remove the run's own containers from a dead process, and any sandbox or grader container that finished over 10 minutes ago. A resume refuses while the state says running and was written in the last 2 minutes |
| Edit wins in the Claude audit | `scripts/audit.py`, [audit-procedure.md](audit-procedure.md) | Each sampled edit episode shows its changed files, how the grader treats each, its grade and its diff, with flags; the procedure says what to look for |
| `test_aware` and scratch files | `rlm_rl/grader.py` | A test-runner check in a new file that nothing loads (the change, the repository and the hidden tests never name it, and it isn't a self-loading file such as `sitecustomize.py`) goes to the Claude audit instead of zeroing the reward. Go stays strict. Replayed over step 11's 54 edit episodes: the one false positive clears and nothing else changes |
| Ceilings and the pot | `config.PRODUCTION_CEILINGS`, `dashboard.POT` | $146 steps, $34 evals, from the $191.01 billing read |
| Dashboard | `dashboard.py`, `dashboard_page.html` | Production's page is titled "RLM Production Run", the KL chart shows the re-score's misaligned and failed counts, and the cost panel shows the new split |
