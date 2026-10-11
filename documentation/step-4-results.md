# Step 4 results: limiter, infra flags, eval logging

> **Step 4 of [production-build-plan.md](production-build-plan.md)** (holes H5, H6 and H9). Built and tested 2026-10-11 on Rishi's PC. Caps applied: **30 requests in flight in total and 20 per model name** (step 2, Rishi 2026-10-10), with root calls going to the training checkpoint and sub-calls to the base model.
>
> **No W&B credit was used.** Every test talks to `tests/fake_model.py`, a local stand-in for the W&B model server, so nothing called W&B. The `.env` file was never opened. Nothing has been committed or pushed, including steps 2 and 3.
>
> **One install, approved by Rishi on 2026-10-11.** A Linux virtual environment, `.venv-linux/` (git-ignored, ~1 GB), sits inside the repo. It holds the pinned packages from `requirements.txt` except the Jupyter ones: ART 0.5.20, wandb, pyarrow, openai, httpx and python-dotenv, all from PyPI.
>
> Anything marked *(mine)* is my choice, not the spec's. Anything marked *(inferred)* hasn't been checked against the live system.

## Summary

**Step 4 is complete.** Every done-when check passes, both with an in-process sandbox stand-in and through real Docker sandboxes in WSL. The full regression suite, 97 tests covering steps 3 and 4, passes in WSL in about 2 minutes. One test skips there because the tokenizer config isn't cached in WSL; it passes on Windows. Running the suite left no container or episode directory behind.

| Done-when check | Result | Evidence |
|---|---|---|
| A load test never has more than the cap in flight | **Pass** | `LoadTest`: 300 mixed root and sub-calls reach exactly 30 in flight and never more, each model stays at or under 20, and there are no 429s. Base-model-only traffic runs at 20. Through Docker, 32 real sandboxes making 320 sub-calls and 64 root turns peaked at 30 on the server (20 base, 10 checkpoint) with no 429s. |
| Forced 429s and a killed kernel produce flagged episodes, and none reach training | **Pass** | `InfraTest.test_none_reach_training`: one 32-episode step with forced 429s and killed sandboxes. All 8 bad episodes are flagged and counted, and none is in what gets trained on. Through Docker, a forced 429 through the real bridge and a real `docker kill` mid-cell each flag the episode. |
| A rerun eval writes one transcript per episode, and the audit script reads them | **Pass** | `EvalLoggingTest.test_rerun_leaves_one_transcript_per_episode`: 12 planned episodes, 4 hit forced 429s, the rerun runs only those 4, and the reader returns exactly 12. `scripts/audit.py` summarizes, samples and shows them. The Docker version passes too. |
| Sub-call queue time doesn't count toward the 120 s cell timeout | **Pass** | Through Docker: 8 sub-calls queued one at a time behind the limiter (over 3.5 s) inside a cell with a 2 s timeout finish normally. |
| Eval logging through ART `model.log(..., split="val")` | **Pass** (local only) | `test_logged_through_art` writes ART's parquet and metrics with W&B reporting off. Infra episodes show up as ART's `val/exception_rate`. |
| One shared limiter for every request (H6) | **Pass** | Root turns and sub-calls go through one `ModelClient`, and several clients can share one `Limiter`. |
| The model is never shown a fake answer (H5) | **Pass** | In every infra case the failed cell's output is never shown. The episode ends, and no observation contains "sub-call failed". |

## 1. What was built

| Path | What it is |
|---|---|
| `rlm_rl/client.py` | `ModelClient`: root turns and sub-calls through one `Limiter`, with retries and `RequestFailed` (an `InfraError`). It also runs the thinking-off checks and the token-id rebuild, and counts tokens per episode (`Usage`) and failures per step (`RequestStats`, with `cf-ray`). |
| `rlm_rl/env.py` | `run_episode` (the episode loop), `Task`, `Episode` with its `infra_error` flag, `run_groups` (one step's rollouts), `filter_step` (spec §5.4 steps 3, 5 and 6), `InfraPause` (§5.4 step 4) and `to_trajectory` for ART |
| `rlm_rl/transcripts.py` | The transcript format, one JSON line per finished episode for training and eval alike, plus the append-only log and its reader |
| `rlm_rl/evals.py` | `run_eval`: runs or resumes one eval and logs every episode, locally and through ART |
| `scripts/audit.py` | Reads transcripts: `summary`, `sample` (wins as Markdown for the Claude audit, 5a) and `show` |
| `tests/test_client.py`, `test_env.py`, `test_evals.py` | 45 tests: 22, 17 and 6 |
| `tests/fake_model.py` | A standard-library stand-in for W&B's OpenAI-compatible server. It counts requests in flight on its own side and answers 429 above 20 per model name, like W&B. It can inject 429s, 5xx errors and hangs. |
| `tests/fakes.py` | `FakeSandbox`, an in-process stand-in for `Sandbox`, so the episode loop can be tested without Docker |
| `tests/test_contract.py` | Now diffs the system prompt in `rlm_rl.env` and `rlm_rl.evals` (H10). A consumer whose dependency is missing now fails instead of being skipped. |
| `.gitignore` | Adds `results/*/transcripts/`, because transcripts hold gold answers |
| `README.md` | The Layout section lists the new modules |

How step 5 and step 6 use it:

```python
client = ModelClient(model.openai_client(), subcall_model=model.get_inference_name(step=0))
groups = await run_groups(tasks, 4, client=client, root_model=model.get_inference_name(step=step),
                          scorer=score, run_id=run)          # each episode is flagged or scored
batch = filter_step(groups)                                  # infra out, then small and all-equal groups
if (why := pause.update(batch)): stop_and_report(why)        # spec §5.4 step 4
await backend.train(model, [art.TrajectoryGroup(map(to_trajectory, g)) for g in batch.groups], ...)
await run_eval(selection_set, name="selection", step=step, run_dir=..., run=run, client=client, ...)
```

The scorer is passed in until step 5 adds `reward.py` and the detectors.

## 2. How it works

### 2.1 The limiter (H6)

- **One `Limiter` for everything.** Every root turn and sub-call waits for a slot. It allows 30 requests in flight in total and 20 for any one model name. During training, root calls (`:stepN`) and sub-calls (`:step0`) can use all 30 between them. An eval of the base model sends both kinds to `:step0`, so it runs at 20.
- **First come, first served,** except that a request blocked only by its own model's cap of 20 doesn't hold up a request for the other model *(mine)*.
- **The slot is held only while a request is on the wire.** A request waiting out a backoff gives its slot back.
- **The OpenAI client's own retries are off** *(mine)*. ART's client retried twice by default, inside the old semaphore and out of sight. Now every retry is counted and waits outside the limiter.
- **Cancelling a waiter never leaks a slot.** A test covers it.
- **Peaks are recorded per step:** total, per model, and time spent waiting. They're for the dashboard.

### 2.2 Retries, and what counts as infra (H5)

Retries use jittered exponential backoff: up to 6 tries, starting at 2 s and doubling, each wait scaled by a random factor between 0.5 and 1.5 and capped at 60 s. A `Retry-After` header is honoured. The worst case is about 93 s of waiting before a request gives up. Each request has a 600 s timeout *(mine)*.

| What happens | Outcome |
|---|---|
| 429, 408, 409, 5xx, a timeout or a dropped connection | Retried. If all 6 tries fail, `RequestFailed`, and the episode is `infra_error` (kind `rate_limited`, `server_error`, `timeout` or `connection`) |
| 401, 403 or 404 on any call (bad key, wrong checkpoint name) | `infra_error` at once (kind `rejected`) |
| Any other 4xx on a root call | `infra_error` at once. A root request the API refuses is never the model's doing *(mine)* |
| 400, 413 or 422 on a sub-call | Shown to the model as `RuntimeError: sub-call failed: the API rejected the request (HTTP 400)`, as step 3 designed (`SubcallError`) |
| The thinking-off check fails (§2.3) | `infra_error` (kind `thinking_on`) |
| No choices, or root logprobs missing | `infra_error` (kind `bad_response`) |
| The sandbox can't start, or is stopped from outside | `infra_error` (kind `sandbox`), from step 3 |
| A bug in our own code, including a scorer that raises | `infra_error` (kind `host_bug`), with the traceback saved. It is never scored as 0 *(mine)* |

Every failure is recorded with its model, HTTP status and Cloudflare `cf-ray` id, which step 2 found to be the only request id W&B returns. The client also counts 429s, 5xx errors, timeouts and retries per step.

### 2.3 Thinking checks and token ids

Both follow step 2 §4, and both are free:
- Every call must report 0 `reasoning_tokens`.
- A root call's rendered prompt, rebuilt from `prompt_logprobs`, must end with the empty think block. The check finds `<think>` and `</think>` (ids 248068 and 248069) in the last few prompt tokens.

The token ids are pinned for Qwen3.8-27B, and a test checks them against the tokenizer config in the local Hugging Face cache. Every call passes `chat_template_kwargs={"enable_thinking": False}`. The token-id rebuild is ported from `rollout.attach_token_ids`, with `<|im_start|>` pinned instead of downloaded each run.

### 2.4 The episode loop and the flag

`run_episode` builds the prompt from `bridge.initial_messages`, and each turn does the following:
1. Asks the root through the client.
2. Runs the first code block in the sandbox.
3. Ends on `submit`, or appends `bridge.observation(...)` and goes on, up to 12 turns.

A reply without a code block uses up the turn with the pinned error. **The episode never raises:**
- An `InfraError` from a root call, a sub-call (through the bridge) or the sandbox stops it, and sets `infra_error` and `infra_kind`.
- Its reward stays `None`, and `to_trajectory` refuses it.

A sub-call's `RequestFailed` is caught on its way through the bridge, so the episode keeps the real kind, for example `rate_limited`, rather than the bridge's generic message.

### 2.5 Before training (spec §5.4)

`filter_step(groups)` runs steps 3, 5 and 6 of §5.4. It:
- removes infra episodes
- drops groups left with fewer than 2 episodes
- drops all-equal groups, counting all-0, all-1 and other-equal groups separately
- sets `skip_train` when fewer than 2 groups remain

Step 2 of §5.4, hack zeroing, is step 5's. It only has to run before `filter_step`.

`InfraPause` runs step 4, pausing above 1%. Rishi chose how it counts (§4): within the step for the second seed, and over the last 10 steps in production.

### 2.6 Transcripts and evals (H9)

- **One JSON line per finished episode**, training and eval alike. It holds:
  - the conversation exactly as the model saw it, and every turn's reply, code, cell result and timing
  - every sub-call's prompt and output
  - the reward, `correct` and reward components, and the infra flag, kind and any traceback
  - the sandbox's audit events and refused bridge requests, timeouts and crashes
  - token use with an estimated cost, the checkpoint (`root_model`) and the sub-call model, and the bridge API, prompt version and prompt hash
- **The context isn't copied** *(mine)*. It's identified by its size and SHA-256, and the first user message already shows its first 3 lines. At ~400K characters, copying it would make each record ~10× bigger.
- **No seed is ever written** (H3). `Task.task_id` must not encode the seed, and the docstring says so.
- **Paths:**
  - `results/<run>/transcripts/train/step0007.jsonl`
  - `results/<run>/transcripts/val/<eval name>/step0010.jsonl`
  - Both are git-ignored, since they hold gold answers.
- **Crash-safe:** each line is flushed and fsynced as it's written, and the reader skips a torn last line.
- **`run_eval` resumes:**
  - Episodes already logged are skipped, and infra episodes run again.
  - The reader keeps the newest line for each episode key (`<eval>/step<N>/<task>/<sample>`).
  - So a rerun ends with exactly one transcript per episode, and earlier attempts stay in the file for the audit.
  - Accuracy leaves out infra episodes and reports them by kind.
  - At most 32 sandboxes run at once; the limiter paces the requests.
- **ART logging:** when an ART model is given, the whole eval is logged with `model.log(groups, split="val", step=step)`, not just this call's episodes.
  - It's built from the transcript file, with plain messages and no token ids, so a resumed eval logs complete groups.
  - Infra episodes go in as each group's exceptions, so ART's `val/exception_rate` counts them.
  - The eval's summary goes in as `<eval name>/accuracy` and similar metrics.
  - This writes `.art/.../trajectories/val/<step>.parquet` locally and the metrics to W&B. Logging metrics doesn't use inference credit.

### 2.7 The audit script

```bash
python3 scripts/audit.py summary results/<run>
python3 scripts/audit.py sample results/<run> --split val -n 5
python3 scripts/audit.py show results/<run>/transcripts/val/selection/step0010.jsonl <episode id>
```

- **`summary`** prints one row per log. It shows episodes, infra errors by kind, accuracy, first-turn submits, sandbox audit events, refused bridge requests and timeouts. It also reports reruns and torn lines.
- **`sample`** renders randomly chosen wins (or every episode with `--all`) as Markdown. That's the input for the Claude audit in decision 5a. It shows the question, gold and submitted answers, each turn's reply and output, the sub-calls, and flags.
- None of this changes a reward. The detectors that zero rewards are step 5's.

## 3. How it was tested

Run from the repo root inside Ubuntu:

```bash
.venv-linux/bin/python scripts/sandbox_ctl.py check
```

| File | Tests | Result |
|---|---|---|
| `tests/test_client.py` | 22: limiter caps, fairness and cancellation; backoff and `Retry-After`; root and sub-call settings; thinking checks; pinned token ids; persistent and transient 429 and 5xx; auth errors; rejected sub-calls; timeouts and dropped connections; the slot is free during backoff; four load tests | All pass in WSL and on Windows. The pinned token-id check skips in WSL, where the tokenizer config isn't cached; it passes on Windows. |
| `tests/test_env.py` | 17: the episode loop (submit, no code block, turn limit); forced 429s on sub-calls and root turns; 5xx; a killed sandbox; a host bug; a full step where none of the flagged episodes reach training; `filter_step`, `InfraPause` and the per-run pause rule. 4 Docker cases: forced 429 through the real bridge, `docker kill` mid-cell, queue time vs the cell timeout, and 32 real sandboxes under load | All pass in WSL. On Windows the 4 Docker cases skip. |
| `tests/test_evals.py` | 6: full transcripts; a rerun leaves one transcript per episode; the audit script reads them; ART logging; paths; transcripts are git-ignored. 1 Docker case | All pass in WSL. On Windows the Docker case skips. |

- **In WSL:** the whole suite, steps 3 and 4 together, is 97 tests. 96 pass and 1 skips (the token-id check), in about 2 minutes.
- **On Windows:** 69 tests run, 55 pass and 14 skip because they need Docker or a Unix socket.
- **Flakiness:** the WSL suite passed twice in a row.

**Two fixes from the WSL runs:**
- **A timing-sensitive test.** `test_slot_is_free_while_backing_off` assumed the first request would finish within 0.15 s. Now it waits until the client has actually started backing off.
- **Slow tests.** The new tests turn asyncio's debug mode off (`tests.fake_model.AsyncCase`). Debug mode records a stack trace for every task, and on WSL's mount of the Windows drive that took the step 4 tests from 82 s to 462 s.

The first model request in a fresh process also takes 5-20 s in WSL, because the venv sits on the Windows drive and its imports are slow there. That's a one-time cost per process.

**One load-test detail.** On the client side the peak is exactly 30. The server sees a little less (22-30), because a slot is also held while the request is sent and the reply parsed. It never sees more. A test confirms the cap of 20 per model is needed: 30 requests to one model name with only a total cap get exactly 10 429s from the fake server, as the step 2 probe found on W&B.

## 4. Defaults I picked, and Rishi's pause-rule decision

Where the plan left a fork, I picked a default:

| Fork | Default |
|---|---|
| Fairness between models | First come, first served, but a request blocked only by its own model's cap doesn't hold up the other model |
| The OpenAI client's built-in retries | Off, so every retry is ours, counted and outside the limiter |
| Request timeout | 600 s |
| Root call refused with a 4xx | Infra, never the model's outcome |
| Our own bug or a scorer error | Infra (`host_bug`), never a zero reward |
| Eval rerun | Resume: skip finished episodes, rerun infra ones, newest line wins |
| Context in transcripts | Size and hash only |
| Sandboxes at once during an eval | 32 |

**Decided by Rishi, 2026-10-11: how the 1% pause rule counts.** The spec says "if more than 1% of the step's episodes were infra_error, pause the run". At 32 episodes a step that can't be measured, because a single flagged episode is 3.1%. Rishi chose:
- **Second seed: per step.** Any infra error pauses the run, so every one gets looked at on the cheap run.
- **Production: a 10-step window.** The run pauses when more than 1% of the last 320 episodes were flagged, that is 4 or more. A stray timeout won't stop the run overnight.

Flagged episodes never reach training either way. The code is `InfraPause.for_run("second_seed")` and `InfraPause.for_run("production")` (`env.PAUSE_WINDOW_STEPS`).

## 5. What changes the spec or plan

For the planning thread, which owns both files.

1. **Spec §1.3:**
   - The cap of 8 becomes **30 in total and 20 per model name** (step 2 already recorded this).
   - Add: "A request gives its slot back while it waits to retry. The API client's own retries are off."
   - Also give the backoff detail: doubling from 2 s, a random factor between 0.5 and 1.5, at most 60 s a wait, `Retry-After` honoured.
2. **Spec §1.2, infra failures,** gains:
   - a failed thinking-off check
   - a root call the API refuses (any 4xx)
   - 401, 403 and 404 on any call
   - a malformed response
   - a bug in host code, scorer included

   A sub-call that gets 400, 413 or 422 is shown to the model as a `RuntimeError` instead.
3. **Spec §5.4 step 4:** "the step's episodes" becomes the last step's for the second seed and the last 10 steps' (320 episodes) for production, as Rishi decided (§4).
4. **Spec §10 code layout:**
   - It gains `rlm_rl/transcripts.py`, `scripts/audit.py`, `tests/fake_model.py` and `tests/fakes.py`.
   - `env.py` also holds `Task`, `Episode`, `filter_step` and `InfraPause`.
   - `client.py` holds the `Usage` cost estimate that step 6's spend counter can build on.
5. **Plan step 4:** "The cap stays at 8 until CoreWeave answers, then we try 16 once" is superseded by Rishi's cap of 30.
6. **Plan step 6, two things it needs to know:**
   - **ART keeps one val parquet per step.** Two evals logged through `model.log` at the same step overwrite each other's `.art/.../trajectories/val/<step>.parquet`. The JSONL keeps both. So step 6 should log only the selection set through `model.log`, or log the capability check under its own ART model name.
   - **The limiter is per process.** Training and an eval must run in the same process, or share one `Limiter`, or the 20-per-model cap can be broken. `run_eval` takes the training run's client, so this is the default.
7. **Spec §8, config.json:** add the limiter caps and the retry policy to what's saved with every checkpoint.

## 6. W&B "Agent Lens": does it fit our audit logs?

**Short answer: not now.** It adds a hosted viewer, but nothing to the audit record. It bills by data volume rather than inference credit. Using it would send every prompt and transcript to CoreWeave's cloud.

**What it is.** It's **CoreWeave Agent Lens**, not a W&B product name; W&B is now part of CoreWeave, and its docs live on docs.coreweave.com.
- Its docs call it "a new agent observability product from the team that builds" W&B Weave. It's in public preview in CoreWeave Forge ([docs](https://docs.coreweave.com/products/agent-lens/what-is-agent-lens)).
- **Weave wasn't renamed.** The two share one backend, so traces sent to one show in the other. Agent Lens is the recommended view for agent traces, while Weave still does evaluations, ops and guardrails.
- Agent Lens adds conversation and turn views, failure clustering, LLM judges and cost and error dashboards.
- My research subagent found the Forge launch dated 2026-09-30 ([CoreWeave news](https://www.coreweave.com/news/coreweave-forge-launches-turning-the-ai-loop-production-run-into-a-better-model-and-agent)). I checked the docs page and pricing page myself.

| Question | Answer |
|---|---|
| **What it adds over our logging** | A hosted UI to browse traces, cluster failures and run LLM judges. It adds nothing to the record itself. It traces model calls automatically, but our audit also needs REPL cells, sandbox events and rewards, which we'd have to instrument by hand. Our JSONL already has all of them. |
| **Extra dependencies** | Weave is already installed: ART 0.5.20 depends on `weave` (0.53.12 in `.venv`). Agent Lens uses a separate CoreWeave Forge SDK. The subagent named it as `coreweave[openai]`, but the docs page I read doesn't name the package. |
| **Does tracing spend inference credit?** | **No source says it does.** Weave bills ingested bytes, and LLM inputs and outputs count ([how ingestion is counted](https://docs.coreweave.com/support/weave/articles/how-is-weave-data-ingestion-calculated)). [Forge pricing](https://www.coreweave.com/forge-pricing):<br>• Free: 1 GB a month<br>• Pro: 1.5 GB a month, then $0.10 per MB<br>• Academic: up to 25 GB a month<br>Agent Lens's LLM judges run models, and the pricing page doesn't say what they cost, so they could use credit *(inferred)*. |
| **Does it send episode data anywhere?** | **Yes.** Traces go to CoreWeave's hosted service (the docs name `trace.wandb.ai`). The subagent found it's a multi-tenant store in North America with no offline mode; I didn't check that myself. W&B already serves every request we make, but traces are stored and shown in the project. |
| **How much data** | At production size, roughly 0.7-3.5 GB for every request of a ~120-step run plus evals, more than the free tier *(inferred: 150-750 KB per episode, from 24K-character sub-call prompts)*. |
| **Is it on today?** | **No.** Tracing starts only when something calls `weave.init()`, and nothing does. ART's own client also switches Weave tracing off for trainable models, because its OpenAI integration "can turn a long RL run into hundreds of GB of trace payload" (ART's `model.py`). Our root turns and sub-calls both go through that client. |

**Recommendation *(mine)*:**
- **Keep the local JSONL as the audit log of record,** and leave Weave uninitialised.
- **If a hosted transcript viewer would help your friends at step 9,** revisit it then for sampled transcripts only, a few MB per eval, which fits the free tier.

Nothing was adopted, installed or traced, and no credit was spent.

## 7. Known limits

- **The thinking check is pinned to Qwen3.8-27B's token ids.** A different base model needs new ids, and the test against the cached tokenizer would catch a mismatch.
- **The fake server is not W&B.** W&B's limit of 20 per model name comes from the step 2 probe. The second seed is the first live check, and the per-step client stats (429 count and peaks) will show it.
- **Server-side peaks sit a little under the client's 30,** because parsing happens inside the slot. That's conservative, which is the safe direction.
- **On Windows, connecting to a closed port times out instead of being refused,** so a test accepts either kind. It doesn't matter in WSL, where the environment runs.
- **The system `python3` in Ubuntu no longer runs the suite.** The step 4 tests need `openai`, so use `.venv-linux/bin/python`, as the README now says.

## 8. Commands

```bash
.venv-linux/bin/python scripts/sandbox_ctl.py check                 # all 97 tests, inside Ubuntu
.venv-linux/bin/python scripts/audit.py summary results/<run>       # read a run's transcripts
.venv-linux/bin/python scripts/audit.py sample results/<run> -n 5   # sampled wins for the Claude audit
```
