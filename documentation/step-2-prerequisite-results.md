# Step 2 results: prerequisite checks

> **Step 2 of [production-build-plan.md](production-build-plan.md).** Run 2026-10-10 on Rishi's PC. The live tests Rishi approved used about **$0.13** of W&B inference credit ($0.03 for the step 2 probes, $0.10 for the thinking replay in §6), and nothing else spent any. Nothing has been committed or pushed. The W&B key in `.env` was loaded by scripts through `python-dotenv` and never printed.
>
> Anything marked *(inferred)* hasn't been checked against the live system yet.

## Summary

| Check | Result | What's left |
|---|---|---|
| Docker runs a `--network none` container | **Pass** | Nothing for step 2. Turning on Docker's WSL integration for Ubuntu is needed only if 3b picks the WSL2 option. |
| LoRA checkpoint downloads from W&B | **Pass** | Nothing. The step-20 GRPO adapter is on disk and its checksums match W&B's. |
| Qwen3.8-27B licence allows publishing a LoRA | **Pass** | Apache-2.0. W&B's terms don't explicitly say who owns trained adapters (§3). |
| W&B serves thinking off | **Pass** | Nothing. The server defaults to off, and the gateway accepts and applies `enable_thinking` (§4). Whether off is the right choice is reviewed in §6. |
| CoreWeave support email | **Dropped by Rishi, 2026-10-10** | Nothing. |
| Concurrency cap of 30 | **Pass if no single model gets more than 20** | The limit is 20 requests in flight per model name. The limiter needs a per-model cap of 20 under the total of 30 (§5). |

**Decision (Rishi, 2026-10-10):** no CoreWeave email, and a cap of **30 requests in flight** across all calls. This replaces the cap of 8 in spec §1.3 and plan step 4 ("8 until CoreWeave answers, then 16").

**What the probe adds:** W&B allows 20 requests in flight per model name, so the limiter also caps each model name at 20 *(mine)*. Root calls go to the training checkpoint (`:stepN`) and sub-calls to the base model (`:step0`), so 30 can be in flight during training. A base-model eval sends root calls and sub-calls to `:step0`, so it runs at 20.

The planning thread owns the spec and the plan, so both changes are recorded here for it to apply.

The plan's "done when" list: Docker runs a `--network none` container (done), a checkpoint is on disk (done), the licence answer is written into the spec (the text is below, for the planning thread to paste in), and the email is sent (dropped by Rishi).

---

## 1. Docker and WSL2

**What's installed:**

| Part | Version |
|---|---|
| Docker Desktop | 4.93.0, engine 29.8.1, context `desktop-linux` |
| Backend | WSL2, kernel 6.6.87.2-microsoft-standard-WSL2 |
| Resources the engine sees | 14 CPUs, 25 GB RAM |
| WSL2 distros | `Ubuntu` (default, WSL 2, Python 3.12.3), `docker-desktop`, `EXL3-Runtime` |

**The test.** This used the `alpine:3.21` image that was already on the PC, so nothing was pulled.

```bash
docker run --rm --pull never --network none --read-only \
  --tmpfs /scratch:rw,size=64m,uid=1000 --user 1000:1000 \
  --cpus 1 --memory 512m --pids-limit 64 \
  --cap-drop ALL --security-opt no-new-privileges alpine:3.21 sh -c '...'
```

| Inside the container | Result |
|---|---|
| User | uid 1000, not root |
| Network interfaces | `lo` only |
| HTTP to 1.1.1.1 | blocked |
| DNS lookup | blocked |
| Write to the root filesystem | blocked |
| Write to `/scratch` | works |
| cgroup limits | pids 64, memory 512 MB, CPU 1 core |

Every limit step 3 needs works on this PC.

**One gap, for 3b only.** The Docker CLI isn't available inside the Ubuntu distro, because Docker Desktop's WSL integration is off for it. The recommended 3b option runs the environment inside WSL2 and mounts a Unix socket into the container, so it needs that integration turned on. To turn it on:

1. Open Docker Desktop.
2. Go to **Settings → Resources → WSL integration**.
3. Turn on the toggle for **Ubuntu**.
4. Click **Apply & restart**.

After that, `wsl -d Ubuntu -- docker version` should print a server version, and the socket test can run in step 3. Whether a socket bind-mounted from the Ubuntu distro reaches a `--network none` container is *(inferred)*: all WSL2 distros share one Linux kernel, so it should.

WSL also printed `Failed to start the systemd user session` when Ubuntu started. It didn't stop any command from running.

---

## 2. LoRA download

**Pulled:** `models-ontario-tech-university/rlm-rlsd-vs-grpo/qwen3-8-27b-grpo-v1:step20`. This is the final checkpoint of the 20-step GRPO run, which scored 0.70 on the held-out eval. It is W&B artifact version v20, type `lora`, with aliases `latest` and `step20`.

**Saved to:** `checkpoints/qwen3-8-27b-grpo-v1/step20/`. I added `checkpoints/` to `.gitignore` so the adapter never gets committed. That edit is uncommitted.

| File | Size |
|---|---|
| `adapter_config.json` | 1.2 KB |
| `adapter_model.safetensors` | 108.9 MB, 800 bf16 tensors, 54.4M parameters |

- The download took 8 s. The MD5s match W&B's manifest. The adapter's SHA-256 is `5f3a0c11b6ba85a1be1ba7cf8d1433bc808e7f293b3689bb1039cd698f2f3701`.
- **Adapter config:**
  - base `Qwen/Qwen3.8-27B`, PEFT LoRA
  - r = 8, alpha = 32, dropout 0, no bias, no rsLoRA or DoRA
  - target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, plus the linear-attention modules `in_proj_qkv`, `in_proj_z` and `out_proj`
- **How it was pulled:** `wandb.Api().artifact(...).download()`, the same call as ART's `ServerlessBackend._experimental_pull_model_checkpoint`. Any `:stepN` checkpoint downloads the same way.

**Notes for step 13 (final eval off W&B):**
- The adapter targets Qwen3.8's linear-attention layers, and its tensor names start with `base_model.model.model.language_model.` (the multimodal wrapper). The server used for the final eval needs LoRA support for those modules, and a text-only loader may need the names remapped *(inferred)*. I haven't loaded it against the base weights, because that needs the 54 GB base model. 13a is the place to check it.
- **The rank is 8.** That is ART serverless's default, and the spec doesn't pin it. If production should use a different rank, it has to be set before the second seed.

**Storage is bigger than the research assumed.** The research says the 5 GB of free storage holds about 30 LoRAs. These are the LoRA artifacts in the `models-ontario-tech-university` entity today:

| Where | Size |
|---|---|
| RLM testing projects (20-step run, three-way, SDPO, smoke tests, probes) | ~11.8 GB |
| `rlm-rlsd-vs-grpo` alone (21 checkpoints per arm × 2 arms) | 4.8 GB |
| `us-finance-sft-lab` (a different project in the same entity) | ~11.2 GB |
| **Total** | **~23 GB** |

Serverless RL keeps every checkpoint unless the script deletes it. At 109 MB a checkpoint, a 120-step production run would add ~13 GB, so the spec's "keep every 5th plus the best" needs an explicit delete call in step 6. I haven't deleted anything. Rishi confirmed the ~23 GB is from his other projects and is fine.

---

## 3. Licence

**Qwen/Qwen3.8-27B is Apache-2.0.** The repo's `LICENSE` file is the standard Apache License 2.0 text ("Copyright 2026 Alibaba Cloud"), and the model card's licence tag says the same. The repo has no `NOTICE` file.

Apache-2.0 allows publishing a LoRA adapter, which is a derivative work, on Hugging Face. The conditions are:
- include a copy of the Apache-2.0 licence
- say the adapter is a modification of `Qwen/Qwen3.8-27B` (`base_model:` in the model card does this)
- keep any attribution notices from the base model (there's no NOTICE file to carry over)
- don't use the Qwen or Alibaba names beyond describing where the model came from

This is a reading of the licence text, not legal advice.

**Two other things decide whether the adapter can be published:**
- **Training data.** Decision 7b covers it. The ticket tasks are synthetic, generated by our own code (`ticket_tasks.py`), so the 20-step adapter is clean.
- **W&B's terms.** W&B's docs say trained LoRAs are stored as artifacts that you can download. I found no page that explicitly says the customer owns them. Rishi dropped the support email that would have asked.

**Text for the spec** (for the planning thread to paste into production-spec.md §11, or a new row in its decisions table):

> **Licence (step 2, 2026-10-10):** Qwen/Qwen3.8-27B is Apache-2.0, so a LoRA adapter can be published on Hugging Face with the licence included, `base_model: Qwen/Qwen3.8-27B` in the model card, and the adapter marked as a modification. Training data must also pass 7b. W&B's terms don't explicitly say who owns a trained adapter.

---

## 4. Thinking off

**The model's own template defaults to thinking on.** Qwen3.8-27B's `chat_template.jinja` thinks unless `enable_thinking` is explicitly `false`. With thinking on, the template also adds "Reasoning effort is set to xhigh…" to the system prompt and opens the reply with `<think>\n`. With `enable_thinking: false`, it opens the reply with an empty `<think>\n\n</think>\n\n` block.

**The W&B server defaults to thinking off.** The 20-step run never passed `enable_thinking`. But the prompt token ids stored with every rollout come from the server's own `prompt_logprobs`, so they show exactly what the server rendered (`rollout.attach_token_ids`). I scanned every stored trajectory: 59 parquet files, 1,888 trajectories, 8,505 root calls, across the 20-step, three-way, SDPO and smoke-test runs.

| Check | Result |
|---|---|
| Prompt ends with `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` (thinking off) | 8,505 of 8,505 |
| Prompt ends with `<think>\n` (thinking on) | 0 |
| System prompt starts with the reasoning-effort line | 0. All 8,505 open with the same tokens, "You are a…" |
| `reasoning` field set on a response | 0 |
| Completion contains a stray `<think>` or `</think>` token | 9 (0.1%) |

- The 9 stray tokens are the model's own output, not thinking mode: one token, mid-reply, then normal text and code. The returned `content` drops them, so they're harmless. The transcript audit should know they exist.
- **Sub-calls weren't logged with token ids, so the scan doesn't cover them.** The live probe below covers them with a sub-call-shaped request.

**The live probe confirms both halves** (run 2026-10-10, approved by Rishi). It sent 4 requests to the base model (`qwen3-8-27b-grpo-v1:step0`), with prompt token ids read back from `prompt_logprobs`:

| Request | HTTP | Rendered prompt | Prompt tokens | Reasoning tokens |
|---|---|---|---|---|
| Root shape, no flag | 200 | thinking off | 25 | 0 |
| Root shape, `enable_thinking: False` | 200 | thinking off | 25 | 0 |
| Root shape, `enable_thinking: True` | 200 | **thinking on** | 61 | 8 of 8 (`content` was empty) |
| Sub-call shape, `enable_thinking: False` | 200 | thinking off | 15 | 0 |

- **The gateway accepts `chat_template_kwargs`** and passes it to the template. With `True`, the prompt grew by the reasoning-effort line, and all 8 output tokens went to reasoning. The spec's pin works.
- **The default is off with or without the flag**, so the pin guards against W&B changing the default, which it controls.

**Recommendation for steps 4 and 6 *(mine)*:** check every call for free, and flag `infra_error` if a check ever fails.
- **Every call:** `usage.completion_tokens_details.reasoning_tokens` must be 0. W&B reports it on every response, including sub-calls.
- **Root calls:** the rendered prompt must end with the thinking-off block. The calls already request `prompt_logprobs`, so this check costs nothing.

---

## 5. Concurrency and the CoreWeave email

**Rishi dropped the email on 2026-10-10** and set the cap to **30 requests in flight**.

**The probe found the limit: 20 requests in flight per model name.** It sent bursts of short requests (64 output tokens, `enable_thinking: False`, no retries), 10 s apart:

| Burst | Got 200 | Got 429 |
|---|---|---|
| 30 × `:step0` | 24 | 6 |
| 24 × `:step0` (run 1) | 20 | 4 |
| 24 × `:step0` (run 2) | 20 | 4 |
| 27 × `:step0` | 20 | 7 |
| 20 × `:step0` (run 1) | 20 | 0 |
| 20 × `:step0` (run 2) | 20 | 0 |
| 15 × `:step0` + 15 × `:step20` | 30 | 0 |

- **Exactly 20 get through on one model name.** The first 30-request burst got 24 through because a few requests finished before the last ones arrived.
- **The limit is per model name, not per user.** 15 + 15 split across two checkpoints all succeeded, which is 30 in flight at once.
- **The 429 comes from the inference service behind the training gateway.** The error body names `https://api.inference.wandb.ai/v1/chat/completions` as the URL that refused.
- **There is no `x-request-id` header,** only Cloudflare's `cf-ray`. Step 4's limiter should log `cf-ray` and count 429s per step.
- This explains the earlier results: 8 never hit a 429, and 32 did.

**What this means for the limiter (step 4) *(mine)*:**
- One shared limiter with a total cap of 30, plus a cap of 20 for each model name.
- In training, root calls go to `:stepN` and sub-calls to `:step0`, so neither cap is hit as long as neither kind needs more than 20 at once.
- Evals of the base model send everything to `:step0`, so they run at 20.
- Jittered backoff on 429 stays as a safety net. The step 4 load test should show zero 429s at these caps.

**Left unasked with the email:**
- Whether Qwen3.8-27B training stays supported. CoreWeave's [public list](https://docs.coreweave.com/products/post-training/serverless-training/available-models) shows only Gemma 4 26B A4B, Llama 3.1 8B and Nemotron 3.5 Lightning.
- Who owns trained adapters (§3).

Neither blocks a step. Storage is fine: Rishi confirmed that the ~23 GB comes from his other projects.

**Credit used by the probes:** about $0.005 for the thinking test and the 30-request burst, $0.018 for the 24, 27 and split bursts, and $0.008 for the two 20-request bursts. That's **about $0.03** in all, from the ~$218 inference pot.

---

## 6. Should thinking be off? (double-check, 2026-10-10)

Rishi asked for a second look at the decision itself, not just whether W&B applies it. **The short answer: thinking off was inherited from the 20-step run, not tested.** The research's only reason for it is that it "changes tokens, cost and behaviour" (research §5.3). Here is what holds up and what doesn't.

### What outside sources say

- **Qwen3.8-27B is built to think.** Its [model card](https://huggingface.co/Qwen/Qwen3.8-27B) says thinking is on by default. The agentic scores it reports (Terminal Bench 73.0, SWE-bench Pro 61.7) use thinking-mode sampling, and the card recommends keeping past thinking in context for agents (`preserve_thinking`). It reports no non-thinking agentic scores. So the model's headline strength comes from the mode we turn off.
- **The RLM paper warns about thinking models.** Qwen3-235B-A22B as an RLM only went from about 30% to 38% on OOLONG, because many trajectories ran out of output tokens ([Zhang et al., App. B](https://arxiv.org/html/2512.24601v3)). Its own RL run used an Instruct model, Qwen3-4B-Instruct, with 4,096 output tokens per turn, and it doesn't state a thinking setting.
- **A reproduction found overthinking failure modes.** It found endless verification and long essays instead of answers. Those came from recursion depth 2, not from a thinking switch ([Think, But Don't Overthink](https://www.alphaxiv.org/abs/2603.02615)).
- **Direct comparisons are thin.** I found no study of thinking on versus off for multi-turn agentic RL. The closest, [AgentWebBench](https://arxiv.org/pdf/2604.10938), finds thinking comparable or slightly better on research-style agent tasks with Qwen3.

### What our own data says (replay test, ~$0.10, approved by Rishi)

I took 6 real root turns from the 20-step run's step-0 (base model) episodes: 2 first turns, 2 middle and 2 late. I re-ran each with thinking on at each `reasoning_effort` level, with up to 8,192 output tokens.

| Mode | Output tokens per turn (mean) | vs off | Over the 3,072 cap | Reply had a code block | Ran out of tokens |
|---|---|---|---|---|---|
| Off (the original turns) | 387 | 1× | 0 of 6 | 6 of 6 | 0 |
| Thinking, `low` | 842 | 2.2× | 0 of 6 | 6 of 6 | 0 |
| Thinking, `medium` | 916 | 2.4× | 0 of 6 | 6 of 6 | 0 |
| Thinking, `xhigh` (the template's default) | 2,940 | 7.6× | 2 of 6 | 5 of 6 | 1 (8,192 tokens of thinking on a first turn, no answer) |

- **Thinking grows with the turn.** Low and medium added little on first turns (~100 reasoning tokens) and 750-1,700 on late turns, where the model decides what to aggregate and submit.
- **The mechanics would work.** The returned logprobs cover every token, thinking included, and the thinking text comes back separately in `message.reasoning`. ART could train on thinking tokens with the existing token-id rebuild.
- **6 turns says nothing about accuracy.** It only measures length and cost.

### What it would cost

- **The 20-step baseline:** episodes averaged 5.2 root turns and ~1,990 root output tokens, about $0.006 of the ~$0.016 per episode.
- **`low` or `medium`:** about $0.026 per episode, ~1.6× today's *(inferred)*. That counts the extra output tokens plus re-reading kept thinking in later turns. The $150 production ceiling would buy ~75 steps instead of ~120.
- **`xhigh`:** about $0.065 per episode, ~4× today's *(inferred)*, so about 30 steps. It also breaks the 3,072-token turn cap and runs out of tokens outright, which is exactly the RLM paper's failure.
- **Training compute:** loss-bearing tokens roughly double at medium. The training pot (~$992) isn't the limit, so this doesn't matter.

### Sub-calls

Sub-calls should stay off whatever the root does *(mine)*. They're capped at 1,024 output tokens, which thinking would eat (the paper's failure again), and they're part of the environment, not the trained model. Each episode makes up to 30 of them, so even ~600 thinking tokens each would add up to ~$0.05 an episode. I didn't measure this; it's an estimate.

### Verdict

- **Keep thinking off for the second seed (step 10).** It has to match the first run to count as a replication.
- **Never use `xhigh`.** It costs ~4×, breaks the turn cap and sometimes never answers.
- **Thinking at `medium` is a real option for the production root, but only if it earns its cost.** It costs ~1.6× per episode (about 75 production steps instead of 120). It never broke the turn cap in the replay, and it's the mode the model is tuned for.
- **The open question is accuracy, and our current environment can't settle it cheaply.** The archived code doesn't run in place.

**Recommendation for the planning thread *(mine)*:** add one arm to the step 11 baselines: the base model as an RLM with thinking on at `medium`, next to the thinking-off arm, on the same 64-task selection set. That's about 64 more episodes, ~$2. Production uses root thinking at `medium` only if it beats thinking off by a clear margin, about 12 points or more, which is roughly 2 SE at n = 64. Otherwise production stays off. Sub-calls stay off either way.

Sampling note: Qwen recommends temperature 0.7, top-p 0.8 for non-thinking use and 1.0, 0.95 for thinking. RL keeps temperature 1.0 and no top-p cut, so that the sampled tokens' logprobs match the policy, as in the 20-step run. That holds for either mode.

---

## Files

- `checkpoints/qwen3-8-27b-grpo-v1/step20/`: the downloaded adapter, git-ignored.
- `.gitignore`: `checkpoints/` added (uncommitted).
- The check scripts are in this session's scratchpad, not the repo: `lora_meta.py`, `lora_pull.py`, `lora_storage.py`, `thinking_scan.py`, `live_probe.py`, `limit_probe.py` and `thinking_replay.py`, plus their JSON results. The thinking-tail check moves into `rlm_rl/client.py` in step 4.
