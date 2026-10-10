# RLM Project Map (high level)

> Whole-project view as of 2026-10-10. Two parts: **Part 1** trains our RLM root model with RL, **Part 2** builds the harness it runs in. This doc is the map; deep dives (starting with Part 1) get their own docs. Visual version: [project-map.html](project-map.html).
>
> Grounded in: [rlm-paper.md](rlm-paper.md), [rlm-inference-strategy.md](archive/rlm-inference-strategy.md), [sdpo-rlm-compatibility.md](archive/sdpo-rlm-compatibility.md), [rl-method-for-rlm.md](archive/rl-method-for-rlm.md), [rlm-three-way-results.md](archive/rlm-three-way-results.md).

---

## 1. The goal in one paragraph

Build a **Recursive Language Model** system: a model that never reads a long input directly, but works on it as a `context` variable inside a Python REPL, writes code to inspect it, and calls itself on pieces of it (RLM paper §3). We get there in two parts. Part 1 trains **Qwen3.8-27B** to be a good RLM *root* with RL, using **RLSD** as the policy update. Part 2 builds a **model-agnostic harness** around a persistent Python REPL and an **HTTP host bridge**, modeled on Claude Science, which later exposes MCP tools, skills, agent profiles, LLM calls and delegation. The trained model then plugs into the harness as its default root.

```
 PART 1: TRAIN THE ROOT                          PART 2: BUILD THE HARNESS
 ─────────────────────                           ─────────────────────────
 Tasks ──► RL environment ──► rollouts           Root loop ◄──► Sandbox REPL
            (REPL + context                         │              │ host.* stubs
             + llm_query/llm_batch                  │              ▼
             + submit)                              │         HTTP bridge
               │                                    │   llm · llm_batch · rlm · submit
               ▼                                    │   later: MCP tools · skills ·
 reward ──► RLSD update (W&B serverless, ART)       │          profiles · delegate
               │                                    │
               ▼                                    ▼
      Trained RLM root (LoRA) ──────────────► default root model in the harness
```

---

## 2. Locked decisions

These are settled. Deep dives build on them rather than reopening them.

| Area | Decision | Source |
|---|---|---|
| Base model | Qwen3.8-27B, LoRA | three-way results, memory |
| Platform | W&B serverless RL via OpenPipe ART (pinned 0.5.20) | sdpo-rlm-compatibility §2 |
| Training style | Straight RL, no SFT bootstrap | strategy doc §9.5 |
| Policy update | RLSD (reward-anchored self-distillation) on a CISPO backbone | rl-method-for-rlm §7, three-way results |
| Sub-calls | Same base model as the root; only the root is trained | strategy doc §9.1 |
| Harness core | Own loop (alexzhang13/rlm is reference only), persistent REPL, HTTP bridge | strategy doc §9.2, §9.4 |
| Harness surface | Use as much of Claude Science's design as we can | strategy doc §9.3 |
| Termination | `submit()` bridge call, not `FINAL(...)` text parsing | strategy doc §6.1 |

---

## 3. Part 1: RL training of the RLM root

**Objective:** a Qwen3.8-27B LoRA that is measurably better than the base model at RLM work: probing the `context`, decomposing, using sub-calls when code alone can't do it, and submitting a correct answer.

### 3.1 Building blocks

| Block | What it is | State today |
|---|---|---|
| **Task suite** | Long-context tasks where the answer lives in `context`. Must include tasks regex can't solve, so sub-calls matter. Curriculum keeps pass rate in the 20-80% band. | Toy tasks in `rlm_rl/archive/tasks.py`, `ticket_tasks.py`. Rishi is researching datasets. |
| **RL environment** | Sandboxed REPL subprocess with `context`, `llm_query`, `llm_batch`, `submit`; limits on turns, sub-calls and output size. The root only sees metadata and truncated output. | Working prototype `rlm_rl/archive/rlm_env.py` (6 turns, 20 sub-calls, 2K chars). Sub-calls already go over a local HTTP endpoint. |
| **Reward** | Correctness on `submit` (exact or partial), small penalties for no submit, syntax errors, budget overruns. | Prototype in the env. |
| **Policy update: RLSD** | Reward sets the *sign* of every token's update; the self-teacher (same checkpoint with feedback in context) only scales the *size*. Teacher sees a successful sibling's code or env feedback, never the gold answer. | Working via logprob injection (`rlm_rl/archive/shaping.py`, `sdpo.py`). |
| **Training infra** | W&B serverless, LoRA + CISPO, group size 8, concurrency cap 8, token ids rebuilt from prompt logprobs. | Working (notebooks 01-03). |
| **Evaluation** | Held-out tasks, including longer contexts than training. Track accuracy, sub-calls per episode, leakage (submit without work). | 48-episode eval exists; needs to grow to 100+. |

### 3.2 Where Part 1 stands

| Step | Status |
|---|---|
| 1.1 Platform smoke tests (GRPO on both Qwens) | Done |
| 1.2 SDPO on serverless via logprob injection | Done |
| 1.3 Pick the method in theory (RLSD) | Done |
| 1.4 Three-way test: base 0.35, GRPO 0.46, SDPO 0.38, **RLSD 0.54** | Done (3 steps, not yet significant vs GRPO) |
| 1.5 Confirm RLSD: 20 steps, sub-call-heavy tasks, larger eval | **Running** in another thread |
| 1.6 Lessons doc from the testing phase | Next |
| 1.7 **Deep dive: RLSD + RL environment design** (tasks, reward, RLSD settings, eval) | Next, this is the following session |
| 1.8 Build the official environment and task suite | After 1.7 |
| 1.9 Small pilot run, then the production run | After 1.8, needs credit sign-off |
| 1.10 Final eval and export the LoRA | End of Part 1 |

**Output of Part 1:** a trained RLM root LoRA, an eval report, and the exact bridge API it was trained on.

---

## 4. Part 2: the RLM-specific, model-agnostic harness

**Objective:** one small loop that can run *any* chat model that writes Python as an RLM, tuned for our trained model, and later able to do real agentic work through Claude Science-style primitives.

### 4.1 Architecture

- **Host (harness process):** the root loop, model clients (OpenAI-compatible: vLLM, ART, hosted APIs), budgets, recursion depth, logging, tool access. Everything that needs trust or control.
- **Sandbox:** a persistent Python kernel holding `context` and the work in progress. No keys, no network. Subprocess first, Docker later.
- **HTTP bridge:** the only way out of the sandbox. Each harness primitive is a plain Python function in the REPL that forwards to the host.
- **Model profiles:** per-model settings (context size, chunk guidance, batching hint). The loop and bridge never change between models.
- **Trajectory log:** every root turn and bridge call as a tree, with tokens, cost, latency, token ids and logprobs. Serves debugging, evals and future RL.

### 4.2 Bridge primitives, in order

| Phase | Primitives | Purpose |
|---|---|---|
| A. RLM core | `llm`, `llm_batch`, `rlm`, `submit` | Sub-calls (batched by default), recursive sub-RLMs, structured termination |
| B. Claude Science | `tool` / `tools` (MCP), `skill_save` / `skill_load`, `profile`, `delegate` | Real tools, reusable helpers, agent profiles, async sub-agents |

### 4.3 Build order

| Step | Deliverable |
|---|---|
| 2.1 | Minimal loop: subprocess REPL, HTTP bridge, `llm` + `submit`, one model |
| 2.2 | `llm_batch`, `rlm`, budgets, truncation, trajectory log |
| 2.3 | Model profiles and clients; same harness runs our LoRA and a hosted model unchanged |
| 2.4 | Claude Science primitives: MCP tools, skills, profiles, delegation |
| 2.5 | Docker sandbox, network off |

---

## 5. How the two parts connect

1. **Same contract.** The RL environment in Part 1 is a narrow version of the harness core in Part 2: REPL, `context`, sub-call functions, `submit`, truncated feedback, budgets. The prototype env already serves sub-calls over a local HTTP endpoint, the same transport the harness bridge uses. The strategy doc's rule is that the environment should *call* the harness loop rather than reimplement it.
2. **The trained model learns the bridge API.** Function names, signatures, prompt format and truncation rules seen in training are what the model will expect at inference. Today the env uses `llm_query` while the harness design uses `llm`. **These must be frozen to one API before the production run.**
3. **Logs feed training.** The harness logs exact token ids and logprobs, so harness transcripts can be re-scored by a teacher and become future RL data.
4. **Integration milestone.** Serve the LoRA behind an OpenAI-compatible endpoint, plug it in as the root, and compare it with the base model inside the full harness.

Practical order: Part 1 leads. Harness steps 2.1-2.2 can be built alongside it, because the official RL environment needs that same core anyway.

---

## 6. Open questions for the deep dives

- **Bridge API freeze:** final names and signatures shared by the env and the harness (`llm` vs `llm_query`, `rlm` at depth > 1 or not during training).
- **Datasets:** which real tasks force sub-calls and long context (Rishi researching).
- **RLSD settings at scale:** clip range, lambda, all-fail-group add-on, teacher lag.
- **Do Claude Science primitives appear in training?** Probably not in Part 1; decide whether a later RL round trains tool use.
- **Serving the trained model:** W&B inference vs a local server for the harness.
