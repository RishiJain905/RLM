# RLM Inference Loop: Recommended Strategy

> High-level design for our model-agnostic RLM harness. Based on the RLM paper ([rlm-paper.md](../rlm-paper.md)) and the Claude Science "Python REPL + host bridge" design. Meant to be talked through before we build.

---

## 1. The Short Answer

Build **one small loop** around **a persistent, sandboxed Python REPL** that talks back to the host through a **host bridge**.

- The **REPL** holds the data (the long prompt as `context`) and the work in progress (variables).
- The **host bridge** is a tiny RPC channel from the REPL to the harness. Every "harness primitive" (LLM calls, sub-RLMs, tools, final answer) is a Python function in the REPL that forwards to the host.
- The **host** owns everything that needs control or trust: model clients, budgets, recursion depth, logging, tool access.
- The **root model** only ever sees metadata and truncated output. It writes code; code does the heavy lifting.

This is exactly the RLM paper's Algorithm 1, with the Claude Science bridge as the mechanism that makes it extensible and model-agnostic.

---

## 2. Why the Claude Science Design Fits

From Alec's (@AlecTPhD) post: *Claude Science has a python repl with a host bridge that gives it access to harness primitives so it can do lots of things programmatically: MCP tool calling, skill creation, agent profile creation, LLM calling, delegation, and more.*

That is a generalization of the RLM idea:

| RLM paper | Claude Science | What it means for us |
|---|---|---|
| `context` variable in a REPL | Python REPL as the workspace | Data lives in the REPL, not the prompt |
| `llm_query` / `rlm_query` from code | LLM calling, delegation via the bridge | Recursion is just a bridge call |
| `FINAL` / `FINAL_VAR` | (harness-defined) | Termination is a bridge call too |
| Fixed toolset | MCP tools, skills, agent profiles | New capabilities = new bridge functions, no loop changes |

The key insight: **the model composes primitives in code instead of calling them one at a time as tool calls.** A loop over 1,000 chunks is one turn, not 1,000 turns. This is what removes the "verbalized sub-call" bottleneck the paper calls out, and it extends the same win to every other primitive (tools, search, delegation).

---

## 3. Architecture

```
┌──────────────────────── Host (harness process) ────────────────────────┐
│                                                                         │
│   Root loop ──► Model client (OpenAI-compatible: vLLM, ART, APIs)       │
│      │                                                                  │
│      │ code                         Bridge (HTTP)                       │
│      ▼                              ├─ llm / llm_batch  → model clients │
│   ┌──────────── Sandbox ────────┐   ├─ rlm (sub-RLM)    → new root loop │
│   │ Persistent Python kernel    │   ├─ tool (MCP)       → MCP clients   │
│   │  context = <long prompt>    │◄─►├─ submit(final)    → ends the loop │
│   │  host.* stubs (RPC client)  │   └─ budgets, depth, logging          │
│   └─────────────────────────────┘                                       │
│                                                                         │
│   Trajectory log (every root turn + every bridge call) ──► eval / RL    │
└─────────────────────────────────────────────────────────────────────────┘
```

Three components, each small:

1. **Root loop** (host): prompt → model → extract code → execute → truncated feedback → repeat.
2. **Sandbox kernel**: a persistent Python process. Start with a subprocess Jupyter kernel; move to Docker later. Never `exec` in the host process for anything beyond local experiments.
3. **Bridge**: a thin HTTP RPC with a fixed list of functions. The sandbox has no API keys and no network; it can only ask the host.

---

## 4. The Loop

```python
def run_rlm(query, context, model, depth=0, budget=Budget()):
    repl = Sandbox.start(bridge=Bridge(model, depth, budget))
    repl.set("context", context)

    history = [system_prompt(model), user_msg(query, metadata(context))]
    for _ in range(budget.max_iterations):
        reply = model.chat(history)
        code = extract_code(reply)
        result = repl.run(code)               # bridge calls happen in here
        if repl.submitted:                    # host.submit(...) was called
            return repl.final_value
        history += [reply, truncate(result)]  # prefix + length only
    return force_final(repl, history)         # ask once for an answer, else fail
```

That is the whole core. Everything else is a bridge function or a budget.

### 4.1 Rules the loop enforces

- **Metadata, not content.** The first user message contains the query plus `len(context)`, type, a short prefix, and how to access it. Never the full context.
- **Truncate all feedback.** stdout, errors, and return values are clipped (e.g. first ~2K chars + total length). The model learns to summarize into variables.
- **Answers come from variables.** The final answer can be built in the REPL and be longer than the context window.
- **Budgets live on the host.** Max root iterations, max depth, max sub-calls, max tokens or dollars, wall-clock timeout. The sandbox cannot bypass them.

---

## 5. Bridge Primitives

Start with the paper's minimum and grow toward the Claude Science set. Each one is a plain function in the REPL.

### Phase 1: the RLM core

| Function | Purpose |
|---|---|
| `llm(prompt, model=None)` | One sub-LM call. Default model is the configured sub-model. |
| `llm_batch(prompts, model=None)` | Parallel sub-calls. **The main performance lever** (paper's runs were all sequential). |
| `rlm(context, query)` | Spawn a sub-RLM with its own REPL, depth + 1. Falls back to `llm` at max depth. |
| `submit(value)` | Return the final answer and end the loop. Replaces `FINAL` / `FINAL_VAR` text parsing. |

### Phase 2: the Claude Science extensions

| Function | Purpose |
|---|---|
| `tool(name, **args)` / `tools()` | Call and list MCP tools through the host's MCP clients. |
| `skill_save(name, code, doc)` / `skill_load(name)` | Persist reusable helpers across runs (e.g. a good chunker). |
| `profile(name, system_prompt, model, tools)` | Define an agent profile once, reuse it for delegation. |
| `delegate(profile, task, context=None)` | Async sub-agent; returns a handle to await. Generalizes `rlm`. |

Keep the list short and stable. A stable bridge API is what makes the harness model-agnostic and what a trained model learns to use.

---

## 6. Key Design Decisions

### 6.1 Termination by function call, not by text

The paper reports final-answer detection is brittle (plans emitted as answers, 16% / 13% of training turns misused `FINAL` / `FINAL_VAR`). Making `submit()` a bridge call removes the parsing problem. Keep a `FINAL(...)` text fallback only for compatibility with the paper's prompts.

### 6.2 Parallel sub-calls from day one

`llm_batch` plus a host-side concurrency limit. This is the single biggest runtime win available and the paper explicitly lists it as future work.

### 6.3 Model-agnostic by interface, tuned by profile

- One client interface: OpenAI-compatible chat completions. That covers vLLM (local Qwen), ART-trained models, and hosted APIs.
- Per-model **profile**: context size, sub-call chunk guidance (paper: ~500K chars for GPT-5-mini vs ~10–15K for Qwen3-8B), batching hint, code-block format. The paper shows one prompt does not fit all models.
- The bridge API and loop never change between models. Only the profile does.

### 6.4 Code extraction and errors

- Accept one fenced ```python block per turn. If none, feed back a short format reminder.
- Exceptions go back as truncated tracebacks. Syntax errors were a major failure mode for Qwen3-Coder, so this feedback path matters, and it is a natural RL penalty signal.

### 6.5 Sandboxing

- Phase 1: subprocess kernel with timeouts per cell and no secrets in its environment.
- Phase 2: Docker (or a cloud sandbox) with network off; the bridge is the only way out.

### 6.6 Logging is a first-class feature

Log every root turn and every bridge call as a tree (root → sub-calls → sub-RLMs) with tokens, cost, latency. This one log serves debugging, cost analysis, evals, and is the raw material for RL trajectories.

---

## 7. How This Connects to Training (first milestone)

- **Train the root; sub-calls use the same base model.** The paper's key insight: a good sub-call model is just a good general model. The root's job (REPL use, decomposition, when to recurse) is what needs learning. Per the decisions in §9, sub-calls go to the same base model and training is straight RL (likely SDPO).
- **Trajectory = root conversation.** Each RL rollout is the root model's multi-turn chat (system, metadata, code turns, truncated outputs). Sub-calls are environment steps, served by the same base model.
- **Same harness for inference and training.** The RL environment should *call this loop*, not reimplement it. ART's OpenAI-compatible client plugs in as the root model client.
- **Reward:** task correctness on submit (exact match / F1), plus small penalties for no submit, syntax errors, and blowing the budget.
- **Starter tasks:** synthetic needle tasks and MRCRv2-style multi-needle retrieval (the paper's RL run generalized from 32–64K / 2 needles to 512K–1M / 8 needles), then OOLONG-style aggregation.
- **No SFT bootstrap.** Go straight to RL (see §9).
- **Start at depth 1.** Deeper recursion helped GPT-5 but hurt weaker coders. Add depth once the root writes reliable code.

---

## 8. Phased Plan

| Phase | Deliverable | Done when |
|---|---|---|
| 0 | Minimal loop: local subprocess REPL, `llm`, `submit`, one model | Solves a synthetic needle task beyond the model's window |
| 1 | `llm_batch`, `rlm`, budgets, truncation, trajectory log | Reproduces paper-shaped gains on a small OOLONG/needle eval |
| 2 | Model profiles, vLLM + API clients, small eval suite | Same harness runs Qwen3 locally and a hosted model unchanged |
| 3 | RL environment wrapping the loop (ART) | First training run on the root model shows reward going up |
| 4 | Claude Science primitives: MCP tools, skills, profiles, delegation, Docker sandbox | Harness handles real agentic tasks, not just long-context QA |

---

## 9. Decisions (agreed with Rishi, 2026-10-10)

1. **Sub-call model during training:** use the **same base model as the root**. Training will likely use SDPO if it works with RL (being checked in a separate thread).
2. **Bridge transport:** **HTTP**.
3. **Claude Science surface:** leverage **as much of Claude Science as we can**. Any information on its infra and surrounding design should feed into the harness, so Phase 4 primitives move up the priority list rather than waiting until after training.
4. **Reference implementation:** read `alexzhang13/rlm` for reference only; **write our own loop** so it fits our training setup.
5. **SFT vs RL:** **straight RL, no SFT bootstrap.** SFT is not effective for the planned base model, Qwen 3.8 27B (being tested in a separate thread).
