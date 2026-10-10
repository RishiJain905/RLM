# Recursive Language Models (RLM): Paper Overview

> Zhang, Kraska, Khattab (MIT CSAIL). *Recursive Language Models.* arXiv:2512.24601v3, May 2026.
> Paper: https://arxiv.org/html/2512.24601v3 · Reference code: https://github.com/alexzhang13/rlm

---

## 1. TL;DR

An RLM is an **inference-time scaffold**, not a new architecture. Instead of stuffing a long prompt into the model's context window, the prompt is loaded as a **variable inside a persistent Python REPL**. The model only sees metadata about it, writes code to inspect and slice it, and **calls itself (or a smaller LM) recursively from inside that code** on pieces of the prompt. The final answer is assembled in REPL variables and returned explicitly.

Headline results:

- Handles inputs **10x+ beyond the base model's context window** (6–11M tokens on BrowseComp-Plus).
- GPT-5 inside an RLM beats vanilla GPT-5 and common scaffolds at comparable cost: median **+26% over compaction, +130% over CodeAct with sub-calls, +13% over Claude Code**.
- A small post-trained model, **RLM-Qwen3-8B** (SFT on ~1K trajectories), beats base Qwen3-8B by a median **28%** and approaches vanilla GPT-5 on three tasks.

---

## 2. Problem: Context Rot and Why Existing Fixes Fail

- **Context rot.** Model quality degrades as prompts get longer, and degrades *faster* the more complex the task is. "Effective context window" depends on the task, not just the advertised limit.
- **Compaction / summarization** (what most agents do when full) assumes early details can be forgotten. That breaks on tasks that need dense access to the whole input.
- **Retrieval / coding agents** can only pull snippets into context, then hit compaction anyway.
- **Self-delegation agents** (the model writes each sub-call out as text) are bottlenecked by output length: the model must verbalize every sub-call one at a time.

The authors frame tasks by **how required work scales with input length**:

| Scaling | Example | What it needs |
|---|---|---|
| Constant | Needle in a haystack | Find one thing |
| Linear | OOLONG | Touch almost every entry |
| Quadratic | OOLONG-Pairs | Reason over pairs of entries |

---

## 3. The Method

### 3.1 Core idea

Treat the prompt as part of the **environment**, not part of the model's input. The model interacts with it symbolically (code) and recursively (sub-calls).

### 3.2 The loop (Algorithm 1)

1. **Init REPL.** Start a persistent Python REPL. Store the full prompt `P` as a variable (`context`). Register a sub-call function (`llm_query`, and `rlm_query` for deeper recursion).
2. **Metadata only.** The root model's first message contains constant-size metadata about `P` (length, short prefix, how to access it). The full prompt never enters the root context.
3. **Iterate.** Each turn the root model emits code. The REPL executes it and updates state. Only a **truncated** view of stdout (prefix + length) is appended to the root's history.
4. **Recurse programmatically.** Code can call the LM on slices or transforms of `P`, inside loops, and store results in variables.
5. **Terminate.** The model returns `FINAL(answer)` (answer written directly) or `FINAL_VAR(var_name)` (return a REPL variable). That value is the response.

If root turns are capped at `c` tokens, there are at most `K/c` root iterations (K = context size), each of which can launch many sub-calls. In practice iterations are capped per recursion level.

### 3.3 Three design choices that matter (vs. a "deceptively similar" scaffold)

| Weak scaffold (Algorithm 2) | RLM fix |
|---|---|
| Prompt placed in the model's context, so it inherits the window limit and needs compaction | Prompt lives in a REPL variable |
| Final answer emitted as a "Finish" action, so output ≤ context window | Output is built in variables and can be arbitrarily long |
| Sub-calls verbalized one by one | Sub-calls are made from code, so they can loop over thousands of slices |

These three points are the real contribution. Everything else is implementation detail.

### 3.4 Recursion depth

- **Depth 0:** REPL only, no sub-calls.
- **Depth 1 (default):** sub-calls go to a plain LM (`llm_query`).
- **Depth > 1:** sub-calls can be full RLMs (`rlm_query(context, query)`), falling back to `llm_query` at max depth.

In the GPT-5 runs, the root is GPT-5 and sub-calls use the cheaper GPT-5-mini.

### 3.5 System prompt (Appendix C)

Prompts are task-agnostic and fixed across tasks. Key elements:

- "You are tasked with answering a query with associated context."
- The `context` variable holds "extremely important information about your query."
- `llm_query` queries a sub-LLM from the REPL; guidance on how much text a sub-call can handle (~500K chars for GPT-5-mini, ~10–15K chars for Qwen3-8B).
- Sub-calls are "strongly encouraged."
- Worked examples: search for a magic number, read a book in chunks, aggregate per-chunk results.
- `FINAL(...)` / `FINAL_VAR(...)` for the answer. When ready, the model "cannot write anything other than the final answer in the FINAL or FINAL_VAR tags."

Model-specific variants exist: Qwen3-Coder needed an extra sentence telling it to **batch sub-calls** (it otherwise over-called); Qwen3-8B got smaller chunk sizes to fit its ~32K window.

---

## 4. Experimental Setup

### 4.1 Benchmarks

| Benchmark | Task | Scaling | Size |
|---|---|---|---|
| S-NIAH | Single needle in a haystack (RULER-style) | Constant | 8K–1M tokens |
| BrowseComp-Plus (1K docs) | Multi-hop QA over 1,000 documents | Constant docs, multi-hop | 6–11M tokens |
| OOLONG (trec_coarse) | Label and aggregate nearly every entry | Linear | ~131K tokens |
| OOLONG-Pairs | Pairwise aggregation, scored by F1 | Quadratic | ~32K tokens |
| LongBench-v2 CodeQA | Multiple-choice repo understanding | Fixed files | 23K–4.2M tokens |
| LongCoT-mini | Long-horizon reasoning (math, chem, CS, logic, chess) | Reasoning depth | n/a |

### 4.2 Models

GPT-5 (root) + GPT-5-mini (sub-calls); Qwen3-Coder-480B-A35B; Qwen3-8B (for fine-tuning); GPT-5.2 for LongCoT-mini.

### 4.3 Baselines

Base model with full prompt; CodeAct (+BM25 retriever, or +sub-call tool, but prompt in context); a compaction/summary agent; OpenCode and Claude Code (with and without offloading context to a file).

---

## 5. Results

### 5.1 Main table (GPT-5, score %, avg cost per query)

| Method | CodeQA | BrowseComp+ | OOLONG | OOLONG-Pairs |
|---|---|---|---|---|
| Base GPT-5 | 24.0* | 0.0* | 44.0 | 0.1 |
| CodeAct + BM25 | 22.0* | 51.0 | 38.0 | 24.7 |
| CodeAct + sub-calls | 24.0* | 0.0* | 40.0 | 28.4 |
| Compaction agent | 58.0 | 70.5 | 46.0 | 0.1 |
| OpenCode + offloading | 64.0 | 94.0 | 52.0 | 4.8 |
| RLM depth 0 | 58.0 | 88.0 | 36.0 | 43.9 |
| **RLM depth 1** | 62.0 ($0.11) | 91.3 ($0.99) | 56.0 ($0.43) | 58.0 ($0.33) |
| RLM depth 2 | 66.0 | 92.0 | 56.5 | 65.5 |
| RLM depth 3 | 58.0 | 92.0 | 58.0 | 76.0 |

`*` = hit context limits at least sometimes. With Qwen3-Coder, RLM depth 1 lifts OOLONG-Pairs from ≤0.1% to 23.1% F1 and OOLONG by 33%.

### 5.2 Key observations

1. **Scales past 10M tokens** at comparable or lower cost (BrowseComp+ RLM ≈ $0.99/query vs. an estimated $1.50–2.75 to just ingest with GPT-5-mini).
2. **The REPL alone does a lot.** Depth 0 already scales past the window; on CodeQA with Qwen3-Coder, depth 0 beat every sub-calling variant. **Sub-calls matter most on information-dense tasks** (OOLONG-Pairs climbs from 43.9 to 76.0 as depth grows).
3. **Degrades slower with complexity.** Beyond ~16K tokens, RLM consistently beats GPT-5, and the gap widens on linear/quadratic tasks.
4. **Cost.** Median RLM run is cheaper than the median base run; mean is higher because of outlier trajectories. All sub-calls in the paper were **blocking and sequential**, so runtime numbers are pessimistic.
5. **Reasoning, not just long context.** On LongCoT-mini, RLM (GPT-5.2) goes from 38.7 to 50.6 overall, and to **65.6 with decomposition hints** (build a dependency graph, solve nodes in parallel batches with `llm_batch`, memoize verified answers, assemble by lookup).
6. **Training transfers.** RLM-Qwen3-8B, trained on unrelated tasks, beats base Qwen3-8B on all four tasks, is cheaper, and runs 3x faster.

---

## 6. Trajectory Analysis (Emergent Behaviors)

- **Probe, then decompose.** Models first peek at the input (print a prefix, count lines, check structure), then split the work into sub-calls.
- **Use priors to filter.** E.g. on BrowseComp+ the model narrows the search space with code (keyword/regex filtering) before spending sub-calls.
- **Stitch long outputs.** Outputs longer than the window are built by concatenating sub-LM outputs in REPL variables (required for OOLONG-Pairs).
- **First decomposition matters.** In-context examples improve the first decomposition, even when unrelated to the task. Models often recover from a bad first plan, but not always.
- **Code quality is a bottleneck.** Qwen3-Coder trajectories contain far more syntax errors than GPT-5's, and errors propagate into sub-calls. That is why deeper recursion *hurt* Qwen3-Coder.

---

## 7. Training a Native RLM

### 7.1 RLM-Qwen3-8B (SFT / rejection fine-tuning)

- **Data:** 750 LongBench-Pro tasks run with RLM(Qwen3-Coder-480B root, Qwen3-8B sub-calls) → 2,250 trajectories → 1,072 kept after dropping zero-score and single-turn runs.
- **Sample unit:** each **root turn** is a separate SFT example. Turns over ~100K chars dropped.
- **Cleanup:** templated errors patched programmatically (16% of turns misused `FINAL`, 13% misused `FINAL_VAR`).
- **Training:** prime-rl, batch 64, 300 steps, ~48 H100-hours.
- **Key insight:** being a good *sub-call* model ≈ being a good general reasoner. So training focuses on the **root model**: REPL manipulation and deciding when and how to recurse.

### 7.2 RL on MRCRv2

- Qwen3-4B-Instruct trained with RL on MRCRv2 (32–64K tokens, 2 needles): 150 steps, batch 128, 4 rollouts, 4096 max output tokens per turn, max 20 RLM iterations.
- Evaluated on the much harder 512K–1M, 8-needle split, and it **generalized**. This is the most directly relevant result for our RL milestone.

---

## 8. Negative Results (Appendix B)

- **One prompt does not fit all models.** A GPT-5 prompt caused Qwen3-Coder to over-call; one added sentence fixed it.
- **Small models need coding ability.** Qwen3-8B struggled as an untrained RLM.
- **Thinking models run out of output tokens** (Qwen3-235B-A22B only went 30% → 38% on OOLONG).
- **Blocking sub-calls are slow.**
- **Final-answer detection is brittle.** Models sometimes emitted a plan as the final answer.

---

## 9. Limitations and Future Work

- Harder, more natural long-context evals and guardrails are underexplored.
- Sub-call cost can explode on bad trajectories.
- Async sub-calls and proper sandboxes would cut runtime and cost, at the price of complexity.
- RLM trajectories are a form of reasoning that can be bootstrapped and trained; **native RLM training could be a new scaling axis.**

---

## 10. Related Work (positioning)

- **Long-context architectures** (retraining, new attention/memory) vs. **scaffolds** (compaction, MemGPT/Mem0-style memory hierarchies). RLMs hand all context management to the model.
- **Sub-LM decomposition** (ViperGPT, THREAD, ReDel, Context Folding, AgentFold) lets the model pick sub-calls but stays bounded by the window. DisCIPL generates the whole program up front and cannot recover from mistakes. RLMs iterate with REPL feedback.

---

## 11. What This Means For Us

- The three design choices in §3.3 are non-negotiable for our harness.
- A REPL-native harness is model-agnostic by construction: any chat model that writes Python can be a root.
- Explicit, structured termination and async/batched sub-calls fix two of the paper's own negative results.
- Training should target the **root** model, with a frozen sub-call model, starting from SFT-style bootstrapping or straight RL on tasks like MRCRv2 / OOLONG.

See [rlm-inference-strategy.md](rlm-inference-strategy.md) for the harness design.
