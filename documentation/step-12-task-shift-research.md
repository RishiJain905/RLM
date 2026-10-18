# Research notes behind the step-12 task shift (2026-10-18)

Compiled from seven read-only research passes in two rounds (round 2 below). Claims attributed to a paper are what it says; lines marked *inference* are ours. The current plan is [step-12-multistep-plan.md](step-12-multistep-plan.md).

## RLMs: when sub-calls help

**"Recursive Language Models"**, Zhang, Kraska and Khattab ([arXiv 2512.24601](https://arxiv.org/abs/2512.24601); [blog](https://alexzhang13.github.io/blog/2025/rlm/)).
- **Benchmarks, ordered by information density:** S-NIAH (constant), BrowseComp-Plus with 1K docs (multi-hop), OOLONG trec_coarse (linear), OOLONG-Pairs (quadratic, 20 queries), LongBench-v2 CodeQA.
- **Without sub-calls** (Table 1):
  - GPT-5 drops from 56.5 to 36 on OOLONG and from 58.0 to 43.9 on Pairs.
  - Base GPT-5 scores 0.1 F1 on Pairs.
  - Qwen3-Coder on CodeQA does *better* without sub-calls: 66 against 56.
- **Recursion depth (v3):** deeper recursion helps GPT-5 on Pairs (43.9 → 76 at depth 3) but hurts Qwen3-Coder on OOLONG.
- **Observed strategies:** peek at the context's structure, filter with regex guided by priors, chunk, map sub-calls over the chunks, stitch the results in REPL variables. Without a cost warning, Qwen3-Coder made thousands of sub-calls, so a batching line was added to its prompt.
- **RLM-Qwen3-8B:** trained by SFT on 1,072 filtered trajectories over 750 LongBench-Pro tasks. Median gain +28%: BrowseComp+ 2 → 14, OOLONG 24 → 32. v3 adds an RLVR run on MRCRv2 that went from 32-64K training contexts to 512K-1M.

**Zhang & Khattab, the "harness" post** ([Jul 2026](https://alexzhang13.github.io/blog/2026/harness/)).
- RL on Qwen3-30B-A3B as an RLM (prime-rl).
- Training tasks: MRCRv2, GraphWalks, LongBench-Pro, OOLONG, OOLONG-Pairs and Ada-LEval, on short inputs. Evaluated at 8-32x longer.
- About 10x the evaluation gain of a plain transformer for the same training gain.
- A "decompose" nudge was needed before a general strategy emerged.

**Prime Intellect RLMEnv** ([blog](https://primeintellect.ai/blog/rlm)).
- Evaluations only, no RL results.
- On OOLONG with labels visible, GPT-5-mini made zero sub-calls and used regex.
- RLMs helped most on real data.

**alphaXiv, "Reinforcing RLMs"** ([May 2026](https://www.alphaxiv.org/blog/reinforcement-learning-for-rlms)).
- Qwen3.5-4B, SFT then RL, with root and child models sharing one policy.
- Rewards came from a rubric judge, and each child inherits the root's advantage.
- Task: verbatim evidence selection across several papers.

**Context-Folding / FoldGRPO** ([arXiv 2510.11967](https://arxiv.org/abs/2510.11967)).
- Trained on BrowseComp-Plus and SWE-Bench Verified, with token-level process penalties.
- Beat ReAct on BrowseComp-Plus: 0.620 against 0.478.

**MemAgent** ([arXiv 2507.02259](https://arxiv.org/abs/2507.02259)).
- HotpotQA needles, with questions the base model solves already filtered out.
- Trained at 32K, it held up to 3.5M.

**Skeptical results.**
- [arXiv 2603.02615](https://arxiv.org/abs/2603.02615): S-NIAH fell to 85-90% inside an RLM.
- SRLM ([arXiv 2603.15653](https://arxiv.org/abs/2603.15653)) argues most of the gain is programmatic access to the context, not recursion.

*Inference:* our needle, callgraph, repo localization and edit families are constant-density and lexical. Under outcome-only CISPO, few sub-calls is the right policy for them.

## Keeping GRPO groups informative

- **GRPO** ([arXiv 2402.03300](https://arxiv.org/abs/2402.03300)): an all-equal group has zero advantage. The paper uses G = 64.
- **DAPO** ([arXiv 2503.14476](https://arxiv.org/abs/2503.14476)):
  - Dynamic sampling (§3.2) over-samples tasks and drops the all-0 and all-1 groups. It was the largest single ablation gain, AIME 42 → 50.
  - It also uses clip-higher (0.2 / 0.28), a token-level loss and overlong shaping.
- **CISPO / MiniMax-M1** ([arXiv 2506.13585](https://arxiv.org/abs/2506.13585)):
  - It clips the importance weight, not the token update.
  - It still uses DAPO's dynamic sampling and filters data by pass rate: strictly between 0 and 0.9 for math, between 0 and 0.5 for logic.
- **Online Difficulty Filtering** ([arXiv 2504.03380](https://arxiv.org/abs/2504.03380)): keeping a 0.3-0.7 pass-rate band works best and reaches the same reward in under half the steps.
- **SPEED** ([arXiv 2506.09016](https://arxiv.org/abs/2506.09016)): screen each task with a few rollouts first, keep it if 0 < p < 1. 2-6x faster.
- **GRESO** ([arXiv 2506.02177](https://arxiv.org/abs/2506.02177)):
  - Over 90% of all-equal tasks stay all-equal on their next draw, and about 20% recover later.
  - It skips a task with a probability that grows with its run of all-equal groups.
- **VCRL** ([arXiv 2509.19803](https://arxiv.org/abs/2509.19803)): filters tasks by reward variance and replays the high-variance ones.
- **RAGEN** ([arXiv 2504.20073](https://arxiv.org/abs/2504.20073)):
  - In the multi-turn "Echo Trap", falling reward std is the first warning.
  - Keeping the top 25% of tasks by std helps.
  - About 4 rollouts each over many distinct tasks works best.
- **rStar2-Agent** ([arXiv 2508.20722](https://arxiv.org/abs/2508.20722)):
  - Python tool use with outcome reward only.
  - Resample-on-Correct prefers successes with fewer tool errors.
  - It dropped problems solved 8/8 offline.
- **Agentic search** — ASearcher ([2508.07976](https://arxiv.org/abs/2508.07976)), DeepResearcher ([2504.03160](https://arxiv.org/abs/2504.03160)), Kimi-Researcher:
  - drop questions that are never solved, or solved 50% of the time or more;
  - drop questions that need too few tool calls (ASearcher's second stage keeps only those needing 5 or more);
  - drop questions the base model solves without the tool.
- **SimpleTIR** ([arXiv 2509.02479](https://arxiv.org/abs/2509.02479)): drops trajectories with "void turns" (no code and no answer). This is close to our broken replies.

## Rewarding the behaviour directly

- **R1-Searcher** ([2503.05592](https://arxiv.org/abs/2503.05592)): a +0.5 retrieval reward in stage 1 only. Without it, Llama never learned to retrieve.
- **OTC** ([2504.14870](https://arxiv.org/abs/2504.14870)): an additive tool reward was gamed. Multiplying it by correctness fixed that.
- **SPEAR** ([2509.22601](https://arxiv.org/abs/2509.22601)): a constant tool reward caused excess turns. A decaying one avoided that.
- **Proof-of-Use** ([2510.10931](https://arxiv.org/abs/2510.10931)): "tool-call hacking" — decorative calls whose output never reaches the answer.

## Task sources (licences checked 2026-10-18; verify before download)

| Source | Licence | Use | Why it needs sub-calls |
|---|---|---|---|
| [MuSiQue](https://github.com/StonyBrookNLP/musique) | CC BY 4.0 (Wikipedia text) | train: research | 2-4 hops; distractors chosen for word overlap; ~25K questions |
| [QAMPARI](https://github.com/samsam3232/qampari) | repo CC0 (Wikipedia text) | train: research lists | one proof passage per answer; list F1 |
| [2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop) | Apache-2.0 repo, data licence unstated | maybe | keep the compositional and inference types |
| [DeepDive](https://huggingface.co/datasets/zai-org/DeepDive) | Apache-2.0 | later | BrowseComp-style; needs an offline corpus |
| [LongBench-Pro](https://huggingface.co/datasets/caskcsg/LongBench-Pro) | Apache-2.0 | maybe | what RLM-Qwen3-8B trained on; check its grading |
| [RepoQA](https://github.com/evalplus/repoqa) | Apache-2.0 | train: code search | functions described without their keywords |
| [SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench) / V2 | CC BY 4.0, per-repo licences | train: bug fixing | already local (V2) |
| [SWE-smith](https://github.com/SWE-bench/SWE-smith) | MIT | train: bug fixing | issues describe behaviour, not location |
| [BrowseComp-Plus](https://huggingface.co/datasets/Tevatron/browsecomp-plus) | MIT tag; corpus copyright uncertain | eval only | RLM paper eval |
| OOLONG / OOLONG-Pairs | no clear licence | eval only | RLM paper evals |
| LongBench-v2 CodeQA, FRAMES | Apache-2.0 | eval only | common evals |
| NoLiMa | Adobe non-commercial | not used | reuse the idea: no word overlap between question and needle |

**Environments to borrow from:** [verifiers](https://github.com/PrimeIntellect-ai/verifiers) RLMEnv (MIT) and [alexzhang13/rlm](https://github.com/alexzhang13/rlm) (MIT), which has an OOLONG environment for prime-rl.

---

# Round 2 (Rishi: "look at other papers too… maybe it's a multi-step fix")

## Decomposition / long-context training (research pass 2, 2026-10-18)

- RAO, Recursive Agent Optimization (Gandhi, Kumar, Neubig et al., May 2026) https://arxiv.org/abs/2605.06639: shared policy at every node, launch_subagent() in a Python REPL, CISPO-style, G=8, 16 tasks/batch, async. Oolong-Real: Qwen3-VL-30B-A3B LoRA r32 lr 3e-5, 32K window, ~60K inputs. Node reward = own success + λ·mean child success (GPT-5-mini judge); leave-one-out baseline over roots shared down the tree; inverse depth-frequency loss weights. λ=0 where base already recursed; λ=0.4 on Oolong to warm start; no SFT; prompt encourages delegation. Single agent learned regex shortcuts; recursive agent had a print-everything phase (steps 40-80) then chunk-and-delegate. Mostly depth 1. No gain on short ART-E.
- Trained Agentic Context Management (Sandlund, Oct 2026) https://arxiv.org/abs/2610.02404: read_chunk + spawn_subagent tools; LoRA r32 SFT of Qwen3.6-35B-A3B on 4,618 synthetic problems with frontier oracle demos; per-agent budgets 3K-12K vs documents 4K-14K (input > budget forces decomposition); root gradient ×4. Families: sum/count tree reduce, sequential folds, top-k frequency, label-then-aggregate over real text, NIAH, NarrativeQA. 8K budget matches GPT-5.4 (1M ctx) on OOLONG-synth >40K. RL after SFT was high-variance and latched onto simpler strategies.
- SearchSwarm (Jun 2026) https://arxiv.org/abs/2606.09730: base never called call_sub_agent; SFT only on correct-root trajectories, sub-agent reports masked; BrowseComp 43.4 → 68.1.
- Context Language Models (Shao, Koh et al., Sep 2026) https://www.alphaxiv.org/abs/2609.37725: stepwise GRPO 8×32, 70 steps, Qwen3.5-9B; efficiency advantage among successes only (weight 0.25); BC-Plus 28.8 → 42.5; plain summarization harness reached 42.1.
- Think, But Don't Overthink (Mar 2026) https://arxiv.org/abs/2603.02615: depth 1 helps on OOLONG; depth 2 and S-NIAH hurt; runtime blow-ups.
- LoongRL (ICLR 2026) https://arxiv.org/abs/2510.19363: KeyChain UUID chains; filter pass@8 0 or 1 (277K → 72K); stage II drops 8/8; GRPO G=8 no SFT; two-way substring reward beat EM/F1/judge; 16K → 128K generalization. KeyChain is greppable.
- QwenLong-L1 https://arxiv.org/abs/2505.17667: SFT 5.3K R1 traces, RL 20K → 60K, carry forward zero-score items; EM-or-judge reward; G=8.
- QwenLong-L1.5 https://arxiv.org/abs/2512.12967: synthesis 42.7K → 14.1K (KG multi-hop, NL2SQL aggregation); drop questions answerable without the document; equal batch share per task type and per-type advantage std; 20K → 60K → 120K.
- MEM1 https://arxiv.org/abs/2506.15841: PPO on N concatenated QA (train 2, test 16); a format reward cut EM 0.709 → 0.466; SFT collapsed at 16 objectives.
- h1 https://arxiv.org/abs/2510.07312: chained problems, horizon curriculum 1 → 5; full-horizon outcome RL needs exponentially many samples, curriculum polynomial.
- LongRLVR https://arxiv.org/abs/2603.02146: verifiable evidence-selection reward; RULER-QA 73.2 → 88.9 on 14B.
- APR https://arxiv.org/abs/2504.15466: SFT 500K spawn/join traces then GRPO G=5; child threads 6.1 → 8.2.
- Parallel-R1 https://arxiv.org/abs/2509.07980: format SFT on easy problems; accuracy-only reward → 13.6% parallel use; parallel-only reward → 80.3% but lower accuracy; alternating (20% of steps tiered +1.2 parallel&correct, +1.0 correct, −1 else) → 63% usage, similar accuracy.
- Chain-of-Agents https://arxiv.org/abs/2406.02818, LLM×MapReduce https://arxiv.org/abs/2410.09342, THREAD https://arxiv.org/abs/2405.17402, ReDel https://arxiv.org/abs/2408.02248 (over/undercommitment). Memory RL: Mem-α https://arxiv.org/abs/2509.25911, Memory-R1 https://arxiv.org/abs/2508.19828, ReMemR1 https://arxiv.org/abs/2509.23040.
- Patterns: make the shortcut fail (budget below input); bootstrap (SFT) then RL; gate any usage bonus on correctness or apply part-time; filter by pass rate and by "solvable without sub-calls"; length/horizon curricula; G=8, per-task advantage normalization; monitor sub-call rate and depth, expect degenerate phases.

## Agentic RL (research pass 2, 2026-10-18)

Credit assignment
- GiGPO https://arxiv.org/abs/2505.10978: episode advantage + step advantage over repeated "anchor" states; ALFWorld +12-13, WebShop +9; reduces to GRPO without repeated states (REPL states rarely repeat).
- ARPO https://arxiv.org/abs/2507.19849 / AEPO https://arxiv.org/abs/2510.14545: branch partial rollouts at high-entropy tool steps (entropy rises after tool results); ARPO ~half tool-call budget; AEPO GAIA 47.6% (Qwen3-14B) from 1K samples.
- Tree-GRPO https://arxiv.org/abs/2509.21240 (step-level trees, ~1/4 budget per secondary sources); RTMC https://arxiv.org/abs/2604.11037 (SWE-V 49.0 → 52.2).
- Turn-level: MT-GRPO https://arxiv.org/abs/2505.11821, IGPO https://arxiv.org/abs/2510.14967 (turn reward = gain in gold-answer probability), Turn-PPO https://arxiv.org/abs/2512.17008, EMPG https://arxiv.org/abs/2509.09265.
- RAGEN-2 https://arxiv.org/abs/2604.06268: template collapse invisible to entropy; MI proxy; SNR prompt filter.
- L0 https://arxiv.org/abs/2506.23667 (REPL agent, pure RL; looping on hard data until all-0/all-1 groups dropped); ReTool https://arxiv.org/abs/2504.11536 (cold-start SFT then PPO lr 1e-6, interpreter output masked; code ratio ~98%); ToRL https://arxiv.org/abs/2503.23383, ZeroTIR https://arxiv.org/abs/2505.07773.

Can RL create a rare behaviour?
- Mostly no without prior probability: Yue et al. https://arxiv.org/abs/2504.13837, Invisible Leash https://arxiv.org/abs/2507.14843, RL Squeezes/SFT Expands https://arxiv.org/abs/2509.21128, Spurious Rewards https://arxiv.org/abs/2506.10947.
- Conditional exceptions: ProRL https://arxiv.org/abs/2505.24864, BroRL https://arxiv.org/abs/2510.01180, Interplay https://arxiv.org/abs/2512.07783 (≥~1% prior exposure, edge-of-competence tasks), f(g(x)) https://arxiv.org/abs/2509.25123, Behavior Injection https://arxiv.org/abs/2505.18917.
- Warm starts are small and correctness-filtered: WebSailor https://arxiv.org/abs/2507.02592 (~2k trajectories, many tool calls; cold start converges higher and keeps tool calls high), DeepDive (858 traces), R1 https://arxiv.org/abs/2501.12948, GLM-4.5 https://arxiv.org/abs/2508.06471 (iterative self-distillation), Kimi-K2 https://arxiv.org/abs/2507.20534. Pure RL works when the base already uses the tool (Search-R1 https://arxiv.org/abs/2503.09516, ZeroSearch https://arxiv.org/abs/2505.04588). Pass@k advantages https://arxiv.org/abs/2508.10751.

Stability (LoRA / agentic)
- LoRA Without Regret https://thinkingmachines.ai/blog/lora/: LoRA LR ≈10x full FT; rank 1 enough for RL; all layers. Tinker RL page: start 1e-5, group 4, num_substeps 1-4.
- Steps per batch: DAPO 16; MiniMax-M1 16 with CISPO, FP32 LM head (corr 0.9 → 0.99), β2 0.95; ScaleRL https://arxiv.org/abs/2510.13786 off-policy k ≤ 8, drop prompts with pass ≥ 0.9; Qwen https://arxiv.org/abs/2512.01374; M2PO https://arxiv.org/abs/2510.01161.
- Entropy: Clip-Cov/KL-Cov https://arxiv.org/abs/2505.22617; LLD https://arxiv.org/abs/2512.04220 (likelihood of correct and incorrect responses both fall in GRPO tool RL → low-confidence outputs → collapse; regularize tokens whose likelihood dropped). Plausibly explains our early-ending turns.
- Group size: ScaleRL 8-32 similar at fixed batch; Knapsack RL https://arxiv.org/abs/2509.25849 adaptive rollouts up to 93 on hard prompts.
- KL: most drop it.

Task synthesis with difficulty control
- SailorFog-QA (WebSailor), DeepDive https://arxiv.org/abs/2509.10446 (KG walks, keep if GPT-4o+search fails 4/4; 3,250 QA; G=16), ASearcher https://arxiv.org/abs/2508.07976 (drop if 0/16, ≥50%, or solved in ≤1 search turn; stage 2 ≥5 tool calls), WebShaper https://arxiv.org/abs/2507.15061, WebExplorer https://arxiv.org/abs/2509.06501, TaskCraft https://arxiv.org/abs/2506.10055, Absolute Zero https://arxiv.org/abs/2505.03335.

## Our harness vs published RLM harnesses (research pass 2, 2026-10-18)

Ours (checked in code): root Qwen3.8-27B thinking medium, temp 1.0, 3,072 tokens/turn read (4,096 edit); 12 turns read / 30 edit; REPL output 8,000 chars/turn; 30 sub-call prompts/episode hard cap with countdown; sub-call prompt cut at 24K chars (end cut, "[prompt truncated]" seen only by the sub-model; root not warned); sub-calls = frozen base step0, thinking off, temp 0.3, 1,024 output tokens; depth 1; prompt nudge: "use code for exact filtering… sub-calls for judgments that need reading comprehension". client.py:65-68 comment "training keeps thinking off" is stale.

| | Ours | alexzhang13/rlm (MIT) | PI RLMEnv | RLM paper / harness post |
|---|---|---|---|---|
| Sub-model | frozen base, no thinking, 1,024 out | training env sub_model = rollout model, 4,096 out, only root trained | defaults to root model, up to 5 tool turns | GPT-5 root, GPT-5-mini subs; RLM-Qwen3-8B root-only training |
| Input per sub-call | 24K chars | ~500K (prompt), ~100K (addendum) | n/a | 500K GPT-5; 10-15K chars per call for Qwen3-8B |
| Cap per episode | 30 prompts | none (optional USD budget); ~20/batch | none (optional token budget) | |
| Turns | 12 / 30 | 30 (20 in training example) | 50 (oolong 30) | 20 in MRCRv2 RL |
| REPL output | 8K | 20K | 8,192 | |
| Nudge | neutral | "orchestrator, not a solver"; "your context window is small"; "delegate everything else"; plan first; turn 0 no final answer | light; env_tips for SFT data then stripped | "strongly encouraged"; harness post adds a decompose nudge |
| Usage shaping | none | optional rubric gate (min_subcall), off by default | none | |

Evidence: under-use common and prompt-driven (PI); sub-calls hurt on CodeQA (RLM paper); over-use risk without warnings (Qwen3-Coder thousands of calls); Kimi K2.5 PARL: trained orchestrator, frozen sub-agents, spawning reward vs "serial collapse" with finish-rate guard, both annealed to zero (https://www.kimi.com/blog/kimi-k2-5); RAO / alphaXiv shared-weight children; Budget Tracker https://arxiv.org/abs/2511.17006 (showing budget helps).

Inferred reasons for under-use: (1) sub-call is weaker than the root (no thinking, 1,024 tokens) so self-reading is safer under exact-match; (2) one sub-call reads only 3x a printed turn; (3) 1,024 output tokens caps labels per call (~150-250), 30 calls ≈ 6K items < L5's ~7K lines; (4) silent truncation drops the question in oversize prompts; (5) neutral/scarcity framing; (6) no warm start.

Suggested harness changes by impact: (1) fix truncation (error to the REPL or keep the instruction), raise input to ~60-100K if context allows, state the real number; (2) sub-call output ~4,096 tokens, consider low thinking for sub-calls, keep frozen base; (3) prompt v4 orchestrator addendum (plan decomposition first, own window is small, batch ~20 prompts, terse structured outputs); (4) budget by tokens/characters or raise cap to 60-100 with per-batch ceiling; (5) if needed: tips-then-strip or PARL-style annealed spawning reward with finish-rate guard on semantic families only.

Sources: https://github.com/alexzhang13/rlm, https://github.com/PrimeIntellect-ai/verifiers, https://github.com/PrimeIntellect-ai/research-environments, https://primeintellect.ai/blog/rlm, https://arxiv.org/abs/2512.24601, https://alexzhang13.github.io/blog/2026/harness/, https://www.kimi.com/blog/kimi-k2-5, https://primeintellect.ai/rl-glossary/subagents-and-compaction, https://alphaxiv.org/blog/reinforcement-learning-for-rlms, https://arxiv.org/abs/2605.06639, https://arxiv.org/abs/2511.17006, https://github.com/NovaSky-AI/SkyRL/tree/main/examples/train/rlm

## Coding with sub-agents (research pass 2, 2026-10-18)

Build from spec
- SWE-Dev https://arxiv.org/abs/2505.16975 (Apache-2.0, 14k train/500 test, Docker per package ~100 GB): requirements doc + repo with target functions removed, hidden tests; Hard = call-tree depth ≥3, 5+ functions, ~190 lines over 3 files. PPO Qwen2.5-7B Hard 6.7 → 12.3; organizer/coder multi-agent fine-tuning 7.4 → 12.7; hand-built pipelines no better than Self-Refine.
- Own generator on SWE-smith repos https://arxiv.org/abs/2504.21798 (MIT, 128 repos, images exist): LM Rewrite strips functions to header+docstring; Breakpoint https://arxiv.org/abs/2506.00172: difficulty rises with call-graph centrality and number of interdependent functions removed (55% → 0%).
- Commit0 https://arxiv.org/abs/2412.01769 (MIT harness, 54 libraries): held-out eval (~29% lite, ~6% full).
- Evals: NL2Repo-Bench https://arxiv.org/abs/2512.12730, ProgramBench https://arxiv.org/abs/2605.03546 (models write monolithic code), FEA-Bench https://arxiv.org/abs/2503.06680, NoCode-bench/Doc2Feat https://arxiv.org/abs/2507.18130, Web-Bench https://arxiv.org/abs/2505.07473, RepoExec https://arxiv.org/abs/2406.11927; ClassEval, BigCodeBench(-Hard) sized for one pass.

Bug fixing
- SWE-smith combined bugs + Breakpoint discovery mode (several corrupted interdependent functions found from failing tests): one sub-call per suspect.
- Real multi-file fixes (≥3 files): SWE-rebench https://arxiv.org/abs/2505.20411 (CC-BY-4.0), R2E-Gym (Apache-2.0); BugPilot https://arxiv.org/abs/2510.19898.
- Evals: SWE-bench Pro https://arxiv.org/abs/2509.16941 (~4 files per fix), Loc-Bench https://arxiv.org/abs/2503.09089.
- Parallel investigation evidence: FastContext https://arxiv.org/abs/2606.14066 (GRPO explorer, F1 + 3-6 parallel calls bonus, +5.5 SWE-Pro subset), SWE-grep https://cognition.com/blog/swe-grep, Context-Folding https://arxiv.org/abs/2510.11967 (58.0% SWE-V with 32K active context; RL +8.8), SWE-Debate https://arxiv.org/abs/2507.23348, SWE-Search https://arxiv.org/abs/2410.20285, SWE-Edit https://arxiv.org/abs/2604.26102, DeLM https://arxiv.org/abs/2606.10662.

Decomposition papers: prompted (Parsel https://arxiv.org/abs/2212.10561, FunCoder https://arxiv.org/abs/2405.20092, CodeChain https://arxiv.org/abs/2310.08992, Self-Organized Agents https://arxiv.org/abs/2404.02183, MapCoder, AgentCoder, CodePlan, MetaGPT, ChatDev); trained (MAGRPO https://arxiv.org/abs/2508.04652 — naive joining underperformed a single 7B; AFM https://arxiv.org/abs/2508.13167; Kimi K2.5 PARL; Puppeteer https://arxiv.org/abs/2505.19591; Agent-as-Tool https://arxiv.org/abs/2507.01489; MAPoRL https://arxiv.org/abs/2502.18439).

Pitfalls: interface mismatch (Cognition https://cognition.com/blog/dont-build-multi-agents; MAGRPO) → root writes signatures/types/docstrings first and passes them to every sub-call, reward only integrated hidden tests; serial collapse and spurious parallelism (PARL needed annealed rewards; Context-Folding branches never returned under outcome-only); overhead (MAST https://arxiv.org/abs/2503.13657 failure modes); Anthropic: coding has fewer truly parallel parts than research, ~15x tokens https://www.anthropic.com/engineering/multi-agent-research-system; tasks too small make delegation pure overhead; leakage (tests shipped, memorised libraries); hidden tests in a pristine container; Docker storage.

