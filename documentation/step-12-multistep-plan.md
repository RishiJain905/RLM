# prod-v1 from checkpoint 20: a multi-step fix for sub-call learning

*2026-10-18. Supersedes [step-12-task-shift-plan.md](step-12-task-shift-plan.md) and [step-12-subcall-plan.md](step-12-subcall-plan.md). Research notes: [step-12-task-shift-research.md](step-12-task-shift-research.md). The run is stopped at checkpoint 20 (selection 66.7%, capability 75%, audit 24/24 clean) and stays there until this is agreed. Nothing here is built yet. Budget left: ~$107 for steps, ~$27 for evals, and the $1,000 training credit (training and SFT GPU-hours only).*

## Rishi's direction (2026-10-18)

Sub-calls are sub-agents. In research, or in coding ("make a calculator": front end, buttons, back end...), the root splits the job, hands each part to a sub-call, and combines the results. Research and coding are what users will run the model for. So both should be families where the rollouts that delegated and got it right are the ones GRPO rewards. Needle and repo localization go. The fix can be multi-step, not only a family swap.

## Why the run stalled: five causes, each with its own fix

| # | Cause | Evidence | Fix (step) |
|---|---|---|---|
| 1 | Most tasks never needed a sub-call | 80% of episodes were needle / callgraph / repo loc / edit, which code or grep solves; they averaged 0.04 sub-calls. The RLM paper's own CodeQA and S-NIAH results show the same | New families (step 1) |
| 2 | The sub-agent is weaker than the root | Sub-calls go to the frozen base, thinking off, 1,024 output tokens, 24K characters in. The root thinks. Reading it yourself is the safer policy | Stronger sub-agents (step 2) |
| 3 | The prompt frames sub-calls as scarce side tools | A neutral nudge plus a hard 30-call countdown. The published harnesses say "you are an orchestrator, your window is small, delegate" and set no count cap | Prompt v4 and budgets (step 2) |
| 4 | Groups are mostly all-right or all-wrong | 3-6 of 8 groups all-solved since step 13. At 4 rollouts a task that passes 80% is all-equal 41% of the time | Task screening and 6 × 6 groups (step 4) |
| 5 | No bootstrap for coding delegation | The base already sub-calls on 85% of semantic tasks, but coding delegation is new. RL mostly amplifies what already happens (Yue et al., RL-squeezes-SFT-expands); agents that needed a new tool habit had a small SFT warm start (WebSailor, SearchSwarm, Parallel-R1, ReTool) | Pilot, then a warm start if needed (step 3) |

## Step 1. Families: every one needs sub-agents by construction

Each family follows two rules from the research.
- **A single agent can't do it within its budget.** RAO and Sandlund set the budget below the input size.
- **A no-sub-call ablation fails it.** ASearcher and QwenLong-L1.5 drop items solvable without the tool. The pilot (step 3) enforces this per level.

| Family | Share | What a task is | Why delegation wins | Grade | Source |
|---|---|---|---|---|---|
| **Research** | 30% | Multi-hop questions (2-4 hops) and list questions over a corpus of 100K-800K characters, packed with look-alike distractors | Each hop or each candidate is its own sub-question; the hop-2 entity only appears after reading hop 1; the corpus is far beyond what the root can print | answer F1; list set-F1 | MuSiQue (CC BY 4.0), QAMPARI (CC0; Wikipedia text), later DeepDive-style KG walks (Apache-2.0) |
| **Context search** | 20% | Aggregation v2: label-then-count, pairs ("users who wrote about X and Y"), timelines, top-k per user; no code-only kinds | Every line needs a semantic label; pairs are quadratic, where the paper shows the biggest RLM gains | OOLONG partial credit 0.75^\|err\|; pair F1 | our generator and licensed labels (CLINC, MASSIVE, LEDGAR, arXiv, news) |
| **Build from a spec** | 30% | A repo with 3-8 interdependent functions or a whole module removed, plus a requirements doc; hidden tests | The root writes the interfaces, each sub-agent drafts one function or file, the root integrates and runs the visible tests | hidden unit tests, pass or fail | our own generator on the permissively licensed repos we already have (SWE-smith-style LM rewrite; Breakpoint's "functions removed" difficulty dial); SWE-Dev (Apache-2.0, 14K tasks) optional, ~100 GB of images |
| **Find and fix bugs** | 20% | Several interdependent functions corrupted at once (Breakpoint "discovery mode"), and real fixes touching 3+ files (SWE-rebench V2, already local) | One sub-agent per suspect function or hypothesis, reporting back before the root patches (FastContext, SWE-grep, Context-Folding gains) | hidden tests | local SWE-smith repos; SWE-rebench V2 |

**Out:** needle, repo localization, callgraph, edit synth, and edit real's single-file levels.

**Held out, never trained on:** BrowseComp-Plus (subset), OOLONG-Pairs, LongBench-v2 CodeQA, Commit0-lite. Evaluated at the end only.

## Step 2. Harness: make sub-agents worth calling

| Change | Now | Proposed | Why |
|---|---|---|---|
| Sub-agent output | 1,024 tokens | 4,096 | Write a real function, or label a whole chunk (alexzhang13/rlm trains at 4,096) |
| Sub-agent input | 24K characters, cut silently | 60K, and an oversize prompt raises an error in the REPL | 3x a printed turn is too little reading per call |
| Sub-agent thinking | off | low, for coding families only | Drafting code needs some reasoning; cost checked in the pilot |
| Sub-agent model | frozen base | keep frozen base | Kimi K2.5 PARL and the RLM paper train the orchestrator only; it keeps credit assignment simple |
| Budget | 30 prompts, countdown | 80 prompts, or a character budget; keep the remaining-count line | Removes the scarcity framing while keeping the cost visible (Budget Tracker) |
| Turns | 12 read / 30 edit | research and context 20, coding 30 | Orchestration needs plan, dispatch and combine turns |
| Prompt v4 | neutral | Orchestrator addendum: "you are an orchestrator, your own window is small; plan the decomposition first; for code write the signatures and contracts first and give them to every sub-call; batch about 20 prompts per `llm_batch`; ask for terse, structured outputs" | The harness post needed a decompose nudge; interface-first prompting addresses the main multi-agent coding failure (Cognition, MAGRPO) |
| REPL output cap | 8,000 characters | keep | Keeps the root from printing everything itself |
| History | full history re-sent each turn | Optional: fold old REPL outputs into one-line summaries | 55% of spend is re-sent history; compaction is standard in Prime Intellect's harness |

## Step 3. Pilot, then a warm start only if needed

**Pilot (~$8 of eval budget).**
- Checkpoint 20 with prompt v4 runs ~12 tasks per family and level, 2 rollouts each, with sub-calls on and with sub-calls disabled.
- Keep a level only if sub-calls clearly win and its pass rate with sub-calls falls between 0.2 and 0.8.
- Also record how often the policy delegates unprompted on the coding families. That is the "prior exposure" the RL papers say decides whether RL alone can grow a behaviour.

**Warm start (only for families where delegation is rare in the pilot).**
- Build a few hundred correct, delegating trajectories by rejection sampling: the pilot's own successes plus targeted extra sampling (≤$15 of step budget).
- Filter: correct, sub-calls used, and their outputs reach the answer (Proof-of-Use).
- Mask sub-call outputs and REPL observations from the loss, then run a short SFT on the **$1,000 training credit**. W&B's SFT jobs bill training GPU-hours, not inference.
- This is the WebSailor, SearchSwarm and ReTool recipe at our scale. If delegation is already common, skip it.

## Step 4. RL mechanics for mixed groups

1. **Screen tasks by history.**
   - Skip a task (or seed) whose last draw was all-equal (GRESO: over 90% stay that way).
   - Retire tasks solved with no sub-calls (the ASearcher filter).
   - Steer each family's level toward a 30-70% pass rate (Online Difficulty Filtering's band).
2. **6 tasks × 6 rollouts** instead of 8 × 4: about the same episodes, more mixed groups.
3. **Advantages per family:** equal family shares in each step, and normalization within a family (QwenLong-L1.5).
4. **Optional delegation bonus, part-time and correctness-gated.**
   - On 1 step in 5, a correct rollout that delegated and used the results scores +0.2 over a correct one that didn't; a wrong rollout gets nothing extra.
   - The bonus is annealed to zero by about step 50.
   - Parallel-R1 (13.6% → 63% usage at the same accuracy), RAO and PARL used variants of this. An ungated usage reward gets farmed (OTC, SPEAR, Proof-of-Use), so the gate and the anneal are not optional.
   - Off unless the first 10 steps show delegation fading.
5. **Keep:**
   - learning rate 1e-5, with the 0.25/0.5 warm-up after any rollback;
   - masking and redrawing broken replies (SimpleTIR's void-turn filter is the same idea);
   - the warm-up call and the 14-minute retries;
   - outcome reward everywhere else.
6. **New dashboard panels:**
   - sub-calls per episode by family;
   - share of correct rollouts that delegated;
   - share of tasks solved without sub-calls;
   - mixed groups per step.
   - Watch for RAO's "print everything" phase and for serial collapse.

## Step 5. Evals
- **selection-v4:** built from the new families' held-out splits. Checkpoint 20 is re-scored on it as the new baseline (~$3). The capability check is unchanged.
- **Final report:** the held-out RLM benchmarks above, run on checkpoint 20 and on the final checkpoint.

## Order of work and cost

| # | Work | Cost |
|---|---|---|
| 1 | Harness (step 2) and prompt v4, with tests | $0 |
| 2 | Generators: research, context search v2, build-from-spec, bug hunt; tests; data checks | $0 (needs downloads: MuSiQue, QAMPARI) |
| 3 | Pilot | ~$8 eval |
| 4 | Warm start, only if the pilot says so | ≤$15 step budget for sampling + training credit for SFT |
| 5 | selection-v4 and checkpoint 20's baseline | ~$3 eval |
| 6 | Dry run, then resume RL from checkpoint 20 | the rest: ~$90, about 45-60 steps at the longer episodes |

**Time:** about 2 days of building before the pilot, most of it the coding generators and their Docker grading.

## Decisions for Rishi
1. **Families and mix:** research 30 / context search 20 / build-from-spec 30 / find-and-fix 20, with needle, repo localization, callgraph and edit synth out.
2. **Harness changes** in step 2, including thinking at low for coding sub-agents (costs more per sub-call).
3. **Downloads:** MuSiQue and QAMPARI. Their Wikipedia text is CC BY-SA underneath, which our licence bar hasn't ruled on yet. SWE-Dev (~100 GB of Docker images on D:) is optional.
4. **Spend:** ~$11 of eval budget for the pilot and baseline, and up to $15 of step budget for warm-start sampling if needed.
5. **The part-time delegation bonus:** keep it in reserve (recommended) or turn it on from the start.
