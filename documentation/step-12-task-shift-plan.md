# prod-v1 from checkpoint 20: new task families that need sub-calls

> **Superseded** by [step-12-multistep-plan.md](step-12-multistep-plan.md) (2026-10-18, after the second research round).

*2026-10-18. Supersedes [step-12-subcall-plan.md](step-12-subcall-plan.md). The run is stopped at checkpoint 20 and stays there until this is agreed and built. Nothing here is built yet. Budget left: ~$107 for steps and ~$27 for evals.*

## What the research says

**The RLM paper** ([arXiv 2512.24601](https://arxiv.org/abs/2512.24601)) and its follow-ups make three points.
- **Sub-calls pay off as information density rises:** a lone needle (constant) < OOLONG (linear: every item needs a label from its meaning) < OOLONG-Pairs (quadratic). On OOLONG-Pairs, GPT-5 scores 0.1 F1 without an RLM and 58 with one.
- **Lexical or code-shaped tasks don't need sub-calls.** On S-NIAH and LongBench-v2 CodeQA, the REPL alone did as well or better, and Qwen3-Coder scored 66 without sub-calls against 56 with them. When an aggregation's labels can be found with regex, GPT-5-mini makes zero sub-calls (Prime Intellect's RLMEnv).
- **RL-trained RLMs used dense tasks.** Zhang & Khattab's July 2026 harness run trained on MRCRv2, GraphWalks, LongBench-Pro, OOLONG, OOLONG-Pairs and Ada-LEval, on short inputs, and transferred to inputs 8-32x longer. They needed a "decompose" nudge in the prompt before a general strategy emerged. alphaXiv's RLM RL used multi-paper evidence selection.

**Keeping groups at 1-3 of 4.** GRPO gives zero gradient to an all-equal group. At 4 rollouts a task that passes 80% of the time comes out all-equal 41% of the time, which is our step-20 picture. Every recipe fixes this the same way:
- DAPO's dynamic sampling was its biggest single gain (AIME 42 → 50).
- MiniMax (CISPO) kept only tasks with pass rates strictly between 0 and 0.9.
- Online Difficulty Filtering keeps 0.3-0.7.
- GRESO skips tasks that were all-equal recently: 90% of them stay that way.
- ASearcher kept only questions that need 5 or more tool calls.

**Rewarding sub-calls directly gets farmed.** OTC, SPEAR and Proof-of-Use all saw this. When it was used at all, it was small, paid only on a correct answer, and decayed. None is proposed here.

Full reports with links: [research notes](step-12-task-shift-research.md).

## Proposed families for the rest of the run

Every family needs sub-calls by construction: either more text than the root can read (12 turns × 8K characters ≈ 96K) or judgments about meaning that regex can't make. Needle, repo localization, callgraph and edit synth leave the mix (Rishi, 2026-10-18).

### 1. Research: multi-hop questions over a large corpus (30%)
- **Source:** MuSiQue answerable questions (CC BY 4.0), 2-4 hops. Each question's own 20 paragraphs, gold plus lexically similar distractors, are packed among paragraphs from other questions to 100K-800K characters.
- **Why sub-calls:** the hop-2 entity only appears after reading hop 1, and keyword hits are mostly look-alike distractors.
- **List questions:** QAMPARI (repo CC0; text from Wikipedia) asks things like "every film directed by X", with one proof passage per answer among distractors. That is a map-reduce over chunks with `llm_batch`.
- **Grading:** answer F1 (exact = 1). List answers by set-F1.
- **Levels:** hops 2 → 4, context 100K → 800K, more distractors.

### 2. Context search: aggregation v2 (30%)
- **Keep:** our generator and its permissively licensed labelled sources (CLINC, MASSIVE, LEDGAR, arXiv, HuffPost news).
- **Drop:** the two code-only kinds.
- **Add the denser kinds the paper shows need sub-calls:**
  - pairs: "which pairs of users both wrote about X and about Y", graded by F1 over pairs;
  - timelines: "the first date user U wrote about X";
  - "most / second-most common topic for user U".
- **Grading:** counts use OOLONG's 0.75^|error| partial credit, and every task is re-drawn when a metadata guess would score.

### 3. Code understanding: semantic search over large repositories (20%)
- **Task:** RepoQA-style (Apache-2.0). A function described in plain words without its identifiers, to be found among thousands in a whole repository dumped into `context` (300K-1M characters). Higher levels ask for every function that does X: semantic classification over many functions, then a list.
- **Source:** our local permissively licensed repositories, with descriptions written once by Claude.
- **Grading:** exact function, or set-F1 for lists.

### 4. Bug fixing in large repositories: edit real (20%)
- **Keep:** the real tasks with hidden tests, which still give mixed results.
- **Change:** the issue describes behaviour only, and from level 2 the repository is too large to read file by file, so locating the bug means scanning many files (`llm_batch` over files) before fixing.
- **Retire:** levels where the base model fails everything.

### 5. Keeping groups mixed
- **Per-task history and pass-rate band:** keep each task's (and level's) pass history, skip tasks that were all-equal on their last draw (GRESO), and aim each family's level at a 30-70% pass rate.
- **Rollouts:** 6 tasks × 6 rollouts instead of 8 × 4. That is about the same episodes per step, and at an 80% pass rate it cuts all-equal groups from 41% to 26%.
- **Cost:** no over-sampling, so step cost doesn't rise.

### 6. Prompt v4: a decomposition nudge
One paragraph: "when the context is bigger than you can read, split it, map `llm_batch` over the pieces, combine the results in code". The harness paper needed this nudge for a general strategy to emerge.

## Checks before resuming
1. **Pilot (~$6):** checkpoint 20 runs ~15 tasks per family and level, with sub-calls on and off. Keep a level only if sub-calls clearly help and the pass rate falls between 0.25 and 0.75 (ASearcher/DeepResearcher-style filtering).
2. **New selection set (selection-v4)** from the new families' held-out splits, with checkpoint 20 re-scored on it as the new baseline (~$3). The capability check is unchanged.
3. **Held out for the final report:** BrowseComp-Plus, OOLONG and OOLONG-Pairs, and LongBench-v2 CodeQA, the RLM paper's own tasks. They never enter training.

## Decisions for Rishi
1. **The four families and the 30 / 30 / 20 / 20 mix.**
2. **Downloads:** MuSiQue (CC BY 4.0), QAMPARI (CC0 repo, Wikipedia text) and RepoQA (Apache-2.0). Wikipedia text is CC BY-SA underneath, and our licence bar has not yet decided on share-alike.
3. **Spend:** ~$9 for the pilot and the new baseline, from the eval budget.
4. **6 × 6 rollouts and prompt v4.**

**Build time:** most of a day: three generators, the grading, selection-v4, the history-based sampling, tests and a dry run.
