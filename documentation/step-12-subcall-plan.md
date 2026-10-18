# prod-v1 at checkpoint 20: why sub-calls aren't growing, and a plan

> **Superseded** by [step-12-task-shift-plan.md](step-12-task-shift-plan.md) (Rishi, 2026-10-18: change the task families, not the grading). The findings below still hold.

*2026-10-18, after checkpoint 20. The run is stopped at checkpoint 20 (stop switch); nothing below is built yet.*

## What the run shows (steps 11-20, branch 2)

| Finding | Evidence |
|---|---|
| Only one task type needs sub-calls | Semantic aggregation (classify lines by topic, then count or list) is 20% of episodes and averages 4.9 sub-calls; every other episode averages 0.04. The policy *does* reach for `llm_batch` when a task needs it (88-100% of semantic aggregation episodes). |
| The other families are solvable without sub-calls | Callgraph by `ast`, edit synth mechanically with grep/read/shell, needle by isolating ~6-32 leaf functions with `ast` (≤8K characters), most of repo loc by grep, edit real as an ordinary coding agent. Needle at level 4 (275K characters): 100%, 0 sub-calls. |
| The sub-call tasks pay almost nothing for being nearly right | Semantic counts are exact-match; a miss earns 0.8 × (1 − err / e_blind), where the blind guess is lines ÷ labels. Answers land within 10% of the truth 67% of the time but are exact 29% of the time, and the blind guess is often close (median 23% off, sometimes 6%), so a count of 110 against 120 earns 0. Semantic aggregation passes 39%. |
| The tasks are too big to count exactly | Level 1 is ~60 lines (LEDGAR) to ~600 (CLINC/MASSIVE); level 5 is up to ~7,000. Sub-calls go to the frozen base model, thinking off; a few percent label error makes exact counts rare. |
| Saturation wastes most of each step | Since step 13, 3-6 of 8 groups are all-solved and 1-3 are trainable; steps 16 and 20 trained nothing. Edit synth (level 5), needle (level 4) and callgraph sit at or near 100%. |
| Reward std falling is saturation, not collapse | Reward std fell from ~0.2 to 0.05-0.09 because most groups are all-equal. The pooled collapse signal is 9.5% (line 30%), capability 75% (76% at checkpoint 0), selection 66.7% (50.0% at the start). KL to base reads 0.006-0.078, measured on only 1-6K tokens a step, so it jumps. |

So the run is learning (selection +9 to +17 points), but what it learns is mostly code-and-tools skill. The one place the RLM method pays off, it can't earn a reward signal.

## Plan (for Rishi's call; ~$0 to build, the run stays stopped while it's built)

### 1. Make semantic aggregation learnable (core)
- **Tolerance grading for counts.** Full credit within max(1, 3%) of the true count, then linear partial credit to 0 at 25% off. Never pay a metadata guess: the generator re-draws any task whose blind guess (lines ÷ labels) lands within 30% of the truth, so guessing can't earn credit. List answers keep set-F1.
- **Noise-robust question kinds.** Add "which topic is most / second-most common (overall, for user X, in a window)" and "which topics have more than N lines". These are argmax- or threshold-style, so a few wrong labels rarely change the answer. Exact credit only.
- **Sizes that make sub-calls the way in.** Keep levels 1-2 small enough to read (warm-up). From level 3 the context passes what the root can see in 12 turns × 8K characters, so labelling has to go through `llm_batch`.
- **Code-only kinds down** from 30% of aggregation to 10%.

### 2. Make needle need reading again from level 3
Today `ast` isolates the candidate functions by their shape. From level 3: behaviour functions also call helpers and live inside classes, fillers come from many more templates, and candidates grow to 60-200. The paraphrase can only be matched by reading many functions, which is a natural `llm_batch` job.

### 3. Rebalance the mix

| Family | Now | Proposed | Why |
|---|---|---|---|
| aggregation | 25% | 35% | the sub-call family |
| repo loc | 15% | 15% | half-semantic already |
| needle | 10% | 10% | semantic from level 3 |
| callgraph | 10% | 5% | pure code skill, near 100% |
| edit real | 25% | 30% | real coding, still mixed results |
| edit synth | 15% | 5% | saturated at its top level |

This replaces the 35% / 5% edit change agreed at checkpoint 20.

### 4. Keep and watch
Masking and redrawing broken replies, the 14-minute retries, the warm-up call, and 1e-5. New dashboard line: sub-calls per episode on semantic tasks vs other tasks. A tolerance-graded episode joins the existing audit, with "answer came from sub-call output" as the clean test.

### Not proposed
- **A reward for using sub-calls.** It would be farmed (research §4.1). Tasks that need them do the teaching.
- **Rolling back.** Checkpoint 20 is the best yet; the new tasks start from it.

## Work and checks before resuming
1. Generator and reward changes with tests (blind-guess rejection, tolerance grading, new kinds, needle level 3+).
2. A data check that every level still builds, plus the existing family tests.
3. A short base-model probe (~$0.05, Rishi's yes): sub-call labelling accuracy per source, to set the tolerance.
4. A dry run, then resume from checkpoint 20 with the new mix.
