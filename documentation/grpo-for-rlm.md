# Does GRPO work well for training an RLM?

> Written 2026-10-10. Based on our 20-step run ([rlsd-vs-grpo-20-step-results.md](archive/rlsd-vs-grpo-20-step-results.md), `results/archive/rlsd_vs_grpo_20.json`, the 1,280 logged training rollouts) and published RLM and long-context RL work.

## Answer

**Yes.** GRPO-family RL is what every published RLM training run uses, and in our own run it worked. It taught Qwen3.8-27B to make *more* self-calls on the tasks that need them and to get more out of each one, while staying inside the sub-call budget.

RLSD's worse result was **not** caused by the budget. Removing every budget-limited episode leaves the gap almost unchanged.

## 1. Did the sub-call budget make RLSD look worse?

No. The budget does cost something: episodes that run out of sub-calls score much lower. But RLSD hit it only a little more often than GRPO, and the gap is the same without those episodes.

| Hard tasks (`tickets_all`), steps 11-20 | GRPO | RLSD |
|---|---|---|
| Episodes that ran out of the 20 sub-calls | 6% | 9% |
| Mean reward, all episodes | **0.63** | 0.48 |
| Mean reward, excluding budget-hit episodes | **0.65** | 0.49 |
| Sub-calls per episode | 8.2 | 8.7 |
| Sub-call tokens per episode | 17.8K | 17.8K |

Both methods spend about the same number of sub-calls and sub-call tokens. GRPO turns them into more correct answers. RLSD's problem is the *quality* of its decomposition and aggregation, not how many calls it can make.

Over the whole run, the budget was hit in 5% of GRPO episodes on each task family, against 11-13% for RLSD.

Two other notes on the earlier findings:
- **The 19% of RLSD teacher scoring that fell back to plain GRPO** made RLSD behave *more* like GRPO, not less. It diluted RLSD's signal, but it cannot explain RLSD doing worse.
- **The 4 RLSD REPL timeouts** (0.6% of its episodes) are worth at most about 0.006 average reward, against a gap of about 0.09.

## 2. Does GRPO actually use self-calls efficiently?

Yes. It learned to recurse more where recursion pays, and each extra self-call bought accuracy.

**On the hard tasks (`tickets_all`), first half vs second half of training:**

| | Steps 1-10 | Steps 11-20 |
|---|---|---|
| Mean reward | 0.40 | **0.63** |
| Exactly correct | 31% | **46%** |
| Sub-calls per episode | 5.9 | 8.2 |
| Episodes with zero sub-calls | 17.5% | **2.5%** |

**Reward by number of sub-calls (GRPO, hard tasks, all steps):**

| Sub-calls | Episodes | Mean reward |
|---|---|---|
| 0 | 32 | 0.05 |
| 1-3 | 68 | 0.44 |
| 4-9 | 137 | 0.59 |
| 10-19 | 72 | **0.64** |
| 20 (budget hit) | 11 | 0.55 |

Not recursing earns almost nothing, and GRPO learned to stop doing it. Returns flatten after about 10 calls, and running out of budget costs a little.

**On the easier tasks (`tickets_customer`, about 25 tickets the root can read directly),** GRPO mostly kept solving them without sub-calls: 68% of episodes in the second half, scoring 0.92 reward. It did not learn to over-recurse where recursion is unnecessary.

**Held-out eval (100 episodes):** GRPO scored 0.70 accuracy using 3.8 sub-calls per episode. RLSD scored 0.59 using 4.8. GRPO got more accuracy from fewer calls.

## 3. What published work says

| Work | Algorithm | Model | What happened |
|---|---|---|---|
| Reinforcing Recursive Language Models ([alphaXiv blog](https://www.alphaxiv.org/blog/reinforcement-learning-for-rlms)) | GRPO-style clipped policy gradient with KL. Child sub-calls inherit the root's advantage, normalised by the number of children. | Qwen3.5-4B | Judge score rose from about 0.6 to 0.8 on single-paper evidence selection, roughly matching Claude Sonnet 4.6 in the same harness and about 9x faster. **Needed a small SFT cold start:** the 4B base scored 0 pass@16, so GRPO had nothing to learn from. |
| RLM paper, Appendix A ([arXiv:2512.24601](https://arxiv.org/abs/2512.24601)) | RL with prime-rl, 4 rollouts per example, group-based. The paper does not name the exact loss. | not stated | Trained 150 steps on MRCRv2 at 32-64K tokens with 2 needles, then evaluated at 512K-1M with 8 needles. The authors present it as generalising, but report no numbers. |
| MemAgent ([arXiv:2507.02259](https://arxiv.org/abs/2507.02259), ICLR 2026) | Multi-conversation DAPO (a GRPO variant), outcome reward only | Qwen2.5 7B/14B | A model that reads long text in chunks with a memory, trained at 8-32K, extrapolated to 3.5M tokens with under 10% loss. This is the closest non-RLM evidence that group-relative RL teaches a model to manage its own context. |
| Context-Folding ([arXiv:2510.11967](https://arxiv.org/abs/2510.11967)) | FoldGRPO: GRPO plus token-level process penalties | Seed-OSS-36B | Outcome-only GRPO failed to teach clean branch/return behaviour. It needed small penalties for not delegating and for failed tool calls. |
| Prime Intellect RLMEnv ([blog](https://www.primeintellect.ai/blog/rlm)) | Not trained yet at time of writing | GPT-5-mini, GLM, INTELLECT-3 (inference only) | Untrained models often under-use sub-calls. They argue RL training is what should fix that. |

**What the literature adds to our result:**
- GRPO is the default for RLM training. No published RLM run used self-distillation.
- GRPO needs a non-zero starting pass rate. Ours is fine: Qwen3.8-27B starts at about 0.5 reward with partial credit, so no SFT cold start is needed, which matches decision 5. A smaller model might need one, as the alphaXiv 4B run did.
- Outcome-only GRPO can miss structural habits such as delegating long work or recovering from tool errors. Context-Folding fixed that with small process penalties, which fit our reward design if we need them.
- Training only the root is fine to start. Training sub-calls with the root's advantage, as alphaXiv does, is a known next step once the root is good.

## 4. Recommendations for the production run with GRPO

1. **Use GRPO (CISPO loss, `scale_rewards=False`) as tested.** Keep the partial-credit reward: it keeps about 5 to 7 of 8 groups mixed, so most groups give a gradient.
2. **Raise the sub-call budget from 20 to about 30.** Budget-hit episodes score 0.25 against 0.53 on hard tasks, and GRPO's use is trending up (8.2 per episode). This is cheap insurance, not a fix for a known failure.
3. **Make tasks harder as accuracy rises.** Use longer contexts and more tickets so the pass rate stays in the 20-80% band. GRPO was at 46% exact on the hard tasks by step 20, with headroom left.
4. **Add process penalties only if a habit stalls.** Candidates are a small penalty for no `submit`, and for REPL errors after the first turn. Watch for this in the logs rather than adding it up front.
5. **Train sub-calls later.** When the root is solid, give child calls the root's advantage, normalised by child count.
6. **Before spending real budget,** fix what the audit found: a real sandbox for the REPL, logged eval transcripts, eval tasks from different templates and longer contexts, and a second seed.

## Caveats

- One seed per arm, 20 steps, one task generator. The GRPO-vs-base gain (about 3 standard errors) is solid. The GRPO-vs-RLSD gap (about 1.6) is suggestive.
- The tasks are synthetic. Real long-context tasks (OOLONG, BrowseComp-Plus, codebase QA) are the next test.
- Some numbers above come from blog posts and abstracts rather than full papers: MemAgent's extrapolation figure, and the prime-rl algorithm details. Check them before citing externally.
