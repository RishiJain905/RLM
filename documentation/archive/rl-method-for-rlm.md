# Which RL method fits RLM best?

> Theory-only decision doc (2026-10-10). Nothing here was run. Builds on [rlm-paper.md](../rlm-paper.md), [rlm-inference-strategy.md](rlm-inference-strategy.md) and [sdpo-rlm-compatibility.md](sdpo-rlm-compatibility.md). Setting: Qwen3.8-27B, W&B serverless RL (ART 0.5.20, one scalar reward per rollout), straight RL with no SFT, sub-calls on the same base model.

---

## 1. Answer

**Use SDPO's self-teacher, but anchored to the reward.** Concretely: a GRPO-style backbone (CISPO loss) where the verifier's reward decides the *sign* of every token's update, and SDPO's self-teacher only decides the *size* (which tokens in a rollout get most of the credit or blame). This is the RLSD idea ([arXiv:2604.03128](https://arxiv.org/html/2604.03128v2)) applied to RLM rollouts. Pure self-distillation is used only as a small extra signal on groups where every rollout failed, since there the reward gives no gradient at all.

So Rishi's lean is mostly right: the self-teacher is the most valuable piece for RLM, and nothing else on the menu gives dense credit inside a multi-turn REPL episode. But **plain SDPO, and the additive hybrid we already have running, are not the best version for RLM.** Since the SDPO paper came out, several papers have shown that letting the self-teacher set the direction of the update breaks on exactly the conditions RLM has: hard tasks, multi-turn tool use, and many valid solution paths (section 4).

| Rank | Method | One-line verdict |
|---|---|---|
| 1 | **Reward-anchored self-distillation** (RLSD-style, on CISPO) | Dense per-turn credit from the self-teacher, without letting it override the reward. Fits every RLM requirement. |
| 2 | SDPO hybrid (current `rlm_rl/sdpo.py`, λ = 0.9) | Close second. Its additive teacher term can flip the sign of a correct rollout's tokens and pull it toward one sibling's path. |
| 3 | GRPO / CISPO alone | Safe, proven baseline (the alphaXiv RLM RL work uses it). One advantage for the whole episode is weak credit assignment for long REPL episodes. |
| 4 | Pure SDPO (λ = 0) | Fails on hard and multi-turn agentic tasks in independent replications. Learns the teacher's path, not correctness. |
| — | PPO with critic, GiGPO, process reward models, on-policy distillation from a bigger teacher | Don't fit the platform, the decisions already made, or RLM's state structure (section 5). |

---

## 2. What RLM asks of an RL method

These come from the RLM paper, our harness design, and the platform. A method is only "best" if it handles all of them.

| # | Requirement | Where it comes from |
|---|---|---|
| R1 | **Credit assignment across many turns.** A rollout is a multi-turn root conversation (probe, decompose, sub-call, aggregate, `submit`) with one terminal reward. A good probe turn and a buggy aggregation turn in the same rollout should not get the same credit. | RLM paper §3.2, strategy doc §7 |
| R2 | **Many valid strategies.** The paper's whole point is that the model picks its own decomposition (regex filter, chunk-and-map, pairwise loop, depth). The method must not collapse onto one reference solution. | RLM paper §6 (emergent behaviors) |
| R3 | **Rich text feedback is free.** Every episode produces tracebacks, truncated REPL output, budget overruns, and a wrong `submit()` value. A method that can learn from that text beats one that only sees a scalar. | SDPO doc §4 |
| R4 | **Hard tasks, low early pass rate.** Long-context tasks start hard. Groups where every rollout fails give GRPO zero gradient. Without SFT, this matters more. | alphaXiv RLM RL blog (small model scored 0 pass@16 without SFT), decision 5 |
| R5 | **No answer leakage, no skipping the REPL.** The model must not learn to guess `submit(answer)` without doing the work. | SDPO doc §5.1 |
| R6 | **Platform fit.** One scalar reward per rollout, `cispo` or `ppo` loss, no critic, no reward-model training, no custom loss. Per-token shaping only through the logprob-injection channel. | SDPO doc §2–3 |
| R7 | **Fits the decisions.** Straight RL (no SFT), same base model for sub-calls, model-agnostic harness. | Strategy doc §9 |

---

## 3. Candidates scored

✓ = handles it, ~ = partly, ✗ = fails or not possible.

| Method | R1 credit | R2 diversity | R3 text feedback | R4 all-fail groups | R5 leakage | R6 platform | R7 decisions |
|---|---|---|---|---|---|---|---|
| GRPO / CISPO | ✗ one advantage per episode | ✓ | ✗ | ✗ | ✓ | ✓ native | ✓ |
| PPO with value critic | ~ value per turn, noisy for long text episodes | ✓ | ✗ | ~ | ✓ | ✗ no critic on serverless | ✓ |
| GiGPO (step-level groups) | ~ only where states repeat | ✓ | ✗ | ✗ | ✓ | ~ via injection | ✓ |
| Process reward model | ✓ | ~ | ~ | ✓ | ~ | ✗ needs a trained PRM | ~ |
| On-policy distillation, bigger teacher | ✓ | ✗ imitates teacher | ✗ | ✓ | ✓ | ~ | ✗ not straight RL from own model |
| Pure SDPO | ✓ | ✗ reference bias | ✓ | ✓ | ✗ needs care | ✓ via injection | ✓ |
| SDPO hybrid (additive, current) | ✓ | ~ | ✓ | ~ | ~ | ✓ via injection | ✓ |
| **Reward-anchored SD (RLSD-style)** | ✓ | ✓ sign fixed by reward | ✓ | ~ needs the all-fail add-on | ✓ | ✓ via injection | ✓ |

The bottom row is the only one with no ✗. The next two sections justify the rows that matter.

---

## 4. Why not plain SDPO

SDPO's own results are real ([arXiv:2601.20802](https://arxiv.org/html/2601.20802v2)): 70.2% vs 66.6% aggregate over GRPO on science QA and tool use, 48.8% vs 41.2% on LiveCodeBench with test feedback, GRPO's 5-hour accuracy reached in about 50 minutes on Chemistry, gaps that widen with model scale, and up to 11x shorter responses. But every one of those settings is **single-turn, one answer, mostly one correct path**. RLM is none of those. Later work tested self-distillation where RLM lives:

1. **It optimizes the wrong thing on hard tasks.** "Privileged, but Biased" ([arXiv:2608.04794](https://arxiv.org/html/2608.04794)) reproduced SDPO's easy-task gains, then ran it on hard tasks including **multi-turn agentic tool use (BFCL) on Qwen3-8B**. Loss went down, accuracy did not go up, and agentic validation fell by up to 3.5 points on average. The per-token loss was nearly the same on correct and incorrect rollouts. Their summary: *"as a lone objective, SD optimizes a signal decoupled from task success."*
2. **It transfers a path, not correctness.** Same paper: the teacher's bias toward the solution in its context scored 0.52, versus ≤0.02 toward a *different* correct solution to the same problem. In RLM terms, a correct rollout that decomposed the task differently from the sibling in the teacher's context gets pushed toward the sibling's decomposition. That directly fights R2. The paper puts it as: *"What the privileged information transfers is a specific trajectory instead of correctness."*
3. **It peaks early and leaks.** RLSD ([arXiv:2604.03128](https://arxiv.org/html/2604.03128v2)) shows on-policy self-distillation peaks within 10–20 steps and then declines, and that the model starts referring to the invisible reference solution. They prove the loss has an irreducible mutual-information term, so it cannot be driven to zero. That is R5's failure mode.
4. **It suppresses exploration and self-correction.** Privileged self-distillation cut thinking models by up to 17% on long reasoning, by penalizing reconsideration and backtracking tokens ([arXiv:2607.05184](https://arxiv.org/abs/2607.05184)). Another study reports reduced output diversity ([arXiv:2606.26091](https://github.com/chrisliu298/awesome-on-policy-distillation)). RLM depends on the model recovering from a bad first decomposition (RLM paper §6).
5. **Naive coupling with RL is unstable for agents.** Skill-SD ([arXiv:2604.10674](https://arxiv.org/html/2604.10674)) reports that naively combining self-distillation with RL for multi-turn agents leads to training collapse. Their working recipe keeps GRPO as the main loss and the distillation term tiny (λ = 0.001), with student-owned rollouts and the teacher synced to the latest checkpoint. It beat GRPO by 14.0 points on AppWorld and 10.9 on Sokoban on Qwen3-4B.

The common thread: **the self-teacher is a good judge of *which tokens* mattered, and a bad judge of *whether* a rollout was right.** Every method that works lets the reward own the "whether".

## 5. Why not the others

- **GRPO / CISPO alone.** It is the safe baseline and the one the alphaXiv RLM work used ([blog](https://alphaxiv.org/blog/reinforcement-learning-for-rlms)): group-normalized reward broadcast to every turn, with child sub-calls inheriting the root's advantage. It works, but it fails R1. An RLM episode is 5–20 code turns; one bad `llm_batch` loop and one perfect regex filter get the same advantage. It also fails R3 and R4. It stays as the **control arm** in any comparison, and its loss is the backbone of the recommendation.
- **PPO with a critic.** A per-turn value function would help R1, but serverless gives us no critic, and value estimates on long text episodes with a 27B base are expensive and noisy. Not available, not worth building.
- **GiGPO** ([arXiv:2505.10978](https://arxiv.org/html/2505.10978v1)). Gets step-level credit by grouping actions taken from the *same* environment state across rollouts (+12% ALFWorld, +9% WebShop over GRPO). In RLM, REPL states almost never match across rollouts after turn 1, because each rollout's variables, printed output and sub-call results differ. Its micro-level groups would be nearly empty. Fails R1 in practice.
- **Process reward models.** Need a separately trained judge of each REPL turn. That is a second model to train and serve, and a new hacking surface. Out of scope under R6.
- **On-policy distillation from a bigger teacher** (e.g. a Qwen 397B model running the harness). Strong dense signal, but it is imitation of another model, which conflicts with decision 5 (straight RL from our own model) and with R2. Keep as a fallback if straight RL stalls at zero reward.
- **SFT / rejection fine-tuning.** What the RLM paper used for RLM-Qwen3-8B. Excluded by decision 5.

---

## 6. Why the current hybrid is close but not best

`rlm_rl/sdpo.py` computes, per token,

`A_t = A_i + tau * delta_t`, where `delta_t = log pi_teacher(y_t) - log pi_student(y_t)` and `tau ≈ 0.11` (λ = 0.9).

The teacher term is **added**, so it can flip the sign. Notebook 02 saw `delta_t` as low as -13 on one token. That gives `0.11 * -13 = -1.43`, which overrides a correct rollout's `A_i = +0.5` and pushes that token *down*, even though the rollout was right. With a sibling's solution in the teacher context, the tokens most likely to get this treatment are the ones where the student took a different but valid route. That is point 2 of section 4, in our own numbers.

## 7. The recommended method

### 7.1 Advantage

For rollout `i` with group advantage `A_i` (from the scalar reward) and teacher log-ratio `delta_t` on each of its root tokens:

```
s_i  = sign(A_i)
w_t  = clip( exp(s_i * delta_t), w_min, w_max )      # e.g. [0.5, 2.0]
w_t  = w_t / mean(w_t over the rollout's root tokens) # keep total credit per rollout
A_t  = A_i * w_t
```

- In a **successful** rollout, tokens the teacher endorses get more credit and tokens it doubts get less, but every token still moves up.
- In a **failed** rollout, the tokens the teacher blames (the bad line of code, the wrong `submit` value) take most of the penalty, and the reasonable probe turns before them take little.
- The sign always comes from the reward, so the teacher can redistribute credit inside a rollout but never decides whether it was right. This is RLSD's principle. The exact clip range and normalization above are our choice; RLSD's own formula is in its paper and repo and should be checked before implementing.

### 7.2 All-fail groups (R4)

When every rollout in a group fails, `A_i = 0` and the reward gives nothing. Only there, add a small pure self-distillation term with the teacher seeing **environment feedback only** (the traceback, "submit returned X, which is wrong", "300 llm() calls, budget 50"), never the gold answer. Keep its weight small and track whether it helps, since section 4 says this term alone does not reliably improve accuracy. HDPO ([arXiv:2603.23871](https://github.com/chrisliu298/awesome-on-policy-distillation)) uses the same "only where RL gradients vanish" gating. The better fix for R4 is task difficulty (7.4), so this add-on is optional and should be ablated.

### 7.3 Teacher context for RLM

- **Successful sibling:** its **code turns only**. Drop its REPL output and its final submitted value. This transfers strategy, not the answer.
- **Failed attempt:** structured environment feedback from the harness log.
- **Never** the gold answer.
- Leave `submit(...)` tokens at `w_t = 1`, so the teacher cannot reshape the answer tokens (R5).
- Root turns only. Sub-call tokens are environment steps for now (same base model, decision 1). Training sub-calls with an inherited root advantage, as the alphaXiv blog does, is a later step.

### 7.4 Backbone settings

- **Loss: CISPO.** It clips and detaches the importance weight but keeps a gradient on every token, including rare high-impact ones (the decision to call `llm_batch`, the first decomposition). It is one of the two loss options serverless exposes.
- **No std scaling of rewards** (`scale_rewards=False`). Simpler, avoids the Dr. GRPO difficulty bias, and makes `A_i` easy to reproduce client-side (needed for 7.5).
- **Group size 8**, matching our serverless concurrency cap.
- **Keep pass rate in the 20–80% band** by curriculum on context length and needle count, as the RLM paper's MRCRv2 run did (trained at 32–64K with 2 needles, generalized to 1M with 8). This removes most all-fail and all-pass groups and is worth more than any loss change.
- **Reward:** correctness on `submit` (exact match or F1), small penalties for no submit, syntax errors and budget overruns. No length penalty at first; SDPO-style shortening already pushes that way.

### 7.5 Does it run on W&B serverless?

Yes, through the same channel as the current SDPO code, with no server changes. ART's loss already uses `A_grpo - tau * (logp_new - logp_old)` and we control `logp_old` per token. Sending

`logp_old_t = logp_student_t + c_t / tau`, with `c_t = A_i * (w_t - 1)`

gives `A_t = A_i + c_t = A_i * w_t` on-policy. In other words, the injection channel can carry **any** per-token advantage, not just SDPO's. Two conditions: compute `A_i` client-side with the same normalization the server uses (simplest with `scale_rewards=False`), and repeat notebook 02 part (a)'s check that the server applies the shaped advantage before trusting a training run.

---

## 8. How we would prove it

The argument above is the theory. The empirical proof is a three-arm run on the RLM environment (not the toy multiplication task, which was too easy to separate anything), same seeds, tasks and budget:

1. CISPO/GRPO (control)
2. SDPO hybrid (current code)
3. Reward-anchored self-distillation (this doc)

Watch: held-out accuracy on longer contexts than training (the generalization the RLM paper showed), steps to a target accuracy, turns and sub-calls per episode, rate of `submit` without any `llm`/REPL work (leakage), and diversity of decomposition strategies across a group. The prediction from this doc: arm 3 ≥ arm 2 > arm 1 on accuracy and sample efficiency, with arm 2 showing more strategy collapse and more early-peak behavior than arm 3. This needs W&B credit, so it waits for Rishi's go-ahead.

## 9. Risks and open questions

- **Evidence is mostly from 4–8B models on single-agent tasks.** None of the cited papers trained an RLM with self-distillation. SDPO's own gap grows with scale, so a 27B model may tolerate the hybrid better than the papers suggest. That is why section 8 keeps the hybrid as its own arm.
- **Thinking mode.** If Qwen3.8-27B serves with reasoning on, the thinking-model degradation in section 4.4 applies more strongly. Our serving config currently returns no reasoning content.
- **Teacher = current checkpoint.** Ours is not EMA-regularized. A lagged checkpoint (`step - k`) is the cheap stabilizer if `delta_t` gets noisy.
- **Cost.** One extra `prompt_logprobs` pass per rollout, the same as current SDPO.
- **ART internals.** Still depends on `kimi_k2_tau` semantics in the pinned ART 0.5.20.

## 10. Sources

- RLM: Zhang, Kraska, Khattab, [arXiv:2512.24601v3](https://arxiv.org/html/2512.24601v3)
- SDPO: Hübotter et al., [arXiv:2601.20802](https://arxiv.org/html/2601.20802v2), [project page](https://self-distillation.github.io/SDPO)
- RLSD, Self-Distilled RLVR: [arXiv:2604.03128](https://arxiv.org/html/2604.03128v2)
- Privileged, but Biased: [arXiv:2608.04794](https://arxiv.org/html/2608.04794)
- Rethinking On-Policy Self-Distillation for Thinking Models: [arXiv:2607.05184](https://arxiv.org/abs/2607.05184)
- Skill-SD: [arXiv:2604.10674](https://arxiv.org/html/2604.10674)
- GiGPO: [arXiv:2505.10978](https://arxiv.org/html/2505.10978v1)
- Reinforcing Recursive Language Models (alphaXiv blog): [link](https://alphaxiv.org/blog/reinforcement-learning-for-rlms)
- Survey list used to find HDPO, SC-SDPO, diversity and forgetting results: [awesome-on-policy-distillation](https://github.com/chrisliu298/awesome-on-policy-distillation)

Claims about RLSD's exact formula, HDPO and the diversity paper come from abstracts and summaries, not full reads. Verify against the papers before implementing.
