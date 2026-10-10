# GRPO vs SDPO vs RLSD on an RLM environment: first run

> Run 2026-10-10, `notebooks/03_rlm_three_way.ipynb`, raw numbers in `results/three_way_rlm.json`. Tests the three-arm comparison proposed in [rl-method-for-rlm.md](rl-method-for-rlm.md) section 8.

## Setup

- **Model:** Qwen3.8-27B LoRA on W&B serverless (ART 0.5.20). Sub-calls go to the untrained base model.
- **Environment** (`rlm_rl/rlm_env.py`): the root model never sees the text. It is a `context` variable in a sandboxed REPL subprocess with `llm_query`, `llm_batch` and `submit`. Limits are 6 turns, 20 sub-calls, and 2,000 characters of output per turn.
- **Tasks:** two-thirds are semantic counting over 200-320 customer messages ("how many report a delivery problem?"). One-third are filtered sums over 300-450 order records. Reward is 1 for an exact answer, up to 0.5 partial credit for a close one, and 0 with no `submit`.
- **Training:** 3 steps per arm, 8 tasks x 4 rollouts per step, the same tasks for every arm, LR 1e-5. Every arm uses CISPO with the importance weight pinned to 1 and no reward std scaling. Only the per-token advantage differs:
  - GRPO: `A_i` on every token.
  - SDPO hybrid: `A_i + 0.11 * delta_t`.
  - RLSD (paper's Algorithm 1): `A_i * (0.5 + 0.5 * clip(exp(sign(A_i) * delta_t), 0.8, 1.2))`.
- **Teacher context** for both self-distillation arms: a successful sibling's code with its submit lines removed, otherwise environment feedback on the failed attempt. Never the gold answer.
- **Eval:** 24 held-out tasks x 2 samples = 48 episodes per model.

## Results

| Model | Eval accuracy | Eval mean reward | Submitted | Sub-calls / episode |
|---|---|---|---|---|
| Base (step 0) | 0.354 | 0.495 | 0.77 | 0.69 |
| GRPO, 3 steps | 0.458 | 0.585 | 0.88 | 1.27 |
| SDPO hybrid, 3 steps | 0.375 | 0.538 | 0.88 | 0.60 |
| **RLSD, 3 steps** | **0.542** | **0.599** | 0.77 | 0.96 |

Training reward per step (rollouts from that step's checkpoint):

| Arm | Step 0 | Step 1 | Step 2 |
|---|---|---|---|
| GRPO | 0.513 | 0.553 | 0.524 |
| SDPO | 0.438 | 0.499 | 0.454 |
| RLSD | 0.520 | 0.524 | 0.593 |

## How much to trust it

- **The ranking matches the prediction** in rl-method-for-rlm.md: RLSD first, then GRPO, then SDPO.
- **RLSD vs GRPO is not significant yet.** With 48 eval episodes, one standard error of accuracy is about 0.07. RLSD is 0.08 above GRPO (about 1 SE) and 0.19 above the base model (about 2.7 SE). The step-0 training rewards also show the noise level: all three arms sampled the same checkpoint on the same tasks and still got 0.44 to 0.52.
- **RLSD's reweighting was small.** The mean absolute change to a token's advantage was about 0.005, against advantages of roughly +/-0.3 to 0.5. With epsilon_w = 0.2 and lambda = 0.5, RLSD is GRPO with a gentle per-token reweighting. A real difference between them should take more steps to show.
- **SDPO is the only arm with no clear gain over the base model**, and its training reward never moved above its own step-0 level. That is consistent with the papers cited in rl-method-for-rlm.md. Three steps cannot separate "SDPO is worse here" from noise.
- **No leakage signal.** First-turn submits fell to 0 in every arm.
- **Little sub-call use.** The base model solves most tasks with regex and code, about one sub-call per episode. That is a valid RLM strategy, but it means these tasks mostly exercise REPL use, not recursion.
- **Plumbing.** Teacher re-scoring aligned on about 88% of rollouts. The rest fell back to plain GRPO.

## What would settle it

- 20 or more steps per arm with a fixed eval set of at least 100 episodes.
- Tasks whose answer needs reading comprehension that regex cannot do, so sub-calls matter.
- A held-out eval on longer contexts than training.
