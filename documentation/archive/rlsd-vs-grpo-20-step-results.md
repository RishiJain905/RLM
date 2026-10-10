# RLSD vs GRPO: 20-step run and audit

> Run 2026-10-10, `scripts/run_rlsd_vs_grpo.py`, raw numbers in `results/rlsd_vs_grpo_20.json`. This settles the open question from [rlm-three-way-results.md](rlm-three-way-results.md).

## Answer

**RLSD did not deliver the hoped-for benefit. Plain GRPO did at least as well, and probably better.** Both beat the untrained model. GRPO's gain is clear; RLSD's is small and not significant. Our recommendation for the production run is GRPO, treating RLSD as an optional later ablation.

| Model (100 held-out episodes each) | Accuracy | Mean reward | Sub-calls / episode | Submitted |
|---|---|---|---|---|
| Base | 0.50 | 0.598 | 2.1 | 0.84 |
| **GRPO, 20 steps** | **0.70** | **0.783** | 3.8 | 0.88 |
| RLSD, 20 steps | 0.59 | 0.693 | 4.8 | 0.92 |

- **GRPO vs base:** +0.20 accuracy, about 3 standard errors. That is a real gain.
- **RLSD vs base:** +0.09, about 1.3 standard errors. Not significant.
- **GRPO vs RLSD:** 0.11 in GRPO's favour, about 1.6 standard errors of the difference. Suggestive, not conclusive. RLSD is clearly not better.

## Setup

- **Model:** Qwen3.8-27B LoRA, W&B serverless, ART 0.5.20. LR 1e-5, 20 steps per arm, 8 tasks x 4 rollouts per step, the same tasks for both arms. CISPO loss with the importance weight pinned to 1 and no reward std scaling.
- **RLSD:** the paper's Algorithm 1, `A_t = A_i * ((1 - lam) + lam * clip(exp(sign(A_i) * delta_t), 0.8, 1.2))`. lam decays from 0.5 toward 0 over 50 steps (0.31 at step 20). The teacher context is a successful sibling's code without its answer, or environment feedback. Never the gold answer.
- **Tasks** (`rlm_rl/ticket_tasks.py`): 250-380 free-text support tickets per context (32-45K characters), assembled from interchangeable parts so there is almost no repeated text. Problem reports are paraphrased without shared keywords, and the distractors reuse the same vocabulary. The sub-model labels them correctly (60 of 60 in a spot check). Two families, half each:
  - `tickets_all`: count one problem type across the whole context. This needs sub-calls.
  - `tickets_customer`: count one customer's problems, about 20-30 tickets. The root can read these directly.
- **Reward:** 1 for an exact count, otherwise `0.8 * max(0, 1 - 4 * relative error)`, and 0 without `submit`.

## Training curves

| | Steps 1-10 mean reward | Steps 11-20 mean reward | Step 20 |
|---|---|---|---|
| GRPO | 0.64 | 0.77 | 0.83 |
| RLSD | 0.59 | 0.68 | 0.65 |

The difference sits almost entirely in the hard family:

| Family | GRPO, first half to second half | RLSD, first half to second half |
|---|---|---|
| `tickets_all` (needs sub-calls) | 0.40 → **0.63** | 0.41 → 0.48 |
| `tickets_customer` | 0.88 → 0.92 | 0.77 → 0.89 |

## Why RLSD did not help here (our reading)

1. **Its reweighting is small by design.** The mean change to a token's advantage was about 0.005, against advantages around +/-0.3. Over 20 steps RLSD is roughly GRPO plus a small nudge, so part of the gap is likely run-to-run variance between two independently sampled training runs.
2. **About 19% of RLSD's teacher re-scoring fell back to plain GRPO** because tokenization did not round-trip (88 of 465 attempts).
3. **RLSD drifted toward more sub-calls and hit the budget more.** The 20-call budget ran out in 77 RLSD episodes against 32 for GRPO. Its eval sub-calls rose to 4.8 per episode against GRPO's 3.8 without better accuracy. The teacher context shows a sibling's code, which often fans out across many sub-calls, so this is consistent with the "transfers a path, not correctness" concern in [rl-method-for-rlm.md](rl-method-for-rlm.md).
4. **Evidence from other papers was on 4-8B single-agent tasks.** On a 27B model with a reward that already separates rollouts well, there may be little dense-credit headroom left for RLSD to use.

## Audit

Every logged training rollout was checked: 1,280 rollouts and 6,429 code cells across both arms.

| Check | Result |
|---|---|
| Reads of `.env`, `.art`, or environment variables | **None** |
| Network access (`socket`, `urllib`, `requests`, `http`) | **None** |
| Other file access | One GRPO rollout at step 1 wrote the ticket text to `C:\tmp\tickets\` (two files, 86 KB). This is harmless, but it confirms the REPL can write anywhere on disk. |
| Other flagged code | `eval()` on sub-model output to parse lists (8 times), one `import sys` for `stdout.flush()`. No harmful use. |
| Guessing | Full-count rollouts with zero sub-calls earned 0.04-0.06 average reward early and 0 late, in both arms. Guessing does not pay on these tasks. Zero-sub-call rollouts on `tickets_customer` score about 0.9, but those tasks are small enough to read directly, so that is not guessing. |
| Infra failures seen by the model | 4 RLSD episodes (0.6%) hit the 120 s cell timeout while sub-calls were queued, which reset their REPL. GRPO had none. No "sub-call failed: rate limited" message ever reached a model. |
| Can guessing or infra explain the gap? | **No.** Guessing earned about 0 in both arms. The 4 timeouts can cost RLSD at most about 0.006 average reward, against a gap of 0.09. |

### Known limitations (from the research thread's audit, confirmed)

- **No real sandbox.** The REPL is a subprocess with keys stripped from its environment and a temp working directory, but it can read and write the filesystem and reach the network. It must be containerised (Docker or a cloud sandbox, no network, read-only except a scratch directory) before any production run.
- **Eval episodes are not logged**, so the 300 eval episodes were not audited. Only aggregate metrics were saved.
- **Eval uses the same generator and templates as training** (different seeds). It measures in-distribution learning, not generalisation.
- **Sub-calls run in a pool of 6 alongside the 8-request root cap**, so up to 14 requests can be in flight. The 429s were absorbed by retries; the cost was the 4 RLSD timeouts above.
- **One seed per arm.** The GRPO-vs-RLSD difference is 1.6 standard errors and could narrow with a second seed.

## Recommendation for production

- Use **GRPO (CISPO, `scale_rewards=False`)** as the training method. It gave a clear +20-point gain in 20 steps on the sub-call-heavy tasks.
- Fix the sandbox, log eval transcripts, and use a held-out eval with different templates and longer contexts before the production run.
- If dense credit is revisited later, test it as an ablation on top of the GRPO run rather than as the default.
