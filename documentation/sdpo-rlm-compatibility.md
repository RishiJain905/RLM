# SDPO x RLM: does it work, and does it work on W&B serverless?

> Findings from checking SDPO ([arXiv:2601.20802](https://arxiv.org/html/2601.20802v2)) against the RLM paper ([arXiv:2512.24601v3](https://arxiv.org/html/2512.24601v3)), the W&B Serverless Training API, and ART 0.5.20. Companion to [rlm-inference-strategy.md](rlm-inference-strategy.md). Code: `rlm_rl/sdpo.py`, `notebooks/02_sdpo_serverless.ipynb`.

---

## 1. Short answer

- **W&B serverless has no native SDPO.** An RL job takes one scalar `reward` per trajectory. `loss_fn` is only `cispo` or `ppo`, and there is no custom-loss or per-token-advantage field.
- **It can still be done, with no server changes.** We inject the self-teacher's logprobs where ART expects the sampling logprobs, then turn on ART's `kimi_k2_tau` term. ART then computes exactly SDPO's hybrid advantage (section 3). The teacher scoring runs on the serverless checkpoint itself through vLLM `prompt_logprobs`.
- **SDPO fits RLM well, arguably better than it fits single-turn tasks.** RLM rollouts already produce the rich, tokenized feedback SDPO is built for: tracebacks, REPL output, a wrong `submit()` value, and successful sibling trajectories. They are also long multi-turn episodes, where GRPO's single scalar gives poor credit assignment. SDPO's dense per-token advantage is the fix.
- **The harness is unaffected.** SDPO only changes training. The trained artifact is a normal LoRA, and the REPL loop, bridge, and `submit()` stay model-agnostic. The one requirement it puts on the harness is to log exact token ids and logprobs per root turn, so a transcript can be re-scored.

## 2. What W&B serverless actually exposes (checked 2026-10-10)

| Item | Finding |
|---|---|
| Trainable bases on this account | 13 models from `GET /v1/preview/base-models`, including `Qwen/Qwen3.8-27B` and `Qwen/Qwen3.6-35B-A3B` |
| RL job input | `trajectory_groups` of `{messages_and_choices, reward, ...}` plus an `experimental_config` |
| Loss options | `cispo`, `ppo`, epsilon clips, IS level, KL penalty vs a reference adapter, `kimi_k2_tau`, reward scaling |
| Per-token advantages | Not in the schema |
| Old logprobs | Taken from the logprobs we send on each `Choice` (`precalculate_logprobs` defaults off) |
| Inference | OpenAI chat endpoint on vLLM. `logprobs` and `prompt_logprobs` work. `/v1/completions` and `echo` return 404 |
| Token ids | The gateway **drops** vLLM `token_ids`/`prompt_token_ids`, but ART's trainer **requires** them. We rebuild them from `prompt_logprobs` keys plus `<|im_start|>` (`rlm_rl/rollout.py::attach_token_ids`). Without this, every serverless training job fails. |
| Concurrency | Per-user, per-model limit. 32 parallel requests to Qwen3.8-27B returned 429s. We cap at 8 and retry. |
| Thinking | With the default serving config, neither Qwen returned reasoning content on our prompts |

## 3. How SDPO runs on serverless

SDPO's hybrid advantage for token *t* of rollout *i* is

`A_t = lambda * A_grpo + (1 - lambda) * log( pi(y_t | x, f, y_<t) / pi(y_t | x, y_<t) )`

where `f` is feedback (a correct sibling solution, or environment output) and the teacher is the same model.

ART's loss already contains `A_t = A_grpo - tau * (logp_new(y_t) - logp_old(y_t))` when `kimi_k2_tau = tau`, and `logp_old` is whatever we put on the `Choice`. So:

1. Sample rollouts as usual (logprobs on).
2. For each rollout with feedback, call the **current checkpoint** with `[teacher context, assistant: student response]`, `continue_final_message=True`, and `prompt_logprobs=0`. The tail of `prompt_logprobs` is the teacher's logprob of every student token. This is one forward pass and no generation.
3. Overwrite the `Choice` token logprobs with the teacher's values. Tokens we leave alone keep zero SDPO advantage, which gives per-token control (section 5).
4. Train with `kimi_k2_tau = (1 - lambda) / lambda` (about 0.11 for the paper's lambda = 0.9) and `epsilon = epsilon_high = 0`, so the clipped importance weight is exactly 1 and the teacher logprobs do not leak into it.

Because the loss is on-policy (`logp_new` is about the student's sampling logprob), this gives `A_grpo + tau * log(pi_teacher / pi_student)`, which is SDPO.

**Verification.** Notebook 02 part (a) trains three identical-at-step-0 LoRAs on the *same* rollouts: two plain GRPO (to measure noise) and one with injected teacher logprobs. See section 7 for the numbers.

### Differences from the paper

| Paper | Ours | Effect |
|---|---|---|
| Top-K (K=100) logit distillation | Sampled-token estimator | Unbiased but higher variance. Fixing it needs server-side loss support |
| EMA / trust-region regularized teacher | Current checkpoint as teacher | Less stable. A lagged checkpoint (`step - k`) is a cheap stand-in |
| One gradient step per batch | ART runs several micro-steps per job (13 in our probe) | Ratio stays near 1 early; we pin the IS weight to 1 anyway |
| Zero-variance groups still learn from feedback | Unclear whether the server drops all-equal-reward groups | Try a tiny reward jitter with `scale_rewards=False` (untested) |

## 4. Mapping SDPO onto an RLM rollout

**Student trajectory** = the root conversation from the harness log: system prompt, context metadata, then alternating `assistant` (code) and `user`/`tool` (truncated REPL output) messages, ending in `submit()`. ART masks non-assistant tokens, so only root code turns are trained. Sub-calls (`llm` / `llm_batch`) are environment steps served by the same base model, which matches decision 1 in the strategy doc.

**Teacher context options** (the feedback `f`), in order of how much signal they carry:

1. **Successful sibling trajectory.** The RLVR mode from the paper. For RLM, include the sibling's *code turns only* and drop its REPL outputs, which keeps the teacher context short and transfers the decomposition strategy rather than the data.
2. **Environment feedback for a failed attempt.** For example: "submit() returned 41, expected 57"; "the episode ended without submit()"; the last traceback; "you issued 300 llm() calls, budget was 50". This is SDPO's rich-feedback (RLRF) mode, where the paper saw its biggest gains (code with test feedback).
3. **Both**, when a sibling succeeded.

**Scoring cost.** One `prompt_logprobs` call per rollout re-scores *every* root turn at once, because the whole student transcript sits after the teacher preface. RLM keeps root transcripts small by design (REPL output is truncated), so the teacher context is roughly sibling-code plus student transcript.

**Tokenization must round-trip.** The re-rendered transcript has to tokenize to the same ids as sampled. `teacher_logprobs()` checks this and falls back to plain GRPO for any rollout that misaligns. In the probe, 1 of 14 failed to align.

## 5. Risks for RLM specifically

1. **Answer leakage and shortcutting.** If the teacher sees the gold answer, it will prefer a direct `submit(answer)` over REPL work, and the student can learn to guess. Mitigations: give the teacher the sibling's strategy and leave out its final value; or zero the SDPO advantage on `submit(...)` tokens by leaving their logprobs untouched. The injection allows that per token.
2. **Exploration collapse.** The paper reports SDPO makes responses shorter. That helps RLM, since thinking tokens exhausted output budgets in the RLM paper. It could also suppress the extra REPL inspection that finds hard answers. Watch turns per episode and sub-call counts.
3. **Thinking tokens.** If reasoning is enabled, prior-turn reasoning must be preserved in the chat template (ART sends `preserve_thinking=True`) or teacher alignment breaks. Thinking tokens get SDPO advantages like any other.
4. **Weak in-context learners.** SDPO underperformed GRPO on Qwen2.5-1.5B in the paper. A 27B or 35B-A3B model should be fine, but it is worth checking on the actual base before scaling up.
5. **Reliance on ART internals.** The trick depends on `kimi_k2_tau` semantics and on old logprobs coming from the client. `openpipe-art` is pinned at 0.5.20 in `requirements.txt`. Re-run notebook 02 part (a) after any upgrade.

## 6. What this means for the harness

- Log exact **token ids and per-token logprobs** for every root turn, and keep chat rendering deterministic, so a transcript can be re-scored by a teacher.
- Emit a structured **feedback string** per episode: final value vs gold, last traceback, budget overruns. That string *is* SDPO's `f`.
- SDPO training needs prompt logprobs from the policy, so it only works for open-weight models served by vLLM-style backends (W&B serverless, local vLLM). Hosted APIs like Claude or GPT can *run* the harness but cannot be SDPO students. That is fine, since the harness is model-agnostic at inference.

## 7. Results so far (2026-10-10)

The task is a toy one, 4-digit multiplication, chosen only to exercise the pipeline. Evals use 32 samples, so accuracy differences of a few points are noise.

**Notebook 01 (GRPO smoke test).** Both models ran rollout, train, new checkpoint, and inference end to end.

| Model | Eval acc, step 0 | Train reward, steps 1 / 2 | Eval acc, step 2 |
|---|---|---|---|
| Qwen3.8-27B | 0.66 | 0.47 / 0.78 | 0.78 |
| Qwen3.6-35B-A3B | 0.78 | 0.69 / 0.81 | 0.91 |

**Notebook 02 (a): does the server use the injected teacher logprobs?** The three LoRAs were trained on identical rollouts.

| Metric | grpo-a | grpo-b | sdpo-check |
|---|---|---|---|
| `loss/probs_corr` | 0.978 | 0.978 | **0.873** |
| `loss/importance_ratio_mean` (raw, before the clip) | 1.000 | 1.000 | **20.5** |
| `loss/train` | -0.0076 | -0.0081 | **-0.0098** |
| `loss/grad_norm` | 1.619 | 1.608 | 1.618 |

- The probs_corr and raw-ratio rows show the server used our logprobs as the "old" policy. Those numbers are the student/teacher ratio, as expected.
- With the importance weight pinned to 1, old logprobs can only change the loss through the `kimi_k2_tau` term. The SDPO loss differs from the GRPO pair by about 4x their own gap, so tau is being applied. This rests on one sample. A larger-tau run would make it unambiguous.
- Teacher re-scoring aligned on 27 of 28 attempts. The mean token log-ratio was -0.05. The minimum was -13, which is the teacher strongly rejecting a wrong digit.

**Notebook 02 (b): three steps of SDPO-hybrid vs GRPO on Qwen3.6-35B-A3B.** Both arms started at 0.75 eval accuracy. After 3 steps GRPO reached 0.84 and SDPO 0.81. That is within noise, so it shows only that SDPO training runs stably. It says nothing yet about which method is better. Most groups were all-correct, so this task is too easy for the 35B model. The real comparison should run on the RLM environment.
