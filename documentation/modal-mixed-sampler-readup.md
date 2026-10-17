# Read-up: RL on mixed-precision, mixed-accelerator samplers (Modal, 2026-10-13)

Sources: Modal's post "Toward RL on mixed-precision, mixed-accelerator samplers" (Nan Jiang, 2026-10-13), the SemiAnalysis tweet that promoted it the same day, ART 0.5.20's source (`art/loss.py`, `art/serverless/`), and our own 20-step run (`results/archive/rlsd_vs_grpo_20.json`). Read-up only: no code changes, no W&B credit, nothing committed. Marked *(inferred)* where I reasoned rather than checked.

## 1. Bottom line for our run

- **No settings change for the second seed.** Our sampler/trainer mismatch is small and stayed flat for 20 steps, which is what the post says matters.
- **Restoring the CISPO clip (decision 1c) is also our mismatch correction.** ART uses W&B's sampler logprobs as the "old" logprobs, so CISPO's importance weight is trainer prob over sampler prob, the same ratio the post corrects. The 20-step run pinned that weight to 1, so it ran uncorrected, the condition that collapsed in the post. The second seed is our first corrected run.
- **The missing piece is watching growth.** The dashboard already shows the ratio tails, but no stop rule or notice looks at whether they grow. That's the cheapest idea worth taking from the post (section 1.4).

### 1.1 Where our mismatch comes from

The post lists seven mismatch sources. For us:

| Source | Our run | Status |
|---|---|---|
| Kernels | W&B's vLLM samples; W&B's ART trainer scores. Both sides are hidden from us. | Present, measured below |
| Precision | Not visible. Probably BF16 on both sides *(inferred)* | Part of the measured gap |
| Accelerator | W&B's fleet, not visible | Part of the measured gap |
| LoRA serving path (not in the post) | vLLM's LoRA kernels vs the trainer's adapter | Small: p99 was already 1.24 at step 0, before the adapter had learned anything, so most of the gap is the base model *(inferred)* |
| Staleness | Synchronous: each step samples from the newest checkpoint. Inside one train call ART takes an optimizer step per packed slice, so later slices are slightly off-policy | Small; the CISPO clip covers it |
| MoE routing | Dense 27B | None |
| Top-p / top-k | Temperature 1.0, no top-p or top-k sent. ART's own vLLM server ignores the model's sampling defaults (`generation_config="vllm"`); W&B likely runs the same *(inferred)* | None |
| Multi-turn context | Token ids rebuilt from the server's own `prompt_logprobs` and sampled ids, and step 6 checked that Qwen3.8's template keeps earlier turns as exact prefixes. That's equivalent to the token-in-token-out approach the post uses | None |

### 1.2 What we measured (20-step GRPO arm, ART's raw ratio stats)

| | Ratio mean | p95 | p99 | `probs_corr` |
|---|---|---|---|---|
| Step 0 | 1.000 | 1.084 | 1.240 | 0.9975 |
| Steps 0-9 avg | 1.000 | 1.082 | 1.231 | ~0.997 |
| Steps 10-19 avg | 1.000 | 1.084 | 1.248 | ~0.997 |

- The ratio is trainer probability over sampler probability per token. The post calls a token extreme when its ratio leaves [0.2, 5]. Our p99 is 1.25, so extreme tokens are well under 1%.
- The p99 crept up by about 0.017 between the two halves. If that continued linearly it would reach roughly 1.35 by step 75, still far under the clip at 5 *(inferred extrapolation)*.
- These stats come from every optimizer step in the train call, so they include a little within-step drift. They overstate the hardware gap slightly, not understate it.
- `rlm_rl/monitor.py` already logs mean, p95, p99, clip fraction and `probs_corr` every step, and the dashboard shows them.

### 1.3 What our loss does, and which of the post's fixes we could use

- ART 0.5.20 CISPO: `loss = -clip(ratio, 1-ε, 1+ε_high).detach() * A * logp`. At our ε = 1.0, ε_high = 4.0 that clips the weight to [0, 5]. That's **truncation**: a token with ratio 8 still trains, at weight 5. The post's MIS **masks** such tokens to zero instead.
- Score centering (SC) is the post's key ingredient. It isn't in ART and we can't change W&B's trainer, so it's out of reach on W&B.
- Top-p mask replay is irrelevant because we don't use top-p.
- W&B's train API does accept these knobs (they're in ART's serverless client schema):
  - `mask_prob_ratio=True` gives MIS. At our ε it would only mask ratios above 5, which is almost nothing today.
  - `precalculate_logprobs=True` with `truncated_importance_sampling=C` splits the correction: the trainer recomputes the old logprobs once, CISPO then sees only within-step drift, and a separate weight capped at C covers the sampler gap.
  - `importance_sampling_level="sequence"` gives a GSPO-style sequence-level ratio.
- None of these is worth turning on now. They're the levers if the second seed shows the ratio tails growing. Whether W&B's server honours each one is untested *(inferred from the client schema)*.

### 1.4 What to watch (proposal, not built)

- The post's collapse signature is falling reward, sampler/trainer KL rising 3x or more, and episodes ending in repeated format errors. Our equivalents:
  - reward and pass rate
  - ratio p99, clip fraction and `probs_corr`
  - submit rate, parse failures and turn-limit hits
- Today `stoprules.collapse` watches entropy and reward std only. `probs_corr > 0.98` is checked once, for the production learning-rate decision.
- A cheap addition would be a "watch" notice when the ratio p99's excess over 1 doubles from its first-10-step level, or when the clip fraction leaves ~0. It's a few lines in `stoprules.py` and costs no credit.
- A sudden step change in these stats, rather than a slow creep, would most likely mean W&B changed its serving stack mid-run. The post's result says a jump in level is survivable when the gap stays bounded.

### 1.5 Other implications for us

- **Model choice.** The alternative root, Qwen3.6-35B-A3B, is the model the post trains. It's a MoE, so routing adds another mismatch source. The post's uncorrected GRPO collapsed around step 100 even on matched hardware. With only CISPO truncation available on W&B, the dense 27B is the safer pick.
- **Final eval off W&B.** The post always evaluates in BF16. Ours should serve both the base model and the LoRA in BF16 on one stack (same stack is already the plan). A quantized local runtime would add a precision shift we never trained under, so it shouldn't produce the headline numbers *(inferred)*.
- **Budget.** No effect. We don't choose W&B's hardware, so the post's ~15% sampling saving isn't a lever for us. Nothing here needs credit.
- **LoRA and run length are on our side.** The post trains for 160+ steps, apparently on full weights (a Megatron trainer, and LoRA isn't mentioned), with fully async sampling about 4 steps stale and no staleness limit. We train a LoRA at 1e-5 to 2e-5, synchronously, for about 75 steps. Each of those slows mismatch growth *(inferred)*.

## 2. What it means for RL in general

### 2.1 The argument

- **The old view: match the sampler to the trainer.** Examples are batch-invariant kernels (Thinking Machines; DeepSeek-V4), SGLang's deterministic mode, and Microsoft's MAI-Thinking-1 matching the trainer's precision. Matching costs throughput and ties both sides to one format and one accelerator.
- **The post's view: let the mismatch exist, but keep it bounded and corrected.** Its headline evidence:
  - Mixed samplers had 4.7x the homogeneous sampler/trainer KL yet matched pass@4.
  - Homogeneous GRPO started with a quarter of the mixed KL and still collapsed around step 100.
  - So stability depends on whether the gap grows, not on how big it is.
- **Two errors, two fixes.** Sampling from q while training p causes:
  - **Wrong weighting.** MIS fixes this: weight each token by p/q and mask extreme ratios.
  - **Drift.** The update pulls p toward q whatever the reward. Score centering fixes this by subtracting the expected score under q at each position.
  - The drift is what feeds growth: the bigger the gap, the harder the update pulls toward the sampler. My derivation: E_q[∇log p] = Σ(q − p)∇log p, which is zero when q = p and grows with the gap. In the post, MIS alone and SC alone both still collapsed. Only SC + MIS was stable.
- **Top-p with mask replay** removes the rarest tokens, the ones with extreme ratios, before they enter the data. It also makes centering exact, because the sampler records the probability of every candidate.

### 2.2 Why it matters beyond Modal

- **One framework for every off-policy source.** Staleness, precision, kernels, accelerator, MoE routing, sampling truncation and re-tokenization are all "sampled from q, not p". One correction stack covers them, in the IMPALA tradition of correcting for lag rather than running in lockstep. That's also how fully async RL can work at scale.
- **The sampling fleet becomes fungible.** Sampling takes about 80-85% of RL GPUs (MAI-Thinking-1). If samplers can run FP8 or NVFP4, speculative decoding, disaggregated prefill/decode, older or spare GPUs, any vendor, then RL compute becomes ordinary inference capacity. The trainer can go to low precision on its own schedule too. The measured saving was only about 15% per update in a setup not tuned for cost. The bigger prize is capacity: using whatever GPUs are free.
- **The data contract is the interface.** Every sampled token has to carry the sampler's logprob, and for exact centering the candidate set's probabilities too. Multi-turn agents need token-in-token-out so the trainer scores exactly the context that was sampled. Anyone building an RL stack should design this contract first.
- **Diagnose by sampler group.** H100 FP8 produced 10% of samples but 29% of the extreme-ratio tokens. Mixed fleets need per-group ratio monitoring, and probably routing away from groups that misbehave.
- **A tension with CISPO.** CISPO's design keeps a gradient on rare "fork" tokens that PPO clipping would drop. MIS and top-p deliberately drop rare, extreme-ratio tokens, because under heavy mismatch those are mostly noise. Which matters more probably depends on how large the mismatch is. Our gap is small enough that neither bites *(inferred)*.

### 2.3 How much to trust it

- One model, one task family, one seed per arm. Only pass@4 is reported (no pass@1), and the learning rate, absolute KL and dollar costs aren't in the text.
- Mixed-sampler entropy split from homogeneous after about step 160 (homogeneous rose, mixed stayed flat), and the post doesn't explain it.
- The corrections are token-level approximations. The post itself names sequence-level corrections (GSPO, DeepSeek-V3.2) or bounded staleness as safeguards at larger scale. Top-p training optimizes the truncated policy, not the full one.
- The arms appear to have run fully async with about 4 steps of staleness *(inferred from the setup description)*. Part of what SC + MIS fixes is therefore staleness, not hardware.
- Modal sells autoscaling serverless GPUs across accelerator types, so heterogeneous fleets are its product. Score centering comes from a recent paper (arXiv 2609.20807) that hasn't been widely replicated yet.

## 3. The SemiAnalysis tweet

> "RL practitioners who got burnt by trainer inference mismatch avoid it like a plague. What if you embrace the mismatch and dance with the fire? Check out what happens when you use different hardware and precision for rollouts in RL training" (@SemiAnalysis_, Oct 13)

- **Why SemiAnalysis cares.** It covers hardware economics, so its interest is the fungibility story: RL demand that could land on A100s, RTX PRO 6000s and FP8 or NVFP4 Blackwell instead of only matched B200s.
- **"Embrace the mismatch" overstates it.** The post's own ablations show plain GRPO collapsing even on matched hardware. The honest version is "accept the mismatch, bound it, correct it, and watch whether it grows".

## 4. Open questions to talk through

1. Add the ratio-growth notice (1.4) before the second seed? It's free and small.
2. Ask W&B what precision and kernels they serve the 27B with, and whether that can change mid-run?
3. If the second seed shows ratio tails growing, which do we try first: `mask_prob_ratio`, or `precalculate_logprobs` with `truncated_importance_sampling`?

## Sources

- Modal post: https://modal.com/blog/toward-rl-on-mixed-precision-mixed-accelerator-samplers
- References the post cites, as given there: MIS arXiv 2510.18855; score centering arXiv 2609.20807; top-p mask replay and DeepSeek-V3.2 arXiv 2512.02556; MoE routing replay arXiv 2510.11370; format rounding arXiv 2510.26788; GSPO arXiv 2507.18071; IMPALA arXiv 1802.01561; Thinking Machines "Defeating nondeterminism in LLM inference"; LMSYS "No token left behind" (token-in-token-out); MAI-Thinking-1 report (sampling share of GPUs)
- ART 0.5.20: `art/loss.py` (CISPO clip, `mask_prob_ratio`, `truncated_importance_sampling`), `art/unsloth/train.py` (`precalculate_logprobs`, one optimizer step per packed slice), `art/serverless/client.py` (the knobs W&B's API accepts), `art/preprocessing/tokenize.py` (sampler logprobs become the old logprobs)
- Ours: `results/archive/rlsd_vs_grpo_20.json`, `rlm_rl/config.py` (`TRAIN_ARGS`), `rlm_rl/stoprules.py`, `rlm_rl/monitor.py`, production-spec.md §5, step-2 and step-6 results
