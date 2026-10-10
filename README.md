# RLM

Training a local Recursive Language Model (RLM) with RL, and building a model-agnostic RLM harness around a Python REPL.

## Setup

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

Put `WANDB_API_KEY=...` in `.env` (gitignored). Serverless RL needs a W&B team entity, not a personal one.

## Layout

- `rlm_rl/` - tasks, rollouts, and the SDPO helpers used by the notebooks
- `notebooks/01_serverless_rl_smoke.ipynb` - GRPO smoke test on `Qwen/Qwen3.8-27B` and `Qwen/Qwen3.6-35B-A3B`
- `notebooks/02_sdpo_serverless.ipynb` - SDPO on W&B serverless via teacher-logprob injection
- `documentation/sdpo-rlm-compatibility.md` - how SDPO fits the RLM setup and the eventual harness
