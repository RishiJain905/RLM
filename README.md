# RLM

Training a local Recursive Language Model (RLM) with RL, and building a model-agnostic RLM harness around a Python REPL.

## Setup

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

Put `WANDB_API_KEY=...` in `.env` (gitignored). Serverless RL needs a W&B team entity, not a personal one.

The production environment runs inside WSL2 (Ubuntu), so it has its own Linux virtual environment in the repo, `.venv-linux/` (gitignored). From Ubuntu, in the repo folder:

```bash
python3 -m venv .venv-linux
.venv-linux/bin/python -m pip install openpipe-art==0.5.20 wandb==0.30.0 pyarrow==26.0.0 openai==2.54.0 httpx==0.28.1 python-dotenv==1.2.4
.venv-linux/bin/python scripts/sandbox_ctl.py check
```

## Layout

Production code (built against `documentation/production-spec.md`):

- `rlm_rl/bridge.py` - bridge API v1 (`context`, `llm`, `llm_batch`, `submit`), the pinned prompt text and its hash, the wire protocol, and the host side of the sub-call socket
- `rlm_rl/repl_worker.py` - the REPL that runs inside the sandbox (standard library only)
- `rlm_rl/sandbox.py` - one Docker container per episode, with no network, no host files and no secrets
- `rlm_rl/client.py` - every model request: one shared limiter (30 in flight, 20 per model name), retries with jittered backoff, infra errors, thinking-off checks and the token-id rebuild
- `rlm_rl/env.py` - the episode loop, the `infra_error` flag on each episode, and what a step does with flagged episodes before training
- `rlm_rl/reward.py` - the reward: exact answers, and partial credit for near misses calibrated against each task's blind guess
- `rlm_rl/detectors.py` - the hack detectors (`early_submit`, `copied_answer`, `escape_attempt`, `timeout_loop`), which zero a reward before the group mean
- `rlm_rl/controller.py` - the difficulty controller: one level per task family, moved by its recent pass rate
- `rlm_rl/transcripts.py` - one JSON line per finished episode, training and eval alike (`results/<run>/transcripts/`, git-ignored)
- `rlm_rl/evals.py` - resumable evals that log every episode, locally and through ART's `model.log(..., split="val")`
- `rlm_rl/tasks/` - host-only task generation (random 64-bit seeds; eval seed lists go in the git-ignored `private/`), the family registry, and `tickets.py`, the ticket tasks with five difficulty levels
- `scripts/sandbox_ctl.py` - build the sandbox image, run the regression tests, clean up after a crash
- `scripts/audit.py` - read transcripts for the hack audit: per-log summary, sampled wins as Markdown, one episode in full, and the Claude audit at each checkpoint (`checkpoint`, `report`; see `documentation/audit-procedure.md`)
- `scripts/reward_check.py` - what blind guessers earn under the reward, per family and level
- `scripts/backtest_detectors.py` - replays the 20-step run's episodes through the detectors to count false alarms
- `tests/` - regression tests; run them inside WSL2 with `.venv-linux/bin/python scripts/sandbox_ctl.py check`. `tests/fake_model.py` is a local stand-in for the W&B model server, so tests never spend credit

The sandbox runs from WSL2 (Ubuntu) with Docker Desktop's WSL integration turned on.

Testing-era code is in the `archive/` folders and doesn't run in place:

- `rlm_rl/archive/` - the 20-step environment (`rlm_env.py`), ticket tasks, rollouts and the SDPO helpers
- `notebooks/archive/` - the smoke test, SDPO and three-way notebooks
- `scripts/archive/run_rlsd_vs_grpo.py` - the resumable 20-step RLSD vs GRPO run
- `documentation/` - research notes, the production spec and build plan, and step results
