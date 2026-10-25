"""The post-training benchmark (plan step 13; Rishi, 2026-10-25: "use the remaining to run the benchmarks").

AA-LCR (rlm_rl/tasks/aalcr.py) in production's set-up: prompt v4, root thinking at medium, sub-calls to the base
model. Two arms (rlm_rl/baselines.py): `bench_ckpt`, a trained checkpoint as the root, and `bench_base`, the base model
in the same set-up, so a difference between them is the training's. Answers are graded with Artificial Analysis's v1.1
equality-checker prompts (the dataset card), the base model as the judge. Run inside WSL2 from the repo root:

    .venv-linux/bin/python scripts/benchmark.py check --source-run prod-v2 --checkpoint 18     # ~$0.01
    .venv-linux/bin/python scripts/benchmark.py run bench-aalcr-v1 --live --arms bench_ckpt --ceiling 9.6 --allow-partial
    .venv-linux/bin/python scripts/benchmark.py run bench-aalcr-v1 --live --arms bench_base --ceiling 14 --allow-partial
    .venv-linux/bin/python scripts/benchmark.py judge bench-aalcr-v1 --live --ceiling 0.6
    .venv-linux/bin/python scripts/benchmark.py summary bench-aalcr-v1

`check` makes sure the checkpoint's LoRA is really attached: it scores one fixed conversation under the base model and
under the checkpoint (prompt logprobs, as the KL check does) and fails if the two agree, which is what a call that fell
back to the base model would give. `run` checks the same before it spends anything.

`run` spends W&B inference credit. `--ceiling` is a hard stop over every arm and every run of this benchmark (as in
scripts/baselines.py), and a rerun resumes: finished episodes are skipped, so running again with a higher ceiling
carries on in the same fixed order. `judge` has a ceiling of its own and resumes the same way. The W&B key is read from
.env and never printed.
"""

import argparse
import asyncio
import dataclasses
import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlm_rl import baselines, config, evals  # noqa: E402
from rlm_rl.client import ModelClient, RequestFailed  # noqa: E402
from rlm_rl.tasks import aalcr  # noqa: E402

RESULTS = ROOT / "results"
EVAL_SET = "aa-lcr-v1"
ARMS = ("bench_ckpt", "bench_base")
JUDGE_THINKING = "low"
JUDGE_MAX_TOKENS = 4096
JUDGE_AT_ONCE = 8
# A LoRA changes every token's probability a little; the base model scored twice agrees to rounding. *(mine)*
ATTACHED_MIN_MEAN_DIFF = 1e-3
PROBE = [{"role": "user", "content": "Count the documents in `context` that mention a refund, then submit the number."},
         {"role": "assistant", "content": "I'll search the documents for the word refund first.\n\n```python\n"
                                          "hits = [d for d in docs if 'refund' in d.lower()]\nprint(len(hits))\n```"}]

# Artificial Analysis's v1.1 judge prompts, verbatim from the dataset card (README.md at the pinned revision)
JUDGE_SYSTEM = """Decide whether the CANDIDATE ANSWER is correct or incorrect against the OFFICIAL ANSWER.
Note the following points when assessing correctness:

- Numbers should still match when they are the same value written differently, e.g., a
  percentage, a count of percentage points, and the equivalent decimal fraction are the same
  value: 0.675, "67.5%" and "67.5 percentage points" all match. So do different scales
  (thousand, million, bn) and different notations (thousands separators, currency symbols,
  LaTeX markup, and numbers written as words).
- Where the question asks for a particular format (e.g., a percentage, a number of decimal
  places, a unit, a rounding, or an ordering) the CANDIDATE ANSWER must meet it. If the
  question asks for no particular format, accept any equivalent form.
- In cases where the question asks for an ordered list, a title, honorific or article added
  to an entry in the CANDIDATE ANSWER can change where that entry sorts. Accept the ordering
  if it is correct either with those additions or without them.
- Grade the value the CANDIDATE ANSWER finally commits to, and it must commit to one. Values
  reached while working, and alternatives it considers and sets aside, do not count. If it
  offers several values without selecting one, it is incorrect even if one of them is right.
  Hedging is fine as long as one clearly definitive answer is given."""
JUDGE_USER = """Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: START QUESTION {question}

END QUESTION

The OFFICIAL ANSWER: {official_answer}

END OFFICIAL ANSWER

BEGIN CANDIDATE ANSWER TO ASSESS

{candidate_answer}

END CANDIDATE ANSWER TO ASSESS

Reply as JSON, with a verdict of CORRECT or INCORRECT."""
_VERDICT_RE = re.compile(r'"verdict"\s*:\s*"(CORRECT|INCORRECT)"', re.I)

# Published AA-LCR v1.1 scores, the whole document set read in one prompt (Artificial Analysis, 2026-10-25)
PUBLISHED = {"Qwen3.8 27B (Xhigh)": 0.82, "Gemma 4 31B (Reasoning)": 0.70}


def run_dir(name: str) -> Path:
    return RESULTS / name


def checkpoint_model(source_run: str, checkpoint: int) -> tuple[str, int]:
    """The branch model and backend step behind a training checkpoint (results/<run>.json)."""
    state = json.loads((RESULTS / f"{source_run}.json").read_text(encoding="utf-8"))
    ckpt = state["checkpoints"][str(checkpoint)]
    model = next(b["model"] for b in state["branches"] if b["branch"] == ckpt["branch"])
    return model, ckpt["backend_step"]


async def open_models(backend, args) -> tuple[str, str]:
    """(base, checkpoint) inference names. The base is the base run's step 0, as in training (prod-v2's sub-calls and
    KL used prod-v1:step0: a freshly registered model's step 0 answered only HTTP 500s, 2026-10-19)."""
    model, step = checkpoint_model(args.source_run, args.checkpoint)
    await backend.open(args.base_run)
    await backend.open(model)
    return backend.inference_name(args.base_run, 0), backend.inference_name(model, step)


def mean_abs_diff(a: list, b: list) -> float | None:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    return statistics.fmean(abs(x - y) for x, y in pairs) if pairs else None


async def attached(client: ModelClient, base: str, ckpt: str, say=print) -> bool:
    """Score PROBE under the base model twice and under the checkpoint once; the LoRA is attached if the checkpoint's
    logprobs differ from the base model's by more than the base model's own repeat does."""
    ids_b, lp_b = await client.prompt_logprobs(PROBE, model=base)
    _, lp_b2 = await client.prompt_logprobs(PROBE, model=base)
    ids_c, lp_c = await client.prompt_logprobs(PROBE, model=ckpt)
    if ids_b != ids_c:
        raise SystemExit("the two models tokenized the probe differently; check the model names")
    noise, diff = mean_abs_diff(lp_b, lp_b2), mean_abs_diff(lp_b, lp_c)
    say(f"base   {base}\ncheckpoint {ckpt}\n{len(ids_b)} probe tokens: base vs base {noise:.2e} mean |dlogprob|, "
        f"base vs checkpoint {diff:.2e}")
    ok = diff is not None and diff > max(ATTACHED_MIN_MEAN_DIFF, 10 * (noise or 0.0))
    say("LoRA attached: the checkpoint scores the probe differently from the base model" if ok else
        "NOT ATTACHED: the checkpoint scores the probe like the base model")
    return ok


async def check(args) -> int:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")  # provides WANDB_API_KEY; never printed
    from rlm_rl.backend import ArtBackend

    backend = ArtBackend(project=config.PROJECT, base_model=config.BASE_MODEL)
    try:
        base, ckpt = await open_models(backend, args)
        client = ModelClient(backend.openai_client(), subcall_model=base)
        ok = await attached(client, base, ckpt)
        print(f"spent ~${client.spent_usd():.4f}")
        return 0 if ok else 1
    finally:
        await backend.close()


def write_meta(out: Path, meta: dict) -> None:
    """A benchmark run sticks to one checkpoint and one base model across its sessions."""
    path = out / "meta.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        changed = {k for k in ("base_model", "checkpoint_model") if old.get(k) != meta[k]}
        if changed:
            raise SystemExit(f"{out.name} ran with a different {', '.join(sorted(changed))}: {old}")
        return
    baselines.write_json(path, meta)


async def run_live(args) -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")  # provides WANDB_API_KEY; never printed
    from rlm_rl.backend import ArtBackend

    tasks, _ = evals.eval_set(EVAL_SET)
    if args.limit:
        tasks = tasks[:args.limit]
    arms = args.arms.split(",")
    for a in arms:  # more samples a question: a rerun keeps sample 0 and runs the new ones
        baselines.ARMS[a] = dataclasses.replace(baselines.ARMS[a], read_samples=args.samples)
    backend = ArtBackend(project=config.PROJECT, base_model=config.BASE_MODEL)
    try:
        base, ckpt = await open_models(backend, args)
        run = baselines.BaselineRun(args.run, RESULTS, openai_client=backend.openai_client(), base_model=base,
                                    ceiling=args.ceiling, eval_set=EVAL_SET, models={"ckpt": ckpt},
                                    max_read=args.max_read)
        if "bench_ckpt" in arms and not await attached(run.clients["bench_ckpt"], base, ckpt):
            raise SystemExit("not started: the checkpoint's LoRA doesn't look attached")
        write_meta(run.run_dir, {"eval_set": EVAL_SET, "source": aalcr.VERSION, "base_model": base,
                                 "checkpoint_model": ckpt, "source_run": args.source_run,
                                 "checkpoint": args.checkpoint})
        proj = baselines.project(run.run_dir, tasks, [baselines.ARMS[a] for a in arms])
        spent = run.budget.spent()
        print(f"projected ${proj['todo_usd']:.2f} more; spent ${spent:.2f}; ceiling ${args.ceiling:g}")
        if spent + proj["todo_usd"] > args.ceiling and not args.allow_partial:
            raise SystemExit("not started: the projection is over the ceiling; pass --allow-partial to run until it")
        await run.run(tasks, arms, retry_infra=1)
        if run.session["skipped_budget"]:
            print(f"STOPPED AT THE CEILING: {run.session['skipped_budget']} episode(s) were not started")
    finally:
        await backend.close()


# ----------------------------------------------------------------------------- judging

def question_rows() -> dict[int, dict]:
    return {int(r["question_id"]): r for r in aalcr._rows()}


def question_id(task_id: str) -> int:
    return int(task_id.rsplit("-q", 1)[1])


def parse_verdict(text: str) -> str | None:
    m = _VERDICT_RE.search(text or "")
    if m:
        return m.group(1).upper()
    words = re.findall(r"\b(INCORRECT|CORRECT)\b", (text or "").upper())
    return words[-1] if words else None


def answer_text(answer) -> str | None:
    if answer is None:
        return None
    if isinstance(answer, (list, tuple)):
        return ", ".join(str(a) for a in answer)
    return str(answer).strip() or None


def judged_path(out: Path) -> Path:
    return out / "judged.json"


def load_judged(out: Path) -> dict:
    try:
        return json.loads(judged_path(out).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"verdicts": {}, "spent_usd": 0.0}


def episodes(out: Path) -> list[dict]:
    recs = []
    for arm in ARMS:
        recs += [r for r in baselines.arm_log(out, arm).records() if not r.get("infra_error")]
    return recs


async def judge(args) -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")  # provides WANDB_API_KEY; never printed
    from rlm_rl.backend import ArtBackend

    out = run_dir(args.run)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    judged = load_judged(out)
    rows = question_rows()
    todo = [r for r in episodes(out) if r["key"] not in judged["verdicts"]]
    backend = ArtBackend(project=config.PROJECT, base_model=config.BASE_MODEL)
    await backend.open(args.base_run)
    client = ModelClient(backend.openai_client(), subcall_model=meta["base_model"])
    before = float(judged.get("spent_usd") or 0.0)
    gate = asyncio.Semaphore(JUDGE_AT_ONCE)
    stopped = 0

    def save():
        judged["spent_usd"] = round(before + client.spent_usd(), 6)
        baselines.write_json(judged_path(out), judged)

    async def one(rec: dict) -> None:
        nonlocal stopped
        row = rows[question_id(rec["task_id"])]
        candidate = answer_text(rec.get("answer"))
        if candidate is None:
            judged["verdicts"][rec["key"]] = {"verdict": "INCORRECT", "why": "no answer submitted"}
            return
        async with gate:
            if before + client.spent_usd() > args.ceiling:
                stopped += 1
                return
            messages = [{"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": JUDGE_USER.format(question=row["question"],
                                                                      official_answer=row["answer"],
                                                                      candidate_answer=candidate)}]
            try:
                reply = await client.chat(messages, model=meta["base_model"], max_tokens=JUDGE_MAX_TOKENS,
                                          thinking=JUDGE_THINKING)
            except RequestFailed as exc:
                print(f"{rec['key']}: judge call failed ({exc.kind}); rerun to retry")
                return
        verdict = parse_verdict(reply)
        if verdict is None:
            print(f"{rec['key']}: no verdict in the judge's reply; rerun to retry")
            return
        judged["verdicts"][rec["key"]] = {"verdict": verdict, "reply": reply[-400:]}
        save()

    try:
        await asyncio.gather(*(one(r) for r in todo))
    finally:
        save()
        await backend.close()
    print(f"judged {len(judged['verdicts'])} episode(s); judge spend ${judged['spent_usd']:.3f}"
          + (f"; {stopped} left at the ceiling" if stopped else ""))


# ----------------------------------------------------------------------------- summary

def mcnemar_p(b: int, c: int) -> float | None:
    """Exact two-sided McNemar test on the discordant pairs."""
    n = b + c
    if n == 0:
        return None
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def summarize(name: str) -> dict:
    out = run_dir(name)
    verdicts = load_judged(out)["verdicts"]
    rows = question_rows()
    arms: dict = {}
    correct_by_arm: dict[str, dict[tuple[int, int], bool]] = {}  # (question, sample) -> judged right
    for rec in episodes(out):
        v = verdicts.get(rec["key"])
        if v is None:
            continue
        arm = rec["arm"]
        qid = question_id(rec["task_id"])
        ok = v["verdict"] == "CORRECT"
        correct_by_arm.setdefault(arm, {})[(qid, rec.get("sample") or 0)] = ok
        a = arms.setdefault(arm, {"n": 0, "correct": 0, "usd": [], "subcalls": [], "turns": [], "by_category": {}})
        a["n"] += 1
        a["correct"] += ok
        a["usd"].append((rec.get("usage") or {}).get("est_cost_usd") or 0.0)
        a["subcalls"].append(rec.get("subcalls_used") or 0)
        a["turns"].append(len(rec.get("turns") or []))
        cat = a["by_category"].setdefault(rows[qid]["document_category"], [0, 0])
        cat[0] += ok
        cat[1] += 1
    table = {}
    for arm, a in arms.items():
        acc = a["correct"] / a["n"]
        samples = {}
        for (_, s), ok in correct_by_arm[arm].items():
            samples.setdefault(s, []).append(ok)
        table[arm] = {"n": a["n"], "correct": a["correct"], "accuracy": round(acc, 4),
                      "by_sample": {s: f"{sum(v)}/{len(v)}" for s, v in sorted(samples.items())},
                      "se": round(math.sqrt(acc * (1 - acc) / a["n"]), 4), "mean_usd": round(statistics.fmean(a["usd"]), 4),
                      "total_usd": round(sum(a["usd"]), 3), "mean_subcalls": round(statistics.fmean(a["subcalls"]), 1),
                      "mean_turns": round(statistics.fmean(a["turns"]), 1),
                      "by_category": {k: f"{v[0]}/{v[1]}" for k, v in sorted(a["by_category"].items())}}
    paired = None
    ck, bs = correct_by_arm.get("bench_ckpt", {}), correct_by_arm.get("bench_base", {})
    both = sorted(set(ck) & set(bs))
    if both:
        b = sum(ck[q] and not bs[q] for q in both)
        c = sum(bs[q] and not ck[q] for q in both)
        paired = {"pairs": len(both), "questions": len({q for q, _ in both}),
                  "checkpoint": sum(ck[q] for q in both), "base": sum(bs[q] for q in both),
                  "only_checkpoint": b, "only_base": c, "mcnemar_p": mcnemar_p(b, c)}
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    result = {"run": name, "meta": meta, "arms": table, "paired": paired, "published": PUBLISHED,
              "judge": {"model": meta["base_model"], "thinking": JUDGE_THINKING, "prompts": "AA-LCR v1.1"}}
    baselines.write_json(out / "summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "run"):
        p = sub.add_parser(name)
        if name == "run":
            p.add_argument("run")
            p.add_argument("--live", action="store_true", required=True, help="spends W&B inference credit")
            p.add_argument("--ceiling", type=float, required=True, help="hard stop over every arm and session, $")
            p.add_argument("--arms", default="bench_ckpt", help=f"comma-separated, from {list(ARMS)}")
            p.add_argument("--allow-partial", action="store_true", help="start even if the projection is over the ceiling")
            p.add_argument("--limit", type=int, help="only the first N questions of the fixed order")
            p.add_argument("--max-read", type=int, default=16, help="episodes at once")
            p.add_argument("--samples", type=int, default=1, help="samples a question (sampling seeds)")
        p.add_argument("--source-run", default="prod-v2")
        p.add_argument("--checkpoint", type=int, default=18)
        p.add_argument("--base-run", default="prod-v1", help="whose step 0 is the base model")
    p = sub.add_parser("judge")
    p.add_argument("run")
    p.add_argument("--live", action="store_true", required=True, help="spends W&B inference credit")
    p.add_argument("--ceiling", type=float, required=True, help="hard stop for the judge's own spend, $")
    p.add_argument("--base-run", default="prod-v1")
    p = sub.add_parser("summary")
    p.add_argument("run")
    p.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.command == "check":
        return asyncio.run(check(args))
    if args.command == "run":
        unknown = [a for a in args.arms.split(",") if a not in ARMS]
        if unknown:
            raise SystemExit(f"unknown arm(s) {unknown}; choose from {list(ARMS)}")
        asyncio.run(run_live(args))
        return 0
    if args.command == "judge":
        asyncio.run(judge(args))
        return 0
    result = summarize(args.run)
    if args.json:
        print(json.dumps(result, indent=1))
        return 0
    for arm, row in result["arms"].items():
        print(f"{arm:11} {row['correct']}/{row['n']} = {row['accuracy']:.1%} (±{row['se']:.1%}), ${row['mean_usd']:.3f} "
              f"an episode, {row['mean_subcalls']} sub-calls, {row['mean_turns']} turns  {row['by_category']}")
    if result["paired"]:
        p = result["paired"]
        print(f"paired on {p['pairs']} (question, sample) pairs over {p['questions']} questions: checkpoint "
              f"{p['checkpoint']}, base {p['base']}; only checkpoint "
              f"{p['only_checkpoint']}, only base {p['only_base']}, McNemar p = {p['mcnemar_p']}")
    print("published (read directly):", ", ".join(f"{k} {v:.0%}" for k, v in PUBLISHED.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
