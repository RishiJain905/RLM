"""The live dashboard (plan step 9): build the page from results/<run>.json, check it, and wait for new steps.

A Claude session on this PC publishes the page as an artifact after each step (decision 9a); the
training script can't. The session runs this loop alongside the run:

    python scripts/dashboard.py wait second-seed-v1            # blocks until there's something new, builds, exits
    (the session publishes results/dashboard/page.html to the dashboard's URL)
    python scripts/dashboard.py wait second-seed-v1 --ack      # records that publish, then waits again

`wait` exits with one line that says what to do: PUBLISH (a new page is built and passed the check),
STALE (the run stopped writing its state without saying why), or BLOCKED (the check found
something secret, so nothing was written). A page is only ever written after `check()` passes.

    python scripts/dashboard.py build prod-dry1 --dry-run      # one build, no waiting
    python scripts/dashboard.py check results/dashboard/page.html --run prod-dry1 --dry-run --deep
    python scripts/dashboard.py waiting second-seed-v1         # the page between runs: the next run, no data

It uses the standard library alone (plus rlm_rl for --deep), so it runs from Windows Python while the
training loop runs in WSL. It never reads .env.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlm_rl import dashboard  # noqa: E402

RESULTS = ROOT / "results"
DRY_RESULTS = RESULTS / "dry"
OUT_DIR = RESULTS / "dashboard"  # git-ignored: page.html, publisher.json and the sample cache


def results_dir(args) -> Path:
    if getattr(args, "results", None):
        return Path(args.results)
    return DRY_RESULTS if getattr(args, "dry_run", False) else RESULTS


def is_dry(args, results: Path) -> bool:
    return bool(getattr(args, "dry_run", False)) or results.resolve() == DRY_RESULTS.resolve()


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_atomic(path: Path, text: str) -> None:
    """Write then swap, so a publish that reads the page mid-build gets the old page or the new one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # a reader holding the file open on Windows
            time.sleep(0.2 * (attempt + 1))
    os.replace(tmp, path)


def pot_arg(args) -> dict | None:
    """--pot-left is read off W&B's billing page when the command starts, so the page dates it today."""
    if args.pot_left is None:
        return None
    return {"inference_left_usd": args.pot_left, "as_of": time.strftime("%Y-%m-%d")}


def seed_roots(results: Path) -> list[Path]:
    return [ROOT / "private" / "seeds", results / "private" / "seeds"]


def deep_items(state: dict) -> tuple[set[tuple[str, str]], list[str], list[str], set[str], set[str]]:
    """Every question of the eval sets this run uses, the held-out tests and the capability prompts,
    and the edit eval tasks' ids and repositories.

    This builds the sets from their private seeds through rlm_rl, which takes a minute and needs the
    data on D:, so it's for a one-off check, not every build.
    """
    from rlm_rl import capability, evals  # noqa: WPS433 (heavy imports kept out of every build)

    cfg = state.get("config") or {}
    items, prompts, notes, task_ids, repos = set(), [], [], set(), set()
    for name in dict.fromkeys(n for n in (cfg.get("selection"), cfg.get("final_eval"), "heldout-test-v1",
                                          "heldout-test-v2") if n):
        try:
            tasks, _ = evals.eval_set(name)
            keys = []
            for t in tasks:
                if dashboard.is_edit_family(t.family):  # keyed by the task text, as eval_items does
                    q = dashboard.scrub(t.question)[:150]
                    keys.append((q, "") if len(q) >= 20 else None)
                    task_ids.add(t.task_id)
                    if isinstance((t.grading or {}).get("repo"), str):
                        repos.add(t.grading["repo"])
                else:
                    keys.append(dashboard.item_key(t.question, "\n".join(t.context.splitlines()[:3])))
            items |= {k for k in keys if k}
            notes.append(f"{name}: {len(tasks)} tasks, {sum(1 for k in keys if k)} keyed")
        except Exception as exc:  # missing data or seeds: say so, don't pretend it was checked
            notes.append(f"{name}: not checked ({type(exc).__name__})")
    if cfg.get("capability_set"):
        try:
            loaded = capability.load(capability.resolve(cfg["capability_set"]))
            prompts += [i.prompt for i in loaded]
            notes.append(f"capability set: {len(loaded)} prompts")
        except Exception as exc:
            notes.append(f"capability set: not checked ({type(exc).__name__})")
    return items, prompts, notes, task_ids, repos


def build(run: str, results: Path, *, dry: bool, pot: dict | None = None, out_dir: Path = OUT_DIR) -> dict:
    """Build the page and check it. The page is written only if the check passes."""
    state = dashboard.read_state(results / f"{run}.json")
    run_dir = results / run
    out = Path(out_dir) / "page.html"
    cache_path = Path(out_dir) / f"cache-{run}.json"
    cache = load_json(cache_path, {})
    samples = dashboard.update_samples(cache, state, run_dir)
    data = dashboard.page_data(state, run_dir=run_dir, samples=samples, dry_run=dry, pot=pot)
    page = dashboard.render(data)
    items, _ = dashboard.eval_items(run_dir)
    task_ids, repos = dashboard.eval_names(run_dir)
    result = dashboard.check(page, seeds=dashboard.private_seeds(seed_roots(results)), items=items,
                             task_ids=task_ids, repos=repos)
    result.update(run=run, step=(state.get("steps") or [{}])[-1].get("step", 0), status=state.get("status"),
                  signature=dashboard.signature(state), heartbeat_age_s=dashboard.heartbeat_age(state))
    if result["ok"]:
        write_atomic(out, page)
        write_atomic(cache_path, json.dumps(cache))
        result["page"] = str(out)
    return result


def waiting_state(run: str, kind: str, lr: float | None = None) -> dict:
    """The state of a run that hasn't started: its config and nothing else, so the page between runs
    carries no numbers, transcripts or flags from the run before. Built in memory; results/<run>.json
    is the training loop's alone."""
    from rlm_rl import config as rconfig  # noqa: WPS433 (kept out of every build)

    cfg = rconfig.second_seed(run) if kind == "second_seed" else rconfig.production(run, lr)
    now = time.time()
    return {"run": run, "kind": kind, "status": "waiting", "created_at": now, "updated_at": now,
            "config": cfg.as_dict(), "latest": 0, "branch": 0,
            "lr": {"current": cfg.learning_rate, "start": cfg.learning_rate, "floor": rconfig.LR_FLOOR,
                   "rule": cfg.lr_rule, "fallback_used": False, "fallback": None},
            "spend": {**dict.fromkeys(("train_usd", "kl_usd", "eval_usd", "interrupted_usd", "total_usd",
                                       "training_gpu_usd_est"), 0.0), "ceilings": dict(cfg.ceilings)}}


def summary_line(kind: str, r: dict) -> str:
    sig = r.get("signature") or {}
    head = (f"{kind} run={r.get('run')} step={r.get('step')} status={r.get('status')} evals={sig.get('evals')} "
            f"notices={sig.get('notices')} size={r.get('bytes', 0) // 1024}KB")
    if r.get("ok"):
        c = r.get("checked") or {}
        return (f"{head} page={r.get('page')} check=passed (seeds {c.get('seeds')}, eval tasks {c.get('eval_tasks')}, "
                f"edit task ids {c.get('task_ids')}, repos {c.get('repos')})")
    return f"{head} check=FAILED {json.dumps(r.get('findings'))} (nothing was written; the last good page stands)"


def cmd_build(args) -> int:
    results = results_dir(args)
    pot = pot_arg(args)
    r = build(args.run, results, dry=is_dry(args, results), pot=pot, out_dir=Path(args.out_dir))
    print(summary_line("BUILT" if r["ok"] else "BLOCKED", r))
    return 0 if r["ok"] else 3


def cmd_waiting(args) -> int:
    if args.kind == "production" and args.lr is None:
        print("production needs --lr (2e-5 or 1e-5, the second seed's call)")
        return 2
    pot = pot_arg(args)
    data = dashboard.page_data(waiting_state(args.run, args.kind, args.lr), pot=pot)
    page = dashboard.render(data)
    r = dashboard.check(page, seeds=dashboard.private_seeds(seed_roots(RESULTS)))
    r.update(run=args.run, step=0, status="waiting", signature={})
    if r["ok"]:
        out = Path(args.out_dir) / "page.html"
        write_atomic(out, page)
        r["page"] = str(out)
    print(summary_line("BUILT" if r["ok"] else "BLOCKED", r))
    return 0 if r["ok"] else 3


def cmd_baselines(args) -> int:
    """Plan step 11's page (rlm_rl/baseline_dashboard.py): its own output folder, so it never touches the training
    dashboard's page or publisher.json."""
    from rlm_rl import baseline_dashboard  # noqa: WPS433

    results = results_dir(args)
    out = Path(args.out_dir) / "page.html"
    r = baseline_dashboard.build([results / run for run in args.runs], out, seed_roots=seed_roots(results),
                                 pot=pot_arg(args), dry_run=is_dry(args, results))
    c = r.get("checked") or {}
    if r["ok"]:
        print(f"BUILT run={'+'.join(args.runs)} page={r['page']} size={r['bytes'] // 1024}KB check=passed (seeds {c.get('seeds')}, "
              f"eval tasks {c.get('eval_tasks')}, edit task ids {c.get('task_ids')}, repos {c.get('repos')})")
        return 0
    print(f"BLOCKED run={'+'.join(args.runs)} check=FAILED {json.dumps(r['findings'])} (nothing was written)")
    return 3


def cmd_check(args) -> int:
    page = Path(args.page).read_text(encoding="utf-8")
    results = results_dir(args)
    seeds = dashboard.private_seeds(seed_roots(results))
    items, prompts, notes, task_ids, repos = set(), [], [], set(), set()
    if args.run:
        items, unkeyed = dashboard.eval_items(results / args.run)
        task_ids, repos = dashboard.eval_names(results / args.run)
        notes += sorted(unkeyed)
        if args.deep:
            # a baseline run (plan step 11) has no training state: its eval set is named instead
            state = ({"config": {"selection": args.eval_set}} if args.eval_set
                     else dashboard.read_state(results / f"{args.run}.json"))
            extra, prompts, more, more_ids, more_repos = deep_items(state)
            items |= extra
            task_ids |= more_ids
            repos |= more_repos
            notes += more
    r = dashboard.check(page, seeds=seeds, items=items, prompts=prompts, task_ids=task_ids, repos=repos)
    print(json.dumps({"ok": r["ok"], "findings": r["findings"], "bytes": r["bytes"], "checked": r["checked"],
                      "deep": notes}, indent=1))
    return 0 if r["ok"] else 3


def cmd_wait(args) -> int:
    results = results_dir(args)
    dry = is_dry(args, results)
    pot = pot_arg(args)
    out_dir = Path(args.out_dir)
    publisher = out_dir / "publisher.json"
    pub = load_json(publisher, {})
    if pub.get("run") != args.run:
        pub = {"run": args.run}
    if args.ack and pub.get("built"):
        pub.update(published=pub["built"], published_at=time.time(), published_step=pub.get("built_step"))
        if pub.get("built_stale"):
            pub["stale_reported"] = pub["built_stale"]
    pub.pop("built", None)
    write_atomic(publisher, json.dumps(pub, indent=1))

    state_file = results / f"{args.run}.json"
    deadline = time.time() + args.timeout if args.timeout else None
    last_mtime = None
    while True:
        if deadline and time.time() > deadline:
            print(f"TIMEOUT run={args.run} nothing new in {args.timeout:g} s; run wait again to keep watching")
            return 4
        try:
            mtime = state_file.stat().st_mtime
        except OSError:
            time.sleep(args.poll)
            continue
        if mtime != last_mtime:
            last_mtime = mtime
            state = dashboard.read_state(state_file)
            sig = dashboard.signature(state)
            age = dashboard.heartbeat_age(state)
            stale = age is not None and age > args.stale_after
            # a stale run is reported once, until it writes again
            stale_new = stale and pub.get("stale_reported") != state.get("updated_at")
            if sig != pub.get("published") or stale_new:
                since = time.time() - float(pub.get("published_at") or 0)
                if since < args.min_interval:
                    time.sleep(args.min_interval - since)
                time.sleep(args.settle)  # let a burst of writes land (a step's end and the next step's start)
                r = build(args.run, results, dry=dry, pot=pot, out_dir=out_dir)
                pub.update(built=r["signature"], built_step=r["step"], built_at=time.time(),
                           built_stale=state.get("updated_at") if stale else None)
                write_atomic(publisher, json.dumps(pub, indent=1))
                if not r["ok"]:
                    print(summary_line("BLOCKED", r))
                    return 3
                if stale_new:
                    print(summary_line("STALE", r) + f" heartbeat_age={age:.0f}s")
                else:
                    print(summary_line("PUBLISH", r))
                return 0
        elif dashboard.heartbeat_age(state) is not None and dashboard.heartbeat_age(state) > args.stale_after \
                and pub.get("stale_reported") != state.get("updated_at"):
            last_mtime = None  # re-read: the run has gone quiet
        time.sleep(args.poll)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--results", help="results folder (default results/, or results/dry/ with --dry-run)")
        p.add_argument("--dry-run", action="store_true", help="a run under results/dry/ (the page says it's a dry run)")
        p.add_argument("--pot-left", type=float, help="W&B inference left before this run, $ (from W&B's billing page)")
        p.add_argument("--out-dir", default=str(OUT_DIR), help="where the page, its sample cache and publisher.json go")

    p = sub.add_parser("build", help="build and check the page once")
    p.add_argument("run")
    common(p)
    p = sub.add_parser("wait", help="wait for something new, then build and check the page")
    p.add_argument("run")
    common(p)
    p.add_argument("--ack", action="store_true", help="the last built page was published")
    p.add_argument("--poll", type=float, default=2.0, help="seconds between looks at the state file")
    p.add_argument("--settle", type=float, default=3.0, help="seconds to let a burst of writes land before building")
    p.add_argument("--min-interval", type=float, default=20.0, help="least seconds between publishes")
    p.add_argument("--stale-after", type=float, default=dashboard.STALE_AFTER_S,
                   help="seconds without a state write before a running run counts as stalled")
    p.add_argument("--timeout", type=float, help="give up after this many seconds")
    p = sub.add_parser("waiting", help="build the page between runs: the next run's name and config, no data")
    p.add_argument("run")
    p.add_argument("--kind", choices=("second_seed", "production"), default="second_seed")
    p.add_argument("--lr", type=float, help="production's learning rate")
    p.add_argument("--pot-left", type=float, help="W&B inference left, $ (from W&B's billing page)")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p = sub.add_parser("baselines", help="build and check plan step 11's baselines page")
    p.add_argument("runs", nargs="+", help="baseline runs, earliest first (step11-v1 step11-v3)")
    common(p)
    p.set_defaults(out_dir=str(RESULTS / "dashboard-step11"))
    p = sub.add_parser("check", help="scan a built page for keys, seeds, eval items and edit eval task ids")
    p.add_argument("page")
    p.add_argument("--run", help="also check against this run's eval transcripts")
    p.add_argument("--deep", action="store_true", help="also build the eval sets from their seeds and check every question")
    p.add_argument("--eval-set", help="with --deep, for a run without training state (the step 11 baselines): its eval set")
    p.add_argument("--results")
    p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return {"build": cmd_build, "wait": cmd_wait, "waiting": cmd_waiting, "check": cmd_check,
            "baselines": cmd_baselines}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
