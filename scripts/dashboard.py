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


def seed_roots(results: Path) -> list[Path]:
    return [ROOT / "private" / "seeds", results / "private" / "seeds"]


def deep_items(state: dict) -> tuple[set[tuple[str, str]], list[str], list[str]]:
    """Every question of the eval sets this run uses, the held-out test and the capability prompts.

    This builds the sets from their private seeds through rlm_rl, which takes a minute and needs the
    data on D:, so it's for a one-off check, not every build.
    """
    from rlm_rl import capability, evals  # noqa: WPS433 (heavy imports kept out of every build)

    cfg = state.get("config") or {}
    items, prompts, notes = set(), [], []
    for name in [n for n in (cfg.get("selection"), cfg.get("final_eval"), "heldout-test-v1") if n]:
        try:
            tasks, _ = evals.eval_set(name)
            keys = [dashboard.item_key(t.question, "\n".join(t.context.splitlines()[:3])) for t in tasks]
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
    return items, prompts, notes


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
    result = dashboard.check(page, seeds=dashboard.private_seeds(seed_roots(results)), items=items)
    result.update(run=run, step=(state.get("steps") or [{}])[-1].get("step", 0), status=state.get("status"),
                  signature=dashboard.signature(state), heartbeat_age_s=dashboard.heartbeat_age(state))
    if result["ok"]:
        write_atomic(out, page)
        write_atomic(cache_path, json.dumps(cache))
        result["page"] = str(out)
    return result


def summary_line(kind: str, r: dict) -> str:
    sig = r.get("signature") or {}
    head = (f"{kind} run={r.get('run')} step={r.get('step')} status={r.get('status')} evals={sig.get('evals')} "
            f"notices={sig.get('notices')} size={r.get('bytes', 0) // 1024}KB")
    if r.get("ok"):
        c = r.get("checked") or {}
        return f"{head} page={r.get('page')} check=passed (seeds {c.get('seeds')}, eval tasks {c.get('eval_tasks')})"
    return f"{head} check=FAILED {json.dumps(r.get('findings'))} (nothing was written; the last good page stands)"


def cmd_build(args) -> int:
    results = results_dir(args)
    pot = {"inference_left_usd": args.pot_left} if args.pot_left is not None else None
    r = build(args.run, results, dry=is_dry(args, results), pot=pot, out_dir=Path(args.out_dir))
    print(summary_line("BUILT" if r["ok"] else "BLOCKED", r))
    return 0 if r["ok"] else 3


def cmd_check(args) -> int:
    page = Path(args.page).read_text(encoding="utf-8")
    results = results_dir(args)
    seeds = dashboard.private_seeds(seed_roots(results))
    items, prompts, notes = set(), [], []
    if args.run:
        items, unkeyed = dashboard.eval_items(results / args.run)
        notes += sorted(unkeyed)
        if args.deep:
            extra, prompts, more = deep_items(dashboard.read_state(results / f"{args.run}.json"))
            items |= extra
            notes += more
    r = dashboard.check(page, seeds=seeds, items=items, prompts=prompts)
    print(json.dumps({"ok": r["ok"], "findings": r["findings"], "bytes": r["bytes"], "checked": r["checked"],
                      "deep": notes}, indent=1))
    return 0 if r["ok"] else 3


def cmd_wait(args) -> int:
    results = results_dir(args)
    dry = is_dry(args, results)
    pot = {"inference_left_usd": args.pot_left} if args.pot_left is not None else None
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
    p = sub.add_parser("check", help="scan a built page for keys, seeds and eval items")
    p.add_argument("page")
    p.add_argument("--run", help="also check against this run's eval transcripts")
    p.add_argument("--deep", action="store_true", help="also build the eval sets from their seeds and check every question")
    p.add_argument("--results")
    p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return {"build": cmd_build, "wait": cmd_wait, "check": cmd_check}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
