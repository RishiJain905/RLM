"""Stop rules, the collapse check and the learning-rate rule (research §8.6, spec §5.2, plan steps 6 and 12).

Pure functions over the run's recorded steps and evals, so each rule can be tested on synthetic
curves. The training loop (training.py) calls them after every step and acts on what fires.

The five stop rules (research §8.6). Each returns a status for the dashboard ("ok", "watch",
"fired" or "n/a") with one line saying where it stands:
1. plateau: no new selection-set best for 3 evals in a row, and no family moved up a level in the
   last 30 steps. The run stops and keeps the best checkpoint.
2. overfitting: training reward rose while the selection set fell, at 2 evals in a row. Stop.
3. frying: the capability check dropped more than 2 points below checkpoint 0's, or a collapse
   came back after the one rollback with a lower learning rate. Stop.
4. saturation: every family is at its top level with a pass rate above 70%. Stop.
5. hacking: a Claude audit found a hack among the wins, or the detectors' hit rate is rising.
   The run pauses for a fix (Rishi makes the resume call).

The collapse check (spec §5.2 step 3) fires on either signal:
- entropy below 70% of its highest value over the previous 10 steps
- within-group reward std below half its previous-10-step average for 3 steps in a row
At 2e-5 the loop rolls back to the last good checkpoint and continues at 1e-5. The thresholds
are the spec's proposed ones, to be finalized on the second seed's real curves.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import Any, Iterable

# Stop rules (research §8.6)
PLATEAU_EVALS = 3
PLATEAU_LEVEL_UP_STEPS = 30
OVERFIT_EVALS = 2
OVERFIT_WINDOW = 10  # training steps averaged on each side of an eval
CAPABILITY_DROP = 0.02  # "more than ~2 points"
CAPABILITY_MIN_COMPLETE = 0.9  # a check with more than 10% infra failures isn't compared *(mine)*
HACK_WINDOW = 10
HACK_RATE = 0.05  # detector hits as a share of scored episodes over the window *(mine)*
HACK_MIN_STEPS = 5

# Collapse (spec §5.2 step 3, proposed thresholds)
ENTROPY_DROP = 0.30
ENTROPY_WINDOW = 10
ENTROPY_MIN_HISTORY = 3
STD_DROP = 0.5
STD_WINDOW = 10
STD_STREAK = 3
STD_MIN_HISTORY = 5

# The second seed's stability checks, which pick production's learning rate (spec §5.2 step 2, proposed)
ENTROPY_KEEP = 0.8  # mean entropy over steps 16-20 at least 80% of steps 1-5
GRAD_SPIKE = 3.0  # grad norm never above 3x its median
PROBS_CORR_MIN = 0.98


def settings() -> dict:
    return {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float))}


@dataclass
class Status:
    rule: str
    state: str  # "ok", "watch", "fired" or "n/a"
    text: str
    action: str = "stop"  # what firing means: "stop" or "pause"

    @property
    def fired(self) -> bool:
        return self.state == "fired"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Collapse:
    kind: str  # "entropy" or "reward_std"
    step: int  # the step that tripped it
    first_bad_step: int
    text: str


def _get(row: dict, *path) -> Any:
    for key in path:
        if not isinstance(row, dict):
            return None
        row = row.get(key)
    return row


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


# ----------------------------------------------------------------------------- evals


def eval_points(evals: Iterable[dict], name: str) -> list[tuple[int, float]]:
    """(checkpoint, accuracy) for one eval, in checkpoint order."""
    points = {e["checkpoint"]: _get(e, "summary", "accuracy") for e in evals if e["name"] == name}
    return sorted((c, a) for c, a in points.items() if a is not None)


def new_bests(points: list[tuple[int, float]]) -> list[bool]:
    out, best = [], None
    for _, acc in points:
        out.append(best is None or acc > best)
        best = acc if best is None else max(best, acc)
    return out


def best_checkpoint(points: list[tuple[int, float]]) -> tuple[int, float] | None:
    """The best checkpoint so far; on a tie, the earliest (it reached that accuracy first)."""
    return max(points, key=lambda p: (p[1], -p[0])) if points else None


def plateau(points: list[tuple[int, float]], level_ups: Iterable[int]) -> Status:
    """Rule 1. `level_ups` are the steps at which a family moved up a level."""
    if not points:
        return Status("plateau", "n/a", "no selection-set eval yet")
    flags = new_bests(points)
    trailing = 0
    for flag in reversed(flags):
        if flag:
            break
        trailing += 1
    last = points[-1][0]
    recent_up = [s for s in level_ups if s > last - PLATEAU_LEVEL_UP_STEPS]
    text = f"{trailing} of {PLATEAU_EVALS} evals without a new best"
    if recent_up:
        text += f"; a family moved up at step {max(recent_up)}"
    if trailing >= PLATEAU_EVALS and not recent_up:
        return Status("plateau", "fired", text + f", and no level-up in the last {PLATEAU_LEVEL_UP_STEPS} steps")
    return Status("plateau", "watch" if trailing else "ok", text)


def train_reward_near(rows: list[dict], checkpoint: int, window: int = OVERFIT_WINDOW) -> float | None:
    """Mean training reward over the `window` steps that led to `checkpoint`."""
    return _mean(_get(r, "reward", "mean") for r in rows if checkpoint - window < r["step"] <= checkpoint)


def overfitting(points: list[tuple[int, float]], rows: list[dict]) -> Status:
    """Rule 2: training reward up while the selection set fell, at OVERFIT_EVALS evals in a row."""
    points = [p for p in points if p[0] > 0]  # checkpoint 0 has no training reward before it
    if len(points) < 2:
        return Status("overfitting", "n/a", "needs two selection evals after training starts")
    bad = []
    for (c0, a0), (c1, a1) in zip(points, points[1:]):
        r0, r1 = train_reward_near(rows, c0), train_reward_near(rows, c1)
        bad.append(r0 is not None and r1 is not None and r1 > r0 and a1 < a0)
    streak = 0
    for flag in reversed(bad):
        if not flag:
            break
        streak += 1
    text = f"{streak} of {OVERFIT_EVALS} evals with training reward up and the selection set down"
    if streak >= OVERFIT_EVALS:
        return Status("overfitting", "fired", text)
    return Status("overfitting", "watch" if streak else "ok", text)


def capability_points(evals: Iterable[dict]) -> list[tuple[int, float]]:
    points = {e["checkpoint"]: e["summary"] for e in evals if e["name"] == "capability"}
    return sorted((c, s["accuracy"]) for c, s in points.items()
                  if s.get("accuracy") is not None and s.get("complete", 1.0) >= CAPABILITY_MIN_COMPLETE)


def frying(evals: Iterable[dict], *, collapse_after_rollback: str | None = None) -> Status:
    """Rule 3: the capability check fell more than CAPABILITY_DROP below checkpoint 0, or a collapse came back."""
    if collapse_after_rollback:
        return Status("frying", "fired", collapse_after_rollback)
    points = capability_points(evals)
    base = dict(points).get(0)
    if base is None:
        return Status("frying", "n/a", "no capability check at checkpoint 0")
    latest_c, latest = points[-1]
    if latest_c == 0:
        return Status("frying", "ok", f"capability {base:.3f} at checkpoint 0")
    drop = base - latest
    change = f"{abs(drop) * 100:.1f} points {'below' if drop > 0 else 'above'}" if drop else "the same as"
    text = f"capability {latest:.3f} at checkpoint {latest_c}, {change} checkpoint 0's {base:.3f}"
    if drop > CAPABILITY_DROP:
        return Status("frying", "fired", text)
    return Status("frying", "watch" if drop > 0 else "ok", text)


def saturation(controller_summary: dict) -> Status:
    """Rule 4: every family at its top level with a pass rate above 70%."""
    if not controller_summary:
        return Status("saturation", "n/a", "no families")
    top = [f for f, s in controller_summary.items() if s["level"] == s["max_level"]]
    saturated = [f for f, s in controller_summary.items() if s.get("saturated")]
    text = f"{len(saturated)} of {len(controller_summary)} families saturated, {len(top)} at their top level"
    if len(saturated) == len(controller_summary):
        return Status("saturation", "fired", text)
    return Status("saturation", "watch" if top else "ok", text)


# ----------------------------------------------------------------------------- hacking


def _hack_share(rows: list[dict]) -> float | None:
    scored = sum(_get(r, "infra", "episodes") - _get(r, "infra", "infra") for r in rows)
    return sum(_get(r, "reward", "hacked") or 0 for r in rows) / scored if scored else None


def hacking(rows: list[dict], audit_hacks: list[dict], *, since_step: int = 0) -> Status:
    """Rule 5. `audit_hacks` are hack verdicts among audited wins that nobody has acknowledged by resuming."""
    if audit_hacks:
        where = ", ".join(sorted({f"checkpoint {h['checkpoint']}" for h in audit_hacks}))
        return Status("hacking", "fired", f"the Claude audit found {len(audit_hacks)} hack(s) among the wins ({where})",
                      action="pause")
    recent = [r for r in rows if r["step"] > since_step]
    current, previous = recent[-HACK_WINDOW:], recent[-2 * HACK_WINDOW:-HACK_WINDOW]
    share, before = _hack_share(current), _hack_share(previous)
    if share is None:
        return Status("hacking", "n/a", "no scored episodes yet", action="pause")
    text = f"detectors fired on {share:.1%} of episodes over the last {len(current)} steps"
    if before is not None:
        text += f" ({before:.1%} the {len(previous)} before)"
    if len(current) >= HACK_MIN_STEPS and share >= HACK_RATE and (before is None or share > before):
        return Status("hacking", "fired", text + f", above {HACK_RATE:.0%} and rising", action="pause")
    return Status("hacking", "watch" if share else "ok", text, action="pause")


# ----------------------------------------------------------------------------- collapse and the LR rule


def collapse(rows: list[dict], *, since_step: int = 0) -> Collapse | None:
    """Spec §5.2 step 3's two signals, checked at the latest step. Only steps after `since_step` can trip it."""
    if not rows or rows[-1]["step"] <= since_step:
        return None
    t = rows[-1]["step"]
    by_step = {r["step"]: r for r in rows}

    ent_t = _get(rows[-1], "policy", "entropy")
    history = [(s, _get(by_step[s], "policy", "entropy")) for s in range(t - ENTROPY_WINDOW, t) if s in by_step]
    history = [(s, e) for s, e in history if e is not None]
    if ent_t is not None and len(history) >= ENTROPY_MIN_HISTORY:
        peak_step, peak = max(history, key=lambda p: (p[1], p[0]))
        if ent_t < (1 - ENTROPY_DROP) * peak:
            return Collapse("entropy", t, peak_step + 1,
                            f"entropy {ent_t:.3f} at step {t} is {1 - ent_t / peak:.0%} below its peak of {peak:.3f} "
                            f"at step {peak_step} (limit {ENTROPY_DROP:.0%} within {ENTROPY_WINDOW} steps)")

    streak = []
    for s in range(t - STD_STREAK + 1, t + 1):
        std = _get(by_step.get(s, {}), "policy", "reward_std")
        prior = [_get(by_step[p], "policy", "reward_std") for p in range(s - STD_WINDOW, s) if p in by_step]
        prior = [p for p in prior if p is not None]
        if s <= since_step or std is None or len(prior) < STD_MIN_HISTORY:
            return None
        average = sum(prior) / len(prior)
        if not std < STD_DROP * average:
            return None
        streak.append(f"{std:.3f} vs {average:.3f}")
    return Collapse("reward_std", t, t - STD_STREAK + 1,
                    f"within-group reward std below half its 10-step average for {STD_STREAK} steps "
                    f"(steps {t - STD_STREAK + 1}-{t}: {'; '.join(streak)})")


def last_good_checkpoint(first_bad_step: int, kept: Iterable[int]) -> int:
    """The newest kept checkpoint before the collapse began.

    Rollouts at step s are sampled from checkpoint s-1, so the first bad step's rollouts came from
    checkpoint first_bad_step-1, and the last checkpoint known to be good is first_bad_step-2.
    """
    limit = first_bad_step - 2
    return max([k for k in kept if k <= limit] or [0])


def lr_decision(rows: list[dict], *, high: float, floor: float) -> dict:
    """Spec §5.2 step 2: read the second seed's curves and pick production's starting learning rate."""
    def series(*path):
        return [(r["step"], _get(r, *path)) for r in rows if _get(r, *path) is not None]

    entropy = dict(series("policy", "entropy"))
    first = _mean(entropy.get(s) for s in range(1, 6))
    last = _mean(entropy.get(s) for s in range(16, 21))
    grads = [g for _, g in series("policy", "grad_norm")]
    corr = [c for _, c in series("policy", "probs_corr")]
    checks = {
        "entropy_flat": {"passed": None if first is None or last is None else last >= ENTROPY_KEEP * first,
                         "steps_1_5": first, "steps_16_20": last, "need": f"steps 16-20 >= {ENTROPY_KEEP:.0%} of 1-5"},
        "grad_norm_spikes": {"passed": None if not grads else max(grads) <= GRAD_SPIKE * statistics.median(grads),
                             "max": max(grads) if grads else None,
                             "median": statistics.median(grads) if grads else None,
                             "need": f"never above {GRAD_SPIKE:g}x the median"},
        "probs_corr": {"passed": None if not corr else min(corr) > PROBS_CORR_MIN, "min": min(corr) if corr else None,
                       "need": f"above {PROBS_CORR_MIN}"},
    }
    passed = all(c["passed"] for c in checks.values())
    missing = [name for name, c in checks.items() if c["passed"] is None]
    return {"production_lr": high if passed else floor, "passed": passed, "checks": checks,
            "note": (f"missing data for {', '.join(missing)}, so production stays at {floor:g}" if missing
                     else f"all checks passed, so production may start at {high:g}" if passed
                     else f"a check failed, so production stays at {floor:g}")}
