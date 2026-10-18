"""The difficulty controller: one level per task family, moved by the family's recent pass rate (spec §6, H13).

After every step, each family's exact-accuracy pass rate is taken over its episodes from its
last 3 steps at the current level:
- above 70%: move up a level, unless it's already at the top (then it counts as saturated, stop rule 4)
- below 20%: move down a level, never below level 1
- otherwise: stay

A move clears the family's window, so it waits 3 steps at the new level before it can move again,
and its pass rate is never mixed across levels. A window that is unanimous (every episode right, or
every one wrong) on at least 8 scored episodes moves at once, without waiting for 3 steps: all of
those groups teach GRPO nothing (prod-v1, 2026-10-18: edit synth went 20 of 20 over its first 4 steps). A step in which a family drew no tasks, or only
infra-error episodes, doesn't count toward its window *(mine)*. Neither does a window with fewer
than 8 scored episodes *(mine)*, so a rarely drawn family isn't moved on a handful of episodes.

A frozen controller (the second seed runs at level 1, plan step 10) records what it would have
done but never moves.

    ctl = DifficultyController({"tickets_all": 5, "tickets_customer": 5})
    level = ctl.level("tickets_all")
    moves = ctl.update(step, [ep for group in groups for ep in group])
"""

from __future__ import annotations

import collections
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

UP_ABOVE = 0.70
# Was 0.20. Raised with the step-12 task shift (2026-10-18): groups only teach when a task is neither always solved
# nor never solved, and Online Difficulty Filtering (arXiv 2504.03380) found a 0.3-0.7 pass-rate band works best.
DOWN_BELOW = 0.30
WINDOW_STEPS = 3
MIN_EPISODES = 8
FIRST_LEVEL = 1


def settings() -> dict:
    """The pinned controller rule, for the run's config.json (spec §8)."""
    return {"up_above": UP_ABOVE, "down_below": DOWN_BELOW, "window_steps": WINDOW_STEPS,
            "min_episodes": MIN_EPISODES, "first_level": FIRST_LEVEL, "unanimous_moves_at_once": True}


@dataclass
class Move:
    step: int
    family: str
    from_level: int
    to_level: int
    pass_rate: float
    episodes: int
    applied: bool  # False when the controller is frozen

    @property
    def direction(self) -> str:
        return "up" if self.to_level > self.from_level else "down"


@dataclass
class FamilyState:
    level: int
    max_level: int
    window: list = field(default_factory=list)  # [step, episodes, passes] per counted step at this level
    moved_at: int | None = None
    saturated: bool = False  # top level with a pass rate above UP_ABOVE

    def pass_rate(self) -> tuple[float | None, int]:
        episodes = sum(n for _, n, _ in self.window)
        passes = sum(p for _, _, p in self.window)
        return (passes / episodes if episodes else None), episodes


class DifficultyController:
    def __init__(self, max_levels: dict[str, int], *, frozen: bool = False, levels: dict[str, int] | None = None):
        if not max_levels:
            raise ValueError("the controller needs at least one family")
        self.frozen = frozen
        self.families: dict[str, FamilyState] = {}
        for family, top in max_levels.items():
            start = (levels or {}).get(family, FIRST_LEVEL)
            if top < FIRST_LEVEL or not FIRST_LEVEL <= start <= top:
                raise ValueError(f"{family}: level {start} is outside 1..{top}")
            self.families[family] = FamilyState(start, top)
        self.history: list[Move] = []

    def level(self, family: str) -> int:
        return self.families[family].level

    def set_families(self, max_levels: dict[str, int]) -> None:
        """Make the controller track exactly these families: a new one starts at level 1, a dropped one is forgotten
        (a resume that changes the training mix, TrainingRun._set_mix)."""
        for family in list(self.families):
            if family not in max_levels:
                del self.families[family]
        for family, top in max_levels.items():
            if family not in self.families:
                self.families[family] = FamilyState(FIRST_LEVEL, top)

    def levels(self) -> dict[str, int]:
        return {family: state.level for family, state in self.families.items()}

    def update(self, step: int, episodes: Iterable[Any]) -> list[Move]:
        """Record one step's episodes and apply the rule to every family. Returns the moves (or would-be moves)."""
        tally: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        for ep in episodes:
            state = self.families.get(ep.family)
            if state is None:
                raise KeyError(f"no level for family {ep.family!r}")
            if getattr(ep, "infra_error", None) is not None or ep.level != state.level:
                continue  # infra episodes aren't scored; episodes drawn at another level don't count here
            tally[ep.family][0] += 1
            tally[ep.family][1] += bool(ep.correct)

        moves = []
        for family, state in self.families.items():
            if family in tally:
                state.window.append([step, *tally[family]])
                state.window = state.window[-WINDOW_STEPS:]
            rate, n = state.pass_rate()
            state.saturated = False
            unanimous = n >= MIN_EPISODES and rate in (0.0, 1.0)  # moves at once (module docstring)
            if (len(state.window) < WINDOW_STEPS and not unanimous) or n < MIN_EPISODES:
                continue
            target = state.level
            if rate > UP_ABOVE:
                if state.level < state.max_level:
                    target = state.level + 1
                else:
                    state.saturated = True
            elif rate < DOWN_BELOW and state.level > FIRST_LEVEL:
                target = state.level - 1
            if target == state.level:
                continue
            move = Move(step, family, state.level, target, round(rate, 4), n, applied=not self.frozen)
            moves.append(move)
            if not self.frozen:
                state.level, state.window, state.moved_at = target, [], step
        self.history.extend(moves)
        return moves

    def saturated(self) -> bool:
        """Stop rule 4 (plan step 12): every family at its top level with a pass rate above 70%."""
        return all(state.saturated for state in self.families.values())

    def summary(self) -> dict:
        """Per-family level, window pass rate and episodes, for the dashboard."""
        out = {}
        for family, state in self.families.items():
            rate, n = state.pass_rate()
            out[family] = {"level": state.level, "max_level": state.max_level, "pass_rate": rate, "episodes": n,
                           "window_steps": len(state.window), "saturated": state.saturated}
        return out

    def state_dict(self) -> dict:
        """Everything needed to resume, saved with each checkpoint (spec §8)."""
        return {"frozen": self.frozen, "families": {f: asdict(s) for f, s in self.families.items()},
                "history": [asdict(m) for m in self.history], "settings": settings()}

    @classmethod
    def from_state(cls, state: dict) -> DifficultyController:
        families = state["families"]
        ctl = cls({f: s["max_level"] for f, s in families.items()}, frozen=state["frozen"],
                  levels={f: s["level"] for f, s in families.items()})
        for family, saved in families.items():
            ctl.families[family] = FamilyState(**saved)
        ctl.history = [Move(**m) for m in state.get("history", [])]
        return ctl
