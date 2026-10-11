"""A dry run of the training loop: the fake model server, a fake backend and the in-process sandbox stand-in.

No W&B call is made and no credit is spent. The tasks are the real ticket generators, and the
policy is an oracle that knows each task's answer (it records every task the loop draws) and
submits it with a probability that rises with the checkpoint and falls with the level. So the
pass rate climbs, families move up levels, entropy drifts down and every stop rule has something
to read, which is what plan step 9's dashboard needs to be built against.

    async with DryRun(results_dir) as dry:
        state = await dry.training_run("dry-prod", cfg=config.production("dry-prod", 2e-5, tasks_from="tickets")).start()

`scripts/train.py --dry-run` uses the same harness.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

from rlm_rl import bridge, capability, evals, tasks as rtasks
from rlm_rl.training import TrainingRun
from tests.fake_model import FakeModelServer
from tests.fakes import FakeBackend, FakeSandbox

_ARITH_RE = re.compile(r"What is (\d+) \+ (\d+)\?")


class Oracle:
    """The dry run's policy. `p_correct(step, level)` is the chance a rollout submits the exact answer."""

    def __init__(self, seed: int = 0, *, base: float = 0.35, slope: float = 0.02, level_penalty: float = 0.1,
                 cap: float = 0.92):
        self.gold: dict[str, tuple] = {}
        self.rng = random.Random(seed)
        self.base, self.slope, self.level_penalty, self.cap = base, slope, level_penalty, cap
        self.capability_ok = lambda step: True  # flip to fake a capability drop

    def record(self, tasks: list) -> list:
        for t in tasks:
            self.gold[bridge.first_user_message(t.question, t.context)] = (t.answer, t.level)
        return tasks

    def draw(self, n: int, mix: dict, levels: dict) -> list:
        return self.record(rtasks.draw(n, mix, levels))

    def eval_set(self, name: str) -> tuple[list, int]:
        tasks, samples = evals.eval_set(name)
        return self.record(tasks), samples

    @staticmethod
    def step_of(model: str) -> int:
        found = re.search(r":step(\d+)$", model)
        return int(found.group(1)) if found else 0

    def p_correct(self, step: int, level: int) -> float:
        return max(0.05, min(self.cap, self.base + self.slope * step - self.level_penalty * (level - 1)))

    def policy_for(self, model: str, messages: list[dict]) -> str:
        if sum(m["role"] == "assistant" for m in messages) == 0:
            return ("Look at the structure first.\n```python\nlines = context.splitlines()\nprint(len(lines))\n"
                    "print(llm('Summarize: ' + lines[0][:200]))\n```")
        answer, level = self.gold.get(messages[1]["content"], (None, 1))
        if answer is None:
            value = "len(context.splitlines()) // 4"
        elif self.rng.random() < self.p_correct(self.step_of(model), level):
            value = repr(answer)
        else:
            value = repr(self.wrong(answer))
        return f"Submit.\n```python\nn = len(context)\nanswer = {value}\nprint(answer)\nsubmit(answer)\n```"

    def wrong(self, answer):
        """A near miss of any answer type: a count off by a little, a list missing an item, a wrong name."""
        if isinstance(answer, int):
            return max(0, answer + self.rng.choice([-3, -2, -1, 1, 2, 3]))
        if isinstance(answer, list):
            return answer[1:] if len(answer) > 1 else answer + ["not-an-answer"]
        return f"{answer}_other"

    def answer_for(self, model: str, prompt: str) -> str:
        found = _ARITH_RE.search(prompt)
        if found:  # a capability-check prompt
            total = int(found.group(1)) + int(found.group(2))
            return str(total if self.capability_ok(self.step_of(model)) else total + 1)
        return "ok"

    def entropy_logprob(self, model: str) -> float:
        """Sampled-token logprob: entropy starts at 0.6 and drifts down slowly."""
        return -max(0.3, 0.6 - 0.003 * self.step_of(model))


def capability_items(n: int = 20, seed: int = 7) -> list[capability.Item]:
    rng = random.Random(seed)
    items = []
    for i in range(n):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        items.append(capability.Item(f"arith-{i:02d}", f"What is {a} + {b}? Reply with the number only.",
                                     [{"kind": "number", "value": a + b}], source="arithmetic"))
    return items


class DryRun:
    """Starts the fake server and backend, and builds TrainingRuns wired to them."""

    def __init__(self, results: Path, *, seed: int = 0, oracle: Oracle | None = None, persist: bool = False,
                 base_logprob: float = -0.8):
        self.results = Path(results)
        self.seed = seed
        self.oracle = oracle or Oracle(seed)
        self.persist = persist
        self.base_logprob = base_logprob
        self.server: FakeModelServer | None = None
        self.backend: FakeBackend | None = None
        self._private = rtasks.PRIVATE_DIR

    async def __aenter__(self) -> DryRun:
        self.results.mkdir(parents=True, exist_ok=True)
        rtasks.PRIVATE_DIR = self.results / "private"  # the dry run's eval seeds never touch private/
        self.server = FakeModelServer()
        self.server.policy_for = self.oracle.policy_for
        self.server.answer_for = self.oracle.answer_for
        self.server.completion_logprob = self.oracle.entropy_logprob
        self.server.prompt_logprob = lambda model: self.base_logprob
        await self.server.start()
        self.backend = FakeBackend(self.server.base_url,
                                   path=str(self.results / "fake-backend.json") if self.persist else None)
        return self

    async def __aexit__(self, *exc) -> None:
        await self.backend.close()
        await self.server.close()
        rtasks.PRIVATE_DIR = self._private

    def training_run(self, run: str, **kwargs) -> TrainingRun:
        kwargs.setdefault("capability_items", capability_items())
        kwargs.setdefault("sandbox_factory", FakeSandbox)
        kwargs.setdefault("rng", random.Random(self.seed))
        return TrainingRun(run, backend=self.backend, results=self.results, draw=self.oracle.draw,
                           eval_sets=self.oracle.eval_set, **kwargs)
