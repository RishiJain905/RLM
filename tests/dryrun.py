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

import base64
import dataclasses
import hashlib
import random
import re
from pathlib import Path

from rlm_rl import bridge, capability, evals, tasks as rtasks
from rlm_rl.client import Pacer
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
        # With the in-process edit sandbox, edit_real's test commands are echoed, not run: they would run a real
        # repository's tests on this machine (and cargo would fetch crates). The fake grader compares files instead.
        self.echo_real_tests = True

    def record(self, tasks: list) -> list:
        for t in tasks:
            if bridge.is_edit(t.family):  # the gold change, as files to write (spec §12)
                lim = bridge.limits(t.family)
                if (t.edit or {}).get("cell_timeout_s"):  # the task's own cell limit, quoted as the environment does
                    lim = dataclasses.replace(lim, cell_timeout_s=int(t.edit["cell_timeout_s"]))
                gold = (EditGold(t, echo=self.echo_real_tests and t.family != "edit_synth"), t.level)
                for version in bridge.PROMPT_VERSIONS:
                    try:
                        first = bridge.initial_messages(t.question, t.context, version, lim, edit=t.edit)[1]
                    except ValueError:  # edit tasks need prompt v2 (spec §12.2)
                        continue
                    self.gold[first["content"]] = gold
                continue
            for version in bridge.PROMPT_VERSIONS:
                first = bridge.first_user_message(t.question, t.context, version, bridge.limits(t.family))
                self.gold[first] = (t.answer, t.level)
        return tasks

    def draw(self, n: int, mix: dict, levels: dict, step: int | None = None, exclude=None) -> list:
        return self.record(rtasks.draw(n, mix, levels, step, exclude=exclude))

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
        answer, level = self.gold.get(messages[1]["content"], (None, 1))
        if isinstance(answer, EditGold):
            return self.edit_policy(model, messages, answer, level)
        if sum(m["role"] == "assistant" for m in messages) == 0:
            return ("Look at the structure first.\n```python\nlines = context.splitlines()\nprint(len(lines))\n"
                    "print(llm('Summarize: ' + lines[0][:200]))\n```")
        if answer is None:
            value = "len(context.splitlines()) // 4"
        elif self.rng.random() < self.p_correct(self.step_of(model), level):
            value = repr(answer)
        else:
            value = repr(self.wrong(answer))
        return f"Submit.\n```python\nn = len(context)\nanswer = {value}\nprint(answer)\nsubmit(answer)\n```"

    def edit_policy(self, model: str, messages: list[dict], gold: "EditGold", level: int) -> str:
        """An edit episode: look at the repository, then write the gold change (or not), run the visible tests, submit."""
        turn = sum(m["role"] == "assistant" for m in messages)
        tools = "read_file(path, start=1" in messages[0]["content"] and gold.use_tools  # prompt v2 lists the coding tools
        if turn == 0:
            if tools:
                look = f"print(read_file({gold.first!r}, n=20))\n" if gold.first else ""
                return ("Survey the repository first.\n```python\nfiles = find_files('**/*')\nprint(len(files), files[:10])\n"
                        f"{look}print(len(grep('def |func |fn |function ', files_only=True)))\n"
                        "print(llm('Which files relate to this task? ' + context[:400]))\n```")
            return ("Survey the repository first.\n```python\nimport os\nfiles = [os.path.join(d, f) for d, _, fs in os.walk('/work/repo') "
                    "if '.git' not in d for f in fs]\nprint(len(files))\nprint(llm('Which files relate to this task? ' + context[:400]))\n```")
        if turn == 1:
            right = self.rng.random() < self.p_correct(self.step_of(model), level)
            review = ("\nprint(llm('Review this change:\\ndiff --git a/x b/x\\n@@ -1 +1 @@\\n-old\\n+new'))" if self.rng.random() < 0.4
                      else "")
            network = "\n# fake: audit network" if self.rng.random() < 0.15 else ""
            if tools:  # half the tasks use the coding tools, now and then with a failed edit_file first
                miss = ("try:\n    edit_file(" + repr(gold.first) + ", 'text that is not there', 'x')\nexcept ValueError as e:\n"
                        "    print(e)\n") if gold.first and self.rng.random() < 0.3 else ""
                writes = gold.write_tools_code() if right else ""
                return (f"Make the change and run the tests.\n```python\nimport base64\n{miss}{writes}{review}{network}\n"
                        f"print(diff()[:300])\nprint(shell({' '.join(gold.test_argv)!r}))\n```")
            writes = gold.write_code() if right else ""
            return (f"Make the change and run the tests.\n```python\nimport subprocess, base64\n{writes}{review}{network}\n"
                    f"r = subprocess.run({gold.test_argv!r}, cwd='/work/repo', capture_output=True, text=True)\n"
                    "print(r.stdout[-300:], r.stderr[-300:])\n```")
        return "Submit.\n```python\nsubmit('done')\n```"

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


class EditGold:
    """What the dry run's oracle knows about an edit task: the files its gold change writes, and how to run its tests."""

    def __init__(self, task, echo: bool = False):
        from tests.fakes import gold_files

        self.files = gold_files(task)
        self.test_argv = (["echo", "dry", "run:"] if echo else []) + task.edit["test_hint"].split()
        self.first = sorted(self.files)[0] if self.files else None
        self.use_tools = int(hashlib.sha256(task.task_id.encode()).hexdigest(), 16) % 2 == 0  # half the tasks

    def write_tools_code(self) -> str:
        """The gold change written with the coding tools' write_file."""
        lines = []
        for path, text in sorted(self.files.items()):
            blob = base64.b64encode(text.encode()).decode()
            lines.append(f"write_file({path!r}, base64.b64decode('{blob}').decode())")
        return "\n".join(lines)

    def write_code(self) -> str:
        lines = []
        for path, text in sorted(self.files.items()):
            blob = base64.b64encode(text.encode()).decode()
            lines.append(f"__import__('pathlib').Path('/work/repo/{path}').write_bytes(base64.b64decode('{blob}'))")
        return "\n".join(lines)


class DryRun:
    """Starts the fake server and backend, and builds TrainingRuns wired to them."""

    def __init__(self, results: Path, *, seed: int = 0, oracle: Oracle | None = None, persist: bool = False,
                 base_logprob: float = -0.8, real_edit: bool = False):
        self.results = Path(results)
        self.seed = seed
        self.oracle = oracle or Oracle(seed)
        self.persist = persist
        self.base_logprob = base_logprob
        self.real_edit = real_edit  # edit episodes in the real edit sandbox and grader (Docker), not the in-process fakes
        self.oracle.echo_real_tests = not real_edit
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
        if not self.real_edit:
            from tests.fakes import FakeEditSandbox, local_grade

            kwargs["sandbox_kwargs"] = {"edit_sandbox_factory": FakeEditSandbox, "grade": local_grade,
                                      **(kwargs.get("sandbox_kwargs") or {})}
        kwargs.setdefault("rng", random.Random(self.seed))
        kwargs.setdefault("pacer", Pacer())  # the fake server has no rate limit to space sub-calls out for
        return TrainingRun(run, backend=self.backend, results=self.results, draw=self.oracle.draw,
                           eval_sets=self.oracle.eval_set, **kwargs)
