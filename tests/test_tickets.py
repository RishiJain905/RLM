"""The ticket tasks, ported with difficulty levels (plan step 5), and the task registry the controller uses.

    python3 -m unittest -v tests.test_tickets
"""

import re
import statistics
import unittest

from rlm_rl import reward, tasks
from rlm_rl.tasks import tickets


class TicketTaskTest(unittest.TestCase):
    def test_levels_grow_to_about_400k_characters(self):
        medians = []
        for level in sorted(tickets.LEVELS):
            sizes = [len(tasks.make_task(f, level).context) for f in tickets.FAMILIES for _ in range(6)]
            medians.append(statistics.median(sizes))
        self.assertEqual(medians, sorted(medians))
        self.assertTrue(30_000 < medians[0] < 50_000, medians[0])  # the 20-step run's size
        self.assertTrue(330_000 < medians[-1] < 450_000, medians[-1])  # ~20 sub-calls of ~20K characters
        self.assertGreater(medians[-1], 130_000)  # plan step 7: top levels above 130K

    def test_every_task_is_scorable_and_carries_its_blind_guess(self):
        for family in tickets.FAMILIES:
            for level in tickets.LEVELS:
                task = tasks.make_task(family, level)
                with self.subTest(family=family, level=level):
                    reward.check_grading(task.grading, task.answer)
                    self.assertIsInstance(task.answer, int)
                    self.assertEqual((task.family, task.level, task.grading["type"]), (family, level, "int"))
                    n_lines = len(task.context.splitlines())
                    share = tickets.CUSTOMER_SHARE if family == "tickets_customer" else None
                    if share is not None:
                        self.assertEqual(task.grading["blind"], round(n_lines * share))
                    self.assertEqual(reward.score_answer(task.answer, task.answer, task.grading).reward, 1.0)

    def test_the_answer_is_what_the_context_says(self):
        """Recount every task from its rendered text alone, by matching each ticket against the templates."""
        items = "(?:" + "|".join(map(re.escape, tickets._ITEMS)) + ")"
        patterns = [(kind, re.compile(re.escape(t[0].upper() + t[1:]).replace(r"\{item\}", items)
                                      .replace(r"\{n\}", r"\d+") + r"\."))
                    for kind, templates in [*tickets._PROBLEMS.items(), ("none", tickets._NON_PROBLEMS)]
                    for t in templates]

        def kind_of(text):
            kinds = {kind for kind, pattern in patterns if pattern.search(text)}
            self.assertEqual(len(kinds), 1, text)
            return kinds.pop()

        for family in tickets.FAMILIES:
            for level in (1, 2):
                task = tasks.make_task(family, level)
                rows = [re.match(r"\[ticket \d+\] from ([^:]+): (.*)", line).groups() for line in task.context.splitlines()]
                if family == "tickets_all":
                    target = next(k for k, label in tickets._LABELS.items() if label in task.question)
                    count = sum(kind_of(text) == target for _, text in rows)
                else:
                    who = re.search(r"tickets from (.+?) report", task.question).group(1)
                    count = sum(name == who and kind_of(text) != "none" for name, text in rows)
                self.assertEqual(count, task.answer, (family, level))

    def test_same_seed_same_task_but_a_fresh_id(self):
        a, b = tickets.make_task(7, "tickets_all", 3), tickets.make_task(7, "tickets_all", 3)
        self.assertEqual((a.context, a.question, a.answer, a.grading), (b.context, b.question, b.answer, b.grading))
        self.assertNotEqual(a.task_id, b.task_id)
        self.assertEqual(tickets.make_task(7, "tickets_all", 3, "sel-001").task_id, "sel-001")

    def test_task_ids_never_hold_the_seed(self):
        for _ in range(50):
            seed = tasks.new_seed()
            task = tickets.make_task(seed, "tickets_all", 1)
            for spelling in (str(seed), f"{seed:x}", f"{seed:X}"):
                self.assertNotIn(spelling, task.task_id)
                self.assertNotIn(spelling, task.context)
                self.assertNotIn(spelling, task.question)

    def test_level_1_reproduces_the_20_step_run(self):
        """legacy_tasks(50, 99_999) is the first seed's held-out eval set, task for task."""
        try:
            from rlm_rl.archive.ticket_tasks import make_ticket_tasks
        except ImportError as exc:
            self.skipTest(f"the archived generator needs ART to import: {exc}")
        for n, seed in ((50, 99_999), (8, 1000), (8, 1019)):
            old, new = make_ticket_tasks(n, seed=seed), tickets.legacy_tasks(n, seed)
            for o, t in zip(old, new):
                self.assertEqual((o.family, o.question, o.context, o.answer), (t.family, t.question, t.context, t.answer))
        self.assertEqual(len({t.task_id for t in tickets.legacy_tasks(50, 99_999)}), 50)

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            tickets.make_task(1, "tickets_all", 6)
        with self.assertRaises(ValueError):
            tickets.make_task(1, "tickets_vip", 1)


class RegistryTest(unittest.TestCase):
    def test_families(self):
        self.assertEqual(tasks.max_levels(tickets.FAMILIES), {"tickets_all": 5, "tickets_customer": 5})
        self.assertEqual(set(tasks.versions(tickets.FAMILIES).values()), {tickets.GENERATOR_VERSION})
        with self.assertRaises(KeyError):
            tasks.generator("oolong")

    def test_allocate(self):
        self.assertEqual(tasks.allocate(8, {"tickets_all": 1, "tickets_customer": 1}),
                         {"tickets_all": 4, "tickets_customer": 4})
        self.assertEqual(tasks.allocate(8, {"a": 0.35, "b": 0.30, "c": 0.20, "d": 0.15}),
                         {"a": 3, "b": 2, "c": 2, "d": 1})
        self.assertEqual(sum(tasks.allocate(7, {"a": 1, "b": 1, "c": 1}).values()), 7)
        with self.assertRaises(ValueError):
            tasks.allocate(8, {"a": 0})

    def test_draw_uses_each_familys_level_and_fresh_seeds(self):
        drawn = tasks.draw(8, {"tickets_all": 1, "tickets_customer": 1}, {"tickets_all": 2, "tickets_customer": 1})
        self.assertEqual(sorted((t.family, t.level) for t in drawn),
                         [("tickets_all", 2)] * 4 + [("tickets_customer", 1)] * 4)
        self.assertEqual(len({t.context for t in drawn}), 8)
        self.assertEqual(len({t.task_id for t in drawn}), 8)


if __name__ == "__main__":
    unittest.main()
