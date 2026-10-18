"""aggregation-v3 (step 12): dense kinds, OOLONG count credit, re-draws, label-free lines, and the frozen eval sets.

Since aggregation-v4 (tests/test_aggregation_v4.py) training draws v4; v3 stays reachable with version=, and these
tests ask for it that way.

The generator tests read the data root (rlm_rl/data.py) and skip without it; the count scorer and template tests
don't need it.

    python3 -m unittest -v tests.test_aggregation_v3
"""

import collections
import datetime as dt
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from rlm_rl import data, reward, tasks
from rlm_rl.tasks import aggregation
from tests.test_data_families import FamilyContract, parse_lines, seed_of

HAVE_DATA = data.available()
needs_data = unittest.skipUnless(HAVE_DATA, "no data root (rlm_rl/data.py)")
ROOT = Path(__file__).resolve().parents[1]
V3 = aggregation.V3_VERSION

V3_TEMPLATES = {"label_count", "window_label_count", "user_label_count", "user_window_label_count", "labels_over",
                "users_with_label", "pairs_both", "pairs_same_top", "timeline_first", "timeline_last", "top_topic",
                "second_topic"}


# ----------------------------------------------------------------------------- the "count" grading type


class CountScoreTest(unittest.TestCase):
    GRADING = {"type": "count", "blind": 40}  # gold 70: the blind guess is 30 off

    def test_oolong_credit(self):
        for answer, expected in ((70, 1.0), ("70 messages", 1.0), (69, 0.6), (72, 0.45), (61, 0.0601),
                                 (79, 0.0601), (60, 0.0), (80, 0.0), (40, 0.0), (500, 0.0)):
            s = reward.score_answer(answer, 70, self.GRADING)
            self.assertAlmostEqual(s.reward, expected, places=4, msg=answer)
            self.assertEqual(s.correct, answer in (70, "70 messages"))
        self.assertAlmostEqual(reward.score_answer(70.5, 70, self.GRADING).reward, round(0.8 * 0.75 ** 0.5, 4))
        # a whole-log count (aggregation-v4): a unit of error is 1% of the gold, so 13 off 1,300 is one unit
        big = {"type": "count", "blind": 900}
        self.assertAlmostEqual(reward.score_answer(1313, 1300, big).reward, 0.6, places=4)
        self.assertEqual(reward.score_answer(1170, 1300, big).reward, 0.0)  # 10 units off

    def test_credit_is_not_calibrated_against_the_blind_guess(self):
        """Unlike "int": a near blind guess doesn't shrink the credit (the generator re-draws such tasks instead)."""
        near = {"type": "count", "blind": 68}
        self.assertAlmostEqual(reward.score_answer(69, 70, near).reward, 0.6)
        self.assertEqual(reward.score_answer(69, 70, {"type": "int", "blind": 68}).reward, 0.4)  # "int" unchanged
        self.assertEqual(reward.score_answer(62, 70, near).components["near_blind"], 1.0)

    def test_unparsed_and_malformed(self):
        for answer in (None, "no idea", [70]):
            self.assertEqual(reward.score_answer(answer, 70, self.GRADING).reward, 0.0)
        for grading in ({"type": "count"}, {"type": "count", "blind": "lots"}):
            with self.assertRaises(ValueError):
                reward.score_answer(70, 70, grading)
        with self.assertRaises(ValueError):
            reward.score_answer(70, "many", self.GRADING)

    def test_a_blind_guess_ten_off_earns_nothing(self):
        """H1 for counts: a blind guess at COUNT_MAX_ERR or more from the gold scores 0, and 0.8 x 0.75^9 (one closer)
        is above BLIND_MAX, which is why the generator re-draws below COUNT_MAX_ERR."""
        self.assertEqual(reward.COUNT_MAX_ERR, 10)
        task = SimpleNamespace(answer=50, grading={"type": "count", "blind": 40})
        self.assertEqual(reward.blind_score(task).reward, 0.0)
        self.assertGreater(reward.NEAR_MISS_MAX * reward.COUNT_DECAY ** 9, aggregation.BLIND_MAX)
        self.assertLess(reward.NEAR_MISS_MAX * reward.COUNT_DECAY ** 10, aggregation.BLIND_MAX)


# ----------------------------------------------------------------------------- templates (no data needed)


def _words(text):
    return re.findall(r"[a-z0-9']+", re.sub(r"\{\w+\}", " ", text.lower()))


def _ngrams(text, n=6):
    w = _words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


class TemplateTest(unittest.TestCase):
    def test_every_v3_template_exists_for_both_families(self):
        for templates in (aggregation.TRAIN_TEMPLATES, aggregation.HELDOUT_TEMPLATES):
            self.assertLessEqual(V3_TEMPLATES, set(templates))

    def test_heldout_templates_share_no_6gram_with_training(self):
        """H7, as scripts/data_check.py measures it."""
        train = set().union(*(_ngrams(t) for ts in aggregation.TRAIN_TEMPLATES.values() for t in ts))
        held = set().union(*(_ngrams(t) for ts in aggregation.HELDOUT_TEMPLATES.values() for t in ts))
        self.assertEqual(held & train, set())

    def test_training_draws_no_code_only_kind(self):
        self.assertFalse(set(aggregation.KINDS_V3) & set(aggregation.CODE_ONLY_KINDS))
        self.assertAlmostEqual(sum(aggregation.KINDS_V3.values()), 1.0)
        # The kind comes from its own sub-seed, so the mix is KINDS' weights exactly, whatever the re-draws.
        n, drawn = 6000, collections.Counter()
        for seed in range(n):
            rng = random.Random(aggregation._sub_seed(seed, "kind"))
            drawn[rng.choices(list(aggregation.KINDS_V3), list(aggregation.KINDS_V3.values()))[0]] += 1
        for kind, weight in aggregation.KINDS_V3.items():
            self.assertAlmostEqual(drawn[kind] / n, weight, delta=0.02, msg=kind)

    def test_frozen_eval_sets_keep_the_v2_generator(self):
        for name in ("selection-v1", "selection-v2", "selection-v3-extra", "heldout-test-v1"):
            self.assertEqual(aggregation.version_for(f"{name}-aggregation-L1-00"), aggregation.LEGACY_VERSION)
        for task_id in (None, "aggregation-L1-1a2b3c4d", "selection-v4-aggregation-L1-00", "selection-v10-x",
                        "selection-v3-aggregation-L1-00"):
            self.assertEqual(aggregation.version_for(task_id), aggregation.GENERATOR_VERSION, task_id)
        self.assertEqual(aggregation.V3_VERSION, "aggregation-v3")

    def test_label_pattern(self):
        p = aggregation.label_pattern(["credit_cards", "Taxes", "FOOD & DRINK", "news", "Base Salary", "work"])
        for text in ("my Credit-Card is lost", "withholding tax applies", "food and drink", "Food & Drink tips",
                     "read the news", "the base salaries", "how does this work"):
            self.assertIsNotNone(p.search(text), text)
        for text in ("a new phone", "my credit score", "network issues", "homework", "taxonomy"):
            self.assertIsNone(p.search(text), text)


# ----------------------------------------------------------------------------- the generator


def recount_v3(task, pools):
    """A v3 task's answer, recomputed from its rendered text and question alone, labelling each line by its pool."""
    rows = parse_lines(task.context)
    labels = re.findall(r"^- (.+?): ", task.question, re.M)
    label_of = {}
    for name in labels:
        for text in pools[name]:
            label_of.setdefault(text, set()).add(name)
    labelled = [(d, u, next(iter(label_of[t]))) for d, u, t in rows if len(label_of.get(t, ())) == 1]
    assert len(labelled) == len(rows), "a line belongs to no listed label, or to two"
    tail = task.question.rsplit("\n", 1)[1]  # the question proper, after the topic list
    quoted = [m for m in re.findall(r"'([^']+)'", tail) if m in labels]
    user = (re.search(r"(?:user|account) (\d{5})", tail) or [None, None])[1]
    window = [dt.datetime.strptime(d, "%B %d, %Y").date() for d in re.findall(r"([A-Z][a-z]+ \d{1,2}, \d{4})", tail)]
    number = [int(a or b) for a, b in re.findall(r"(?:more than|exceeds|least) (\d+)|(\d+) or more", tail)]
    template = task.grading["template"]

    def in_window(d):
        return not window or window[0] <= d <= window[1]

    def per_user(name):
        out = collections.Counter()
        for _, u, lab in labelled:
            out[u] += lab == name
        return out

    def ranked(u):
        c = collections.Counter(lab for _, uu, lab in labelled if uu == u)
        return sorted(c, key=lambda name: (-c[name], name))

    if template in ("label_count", "window_label_count"):
        return sum(lab == quoted[0] and in_window(d) for d, _, lab in labelled)
    if template in ("user_label_count", "user_window_label_count"):
        return sum(u == user and lab == quoted[0] and in_window(d) for d, u, lab in labelled)
    if template == "labels_over":
        return sorted(n for n in labels if sum(lab == n for _, _, lab in labelled) > number[0])
    if template == "users_with_label":
        return sorted(u for u, c in per_user(quoted[0]).items() if c >= number[0])
    if template == "pairs_both":
        x, y = per_user(quoted[0]), per_user(quoted[1])
        members = sorted((u for u in x if x[u] >= number[0] and y[u] >= number[0]), key=int)
        return sorted(f"{a}|{b}" for i, a in enumerate(members) for b in members[i + 1:])
    if template == "pairs_same_top":
        listed = sorted(set(re.findall(r"(?<!\d)\d{5}(?!\d)", tail)), key=int)
        top = {u: ranked(u)[0] for u in listed}
        return sorted(f"{a}|{b}" for i, a in enumerate(listed) for b in listed[i + 1:] if top[a] == top[b])
    if template in ("timeline_first", "timeline_last"):
        days = [d for d, u, lab in labelled if u == user and lab == quoted[0]]
        return (min(days) if template == "timeline_first" else max(days)).isoformat()
    if template in ("top_topic", "second_topic"):
        return ranked(user)[0 if template == "top_topic" else 1]
    raise AssertionError(template)


def _all_pools():
    return {**data.clinc_domain_pools(), **data.massive_scenario_pools(), **data.ledgar_pools(),
            **data.arxiv_title_pools(), **data.huffpost_pools()}


def _canon_digest(task) -> str:
    """The task with every 5-digit number (the user ids) renamed by first appearance in the context."""
    ids = re.compile(r"(?<!\d)\d{5}(?!\d)")
    mapping = {}
    for m in ids.finditer(task.context):
        mapping.setdefault(m.group(), f"U{len(mapping):03d}")

    def sub(text):
        return ids.sub(lambda m: mapping.get(m.group(), m.group()), text)

    answer = sorted(sub(str(a)) for a in task.answer) if isinstance(task.answer, list) else sub(str(task.answer))
    blob = json.dumps([task.family, task.level, sub(task.question), sub(task.context), answer], default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _raw_digest(task) -> str:
    blob = json.dumps([task.task_id, task.family, task.level, task.question, task.context, task.answer, task.grading],
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


# Frozen eval-set tasks: (set, family, level, index) -> (digest of the pre-v3 code's task with the users renamed,
# digest of today's task). The seeds are random.Random(777)'s draws in this order (as in the step 12 check).
_FROZEN_ORDER = [(name, family, level, i)
                 for name, family in (("selection-v2", "aggregation"), ("selection-v3-extra", "aggregation"),
                                      ("selection-v1", "aggregation"), ("heldout-test-v1", "aggregation_heldout"))
                 for level in range(1, 6) for i in range(3)]
_FROZEN_PINNED = {
    ("selection-v2", "aggregation", 1, 0): ("b5a525ede88002d5c14220863e17f12d2f2373b5c0bf6134d6bc63d062408af8",
                                            "48d9a6338ee7c6ef480148a57df1ea5b91f45285b58e9be68e76fdeb76353266"),
    ("selection-v3-extra", "aggregation", 3, 1): ("348ea03151c08b7dc2de30382e6820d3e6e8475b69a38f370657b2bb27c8f186",
                                                  "b84f41341dcc39f42c74936cc5b43df189c69523c90d2118afd0dc5a81c44139"),
    ("selection-v1", "aggregation", 5, 2): ("5f7badcf026c2872f7ed1c436f984fb45cce69e95e9d9c33bab92c0fd72ea560",
                                            "20f7a0296db841294bb84fc5f5bd41103bba709fa1dac58de4bdf72a798ca137"),
    ("heldout-test-v1", "aggregation_heldout", 2, 1): (
        "e96b192ec86bd293ba5041e1c8fc0641e636a09e4fc8dbe754ff2488748f58bc",
        "bed7df8a43b7825ddb5b4129bc82d7638cd68f52b07d17f31050ce9cd2c865d8"),
}

_SUBPROCESS = """
import hashlib, json, sys
sys.path.insert(0, {root!r})
from rlm_rl import tasks
out = {{}}
for family, level, task_id in (("aggregation", 2, "t"), ("aggregation", 3, "selection-v2-aggregation-L3-00")):
    t = tasks.make_task(family, level, 4242, task_id)
    blob = json.dumps([t.question, t.context, t.answer, t.grading], sort_keys=True, default=str)
    out[task_id] = hashlib.sha256(blob.encode()).hexdigest()
print(json.dumps(out))
"""


@needs_data
class AggregationV3Test(FamilyContract, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng, cls.specs, cls.sample = random.Random(31), [], []
        for family in aggregation.FAMILIES:
            for level in aggregation.LEVELS:
                for kind in aggregation.KINDS_V3:
                    for _ in range(2):
                        cls.specs.append((seed_of(rng), family, level, kind))
                        cls.sample.append(aggregation.make_task(*cls.specs[-1][:3], kind=kind, version=V3))

    def test_every_kind_family_and_level(self):
        for task in self.sample:
            g = task.grading
            with self.subTest(family=task.family, level=task.level, kind=g["kind"]):
                self.check_task(task, task.family, task.level)
                lo, hi = aggregation.LEVELS[task.level]
                self.assertTrue(lo * 0.95 <= len(task.context) <= hi * 1.05, len(task.context))
                self.assertEqual((g["generator"], g["code_only"]), ("aggregation-v3", False))
                self.assertIn(g["kind"], aggregation.KINDS_V3)
                self.assertIn(g["template"], V3_TEMPLATES)
                labels = re.findall(r"^- (.+?): ", task.question, re.M)
                self.assertEqual(len(labels), aggregation.LABELS_V3[task.level])

    def test_the_answer_is_what_the_context_says(self):
        pools, seen = _all_pools(), set()
        for task in self.sample:
            seen.add(task.grading["template"])
            with self.subTest(family=task.family, level=task.level, template=task.grading["template"]):
                self.assertEqual(recount_v3(task, pools), task.answer)
        self.assertEqual(seen, V3_TEMPLATES)

    def test_no_line_names_a_label_the_question_lists(self):
        """Lexically, by the generator's own rule and by a plain word match."""
        for task in self.sample:
            labels = re.findall(r"^- (.+?): ", task.question, re.M)
            pattern = aggregation.label_pattern(labels)
            plain = [re.compile(r"\b" + re.escape(name.lower().replace("_", " ")) + r"s?\b") for name in labels]
            texts = [t for _, _, t in parse_lines(task.context)]
            with self.subTest(source=task.grading["source"], level=task.level):
                self.assertFalse([t for t in texts if pattern.search(t)][:3])
                self.assertFalse([t for t in texts if any(p.search(t.lower()) for p in plain)][:3])

    def test_metadata_guesses_cannot_score(self):
        for task in self.sample:
            g = task.grading
            with self.subTest(family=task.family, level=task.level, template=g["template"]):
                self.assertEqual(reward.blind_score(task).reward, 0.0)
                if g["type"] == "count":
                    self.assertGreaterEqual(abs(task.answer - g["blind"]), reward.COUNT_MAX_ERR)
                    self.assertGreaterEqual(task.answer, reward.COUNT_MAX_ERR)  # guessing 0 earns nothing
                    self.assertEqual(g["blind"], round(g["scope_lines"] / aggregation.LABELS_V3[task.level]))
                elif g["type"] == "list" and "blind_f1" in g:
                    self.assertLessEqual(g["blind_f1"], aggregation.LIST_BLIND_F1_MAX)
                    self.assertAlmostEqual(reward.set_f1(set(g["blind"]), set(task.answer)), g["blind_f1"], places=3)
                elif g["template"].startswith("timeline"):
                    self.assertNotEqual(g["blind"], task.answer)

    def test_no_fixed_shortcut_earns_much(self):
        """Strategies that only use what code counts: one share of the count's scope for every count task, one place
        among the user's dates for every timeline task. In v3's first draft these earned up to ~0.18 (level 1 counts)
        and ~0.28 (timelines, "the user's 4th date"); SCOPE_LINES, topic periods and TIMELINE_SLACK brought them
        to ~0.07-0.10 and ~0.03-0.09."""
        rng = random.Random(41)
        counts, places = [], collections.Counter()
        for level in aggregation.LEVELS:
            for kind in aggregation.COUNT_KINDS:
                counts += [aggregation.make_task(seed_of(rng), "aggregation", level, kind=kind, version=V3)
                           for _ in range(12)]
            for _ in range(12):
                task = aggregation.make_task(seed_of(rng), "aggregation", level, kind="timeline", version=V3)
                user = re.search(r"user (\d{5})", task.question)[1]
                dates = sorted({d.isoformat() for d, u, _ in parse_lines(task.context) if u == user})
                seq = dates if task.grading["template"] == "timeline_first" else dates[::-1]
                places[seq.index(task.answer)] += 1
        best_share = max(statistics.fmean(
            reward.score_answer(round(c / 40 * t.grading["scope_lines"]), t.answer, t.grading).reward for t in counts)
            for c in range(1, 41))
        self.assertLess(best_share, 0.15)
        self.assertLess(places.most_common(1)[0][1] / sum(places.values()), 0.15)

    def test_count_scopes_fit_the_level(self):
        for task in self.sample:
            g = task.grading
            if g["type"] == "count":
                lo, hi = aggregation.SCOPE_LINES[task.level]
                with self.subTest(level=task.level, template=g["template"]):
                    self.assertLessEqual(g["scope_lines"], hi)
                    if g["template"] != "label_count":  # a whole log may be shorter than the floor
                        self.assertGreaterEqual(g["scope_lines"], lo)
                    else:
                        self.assertEqual(g["scope_lines"], len(task.context.splitlines()) - 1)

    def test_deterministic_per_seed_including_redraws(self):
        self.check_deterministic("aggregation", 2)
        redrawn = [(spec, t) for spec, t in zip(self.specs, self.sample) if t.grading["redraws"] > 0]
        self.assertTrue(redrawn)
        for (seed, family, level, kind), task in redrawn[:4]:
            again = aggregation.make_task(seed, family, level, task.task_id, kind=kind, version=V3)
            self.assertEqual((again.question, again.context, again.answer, again.grading),
                             (task.question, task.context, task.answer, task.grading))

    def test_same_tasks_in_every_process(self):
        """Python's string hash seed changes per process; the tasks must not (v2's user ids once did)."""
        code = _SUBPROCESS.format(root=str(ROOT))
        outs = []
        for hash_seed in ("1", "2"):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed)
            proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            outs.append(json.loads(proc.stdout.strip().splitlines()[-1]))
        self.assertEqual(outs[0], outs[1])

    def test_frozen_eval_sets_build_the_same_tasks(self):
        """selection-v1/v2/v3 and heldout-test-v1/v2 keep the v2 generator: the same tasks as before v3 (up to the
        user ids, which used to change per process), and byte for byte the same from now on."""
        rng = random.Random(777)
        for key in _FROZEN_ORDER:
            seed = rng.getrandbits(64)
            if key not in _FROZEN_PINNED:
                continue
            name, family, level, i = key
            task = tasks.make_task(family, level, seed, f"{name}-{family}-L{level}-{i:02d}")
            with self.subTest(key=key):
                self.assertNotIn("generator", task.grading)  # v2
                self.assertEqual((_canon_digest(task), _raw_digest(task)), _FROZEN_PINNED[key])

    def test_training_reads_only_training_sources(self):
        huff = {t for texts in data.huffpost_pools().values() for t in texts}
        train_names = {s.name for s in aggregation.TRAIN_SET}
        for task in self.sample:
            texts = {t for _, _, t in parse_lines(task.context)}
            with self.subTest(family=task.family, level=task.level):
                if task.family == "aggregation":
                    self.assertIn(task.grading["source"], train_names)
                    self.assertFalse(texts & huff)
                else:
                    self.assertEqual(task.grading["source"], "huffpost")
                    self.assertLessEqual(texts, huff)

    def test_every_level_draws_several_sources(self):
        """The label-name filter must not leave a level to one source (LABELS_V3 keeps level 5 at 7 labels)."""
        by_level = collections.defaultdict(set)
        for task in self.sample:
            if task.family == "aggregation":
                by_level[task.level].add(task.grading["source"])
        for level, sources in by_level.items():
            self.assertGreaterEqual(len(sources), 3, (level, sources))

    def test_v3_draws_its_own_kinds(self):
        rng = random.Random(51)
        drawn = [aggregation.make_task(seed_of(rng), "aggregation", rng.choice([1, 3, 5]), version=V3) for _ in range(16)]
        self.assertTrue(all(t.grading["generator"] == "aggregation-v3" for t in drawn))
        self.assertGreaterEqual(len({t.grading["kind"] for t in drawn}), 4)
        with self.assertRaises(ValueError):
            aggregation.make_task(1, "aggregation", 1, kind="most_active_user", version=V3)
        with self.assertRaises(ValueError):
            aggregation.make_task(1, "aggregation", 1, version=aggregation.LEGACY_VERSION, kind="pairs")


if __name__ == "__main__":
    unittest.main()
