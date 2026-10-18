"""aggregation-v4 (the step 12 pilot): every question needs more lines labelled than the root can read itself.

The pilot (checkpoint 20, prompt v4) found no sub-call gain on v3: the root filtered by the question's user id or
dates down to 48-165 lines and read them. Here every question's must-label scope, measured from the rendered text
after the best filter by user id or date, holds at least its floor (1,500 lines at level 1 up to 4,100 at level 5,
or 157K-431K characters for long lines). Reads the data root and skips without it.

    python3 -m unittest -v tests.test_aggregation_v4
"""

import collections
import datetime as dt
import hashlib
import json
import random
import re
import statistics
import unittest

from rlm_rl import data, reward, tasks
from rlm_rl.tasks import aggregation
from tests.test_aggregation_v3 import _all_pools, recount_v3
from tests.test_data_families import FamilyContract, parse_lines, seed_of

needs_data = unittest.skipUnless(data.available(), "no data root (rlm_rl/data.py)")
V4_TEMPLATES = {"label_count", "window_label_count", "labels_over", "users_with_label", "pairs_both",
                "pairs_same_top", "top_topics"}
LIST_CAPS = {"users_with_label": aggregation.LIST_BLIND_F1_MAX, "pairs_both": aggregation.LIST_BLIND_F1_MAX,
             "pairs_same_top": aggregation.SAME_TOP_BLIND_F1_MAX, "top_topics": aggregation.TOP_SHARE_MAX}


def _labelled(task, pools):
    """(date, user, label, text, characters of the whole line) per line, in the context's order, each line
    labelled by its pool."""
    labels = re.findall(r"^- (.+?): ", task.question, re.M)
    label_of = {}
    for name in labels:
        for text in pools[name]:
            label_of.setdefault(text, set()).add(name)
    lines = task.context.splitlines()[1:]
    rows = [(d, u, next(iter(label_of[t])), t, len(line)) for (d, u, t), line in zip(parse_lines(task.context), lines)
            if len(label_of.get(t, ())) == 1]
    assert len(rows) == len(task.context.splitlines()) - 1, "a line belongs to no listed label, or to two"
    return labels, rows


def _question_terms(task, labels):
    tail = task.question.rsplit("\n", 1)[1]
    quoted = [m for m in re.findall(r"'([^']+)'", tail) if m in labels]
    window = [dt.datetime.strptime(d, "%B %d, %Y").date() for d in re.findall(r"([A-Z][a-z]+ \d{1,2}, \d{4})", tail)]
    number = [int(a or b) for a, b in re.findall(r"(?:more than|exceeds|least) (\d+)|(\d+) or more", tail)]
    listed = sorted(set(re.findall(r"(?<!\d)\d{5}(?!\d)", tail)), key=int)
    return quoted, window, number, listed


def _first_settled(seq, decided):
    """Lines of `seq` labelled, in order, before decided(counts, lines left) holds (written apart from the generator's)."""
    counts = collections.Counter()
    for j in range(len(seq) + 1):
        if j == len(seq) or decided(counts, len(seq) - j):
            return j
        counts[seq[j][2]] += 1


def must_label(task, pools) -> dict:
    """The lines a question needs labelled, from its text alone: after the best filter by user id or date (`filtered`),
    and the stricter count of lines labelled in date order before the answer can't change (`strict`), with their
    characters and distinct texts."""
    labels, rows = _labelled(task, pools)
    quoted, window, number, listed = _question_terms(task, labels)
    template = task.grading["template"]
    by_user = collections.defaultdict(list)
    for row in rows:
        by_user[row[1]].append(row)
    if template in ("label_count", "window_label_count"):
        scope = [r for r in rows if not window or window[0] <= r[0] <= window[1]]
        filtered, strict = scope, scope
    elif template == "labels_over":
        t = number[0]
        j = _first_settled(rows, lambda c, left: all(c[n] > t or c[n] + left <= t for n in labels))
        filtered, strict = rows, rows[:j]
    elif template in ("users_with_label", "pairs_both"):
        needs = {quoted[0]: number[0]} if template == "users_with_label" else {quoted[0]: number[0], quoted[1]: number[0]}
        filtered = [r for u, rs in by_user.items() if len(rs) >= sum(needs.values()) for r in rs]
        strict = []
        for rs in by_user.values():
            j = _first_settled(rs, lambda c, left: all(c[n] >= t for n, t in needs.items())
                               or any(c[n] + left < t for n, t in needs.items()))
            strict += rs[:j]
    else:  # pairs_same_top, top_topics: the listed users' lines
        filtered, strict = [r for u in listed for r in by_user[u]], []
        for u in listed:
            def top_settled(c, left):
                two = sorted(c.values(), reverse=True)[:2] + [0, 0]
                return two[0] - two[1] > left
            strict += by_user[u][:_first_settled(by_user[u], top_settled)]
    return {"filtered": len(filtered), "strict": len(strict), "chars": sum(r[4] for r in strict),
            "distinct": len({r[3] for r in strict})}


def recount_v4(task, pools):
    if task.grading["template"] != "top_topics":
        return recount_v3(task, pools)
    labels, rows = _labelled(task, pools)
    listed = _question_terms(task, labels)[3]
    out = []
    for u in listed:
        c = collections.Counter(r[2] for r in rows if r[1] == u)
        out.append(f"{u}|{sorted(c, key=lambda n: (-c[n], n))[0]}")
    return sorted(out)


@needs_data
class AggregationV4Test(FamilyContract, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng, cls.specs, cls.sample = random.Random(61), [], []
        for family in aggregation.FAMILIES:
            for level in aggregation.LEVELS:
                for kind in aggregation.KINDS_V4:
                    for _ in range(2):
                        cls.specs.append((seed_of(rng), family, level, kind))
                        cls.sample.append(aggregation.make_task(*cls.specs[-1][:3], kind=kind))
        cls.pools = _all_pools()
        cls.scopes = [must_label(t, cls.pools) for t in cls.sample]

    def test_versions(self):
        self.assertEqual(aggregation.GENERATOR_VERSION, "aggregation-v4")
        self.assertEqual(tasks.versions(["aggregation"]), {"aggregation": "aggregation-v4"})
        self.assertIs(aggregation.KINDS, aggregation.KINDS_V4)
        self.assertEqual(aggregation.version_for("selection-v2-aggregation-L1-00"), aggregation.LEGACY_VERSION)
        for task_id in (None, "aggregation-L1-1a2b3c4d", "selection-v4-aggregation-L1-00"):
            self.assertEqual(aggregation.version_for(task_id), "aggregation-v4")

    def test_every_kind_family_and_level(self):
        for task in self.sample:
            g = task.grading
            with self.subTest(family=task.family, level=task.level, kind=g["kind"]):
                self.check_task(task, task.family, task.level)
                self.assertLessEqual(len(task.context), aggregation.MAX_CHARS)
                self.assertEqual((g["generator"], g["code_only"]), ("aggregation-v4", False))
                self.assertIn(g["template"], V4_TEMPLATES)
                self.assertEqual(len(re.findall(r"^- (.+?): ", task.question, re.M)), aggregation.LABELS_V3[task.level])

    def test_the_answer_is_what_the_context_says(self):
        seen = set()
        for task in self.sample:
            seen.add(task.grading["template"])
            with self.subTest(family=task.family, level=task.level, template=task.grading["template"]):
                self.assertEqual(recount_v4(task, self.pools), task.answer)
        self.assertEqual(seen, V4_TEMPLATES)

    def test_must_label_scope_exceeds_what_the_root_can_read(self):
        """The pilot's root read 48-165 lines it had filtered by id or date; ~1,000 short lines is its own limit."""
        for task, scope in zip(self.sample, self.scopes):
            g, level = task.grading, task.level
            with self.subTest(family=task.family, level=level, template=g["template"], source=g["source"]):
                self.assertEqual(scope["strict"], g["scope_lines"])
                self.assertGreaterEqual(scope["filtered"], scope["strict"])
                self.assertGreaterEqual(scope["strict"], g["scope_floor"])
                if g["source"] in ("ledgar", "huffpost") and g["scope_floor"] < aggregation.SCOPE_FLOOR_LINES[level]:
                    # long lines: the floor is in characters (and still several times the root's ~160K per episode)
                    self.assertGreaterEqual(scope["chars"], 0.85 * aggregation.SCOPE_FLOOR_CHARS[level])
                else:
                    self.assertEqual(g["scope_floor"], aggregation.SCOPE_FLOOR_LINES[level])
                # Texts repeat only after a label's pool runs out, so labelling each distinct text once saves little.
                self.assertGreaterEqual(scope["distinct"], 0.9 * g["scope_floor"])

    def test_floors_rise_with_the_level(self):
        floors = {level: aggregation.SCOPE_FLOOR_LINES[level] for level in aggregation.LEVELS}
        self.assertEqual(floors[1], 1500)
        self.assertGreaterEqual(floors[5], 4000)
        self.assertEqual(list(floors.values()), sorted(floors.values()))
        by_level = collections.defaultdict(list)
        for task, scope in zip(self.sample, self.scopes):
            if task.grading["source"] in ("clinc150", "arxiv-titles"):
                by_level[task.level].append(scope["strict"])
        for level, scopes in by_level.items():
            self.assertGreaterEqual(min(scopes), floors[level], level)

    def test_no_line_names_a_label_the_question_lists(self):
        for task in self.sample:
            labels = re.findall(r"^- (.+?): ", task.question, re.M)
            pattern = aggregation.label_pattern(labels)
            with self.subTest(source=task.grading["source"], level=task.level):
                self.assertFalse([t for _, _, t in parse_lines(task.context) if pattern.search(t)][:3])

    def test_metadata_guesses_cannot_score(self):
        for task in self.sample:
            g = task.grading
            with self.subTest(family=task.family, level=task.level, template=g["template"]):
                self.assertEqual(reward.blind_score(task).reward, 0.0)
                if g["type"] == "count":
                    self.assertGreaterEqual(abs(task.answer - g["blind"]), reward.COUNT_MAX_ERR)
                    self.assertEqual(g["blind"], round(g["scope_lines"] / aggregation.LABELS_V3[task.level]))
                elif g["template"] in LIST_CAPS:
                    self.assertLessEqual(g["blind_f1"], LIST_CAPS[g["template"]])
                    self.assertAlmostEqual(reward.set_f1(set(g["blind"]), set(task.answer)), g["blind_f1"], places=3)

    def test_thresholds_a_sample_cannot_settle(self):
        """labels_over's gap and each listed user's lead: big enough for a few labelling errors, small enough that
        labelling a sample of the lines doesn't settle them."""
        for task in self.sample:
            labels, rows = _labelled(task, self.pools)
            template = task.grading["template"]
            if template == "labels_over":
                counts = sorted(collections.Counter(r[2] for r in rows).values())
                t = _question_terms(task, labels)[2][0]
                gap = min(c for c in counts if c > t) - max(c for c in counts if c <= t)
                self.assertTrue(6 <= gap <= aggregation.LABELS_OVER_GAP_V4[1] * len(rows), gap)
            elif template in ("top_topics", "pairs_same_top"):
                for u in _question_terms(task, labels)[3]:
                    c = sorted(collections.Counter(r[2] for r in rows if r[1] == u).values(), reverse=True) + [0]
                    n = sum(c)
                    self.assertTrue(max(3, 0.02 * n) <= c[0] - c[1] <= 0.2 * n, (u, c))

    def test_training_draws_v4_kinds_and_sources_only(self):
        self.assertAlmostEqual(sum(aggregation.KINDS_V4.values()), 1.0)
        self.assertFalse(set(aggregation.KINDS_V4) & {"user_label_count", "timeline", "top_topic",
                                                       *aggregation.CODE_ONLY_KINDS})
        n, drawn = 6000, collections.Counter()
        for seed in range(n):
            rng = random.Random(aggregation._sub_seed(seed, "kind", aggregation.GENERATOR_VERSION))
            drawn[rng.choices(list(aggregation.KINDS_V4), list(aggregation.KINDS_V4.values()))[0]] += 1
        for kind, weight in aggregation.KINDS_V4.items():
            self.assertAlmostEqual(drawn[kind] / n, weight, delta=0.02, msg=kind)
        huff = {t for texts in data.huffpost_pools().values() for t in texts}
        train = {s.name for s in aggregation.TRAIN_SET_V4}
        for task in self.sample:
            texts = {t for _, _, t in parse_lines(task.context)}
            if task.family == "aggregation":
                self.assertIn(task.grading["source"], train)
                self.assertFalse(texts & huff)
            else:
                self.assertLessEqual(texts, huff)
        self.assertEqual({t.grading["source"] for t in self.sample if t.family == "aggregation"}, train)

    def test_deterministic_including_redraws(self):
        self.check_deterministic("aggregation", 1)
        redrawn = [(spec, t) for spec, t in zip(self.specs, self.sample) if t.grading["redraws"] > 0]
        self.assertTrue(redrawn)
        for (seed, family, level, kind), task in redrawn[:3]:
            again = aggregation.make_task(seed, family, level, task.task_id, kind=kind)
            self.assertEqual((again.context, again.question, again.answer, again.grading),
                             (task.context, task.question, task.answer, task.grading))

    def test_v3_is_still_built_the_same(self):
        """v3 stays reachable (version=), byte for byte: the pilot's task lists hold v3 seeds."""
        rng, pinned = random.Random(99), {}
        for family in aggregation.FAMILIES:
            for level in range(1, 6):
                for kind in list(aggregation.KINDS_V3) + [None]:
                    seed = rng.getrandbits(64)
                    if (family, level, kind) in _V3_PINNED:
                        pinned[(family, level, kind)] = seed
        for key, seed in pinned.items():
            family, level, kind = key
            t = aggregation.make_task(seed, family, level, f"t-{family}-{level}-{kind}", version="aggregation-v3",
                                      kind=kind)
            blob = json.dumps([t.question, t.context, t.answer, t.grading], sort_keys=True, default=str)
            self.assertEqual(hashlib.sha256(blob.encode()).hexdigest(), _V3_PINNED[key], key)

    def test_sizes(self):
        lines = collections.defaultdict(list)
        for task in self.sample:
            lines[task.level].append(len(task.context.splitlines()) - 1)
        self.assertEqual(sorted(statistics.median(v) for v in lines.values()),
                         [statistics.median(lines[level]) for level in sorted(lines)])


# v3 tasks as the v3 commit built them (09fee8f): digests of [question, context, answer, grading].
_V3_PINNED = {
    ("aggregation", 2, "pairs"): "b2f77604847c3ec196c0384909ad45a1651ee7b872312b30dec677791cb38380",
    ("aggregation", 5, None): "b835064a78f5b59a0d969e064df107820e528257894097cfa09dc859229475c8",
    ("aggregation_heldout", 4, "top_topic"): "bfc1d84c30e9396a7ce5fd63d94790959fc66f9358dbccc9e5a45b7da51f36e0",
}


if __name__ == "__main__":
    unittest.main()
