"""The research family (step 12 multistep plan, step 1) and its free-text grading type.

MuSiQue multi-hop and QAMPARI list questions over a corpus far bigger than the root can print. The grading tests need
nothing outside the repo. The generator tests read the derived pools (scripts/build_research_data.py, under the data
root's derived/research/) and skip without them.

    python3 -m unittest -v tests.test_research
"""

import math
import random
import re
import statistics
import unittest
from types import SimpleNamespace
from unittest import mock

from rlm_rl import data, reward, tasks
from rlm_rl.reward import blind_score, normalize_text, score_answer
from rlm_rl.tasks import research

HAVE_POOLS = data.available() and all(data.derived(p).exists() for p in data.RESEARCH_POOLS.values())
needs_pools = unittest.skipUnless(HAVE_POOLS, "no research pools (scripts/build_research_data.py)")

# What tasks/__init__.py's GENERATORS gets for this family.
RESEARCH_GENERATORS = {"research": "research", "research_select": "research", "research_test": "research"}
DOC_RE = re.compile(r"^# ===== doc (\d+): (.*)$", re.M)


def docs_of(context: str) -> list[tuple[int, str, str]]:
    parts = DOC_RE.split(context)
    return [(int(parts[i]), parts[i + 1], parts[i + 2].strip()) for i in range(1, len(parts), 3)]


# ----------------------------------------------------------------------------- grading


class TextGradingTest(unittest.TestCase):
    G = {"type": "text", "aliases": ["Bombardier"], "blind": None, "ignore": ["company", "train", "what", "built"]}

    def test_exact_match_against_the_answer_or_an_alias(self):
        for answer in ("Bombardier Inc.", "bombardier inc", ' "Bombardier" ', "BOMBARDIER", "The Bombardier Inc"):
            s = score_answer(answer, "Bombardier Inc.", self.G)
            self.assertEqual((s.reward, s.correct), (1.0, True), answer)

    def test_normalization(self):
        cases = {"The Beatles": "beatles", "Rashād": "rashad", "Saint-Denis": "saint denis", "U.S.": "us",
                 1961: "1961", 1961.0: "1961", "AC/DC": "ac dc", "  Miquette   Giraudy ": "miquette giraudy"}
        for value, expected in cases.items():
            self.assertEqual(normalize_text(value), expected, value)
        grading = {"type": "text", "aliases": [], "blind": None}
        self.assertTrue(score_answer(1961, "1961", grading).correct)
        self.assertTrue(score_answer("Krivacevic", "Krivačević", grading).correct)

    def test_partial_credit_is_0_8_token_f1_against_the_best_alias(self):
        s = score_answer("Bombardier Transportation", "Bombardier Inc.", self.G)  # vs alias "bombardier": F1 2/3
        self.assertAlmostEqual(s.reward, round(0.8 * 2 / 3, 4))
        self.assertFalse(s.correct)
        grading = {"type": "text", "aliases": [], "blind": None}
        self.assertAlmostEqual(score_answer("Giraudy", "Miquette Giraudy", grading).reward, round(0.8 * 2 / 3, 4))
        self.assertEqual(score_answer("Steve Hillage", "Miquette Giraudy", grading).reward, 0.0)

    def test_the_questions_own_words_earn_nothing(self):
        grading = {"type": "text", "aliases": [], "blind": None,
                   "ignore": sorted(set(normalize_text("Which university did the Green performer attend?").split()))}
        self.assertEqual(score_answer("Yale University", "Harvard University", grading).reward, 0.0)
        self.assertEqual(score_answer("the university", "Harvard University", grading).reward, 0.0)
        self.assertEqual(score_answer("Which university did the Green performer attend", "Harvard University",
                                      grading).reward, 0.0)
        self.assertEqual(score_answer("Harvard", "Harvard University", grading).reward, 0.8)
        # a gold made only of question words is still scored on its own words
        self.assertEqual(score_answer("Green", "Green", grading).reward, 1.0)

    def test_empty_or_generic_answers_score_0(self):
        for answer in (None, "", "   ", "unknown", "Unknown.", "N/A", "I don't know", "none", "not found",
                       ["Bombardier"], True, {"a": 1}, "the", "?!"):
            s = score_answer(answer, "Bombardier Inc.", self.G)
            self.assertEqual((s.reward, s.correct), (0.0, False), answer)
        # unless the gold itself says so
        self.assertTrue(score_answer("None", "None", {"type": "text", "aliases": [], "blind": None}).correct)

    def test_hedging_pays_less_than_answering(self):
        hedge = score_answer("Bombardier or Alstom or Siemens", "Bombardier Inc.", self.G).reward
        self.assertLess(hedge, 0.5)
        self.assertGreater(hedge, 0.0)

    def test_grading_must_be_well_formed(self):
        for grading, gold in (({"type": "text", "aliases": [], "blind": None}, ""),
                              ({"type": "text", "aliases": [], "blind": "Bombardier"}, "Bombardier"),
                              ({"type": "text", "aliases": "Bombardier", "blind": None}, "Bombardier")):
            with self.assertRaises(ValueError, msg=grading):
                score_answer("Bombardier", gold, grading)

    def test_no_blind_guess_and_h1(self):
        task = SimpleNamespace(answer="Bombardier Inc.", grading=self.G)
        s = blind_score(task)
        self.assertEqual((s.reward, s.correct), (0.0, False))
        self.assertIn("text", reward.ANSWER_TYPES)
        self.assertIn("text", reward.settings()["answer_types"])

    def test_one_date_in_any_format_is_exact(self):
        grading = {"type": "text", "aliases": [], "blind": None}
        for gold in ("29 March 2004", "March 29, 2004", "2004-03-29"):
            for answer in ("29 March 2004", "March 29, 2004", "2004-03-29", "29th of March, 2004", "Mar 29 2004",
                           "29 march 2004."):
                with self.subTest(gold=gold, answer=answer):
                    self.assertTrue(score_answer(answer, gold, grading).correct)
        self.assertTrue(score_answer("Sept 2001", "September 2001", grading).correct)
        for answer in ("30 March 2004", "29 March 2005", "March 2004", "2004"):  # another date, or less of it
            s = score_answer(answer, "29 March 2004", grading)
            self.assertFalse(s.correct, answer)
            self.assertLess(s.reward, 1.0)
        self.assertGreater(score_answer("March 2004", "29 March 2004", grading).reward, 0.0)  # still near

    def test_a_number_matches_it_with_its_unit(self):
        grading = {"type": "text", "aliases": [], "blind": None}
        for answer, gold in (("53", "53 member states"), ("53 member states", "53"), ("1,000", "1000 people"),
                             ("12%", "12"), ("3.5 km", "3.5"), (53, "53 member states")):
            self.assertTrue(score_answer(answer, gold, grading).correct, (answer, gold))
        for answer, gold in (("54", "53 member states"), ("53 countries", "53 member states"),
                             ("3", "3 Doors Down"), ("50", "50 Cent"), ("53 or 54", "53")):
            self.assertFalse(score_answer(answer, gold, grading).correct, (answer, gold))

    def test_a_countrys_common_and_official_names_match(self):
        grading = {"type": "text", "aliases": [], "blind": None}
        for answer, gold in (("Italy", "Italian Republic"), ("the Italian Republic", "Italy"),
                             ("USA", "United States"), ("U.S.", "United States of America"),
                             ("UK", "United Kingdom"), ("People's Republic of China", "China"),
                             ("Burma", "Myanmar"), ("Czechia", "Czech Republic"), ("Russian Federation", "Russia")):
            self.assertTrue(score_answer(answer, gold, grading).correct, (answer, gold))
        for answer, gold in (("England", "United Kingdom"), ("France", "Italy"), ("Italian", "Italy"),
                             ("South Korea", "North Korea"), ("Rome", "Italian Republic")):
            self.assertFalse(score_answer(answer, gold, grading).correct, (answer, gold))

    def test_equivalent_forms_keep_generic_answers_at_0(self):
        for gold in ("Italian Republic", "29 March 2004", "53 member states"):
            for answer in ("unknown", "", None, "N/A", "none", ["Italy"]):
                self.assertEqual(score_answer(answer, gold, {"type": "text", "aliases": [], "blind": None}).reward,
                                 0.0, (answer, gold))

    def test_episode_scoring_uses_only_a_valid_submit(self):
        task = SimpleNamespace(answer="Bombardier Inc.", grading=self.G)
        self.assertEqual(reward.score(task, SimpleNamespace(submitted=True, answer="Bombardier",
                                                            answer_valid=True)).reward, 1.0)
        self.assertEqual(reward.score(task, SimpleNamespace(submitted=False, answer="Bombardier",
                                                            answer_valid=True)).reward, 0.0)


# ----------------------------------------------------------------------------- the family


@needs_pools
class ResearchFamilyTest(unittest.TestCase):
    N = 3  # tasks per family and level, plus 2N more training tasks for the statistical checks
    LEVELS = range(1, research.MAX_LEVEL + 1)
    # Keyword search on the question's content words: the median number of distractor documents it hits, by level
    # (research-v3 measured 149-183 at level 1 up to 1,086-2,003 at level 5), and the fewest in any one task.
    MIN_HITS = {1: 90, 2: 140, 3: 250, 4: 350, 5: 600}
    MIN_TASK_HITS = 40

    @classmethod
    def setUpClass(cls):
        cls.built = {}
        for family in research.FAMILIES:
            for level in cls.LEVELS:
                cls.built[family, level] = [research.build(1_000 * level + i, family, level) for i in range(cls.N)]
        # more training tasks for the statistical checks
        cls.train = {level: cls.built["research", level] +
                     [research.build(50_000 + 1_000 * level + i, "research", level) for i in range(2 * cls.N)]
                     for level in cls.LEVELS}

    def all_built(self):
        for (family, level), pairs in self.built.items():
            for task, info in pairs:
                yield family, level, task, info

    def test_contract(self):
        for family, level, task, info in self.all_built():
            with self.subTest(family=family, level=level, task=task.task_id):
                reward.check_grading(task.grading, task.answer)
                self.assertEqual((task.family, task.level), (family, level))
                self.assertEqual(score_answer(task.answer, task.answer, task.grading).reward, 1.0)
                blind = blind_score(task)
                self.assertLessEqual(blind.reward, float(blind.correct))  # H1
                self.assertEqual(blind.reward, 0.0)
                self.assertEqual(task.grading["kind"], info["kind"])
                self.assertFalse(task.grading["code_only"])

    def test_documents_are_well_formed(self):
        for family, level, task, info in self.all_built():
            docs = docs_of(task.context)
            with self.subTest(family=family, level=level, task=task.task_id):
                self.assertEqual([n for n, _, _ in docs], list(range(1, len(docs) + 1)))
                self.assertEqual(len(docs), task.grading["docs"])
                self.assertTrue(task.context.startswith("# ===== doc 1: "))
                self.assertTrue(all(title and text for _, title, text in docs))
                self.assertTrue(info["gold_docs"])
                if info["kind"] == "list":
                    titles = [t for _, t, _ in docs]
                    self.assertEqual(len({t.casefold() for t in titles}), len(titles))  # a title names one document
                    self.assertEqual(sorted(titles[n - 1] for n in info["gold_docs"]), task.answer)

    def test_contexts_hit_their_level_sizes(self):
        for family, level, task, info in self.all_built():
            lo, hi = research.LEVELS[level]
            with self.subTest(family=family, level=level):
                self.assertGreaterEqual(len(task.context), lo)
                self.assertLessEqual(len(task.context), hi)
        for level, pairs in self.train.items():
            lo, hi = research.LEVELS[level]
            self.assertTrue(all(lo <= len(t.context) <= hi for t, _ in pairs))

    def test_hops_and_answer_counts_by_level(self):
        for level, pairs in self.train.items():
            for task, info in pairs:
                with self.subTest(level=level, task=task.task_id):
                    if info["kind"] == "multihop":
                        self.assertIn(info["hops"], research.HOPS[level])
                        self.assertEqual(task.grading["hops"], info["hops"])
                    else:
                        lo, hi = research.LIST_ANSWERS[level]
                        self.assertTrue(lo <= len(task.answer) <= hi, len(task.answer))
                        self.assertGreaterEqual(info["negatives"],
                                                min(research.NEG_CAP[level],
                                                    math.ceil(research.NEG_RATIO[level] * len(task.answer))))

    def test_about_30_percent_list_questions(self):
        share = statistics.fmean(research._kind(random.Random(seed)) == "list" for seed in range(4_000))
        self.assertAlmostEqual(share, research.LIST_SHARE, delta=0.025)
        kinds = [info["kind"] for pairs in self.train.values() for _, info in pairs]
        self.assertTrue({"list", "multihop"} <= set(kinds))

    def test_deterministic_per_seed(self):
        for family in research.FAMILIES:
            a = research.make_task(12345, family, 2, task_id="t")
            b = research.make_task(12345, family, 2, task_id="t")
            c = research.make_task(12346, family, 2, task_id="t")
            self.assertEqual((a.question, a.context, a.answer, a.grading), (b.question, b.context, b.answer, b.grading))
            self.assertNotEqual(a.context, c.context)
            self.assertNotIn("12345", research.make_task(12345, family, 2).task_id)

    def test_keyword_search_hits_many_distractors(self):
        """A keyword search on the question's content words matches many documents that aren't gold, at every level,
        and ranking documents by how many of those words they hold almost never puts exactly the gold on top."""
        for level, pairs in self.train.items():
            hits, top_solves = [], []
            for task, info in pairs:
                words = research.content_words(info["question"])
                docs = docs_of(task.context)
                gold = set(info["gold_docs"])
                pats = [re.compile(rf"(?<![^\W_]){re.escape(w)}(?![^\W_])", re.I) for w in words]
                score = {n: sum(bool(p.search(f"{title} {text}")) for p in pats) for n, title, text in docs}
                hits.append(sum(1 for n in score if score[n] and n not in gold))
                top = set(sorted(score, key=lambda n: (-score[n], n))[:len(gold)])
                top_solves.append(top == gold)
            with self.subTest(level=level, hits=sorted(hits)):
                self.assertGreaterEqual(statistics.median(hits), self.MIN_HITS[level])
                self.assertGreaterEqual(min(hits), self.MIN_TASK_HITS)
                self.assertLessEqual(statistics.fmean(top_solves), 0.2)

    def test_no_level_can_be_read_by_printing(self):
        """Research-v3: the pilot read level 1 (60-90K) through grep and a few prints. The smallest context is now
        bigger than everything the root can print in an episode (20 turns x the REPL's output cap)."""
        from rlm_rl import bridge

        self.assertGreater(research.LEVELS[1][0], 20 * bridge.OUTPUT_MAX_CHARS)
        self.assertEqual(research.HOPS[1], (2,))
        self.assertTrue(all(2 not in research.HOPS[level] for level in self.LEVELS if level > 1))
        lows = [research.LEVELS[level][0] for level in self.LEVELS]
        self.assertEqual(lows, sorted(lows))
        self.assertGreaterEqual(research.LEVELS[research.MAX_LEVEL][1], 1_200_000)

    def test_list_answers_name_the_entity_and_padding_never_does(self):
        """Gold documents can be checked by reading (each names the question's entity). Apart from them, only the hard
        negatives do, so a search for the entity can't tell the answers from those."""
        for level, pairs in self.train.items():
            for task, info in pairs:
                if info["kind"] != "list":
                    continue
                q = next(q for q in research._qampari("train")["questions"] if q["q"] == info["question"])
                entity = research.entity_re(q["entities"])
                naming = {n for n, title, text in docs_of(task.context) if entity.search(f"{title} {text}")}
                with self.subTest(level=level, question=info["question"]):
                    self.assertLessEqual(set(info["gold_docs"]), naming)
                    self.assertEqual(len(naming), len(info["gold_docs"]) + info["negatives"])
                    # that search is the blind guess, so it earns nothing unless it is exactly right
                    grep = blind_score(task)
                    self.assertEqual(grep.reward, 1.0 if not info["negatives"] else 0.0)

    def test_nothing_the_root_sees_holds_ids_gold_flags_or_answers(self):
        for family, level, task, info in self.all_built():
            with self.subTest(family=family, level=level, task=task.task_id):
                for marker in ("hop__", "__wikidata", '"is_supporting"', '"answerable"', "paragraph_support_idx"):
                    self.assertNotIn(marker, task.context)
                    self.assertNotIn(marker, task.question)
                self.assertFalse({"id", "qid", "subq", "gold", "gold_docs", "entities"} & set(task.grading))
                if info["kind"] == "multihop":
                    gold = normalize_text(task.answer)
                    self.assertNotRegex(normalize_text(info["question"]), rf"(?<!\w){re.escape(gold)}(?!\w)")
                    self.assertFalse(set(task.grading["ignore"]) - set(normalize_text(task.question).split()))
                else:
                    self.assertFalse([t for t in task.answer if t.casefold() in info["question"].casefold()])

    def test_each_family_draws_from_its_own_split(self):
        questions = {split: {q["q"] for q in research._musique(split)["questions"]} |
                     {q["q"] for q in research._qampari(split)["questions"]} for split in ("train", "select", "test")}
        for family, level, task, info in self.all_built():
            with self.subTest(family=family, level=level):
                self.assertIn(info["question"], questions[research.SPLIT[family]])
                held_out = "reference corpus" in task.question
                self.assertEqual(held_out, family == "research_test")  # the test split's own templates

    def test_splits_share_no_source_question(self):
        mq = {s: research._musique(s)["questions"] for s in ("train", "select", "test")}
        qp = {s: research._qampari(s)["questions"] for s in ("train", "select", "test")}
        for a, b in (("train", "select"), ("train", "test"), ("select", "test")):
            with self.subTest(a=a, b=b):
                self.assertFalse({q["id"] for q in mq[a]} & {q["id"] for q in mq[b]})
                self.assertFalse({i for q in mq[a] for i in q["subq"]} & {i for q in mq[b] for i in q["subq"]})
                self.assertFalse({q["q"] for q in mq[a]} & {q["q"] for q in mq[b]})
                self.assertFalse({q["qid"] for q in qp[a]} & {q["qid"] for q in qp[b]})
                self.assertFalse({q["q"].casefold() for q in qp[a]} & {q["q"].casefold() for q in qp[b]})
                self.assertFalse({u for q in qp[a] for u in q["entity_urls"]} & {u for q in qp[b] for u in q["entity_urls"]})
        # an earlier split never pads with a later split's answers
        for split, later in (("train", ("select", "test")), ("select", ("test",))):
            pool = research._musique(split)
            padding = {text for (_, text), ok in zip(pool["paragraphs"], pool["pad_ok"]) if ok}
            pages = {title for title, _ in research._qampari(split)["pages"]}
            for other in later:
                paras = research._musique(other)["paragraphs"]
                support = {paras[i][1] for q in research._musique(other)["questions"] for i in q["gold"]}
                answer_pages = {research._qampari(other)["pages"][pid][0]
                                for q in research._qampari(other)["questions"] for pid, _ in q["gold"]}
                with self.subTest(split=split, later=other):
                    self.assertFalse(padding & support)
                    self.assertFalse(pages & answer_pages)

    def test_blind_guesser_earns_nothing(self):
        """H1, as tests/test_reward.py's BlindGuesserTest checks for the families in GENERATORS."""
        rewards = [blind_score(t).reward for pairs in self.train.values() for t, _ in pairs]
        self.assertEqual(set(rewards), {0.0})

    def test_licence_bar_and_registry(self):
        self.assertEqual(research.training_sources(), ("musique", "qampari"))
        data.check_training_sources("research", list(research.training_sources()))
        for source in ("musique-dev", "qampari-eval"):
            with self.assertRaisesRegex(ValueError, "eval-only"):
                data.check_training_sources("research", [source])
        for source in ("musique", "musique-dev", "qampari", "qampari-eval"):
            self.assertIn("CC BY-SA", data.SOURCES[source]["licence"])
            for rel in data.SOURCES[source]["files"]:
                self.assertTrue(data.path(rel).exists(), rel)
        with mock.patch.dict(tasks.GENERATORS, RESEARCH_GENERATORS):
            tasks.check_mix({"research": 1.0})
            for family in ("research_select", "research_test"):
                self.assertTrue(tasks.eval_only(family))
                with self.assertRaisesRegex(ValueError, "eval-only"):
                    tasks.check_mix({family: 1.0})
            self.assertEqual(tasks.max_levels(research.FAMILIES), dict.fromkeys(research.FAMILIES, 5))
            self.assertEqual(set(tasks.versions(["research"]).values()), {research.GENERATOR_VERSION})
            task = tasks.make_task("research", 1, seed=7)
            self.assertEqual((task.family, task.level), ("research", 1))
            self.assertEqual(tasks.training_sources("research"), ("musique", "qampari"))


# ----------------------------------------------------------------------------- the open-book check (research-v3)


def _clear_research_caches():
    for fn in (research.closed_book, research.open_book, research.usable, research._by_hops, research.list_pool):
        fn.cache_clear()


class UsableTest(unittest.TestCase):
    """research.usable(): checked closed-book AND not known AND, once the open-book file exists, valid."""

    CLOSED = {"version": "closed-book-v1",
              "results": {"musique": {"select": {"checked": [0, 1, 2, 3, 5], "known": [1]}}}}

    def usable(self, closed, opened):
        _clear_research_caches()
        try:
            with mock.patch.object(research, "closed_book", lambda: closed), \
                    mock.patch.object(research, "open_book", lambda: opened):
                return research.usable.__wrapped__("musique", "select")
        finally:
            _clear_research_caches()

    def test_rules(self):
        self.assertIsNone(self.usable(None, None))  # no check yet: every question
        self.assertEqual(self.usable(self.CLOSED, None), {0, 2, 3, 5})
        opened = {"version": "open-book-v1",
                  "results": {"musique": {"select": {"checked": [0, 1, 2, 3, 5], "valid": [0, 1, 5]}}}}
        self.assertEqual(self.usable(self.CLOSED, opened), {0, 5})
        other_split = {"version": "open-book-v1", "results": {"musique": {"test": {"checked": [0], "valid": []}}}}
        self.assertEqual(self.usable(self.CLOSED, other_split), {0, 2, 3, 5})


def _open_book_script():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "open_book", Path(__file__).resolve().parents[1] / "scripts" / "open_book.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@needs_pools
@unittest.skipUnless(HAVE_POOLS and data.derived(research.CLOSED_BOOK).exists(), "no closed-book results")
class OpenBookScriptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = _open_book_script()
        cls.items = cls.script.plan()

    def test_plan_is_every_question_the_closed_book_check_checked(self):
        closed = research.closed_book()["results"]
        expected = {(source, split, i) for source, by in closed.items() for split, row in by.items()
                    for i in row["checked"]}
        self.assertEqual(set(self.items), expected)
        self.assertEqual(len(self.items), len(expected))

    def test_documents_are_the_gold_and_nothing_else(self):
        for source, split, i in [item for item in self.items if item[1] == "select"][:: 50]:
            text, gold = self.script.documents(source, split, i)
            titles = [t for _, t, _ in docs_of(text)]
            with self.subTest(source=source, i=i):
                if source == "musique":
                    q = research._musique(split)["questions"][i]
                    self.assertEqual(len(titles), len(set(q["gold"])))
                    self.assertEqual(gold, [])
                else:
                    q = research._qampari(split)["questions"][i]
                    self.assertEqual(len(titles), len(q["gold"]) + len(q["neg"]))
                    self.assertLessEqual(set(gold), set(titles))
                self.assertIn(research._musique(split)["questions"][i]["q"] if source == "musique" else
                              research._qampari(split)["questions"][i]["q"], self.script.prompt(source, split, i))

    def test_judge(self):
        musique = [item for item in self.items if item[0] == "musique" and item[1] == "test"][:20]
        for source, split, i in musique:
            q = research._musique(split)["questions"][i]
            self.assertTrue(self.script.judge(source, split, i, f" {q['a']} ")[0])
            self.assertFalse(self.script.judge(source, split, i, "unknown")[0])
        lists = [item for item in self.items if item[0] == "qampari" and item[1] == "test"][:20]
        for source, split, i in lists:
            _, gold = self.script.documents(source, split, i)
            self.assertTrue(self.script.judge(source, split, i, "\n".join(f"- {t}" for t in gold))[0])
            self.assertTrue(self.script.judge(source, split, i, "\n".join(f"{n}. {t}" for n, t in enumerate(gold, 1)))[0])
            half = gold[: len(gold) // 2]
            self.assertFalse(self.script.judge(source, split, i, "\n".join(half))[0])  # F1 under 0.8


if __name__ == "__main__":
    unittest.main()
