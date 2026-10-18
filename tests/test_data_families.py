"""Plan step 7's task families, mix, licence bar and eval sets.

The synthetic code families need nothing outside the repo. The aggregation, repo-localization, capability
and eval-set tests read the data root (D:\\RLM-data, /mnt/d/RLM-data in WSL, or $RLM_DATA_DIR) and skip
without it.

    python3 -m unittest -v tests.test_data_families
"""

import ast
import copy
import datetime as dt
import random
import re
import shutil
import statistics
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rlm_rl import capability, capability_set, config, data, evals, reward, tasks
from rlm_rl.tasks import aggregation, behaviours, editreal, repoloc, synthcode

HAVE_DATA = data.available()
HAVE_LCA = HAVE_DATA and (data.derived("lca", "index-train.jsonl.gz")).exists() and shutil.which("git") is not None
needs_data = unittest.skipUnless(HAVE_DATA, "no data root (rlm_rl/data.py)")
needs_lca = unittest.skipUnless(HAVE_LCA, "no LCA index or git (scripts/build_data.py lca-git lca-index)")
needs_edit_index = unittest.skipUnless(HAVE_DATA and editreal.index_path().exists(),
                                       "no edit_real task index (scripts/build_edit_data.py)")

FILE_RE = re.compile(r"^# ===== file: (.+?) =====$", re.M)


def files_of(context: str) -> dict[str, str]:
    parts = FILE_RE.split(context)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts), 2)}


def seed_of(rng):
    return rng.getrandbits(64)


# ----------------------------------------------------------------------------- shared properties


class FamilyContract:
    """What every generated task must satisfy (spec §3, H3)."""

    def check_task(self, task, family, level):
        reward.check_grading(task.grading, task.answer)
        self.assertEqual((task.family, task.level), (family, level))
        self.assertEqual(reward.score_answer(task.answer, task.answer, task.grading).reward, 1.0)
        blind = reward.score_answer(task.grading.get("blind"), task.answer, task.grading)
        self.assertLessEqual(blind.reward, float(blind.correct))  # H1: the blind guess earns only an exact hit
        self.assertNotIn(str(task.answer), task.question) if isinstance(task.answer, (int, str)) and \
            len(str(task.answer)) > 4 else None

    def check_deterministic(self, family, level):
        a = tasks.make_task(family, level, seed=12345, task_id="t")
        b = tasks.make_task(family, level, seed=12345, task_id="t")
        c = tasks.make_task(family, level, seed=12346, task_id="t")
        self.assertEqual((a.question, a.context, a.answer), (b.question, b.context, b.answer))
        self.assertNotEqual(a.context, c.context)
        self.assertNotIn("12345", tasks.make_task(family, level, seed=12345).task_id)


# ----------------------------------------------------------------------------- synthetic code


class BehaviourLibraryTest(unittest.TestCase):
    def test_every_template_matches_its_reference(self):
        names = {"a": "items", "b": "k", "v": "acc", "w": "out", "it": "elem"}
        for b in behaviours.TRAIN + behaviours.HELDOUT:
            src = f"def f({'items' if b.args == 1 else 'items, k'}):\n" + behaviours.render(b, names) + "\n"
            ns = {}
            exec(compile(src, b.key, "exec"), ns)
            for sample in b.samples:
                with self.subTest(behaviour=b.key, sample=sample):
                    self.assertEqual(ns["f"](*copy.deepcopy(sample)), b.reference(*copy.deepcopy(sample)))

    def test_decoys_differ_from_each_other_on_some_sample(self):
        for group in list(behaviours.TRAIN_GROUPS.values()) + list(behaviours.HELDOUT_GROUPS.values()):
            for x in group:
                for y in group:
                    if x is not y and x.args == y.args:
                        self.assertTrue(any(x.reference(*copy.deepcopy(s)) != y.reference(*copy.deepcopy(s))
                                            for s in x.samples), (x.key, y.key))

    def test_heldout_behaviours_are_never_trained(self):
        self.assertFalse({b.key for b in behaviours.TRAIN} & {b.key for b in behaviours.HELDOUT})
        self.assertFalse(set(behaviours.TRAIN_GROUPS) & set(behaviours.HELDOUT_GROUPS))


def ast_callers(context: str, target: str) -> set[str]:
    """Transitive callers from plain calls only, the way a policy using `ast` would find them."""
    calls: dict[str, set[str]] = {}
    for path, text in files_of(context).items():
        module = path.rsplit("/", 1)[1].removesuffix(".py")
        tree = ast.parse(text)
        local = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        for fn in (n for n in tree.body if isinstance(n, ast.FunctionDef)):
            out = calls.setdefault(f"{module}.{fn.name}", set())
            for node in ast.walk(fn):
                if isinstance(node, ast.Call):
                    f = node.func
                    if isinstance(f, ast.Name) and f.id in local:
                        out.add(f"{module}.{f.id}")
                    elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                        out.add(f"{f.value.id}.{f.attr}")
    found, frontier = set(), [target]
    while frontier:
        node = frontier.pop()
        for caller, callees in calls.items():
            if node in callees and caller not in found:
                found.add(caller)
                frontier.append(caller)
    return found


class SynthCodeTest(FamilyContract, unittest.TestCase):
    def test_every_family_and_level(self):
        rng = random.Random(1)
        for family in synthcode.FAMILIES:
            for level in synthcode.LEVELS:
                task = tasks.make_task(family, level, seed=seed_of(rng))
                with self.subTest(family=family, level=level):
                    self.check_task(task, family, level)
                    for text in files_of(task.context).values():
                        ast.parse(text)  # every file is valid Python
                    if family.startswith("callgraph"):
                        names = {f"{p.rsplit('/', 1)[1][:-3]}.{m}" for p, t in files_of(task.context).items()
                                 for m in re.findall(r"^def (\w+)\(", t, re.M)}
                        self.assertTrue(set(task.answer) <= names)
                    else:
                        module, name = task.answer.split(".")
                        body = files_of(task.context)[next(p for p in files_of(task.context) if p.endswith(f"/{module}.py"))]
                        self.assertIsNotNone(re.search(rf"^def {name}\(", body, re.M))

    def test_deterministic_per_seed(self):
        for family in ("callgraph", "needle"):
            self.check_deterministic(family, 2)

    def test_plain_call_graphs_are_what_ast_finds(self):
        rng = random.Random(2)
        for _ in range(8):
            task = tasks.make_task("callgraph", 1, seed=seed_of(rng))
            target = re.search(r"`(\w+\.\w+)`", task.question).group(1)
            self.assertTrue(task.grading["code_only"])
            self.assertEqual(ast_callers(task.context, target), set(task.answer))

    def test_hidden_calls_defeat_plain_ast_above_level_1(self):
        rng, differ = random.Random(3), 0
        for _ in range(12):
            task = tasks.make_task("callgraph", 5, seed=seed_of(rng))
            target = re.search(r"`(\w+\.\w+)`", task.question).group(1)
            if not task.grading["code_only"]:
                differ += ast_callers(task.context, target) != set(task.answer)
        self.assertGreaterEqual(differ, 4)

    def test_needle_target_does_what_the_question_says_and_decoys_dont(self):
        rng = random.Random(4)
        by_key = {b.key: b for b in behaviours.TRAIN + behaviours.HELDOUT}
        for family in ("needle", "needle_heldout"):
            for level in (1, 3, 5):
                task = tasks.make_task(family, level, seed=seed_of(rng))
                behaviour = by_key[task.grading["behaviour"]]
                self.assertIn(task.question.split("function that ")[-1].split(". ")[0][:30], task.question)
                sources = {}
                for path, text in files_of(task.context).items():
                    module = path.rsplit("/", 1)[1][:-3]
                    for node in ast.parse(text).body:
                        if isinstance(node, ast.FunctionDef):
                            sources[f"{module}.{node.name}"] = ast.get_source_segment(text, node)

                def run(qname, sample):
                    ns = {}
                    exec(sources[qname], ns)  # our own generated code: pure functions, no calls
                    return ns[qname.split(".")[1]](*copy.deepcopy(sample))

                with self.subTest(family=family, level=level):
                    for sample in behaviour.samples:
                        self.assertEqual(run(task.answer, sample), behaviour.reference(*copy.deepcopy(sample)))
                    for decoy in task.grading["decoys"]:
                        try:
                            same = all(run(decoy, s) == behaviour.reference(*copy.deepcopy(s)) for s in behaviour.samples)
                        except Exception:  # a decoy with other argument types
                            same = False
                        self.assertFalse(same, decoy)

    def test_heldout_uses_its_own_questions_and_call_styles(self):
        rng = random.Random(5)
        train_q = {t.split("{")[0] for ts in synthcode.TRAIN_QUESTIONS.values() for t in ts}
        styles = set()
        for _ in range(6):
            task = tasks.make_task("callgraph_heldout", 4, seed=seed_of(rng))
            self.assertFalse(any(task.question.startswith(q) for q in train_q))
            styles |= {"partial"} if "functools.partial(" in task.context else set()
            styles |= {"lambda"} if "lambda v:" in task.context else set()
        self.assertEqual(styles, {"partial", "lambda"})
        for _ in range(6):
            task = tasks.make_task("callgraph", 5, seed=seed_of(rng))
            self.assertNotIn("functools.partial(", task.context)
            self.assertNotIn("lambda v:", task.context)

    def test_sizes_grow_to_about_400k(self):
        rng = random.Random(6)
        for family in ("callgraph", "needle"):
            medians = [statistics.median(len(tasks.make_task(family, lv, seed=seed_of(rng)).context) for _ in range(3))
                       for lv in synthcode.LEVELS]
            self.assertEqual(medians, sorted(medians))
            self.assertTrue(25_000 < medians[0] < 50_000, medians)
            self.assertTrue(130_000 < medians[-1] < 470_000, medians)


# ----------------------------------------------------------------------------- aggregation


_LINE_RES = [re.compile(r"^(?P<d>\d{4}-\d{2}-\d{2}) \| user (?P<u>\d+) \| (?P<t>.*)$"),
             re.compile(r"^\[(?P<d>\w{3} \d+, \d{4})\] user#(?P<u>\d+): (?P<t>.*)$"),
             re.compile(r"^(?P<d>\d{2}/\d{2}/\d{4}) - id (?P<u>\d+) - (?P<t>.*)$"),
             re.compile(r"^On \w+ (?P<d>\d+ \w+ \d{4}), account (?P<u>\d+) posted: (?P<t>.*)$"),
             re.compile(r"^(?P<d>\d{4}\.\d{2}\.\d{2}) <(?P<u>\d+)> (?P<t>.*)$")]
_DATE_FORMATS = ("%Y-%m-%d", "%b %d, %Y", "%d/%m/%Y", "%d %B %Y", "%Y.%m.%d")


def parse_lines(context):
    rows = []
    for line in context.splitlines()[1:]:
        for pattern, fmt in zip(_LINE_RES, _DATE_FORMATS):
            m = pattern.match(line)
            if m:
                rows.append((dt.datetime.strptime(m["d"], fmt).date(), m["u"], m["t"]))
                break
        else:
            raise AssertionError(f"unparsed line: {line}")
    return rows


def recount(task, pools):
    """Recompute a task's answer from its rendered text alone, labelling each message by its source pool."""
    rows = parse_lines(task.context)
    labels = re.findall(r"^- (.+?): ", task.question, re.M)
    label_of = {}
    for name in labels:
        for text in pools[name]:
            label_of.setdefault(text, set()).add(name)
    q, kind = task.question, task.grading["kind"]
    if task.grading["code_only"]:  # no labels in the question, and none needed
        labelled = [(d, u, None) for d, u, _ in rows]
    else:
        labelled = [(d, u, next(iter(label_of[t]))) for d, u, t in rows if len(label_of.get(t, ())) == 1]
        assert len(labelled) == len(rows), "a message belongs to no label, or to two"
    quoted = re.findall(r"'([^']+)'", q)
    user = (re.search(r"(?:user|account) (\d{5})", q) or [None, None])[1]
    dates = re.findall(r"(\w+ \d+, \d{4})", q)
    window = [dt.datetime.strptime(d, "%B %d, %Y").date() for d in dates]
    number = [int(n) for n in re.findall(r"(?:more than|exceeds|least|or more) (\d+)|(\d+) or more", q)[0] if n] \
        if kind in ("labels_over", "users_with_label") else None
    if kind == "label_count":
        return sum(lab == quoted[-1] for _, _, lab in labelled)
    if kind == "user_label_count":
        return sum(u == user and lab == quoted[-1] for _, u, lab in labelled)
    if kind == "window_label_count":
        return sum(lab == quoted[-1] and window[0] <= d <= window[1] for d, _, lab in labelled)
    if kind == "user_window_count":
        return sum(u == user and window[0] <= d <= window[1] for d, u, _ in labelled)
    if kind == "most_active_user":
        counts = {}
        for _, u, _ in labelled:
            counts[u] = counts.get(u, 0) + 1
        return max(counts, key=counts.get)
    if kind == "labels_over":
        return sorted(n for n in labels if sum(lab == n for _, _, lab in labelled) > number[0])
    if kind == "users_with_label":
        per = {}
        for _, u, lab in labelled:
            per[u] = per.get(u, 0) + (lab == quoted[-1])
        return sorted(u for u, c in per.items() if c >= number[0])
    raise AssertionError(kind)


@needs_data
class AggregationTest(FamilyContract, unittest.TestCase):
    def test_every_family_and_level(self):
        rng = random.Random(1)
        for family in aggregation.FAMILIES:
            for level in aggregation.LEVELS:
                task = tasks.make_task(family, level, seed=seed_of(rng))
                with self.subTest(family=family, level=level):
                    self.check_task(task, family, level)
                    lo, hi = aggregation.LEVELS[level]
                    self.assertTrue(lo * 0.95 <= len(task.context) <= hi * 1.05, len(task.context))

    def test_the_answer_is_what_the_context_says(self):
        pools = {**data.clinc_domain_pools(), **data.massive_scenario_pools(), **data.ledgar_pools(),
                 **data.arxiv_title_pools(), **data.huffpost_pools()}
        rng, kinds = random.Random(2), set()
        for family in aggregation.FAMILIES:
            for _ in range(30):
                task = tasks.make_task(family, rng.choice([1, 2]), seed=seed_of(rng))
                kinds.add(task.grading["kind"])
                with self.subTest(family=family, kind=task.grading["kind"]):
                    self.assertEqual(recount(task, pools), task.answer)
        self.assertEqual(kinds, set(aggregation.KINDS))

    def test_no_label_appears_in_the_context(self):
        rng = random.Random(3)
        for _ in range(10):
            task = tasks.make_task("aggregation", 1, seed=seed_of(rng))
            for name in re.findall(r"^- (.+?): ", task.question, re.M):
                self.assertNotIn(f" {name} ", f" {task.context.splitlines()[0]} ")

    def test_deterministic_per_seed(self):
        self.check_deterministic("aggregation", 1)

    def test_every_training_source_is_drawn(self):
        """Decision 7c raised aggregation to 45%, so it draws on four registers, not one (LEDGAR and arXiv added)."""
        rng = random.Random(5)
        drawn = {tasks.make_task("aggregation", rng.choice([1, 2, 3]), seed=seed_of(rng)).grading["source"]
                 for _ in range(60)}
        self.assertEqual(drawn, {s.name for s in aggregation.TRAIN_SET})
        names = [label for s in aggregation.TRAIN_SET + (aggregation.HELDOUT,) for label in s.labels]
        self.assertEqual(len(names), len(set(names)))  # no label name is shared between sources

    def test_training_never_reads_heldout_sources(self):
        rng = random.Random(4)
        huff = {t for texts in data.huffpost_pools().values() for t in texts}
        for _ in range(6):
            rows = parse_lines(tasks.make_task("aggregation", 2, seed=seed_of(rng)).context)
            self.assertFalse({t for _, _, t in rows} & huff)


# ----------------------------------------------------------------------------- repo localization


@needs_lca
class RepoLocTest(FamilyContract, unittest.TestCase):
    def test_every_family_and_level(self):
        rng = random.Random(1)
        for family in ("repo_loc", "repo_loc_select"):
            for level in repoloc.LEVELS:
                task = tasks.make_task(family, level, seed=seed_of(rng))
                with self.subTest(family=family, level=level):
                    self.check_task(task, family, level)
                    files = files_of(task.context)
                    self.assertTrue(set(task.answer) <= set(files))
                    self.assertEqual(len(files), task.grading["files"])

    def test_selection_repos_are_never_trained_on(self):
        rng = random.Random(2)
        for _ in range(40):
            task = tasks.make_task("repo_loc", rng.choice([1, 3, 5]), seed=seed_of(rng))
            self.assertNotIn(task.grading["repo"], repoloc.SELECTION_REPOS)
        self.assertFalse({it["id"] for it in repoloc.pool("repo_loc_test")} & {it["id"] for it in repoloc.items("train")})

    def test_deterministic_per_seed(self):
        self.check_deterministic("repo_loc", 1)

    def test_test_split_uses_its_own_question(self):
        tasks_ = repoloc.test_tasks([tasks.new_seed() for _ in repoloc.items("test")], "t")
        self.assertEqual(len(tasks_), len(repoloc.items("test")))
        self.assertTrue(all(t.question.startswith("Bug report:") for t in tasks_))


# ----------------------------------------------------------------------------- mix, licences, eval sets


class MixAndLicenceTest(unittest.TestCase):
    def test_mix_is_spec_v2(self):
        """Spec v2 §12.7 (decision E-a): editing 40% (edit_real 25, edit_synth 15), aggregation 25, code reading 20,
        repo localization 15, in that order, so allocate_step splits 40 tasks 10/4/4/6/10/6."""
        self.assertEqual(tasks.DOMAIN_MIX, {"aggregation": 0.25, "code": 0.20, "repo": 0.15, "editing": 0.40})
        mix = tasks.production_mix()
        self.assertEqual(list(mix), ["aggregation", "callgraph", "needle", "repo_loc", "edit_real", "edit_synth"])
        self.assertAlmostEqual(sum(mix.values()), 1.0)
        for family, share in {"aggregation": 0.25, "callgraph": 0.10, "needle": 0.10, "repo_loc": 0.15,
                              "edit_real": 0.25, "edit_synth": 0.15}.items():
            self.assertAlmostEqual(mix[family], share, msg=family)

    @needs_edit_index
    def test_production_config(self):
        cfg = config.production("p", 2e-5)
        self.assertEqual(cfg.selection, "selection-v3")
        self.assertEqual(cfg.prompt_version, "v3")
        self.assertEqual(cfg.root_thinking, "medium")  # decision 11b (Rishi, 2026-10-17)
        self.assertIsNone(config.second_seed().root_thinking)
        with self.assertRaises(ValueError):
            config.RunConfig(run="x", kind="production", learning_rate=2e-5, lr_rule="fixed", root_thinking="xhigh")
        self.assertEqual(cfg.ceilings, {"train": 146.0, "eval": 34.0})  # from $191.01 left (Rishi, 2026-10-17)
        self.assertEqual(cfg.capability_set, config.CAPABILITY_SET)
        self.assertEqual(cfg.mix, tasks.production_mix())
        self.assertEqual(config.production("p", 2e-5, tasks_from="tickets").selection, "selection-tickets-v1")
        self.assertEqual(config.manifest(cfg)["mix"], cfg.mix)

    @needs_edit_index
    def test_licence_bar(self):
        tasks.check_mix(tasks.production_mix())
        for family in ("aggregation_heldout", "needle_heldout", "repo_loc_test", "repo_loc_select"):
            with self.assertRaisesRegex(ValueError, "eval-only"):
                tasks.check_mix({family: 1.0})
        for source in ("oolong-synth", "huffpost", "ifeval"):
            with self.assertRaisesRegex(ValueError, "eval-only"):
                data.check_training_sources("x", [source])
        with mock.patch.object(tasks, "training_sources", return_value=("oolong-synth",)):
            with self.assertRaisesRegex(ValueError, "7b"):
                tasks.check_mix({"aggregation": 1.0})
        for family in tasks.production_mix():
            for source in tasks.training_sources(family):
                self.assertTrue(data.SOURCES[source]["permissive"], source)

    def test_selection_plan_is_64_tasks_and_heldout_families_stay_out_of_training(self):
        self.assertEqual(sum(evals.SELECTION_V1.values()), 64)
        self.assertEqual(sum(evals.SELECTION_V2_READ.values()) + len(evals.SELECTION_V2_SYNTH), 48 + 6)  # + 10 edit_real
        self.assertEqual(len(evals.HELDOUT_V2_SYNTH), 20)
        trained = set(tasks.production_mix())
        self.assertFalse({f for f, _ in evals.HELDOUT_TEST_V1} & trained)
        self.assertTrue({"tickets_all", "tickets_customer"} & {f for f, _ in evals.SELECTION_V1})
        self.assertFalse({"tickets_all", "tickets_customer"} & trained)

    @needs_lca
    def test_eval_sets_build_from_private_seeds(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(tasks, "PRIVATE_DIR", Path(tmp)):
            first, samples = evals.eval_set("selection-v1")
            again, _ = evals.eval_set("selection-v1")
            self.assertEqual((len(first), samples), (64, 1))
            self.assertEqual([t.context for t in first], [t.context for t in again])
            self.assertTrue((Path(tmp) / "seeds" / "selection-v1.json").exists())

    @needs_data
    def test_capability_set(self):
        items = capability.load(capability.resolve(config.CAPABILITY_SET))
        self.assertEqual(len(items), 100)
        self.assertEqual({i.source for i in items}, {"ifeval", "mmlu_pro", "cruxeval", "chat"})
        chat = {i.id: i for i in items if i.source == "chat"}
        self.assertTrue(capability.grade(chat["chat-00"], "ready"))
        self.assertTrue(capability.grade(chat["chat-01"], "391"))
        self.assertFalse(capability.grade(chat["chat-01"], "392"))
        self.assertEqual(capability.resolve(config.CAPABILITY_SET), capability_set.path())


if __name__ == "__main__":
    unittest.main()
