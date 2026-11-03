"""Parity with training (rlm-harness v0, step 8): the pure parts of scripts/harness_parity.py, on synthetic records.

No engine, no network, no transcripts, no data drive and no Docker: the records are built with rlm_rl.bridge, the
comparison, sub-call book, classifier, eligibility rules, exit codes and summary guard run as the script runs them.
"""

import asyncio
import collections
import contextlib
import copy
import hashlib
import importlib.util
import io
import itertools
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rlm_rl import bridge
from rlm_rl.client import ROOT_TEMPERATURE

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location("harness_parity", ROOT / "scripts" / "harness_parity.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses use `from __future__ import annotations` and look the module up
    spec.loader.exec_module(module)
    return module


hp = _load_script()

QUESTION = "Which carrier had the most delayed departures in the March operations report, and by how many minutes?"
CONTEXT = ("OPERATIONS REPORT 2026-03: carrier delays and cancellations by region, compiled for the review board\n"
           "second line of the report\nthird line\nfourth line\n")
EDIT = {"language": "python", "n_files": 4, "test_hint": "pytest -q tests"}
FAKE_ENV = {"PATH": "/usr/bin", "LANG": "C.UTF-8"}


def flip(text, at=5):
    """The same text with one character changed."""
    return text[:at] + ("~" if text[at] != "~" else "^") + text[at + 1:]


def make_record(family="aggregation", n_turns=3, split="val", pace=None, **over):
    """A clean v4 episode record, as transcripts.episode_record writes it: the conversation (system, first message,
    then each reply and observation; the last reply is a submit and has no observation), the turns and the limits."""
    edit = bridge.is_edit(family)
    lim = bridge.limits(family, "v4")
    messages = bridge.initial_messages(QUESTION, CONTEXT, "v4", lim, edit=EDIT if edit else None)
    turns = []
    for i in range(1, n_turns + 1):
        reply = f"Step {i}: look at the data.\n```python\nprint({i})\n```"
        last = i == n_turns
        obs = None if last else bridge.observation(f"{i}\n", 2, None, lim.max_turns - i, lim.max_subcalls,
                                                   pace=pace if i == n_turns - 1 else None)
        turns.append({"reply": reply, "finish_reason": "stop", "code": f"print({i})",
                      "cell": {"error": None, "restarted": False}, "observation": obs})
        messages.append({"role": "assistant", "content": reply})
        if obs is not None:
            messages.append({"role": "user", "content": obs})
    record = {
        "episode_id": f"ep{family}{n_turns}".replace("_", ""), "run": "prod-v2", "split": split, "step": 3,
        "eval": "selection-v4" if split == "val" else None, "family": family, "task_id": f"{family}-task",
        "question": QUESTION, "prompt_version": "v4", "limits": lim.as_dict(),
        "language": EDIT["language"] if edit else None,
        "context_chars": len(CONTEXT), "context_sha256": hashlib.sha256(CONTEXT.encode("utf-8", "replace")).hexdigest(),
        "messages": messages, "turns": turns, "subcalls": [], "subcalls_used": 0, "submitted": True, "answer": "42",
        "hacks": [], "flags": [], "events": [], "infra_error": None, "timeouts": 0, "crashes": 0, "restarts": 0,
        "diff": None,
    }
    record.update(over)
    return record


def root_requests(record, n=None):
    """The root model.complete params a faithful engine sends: request k carries messages[:2 + 2*(k-1)]."""
    lim = record["limits"]
    out = []
    for k in range(1, (len(record["turns"]) if n is None else n) + 1):
        out.append({"run_id": "r1", "role": "root", "turn": k, "model": dict(hp.MODEL_REF),
                    "messages": copy.deepcopy(record["messages"][:2 + 2 * (k - 1)]), "tools": [],
                    "max_tokens": lim["root_max_tokens"], "temperature": ROOT_TEMPERATURE, "thinking": "off"})
    return out


def change_from(requests, index, first_request):
    """Change one character of message `index` in every request from `first_request` (1-based) on, as a real run
    would carry it: a message stays in the conversation once it has been sent."""
    for params in requests[first_request - 1:]:
        message = params["messages"][index]
        message["content"] = flip(message["content"])


def diff(accepted=False, classes=None, request=2, index=3, recorded="a", got="b"):
    return hp.Difference(request, index, "user", recorded, got, classes if classes is not None else ["unexplained"],
                         accepted)


def result(eid="e1", family="aggregation", differences=(), **kw):
    mode = "repo" if bridge.is_edit(family) else "read"
    r = hp.Result(eid, "prod-v2", family, mode, "training-v4", 3, 0)
    r.differences = list(differences)
    r.requests, r.matched = kw.pop("requests", 3), kw.pop("matched", 3)
    for key, value in kw.items():
        setattr(r, key, value)
    return r


class CompareTests(unittest.TestCase):
    def test_message_prefix_length(self):
        rec = make_record(n_turns=4)
        self.assertEqual(len(rec["messages"]), 9)  # 4 replies and 3 observations behind the first two messages
        limits = rec["limits"]
        reqs = root_requests(rec)
        self.assertEqual([len(r["messages"]) for r in reqs], [2, 4, 6, 8])  # request k: messages[:2 + 2*(k-1)]
        self.assertEqual(hp.compare_requests(rec["messages"], reqs, limits), ([], [], 4))

        # One changed character in request 2 alone: that request fails, the others stay byte for byte.
        reqs = root_requests(rec)
        reqs[1]["messages"][3]["content"] = flip(reqs[1]["messages"][3]["content"])
        diffs, failures, matched = hp.compare_requests(rec["messages"], reqs, limits)
        self.assertEqual(failures, [])
        self.assertEqual(matched, 3)
        self.assertEqual([(d.request, d.index, d.role) for d in diffs], [(2, 3, "user")])
        self.assertEqual(diffs[0].recorded, rec["messages"][3]["content"])
        self.assertEqual(diffs[0].got, reqs[1]["messages"][3]["content"])
        self.assertNotEqual(diffs[0].recorded, diffs[0].got)

        # The same change carried into every later request (the conversation keeps the message): reported once, at
        # request 2, where it first differs. Requests 3 and 4 contain it too, so only request 1 is byte for byte.
        reqs = root_requests(rec)
        change_from(reqs, 3, first_request=2)
        diffs, failures, matched = hp.compare_requests(rec["messages"], reqs, limits)
        self.assertEqual(failures, [])
        self.assertEqual([(d.request, d.index) for d in diffs], [(2, 3)])
        self.assertEqual(matched, 1)

        # Two messages that differ are two differences, each at its own first request.
        reqs = root_requests(rec)
        change_from(reqs, 3, first_request=2)
        change_from(reqs, 4, first_request=3)
        diffs, _, matched = hp.compare_requests(rec["messages"], reqs, limits)
        self.assertEqual([(d.request, d.index, d.role) for d in diffs], [(2, 3, "user"), (3, 4, "assistant")])
        self.assertEqual(matched, 1)

        # The system prompt changed from the first request on.
        reqs = root_requests(rec)
        change_from(reqs, 0, first_request=1)
        diffs, _, matched = hp.compare_requests(rec["messages"], reqs, limits)
        self.assertEqual([(d.request, d.index, d.role) for d in diffs], [(1, 0, "system")])
        self.assertEqual(matched, 0)

        # A wrong message count: too many in request 1 (the whole conversation), too few in request 3.
        reqs = root_requests(rec)
        reqs[0]["messages"] = copy.deepcopy(rec["messages"][:4])
        reqs[2]["messages"] = reqs[2]["messages"][:5]
        diffs, failures, matched = hp.compare_requests(rec["messages"], reqs, limits)
        self.assertEqual(diffs, [])
        self.assertEqual(matched, 2)
        self.assertEqual(len(failures), 2)
        self.assertTrue(failures[0].startswith("message_count: request 1 has 4 messages, training had 2"), failures)
        self.assertTrue(failures[1].startswith("message_count: request 3 has 5 messages, training had 6"), failures)
        reqs = root_requests(rec)
        del reqs[1]["messages"]  # no messages at all
        _, failures, matched = hp.compare_requests(rec["messages"], reqs, limits)
        self.assertEqual(matched, 3)
        self.assertEqual(len(failures), 1)
        self.assertIn("request 2 has 0 messages", failures[0])

        # A message that is not exactly {role, content} (or has another role) is a shape failure, reported once.
        reqs = root_requests(rec)
        for params in reqs:
            params["messages"][1]["name"] = "extra"
        diffs, failures, matched = hp.compare_requests(rec["messages"], reqs, limits)
        self.assertEqual(diffs, [])
        self.assertEqual(matched, 0)
        self.assertEqual(len(failures), 1)
        self.assertTrue(failures[0].startswith("message_shape: request 1, message 1"), failures)
        reqs = root_requests(rec)
        reqs[1]["messages"][2]["role"] = "user"
        _, failures, matched = hp.compare_requests(rec["messages"], reqs, limits)
        self.assertEqual(matched, 3)
        self.assertEqual(len(failures), 1)
        self.assertTrue(failures[0].startswith("message_shape: request 2, message 2"), failures)

        # A request parameter that is wrong fails with its name, but does not change which messages differ.
        for key, value in (("temperature", 0.7), ("max_tokens", 1024), ("tools", [{"name": "python"}]),
                           ("thinking", "low")):
            with self.subTest(key):
                reqs = root_requests(rec)
                reqs[1][key] = value
                diffs, failures, matched = hp.compare_requests(rec["messages"], reqs, limits)
                self.assertEqual((diffs, matched), ([], 4))
                self.assertEqual(len(failures), 1)
                self.assertTrue(failures[0].startswith(f"request_param: request 2 has {key} {value!r}"), failures)


class SubcallTests(unittest.TestCase):
    RECORDED = [{"prompt": "A", "output": "a1"}, {"prompt": "B", "output": "b"}, {"prompt": "A", "output": "a2"},
                {"prompt": "C", "output": None}]

    def test_subcall_matching(self):
        book = hp.SubcallBook(self.RECORDED)
        self.assertEqual(book.unused(), 4)
        out = book.answer("B")
        self.assertEqual(out["text"], "b")
        self.assertEqual((out["tool_call"], out["reasoning"], out["finish_reason"], out["served_model"],
                          out["seconds"]), (None, None, "stop", "recorded", 0))
        self.assertEqual(out["usage"], hp.ZERO_USAGE)
        self.assertEqual(book.answer("C")["text"], "")  # a recorded output of None
        self.assertEqual(book.unused(), 2)

        # Exact prompts only: a case or whitespace difference is not a match.
        for prompt in ("a", "A ", " A", "AA", ""):
            with self.subTest(prompt):
                with self.assertRaises(hp.RpcReply):
                    book.answer(prompt)
        self.assertEqual(len(book.unmatched), 5)
        self.assertEqual(book.unused(), 2)  # an unmatched prompt consumes nothing

        # Duplicates are consumed in the order recorded, whatever order the calls arrive in.
        orders = set(itertools.permutations(["A", "B", "A", "C"]))
        self.assertEqual(len(orders), 12)
        for order in orders:
            with self.subTest(order):
                book = hp.SubcallBook(self.RECORDED)
                got = [(p, book.answer(p)["text"]) for p in order]
                self.assertEqual([t for p, t in got if p == "A"], ["a1", "a2"])
                self.assertEqual(dict(got)["B"], "b")
                self.assertEqual((book.unused(), book.unmatched), (0, []))

        # A prompt recorded twice matches twice; a third arrival is unmatched.
        book = hp.SubcallBook(self.RECORDED)
        book.answer("A"), book.answer("A")
        with self.assertRaises(hp.RpcReply):
            book.answer("A")
        self.assertEqual(len(book.unmatched), 1)
        self.assertEqual(book.unused(), 2)  # B and C are still waiting

        # An unknown prompt: the `other` error that stops the run (D-06), and an entry for the report with its length,
        # its hash and the longest prefix it shares with a recorded prompt, but no text.
        recorded = [{"prompt": "Other prompt entirely", "output": "x"},
                    {"prompt": "Summarize the section: alpha beta", "output": "y"}]
        book = hp.SubcallBook(recorded)
        prompt = "Summarize the section: alpha GAMMA delta"
        with self.assertRaises(hp.RpcReply) as cm:
            book.answer(prompt)
        err = cm.exception
        self.assertEqual(err.code, -32001)
        self.assertEqual(err.data["kind"], "other")  # the run stops with infra_error (D-06)
        self.assertEqual(err.to_json()["code"], -32001)
        self.assertEqual(len(book.unmatched), 1)
        entry = book.unmatched[0]
        self.assertEqual(entry["chars"], len(prompt))
        self.assertEqual(entry["sha256"], hashlib.sha256(prompt.encode("utf-8")).hexdigest())
        self.assertEqual(entry["common_prefix"], len("Summarize the section: alpha "))  # the closest recorded prompt
        self.assertEqual(entry["closest_chars"], len(recorded[1]["prompt"]))
        self.assertNotIn("alpha", json.dumps(entry))  # lengths and a hash, no text
        self.assertEqual(book.unused(), 2)

        # No shared prefix at all, and a book with nothing recorded.
        self.assertEqual(book.describe("zzz")["common_prefix"], 0)
        empty = hp.SubcallBook([])
        entry = empty.describe("anything")
        self.assertEqual((entry["common_prefix"], entry["closest_chars"]), (0, None))
        with self.assertRaises(hp.RpcReply):
            empty.answer("anything")
        self.assertEqual(empty.unused(), 0)

    def test_recorded_subcall_error(self):
        msg = "the API rejected the request (HTTP 400)"
        error = repr(bridge.SubcallError(msg))  # what training logs
        self.assertEqual(error, f"SubcallError({msg!r})")
        self.assertEqual(error, "SubcallError('the API rejected the request (HTTP 400)')")
        book = hp.SubcallBook([{"prompt": "p", "output": None, "error": error}])
        with self.assertRaises(hp.RpcReply) as cm:
            book.answer("p")
        err = cm.exception
        self.assertEqual(err.code, -32001)
        self.assertEqual(err.message, msg)  # the engine adds the "sub-call failed: " prefix itself
        self.assertEqual(err.data, {"kind": "bad_request", "provider": "parity", "model": "recorded", "status": 400,
                                    "message": msg})
        self.assertEqual(err.to_json(), {"code": -32001, "message": msg, "data": err.data})
        self.assertEqual(book.unmatched, [])  # it matched; it is an answer, not a miss
        self.assertEqual(book.unused(), 0)

        # An error wins over an output, and is consumed in order like any other record.
        book = hp.SubcallBook([{"prompt": "p", "output": "ignored", "error": error}, {"prompt": "p", "output": "ok"}])
        with self.assertRaises(hp.RpcReply):
            book.answer("p")
        self.assertEqual(book.answer("p")["text"], "ok")

        # subcall_error_message unwraps the repr (quotes, escapes) and leaves anything else alone.
        for text in ("it's quoted", 'say "hi"', "two\nlines", "tab\tand \\ backslash", "ünïcode ✓"):
            with self.subTest(text):
                self.assertEqual(hp.subcall_error_message(repr(bridge.SubcallError(text))), text)
        for plain in ("plain failure text", "SubcallError", "RuntimeError('x')", "SubcallError(5)",
                      "SubcallError('a', 'b')", "SubcallError(not a literal)", ""):
            with self.subTest(plain):
                self.assertEqual(hp.subcall_error_message(plain), plain)
        book = hp.SubcallBook([{"prompt": "q", "error": "plain failure text"}])
        with self.assertRaises(hp.RpcReply) as cm:
            book.answer("q")
        self.assertEqual((cm.exception.message, cm.exception.data["kind"], cm.exception.data["status"]),
                         ("plain failure text", "bad_request", 400))

    def test_replay_answers(self):
        rec = make_record(n_turns=2, subcalls=[{"prompt": "P1", "output": "O1", "error": None}], subcalls_used=1)

        async def go():
            replay = hp.Replay(rec)
            first = await replay.complete({"role": "root", "turn": 1, "messages": []})
            second = await replay.complete({"role": "root", "turn": 2, "messages": []})
            sub = await replay.complete({"role": "sub", "messages": [{"role": "user", "content": "P1"}]})
            with self.assertRaises(hp.RpcReply) as extra:
                await replay.complete({"role": "root", "turn": 3, "messages": []})
            with self.assertRaises(hp.RpcReply) as lost:
                await replay.complete({"role": "sub", "messages": [{"role": "user", "content": "P2"}]})
            approved = await replay.approve({"kind": "write"})
            await replay.approve({"kind": "write"})
            await replay.approve({"kind": "program"})
            replay.event("run.cell", {"turn": 1})
            replay.event("model.complete", {"ignored": True})
            self.assertFalse(replay.finished.is_set())
            replay.event("run.finished", {"turns": 2})
            return replay, first, second, sub, extra.exception, lost.exception, approved

        replay, first, second, sub, extra, lost, approved = asyncio.run(go())
        self.assertEqual((first["text"], first["finish_reason"]), (rec["turns"][0]["reply"], "stop"))
        self.assertEqual(second["text"], rec["turns"][1]["reply"])
        self.assertEqual((first["tool_call"], first["reasoning"], first["usage"]), (None, None, hp.ZERO_USAGE))
        self.assertEqual(sub["text"], "O1")
        self.assertEqual((extra.code, extra.data["kind"]), (-32001, "other"))
        self.assertEqual(lost.data["kind"], "other")
        self.assertEqual(len(replay.book.unmatched), 1)
        self.assertEqual([f.split(":")[0] for f in replay.failures], ["extra_root_call"])
        self.assertEqual(approved, {"allow": True, "remember_for_run": False})
        self.assertEqual(dict(replay.approvals), {"write": 2, "program": 1})
        self.assertTrue(replay.finished.is_set())
        self.assertEqual([m for m, _ in replay.events], ["run.cell", "run.finished"])
        self.assertEqual(replay.last("run.finished"), {"turns": 2})
        self.assertIsNone(replay.last("run.started"))

        async def wrong_turn():  # the engine's turn counter must follow the root calls
            replay = hp.Replay(rec)
            await replay.complete({"role": "root", "turn": 5, "messages": []})
            return replay

        self.assertTrue(asyncio.run(wrong_turn()).failures[0].startswith("turn_number: root call 1"))


class ClassifyTests(unittest.TestCase):
    def test_classify_timing_and_version(self):
        def classify(a, b, **kw):
            return hp.classify(a, b, **kw)

        self.assertEqual(classify("same text", "same text"), ([], True))

        # unittest, go, cargo and plain time units.
        for a, b in (("Ran 4 tests in 0.001s\n\nOK", "Ran 4 tests in 0.002s\n\nOK"),
                     ("Ran 4 tests in 0.001s\n\nOK", "Ran 4 tests in 12.340s\n\nOK"),
                     ("ok  \texample.com/pkg\t0.003s", "ok  \texample.com/pkg\t0.431s"),
                     ("test result: ok. 3 passed; 0 failed; finished in 0.00s", "test result: ok. 3 passed; 0 failed; "
                                                                                 "finished in 0.01s"),
                     ("took 12 ms", "took 3 s"), ("waited 2 seconds", "waited 7 sec")):
            with self.subTest(a=a):
                self.assertEqual(classify(a, b), (["timing"], True))

        # pytest: the '=' padding follows the width of the time, so it is masked with it; the counts stay.
        short = "===== 3 passed in 0.12s ====="
        long = "==== 3 passed in 10.25s ===="
        self.assertEqual(classify(short, long), (["timing"], True))
        self.assertEqual(classify(f"collected 3 items\n\n{short}\n", f"collected 3 items\n\n{long}\n"),
                         (["timing"], True))
        self.assertEqual(classify(short, "===== 2 passed in 0.12s ====="), (["unexplained"], False))
        self.assertEqual(classify(short, "===== 3 passed, 1 failed in 0.12s ====="), (["unexplained"], False))
        self.assertEqual(classify("== 3 passed ==", "==== 3 passed ===="), (["unexplained"], False))  # no time: no mask
        self.assertEqual(classify("Ran 4 tests in 0.001s", "Ran 5 tests in 0.001s"), (["unexplained"], False))

        # sys.version: the patch release and the whole build string.
        a = "3.12.15 (main, Oct  6 2026, 02:02:22) [GCC 14.2.0]"
        b = "3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]"
        self.assertEqual(classify(a, b), (["python-version"], True))
        self.assertEqual(classify(f"python {a}\nok\n", f"python {b}\nok\n"), (["python-version"], True))
        self.assertEqual(classify("3.12.15", "3.12.3"), (["python-version"], True))
        windows = "3.12.3 (tags/v3.12.3:f6650f9, Apr  9 2024, 14:05:25) [MSC v.1938 64 bit (AMD64)]"  # this PC's Python
        self.assertEqual(classify(a, windows), (["python-version"], True))
        self.assertEqual(hp.mask_version(a), "3.12.x")
        self.assertEqual(classify("3.12.15", "3.11.9"), (["unexplained"], False))  # another minor release
        self.assertEqual(classify("host 13.12.5", "host 13.12.9"), (["unexplained"], False))  # not a Python version

        # Both at once land in both classes. A mask that changes both texts alike but isn't needed is left out.
        self.assertEqual(classify(f"{a}\nRan 4 tests in 0.001s", f"{b}\nRan 4 tests in 0.5s"),
                         (["timing", "python-version"], True))
        self.assertEqual(classify(f"{a}\nRan 4 tests in 0.001s", f"{b}\nRan 4 tests in 0.001s"),
                         (["python-version"], True))

        # A one-digit change in a number that is not a time is unexplained, even next to a masked time.
        self.assertEqual(classify("total: 4242 items", "total: 4243 items"), (["unexplained"], False))
        self.assertEqual(classify("12 steps in 0.5s", "13 steps in 0.5s"), (["unexplained"], False))
        self.assertEqual(classify("rows=1234\nRan 4 tests in 0.1s", "rows=1235\nRan 4 tests in 0.2s"),
                         (["unexplained"], False))
        self.assertEqual(classify("done", "done!"), (["unexplained"], False))

    def test_classify_traceback_keeps_repl_frames(self):
        def tb(repl_line, frame):
            return ("Traceback (most recent call last):\n"
                    f'  File "<repl>", line {repl_line}, in <module>\n    read_file("missing.txt")\n'
                    f"{frame}\n    raise FileNotFoundError(path)\nFileNotFoundError: missing.txt")

        training = tb(3, '  File "/opt/rlm/repl_tools.py", line 120, in read_file')
        harness = tb(3, '  File "C:\\Users\\rishi\\rlm\\engine\\rlm_engine\\worker\\tools.py", line 77, in read_file')
        self.assertEqual(hp.classify(training, harness), (["traceback-path"], True))
        self.assertEqual(hp.classify(training, tb(3, '  File "/x/tools.py", line 9, in read_file')),
                         (["traceback-path"], True))

        # The cell's own frame is never masked: a different line number there is a real difference.
        self.assertEqual(hp.mask_traceback('File "<repl>", line 3'), 'File "<repl>", line 3')
        self.assertEqual(hp.classify(training, tb(4, '  File "/opt/rlm/repl_tools.py", line 120, in read_file')),
                         (["unexplained"], False))
        self.assertEqual(hp.classify('File "<repl>", line 3', 'File "<repl>", line 4'), (["unexplained"], False))
        self.assertEqual(hp.classify(training, tb(4, '  File "C:\\x\\tools.py", line 77, in read_file')),
                         (["unexplained"], False))
        self.assertEqual(hp.classify('File "<repl>", line 3', 'File "<repl>", line 3'), ([], True))
        # Worker frames mask to one form; the line number goes with the path.
        self.assertEqual(hp.mask_traceback('File "/opt/rlm/repl_worker.py", line 410, in run'),
                         'File "<F>", line <N>, in run')

    def test_classify_path_form(self):
        with tempfile.TemporaryDirectory() as tmp:
            native = os.path.join(tmp, "ws", "abcd1234")
            scratch = os.path.join(tmp, "home", "scratch", "run1")
            roots = [(native, bridge.EDIT_REPO_PATH), (scratch, "/scratch")]
            recorded = "cwd=/work/repo\nfile=/work/repo/src/app.py\nout=/scratch/out.txt"
            got = f"cwd={native}\nfile={native}{os.sep}src{os.sep}app.py\nout={scratch}{os.sep}out.txt"
            self.assertEqual(hp.mask_paths(got, roots), recorded)

            # Accepted when the cell's own code computed the path (os.getcwd(), __file__, ...) ...
            for code in ("import os\nprint(os.getcwd())", "print(__file__)", "print(Path('.').resolve())",
                         "print(os.path.abspath('x'))"):
                with self.subTest(code=code):
                    self.assertEqual(hp.classify(recorded, got, roots=roots, code=code), (["path-form"], True))
            # ... and reported as a tool's return value, and not accepted, when it didn't.
            for code in (None, "", "print(read_file('src/app.py'))", "print(shell('pwd'))"):
                with self.subTest(code=code):
                    self.assertEqual(hp.classify(recorded, got, roots=roots, code=code),
                                     (["path-form (tool)"], False))
            self.assertNotIn("path-form (tool)", hp.ACCEPTED)
            # Without roots the paths are just different text.
            self.assertEqual(hp.classify(recorded, got, code="print(os.getcwd())"), (["unexplained"], False))
            # A path that is not under either root is not masked.
            self.assertEqual(hp.classify(recorded, got.replace(native, "/elsewhere"), roots=roots,
                                         code="print(os.getcwd())"), (["unexplained"], False))
            # native_roots: the workspace for repo mode, always the scratch folder.
            work, home = Path(tmp) / "ws" / "abcd1234", Path(tmp) / "home"
            self.assertEqual(hp.native_roots(True, work, home, "run1"),
                             [(str(work), "/work/repo"), (str(home / "scratch" / "run1"), "/scratch")])
            self.assertEqual(hp.native_roots(False, None, home, "run1"), [(str(home / "scratch" / "run1"), "/scratch")])

        # A Windows folder, as printed raw, with forward slashes, or doubled inside a repr.
        windows = [("C:\\rlm\\ws\\abcd1234", "/work/repo")]
        for text in ("C:\\rlm\\ws\\abcd1234\\src\\a.py", "C:/rlm/ws/abcd1234/src/a.py",
                     "C:\\\\rlm\\\\ws\\\\abcd1234\\\\src\\\\a.py"):
            with self.subTest(text):
                self.assertEqual(hp.mask_paths(text, windows), "/work/repo/src/a.py")
        self.assertEqual(hp.mask_paths("path: 'C:\\rlm\\ws\\abcd1234' ok", windows), "path: '/work/repo' ok")

    def test_toolchain_gap(self):
        self.assertTrue(hp.toolchain_gap("ok", "bash: cargo: command not found"))
        self.assertTrue(hp.toolchain_gap("ok", "ModuleNotFoundError: No module named 'numpy'"))
        self.assertTrue(hp.toolchain_gap("", "'go' is not recognized as an internal or external command"))
        self.assertFalse(hp.toolchain_gap("bash: cargo: command not found", "bash: cargo: command not found"))
        self.assertFalse(hp.toolchain_gap("ok", "AssertionError"))


class EligibilityTests(unittest.TestCase):
    def test_pace_table(self):
        pace = hp.pace_state
        self.assertEqual(pace("prod-v2", {"split": "train", "step": 1}), "unknown")  # before checkpoint 1's notice
        self.assertEqual(pace("prod-v2", {"split": "train", "step": 2}), "on")
        self.assertEqual(pace("prod-v2", {"split": "train", "step": 40}), "on")
        self.assertEqual(pace("prod-v2", {"split": "val", "step": 1}), "on")  # any val transcript of prod-v2
        self.assertEqual(pace("prod-v2", {"split": "val"}), "on")
        self.assertEqual(pace("prod-v2", {"split": "train"}), "unknown")  # no step recorded
        self.assertEqual(pace("prod-v2", {"split": "train", "step": "2"}), "unknown")
        self.assertEqual(pace("prod-v2", {"split": "test", "step": 5}), "unknown")
        self.assertEqual(pace("prod-v1", {"split": "train", "step": 1}), "off")
        self.assertEqual(pace("prod-v1", {"split": "val", "step": 9}), "off")
        for other in ("prod-v3", "pilot", "prod-v2-extra", ""):
            self.assertEqual(pace(other, {"split": "val", "step": 9}), "unknown", other)

        # A record the table calls off must not carry a pacing line: the table is wrong, so setup fails (exit 2).
        edit_pace = make_record("edit_bugs", n_turns=4, pace=bridge.OBS_PACE_EDIT)
        test_pace = make_record("edit_bugs", n_turns=4, pace=bridge.OBS_PACE_TEST.format(turns_left=5))
        clean = make_record("edit_bugs", n_turns=4)
        for rec in (edit_pace, test_pace):
            with self.assertRaises(hp.SetupError):
                hp.check_pace_text("prod-v1", rec, "off")
        self.assertIsNone(hp.check_pace_text("prod-v1", clean, "off"))
        # on and unknown records may carry it; so may a model's own reply that quotes it.
        for state in ("on", "unknown"):
            self.assertIsNone(hp.check_pace_text("prod-v2", edit_pace, state))
        quoted = make_record("edit_bugs", n_turns=4)
        quoted["messages"][2]["content"] += bridge.OBS_PACE_EDIT
        self.assertIsNone(hp.check_pace_text("prod-v1", quoted, "off"))

    def test_eligibility(self):
        read, edit = make_record("aggregation"), make_record("edit_bugs")
        self.assertIsNone(hp.ineligible(read, "off"))
        self.assertIsNone(hp.ineligible(read, "unknown"))  # pace matters to edit episodes only
        for pace in ("on", "off"):
            self.assertIsNone(hp.ineligible(edit, pace))

        blocked = [{"kind": "write", "blocked": True}]
        cases = [
            ("prompt_version", read, {"prompt_version": "v3"}),
            ("infra_error", read, {"infra_error": "container died"}),
            ("hacks", read, {"hacks": ["peeked at gold"]}),
            ("flags", read, {"flags": ["suspicious"]}),
            ("events", read, {"events": blocked}),
            ("events", edit, {"events": [{"kind": "tool", "blocked": False}, {"kind": "write", "blocked": True}]}),
            ("events", edit, {"events": [{"kind": "write"}]}),  # an event without the flag was blocked
            ("events", edit, {"events": [{"kind": "write", "blocked": "True"}]}),
            ("timeouts", read, {"timeouts": 1}),
            ("crashes", read, {"crashes": 2}),
            ("restarts", read, {"restarts": 1}),
            ("no_turns", read, {"turns": []}),
            ("limits", read, {"limits": dict(read["limits"], max_turns=12)}),
            ("limits", edit, {"limits": dict(edit["limits"], cell_timeout_s=600)}),  # a task with its own timeout
            ("limits", read, {"limits": {}}),
            ("system_prompt", read, {"messages": [{"role": "system", "content": "warm-start hint"}] +
                                                 read["messages"][1:]}),
        ]
        for code, base, change in cases:
            with self.subTest(code, change=sorted(change)):
                self.assertEqual(hp.ineligible({**base, **change}, "on"), code)
        self.assertEqual(hp.ineligible(dict(read, messages=[]), "on"), "system_prompt")
        self.assertEqual(hp.ineligible(edit, "unknown"), "pace_unknown")  # an edit episode with no known pace

        # Edit-mode REPLs log every tool call and repo access as an unblocked event; those replay fine.
        for events in ([{"kind": "tool", "blocked": False}],
                       [{"kind": "tool", "blocked": False}, {"kind": "repo", "blocked": False},
                        {"kind": "program", "blocked": "False"}], None, []):
            with self.subTest(events=events):
                self.assertIsNone(hp.ineligible(dict(edit, events=events), "on"))
                self.assertIsNone(hp.ineligible(dict(read, events=events), "on"))

        # The system prompt and limits come from bridge: the records here are the ones that pass.
        self.assertEqual(read["messages"][0]["content"], bridge.system_prompt("v4"))
        self.assertEqual(edit["limits"], bridge.limits("edit_bugs", "v4").as_dict())

        # Training-split read tasks use seeds that were never recorded: skipped before anything is loaded or written.
        train_read = make_record("aggregation", split="train")
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("rlm_rl.tasks.PRIVATE_DIR", Path(tmp)), \
                mock.patch("rlm_rl.evals.eval_set", side_effect=AssertionError("eval_set must not run")) as eval_set, \
                mock.patch.object(hp.Rebuilder, "_eval_task", side_effect=AssertionError("no eval task")) as ev:
            with self.assertRaises(hp.Skip) as cm:
                hp.Rebuilder().task(train_read)
            self.assertEqual(cm.exception.code, "seed_not_recorded")
            eval_set.assert_not_called()
            ev.assert_not_called()
            self.assertEqual(os.listdir(tmp), [])  # no seed file, no folder

        # A val task whose private seed file is missing: eval_set would draw and save new seeds, so it must not run.
        val_read = make_record("aggregation", split="val")
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("rlm_rl.tasks.PRIVATE_DIR", Path(tmp)), \
                mock.patch("rlm_rl.evals.eval_set", side_effect=AssertionError("eval_set must not run")) as eval_set:
            with self.assertRaises(hp.Skip) as cm:
                hp.Rebuilder().task(val_read)
            self.assertEqual(cm.exception.code, "seed_not_recorded")
            eval_set.assert_not_called()
            self.assertEqual(os.listdir(tmp), [])

    def test_check_task(self):
        rec = make_record("aggregation")
        task = SimpleNamespace(question=QUESTION, context=CONTEXT, edit=None)
        self.assertIsNone(hp.check_task(rec, task))

        def code(task_, record=rec):
            with self.assertRaises(hp.Skip) as cm:
                hp.check_task(record, task_)
            return cm.exception.code

        self.assertEqual(code(SimpleNamespace(question=QUESTION, context=CONTEXT + "x", edit=None)),
                         "context_not_rebuildable")
        self.assertEqual(code(SimpleNamespace(question=QUESTION, context=flip(CONTEXT), edit=None)),
                         "context_not_rebuildable")
        self.assertEqual(code(SimpleNamespace(question=QUESTION + "?", context=CONTEXT, edit=None)),
                         "first_message_differs")
        erec = make_record("edit_spec")
        etask = SimpleNamespace(question=QUESTION, context=CONTEXT, edit=dict(EDIT))
        self.assertIsNone(hp.check_task(erec, etask))
        self.assertEqual(code(SimpleNamespace(question=QUESTION, context=CONTEXT, edit=dict(EDIT, test_hint="go test")),
                              erec), "first_message_differs")

    def test_selection(self):
        def cand(eid, family, task="t", subcalls=0, language=None):
            record = {"episode_id": eid, "family": family, "task_id": task, "subcalls_used": subcalls,
                      "language": language}
            return hp.Candidate("prod-v2", record, "on")

        pool = sorted([cand("e1", "aggregation", subcalls=2), cand("e2", "aggregation", "u", subcalls=9),
                       cand("e3", "research_select", subcalls=5), cand("e4", "edit_spec_go", language="go"),
                       cand("e5", "edit_spec_py", language="python"), cand("e6", "edit_bugs", language="java"),
                       cand("e7", "edit_bugs", "a", language="python")], key=lambda c: c.order)
        self.assertEqual([c.id for c in pool], ["e1", "e2", "e7", "e6", "e4", "e5", "e3"])  # family, task_id, id
        self.assertEqual([c.edit for c in pool], [False, False, True, True, True, True, False])

        plan = hp.slots(3, 2)
        self.assertEqual([s.name for s in plan], ["aggregation", "research_select", "read with the most sub-calls",
                                                  "edit_spec*", "edit_bugs*"])
        ids = {s.name: [c.id for c in s.pick(pool)] for s in plan}
        self.assertEqual(ids["aggregation"], ["e1", "e2"])
        self.assertEqual(ids["research_select"], ["e3"])
        self.assertEqual(ids["read with the most sub-calls"], ["e2", "e3", "e1"])  # the llm_batch episode first
        self.assertEqual(ids["edit_spec*"], ["e5", "e4"])  # python first
        self.assertEqual(ids["edit_bugs*"], ["e7", "e6"])
        self.assertEqual([s.name for s in hp.slots(4, 3)][-4:], ["any read", "edit_spec*", "edit_bugs*", "any edit"])
        self.assertEqual([s.name for s in hp.slots(1, 0)], ["aggregation"])
        self.assertEqual(hp.slots(0, 0), [])
        pinned = hp.slots(3, 2, ["e6", "e3"])
        self.assertEqual([s.name for s in pinned], ["e6", "e3"])
        self.assertEqual([[c.id for c in s.pick(pool)] for s in pinned], [["e6"], ["e3"]])


class ReportTests(unittest.TestCase):
    HEADER = {"date": "2026-10-31", "harness_commit": "abc123 dirty", "rlm_commit": "def456",
              "engine": "0.1.0, python 3.12.3", "os": "Linux-6", "git": "git version 2.45.0", "shell": "sh",
              "permission_mode": "accept-edits", "digest": "0123456789ab"}

    def results(self):
        timing = diff(accepted=True, classes=["timing"], recorded=f"Question: {QUESTION}\nRan 4 tests in 0.1s",
                      got=f"Question: {QUESTION}\nRan 4 tests in 0.2s")
        return [result("e1", "aggregation", [timing], requests=5, matched=4),
                result("e2", "edit_bugs", requests=7, matched=7, approvals={"write": 2})]

    def test_unified_diff_shows_line_ending_differences(self):
        """A CRLF or final-newline difference (the spec's Windows case) is visible in the report's diff."""
        self.assertIn("+a\\r\\n", hp.unified("a\nb\n", "a\r\nb\r\n").splitlines())
        self.assertIn("-b [no newline at end]", hp.unified("a\nb", "a\nb\n").splitlines())
        self.assertEqual(hp.unified("x\n", "x\n"), "")

    def test_turn_code(self):
        """An observation's message index maps to the code of the turn that printed it."""
        turns = [{"code": "a"}, {"code": "b"}, {"code": None}]
        self.assertEqual([hp.turn_code(turns, i) for i in (0, 1, 2, 3, 4, 5, 7, 9)],
                         [None, None, None, "a", None, "b", None, None])

    def test_exit_codes(self):
        passing = [result("a"), result("b", "edit_spec")]
        for r in passing:
            self.assertEqual(r.verdict(), "PASS")
        self.assertEqual(hp.exit_code(passing, 2, strict=False), 0)
        self.assertEqual(hp.exit_code(passing, 1, strict=False), 0)  # more than asked is fine

        # One unexplained difference.
        bad = [result("a"), result("b", differences=[diff(accepted=False, classes=["unexplained"])])]
        self.assertEqual((bad[1].unexplained, bad[1].verdict()), (1, "FAIL"))
        self.assertEqual(hp.exit_code(bad, 2, strict=False), 1)

        # An accepted class passes, and --strict turns it into a failure (the byte-for-byte count on its own).
        accepted = [result("a", differences=[diff(accepted=True, classes=["timing"])]), result("b")]
        self.assertEqual(accepted[0].verdict(), "PASS")
        self.assertEqual(hp.exit_code(accepted, 2, strict=False), 0)
        self.assertEqual(accepted[0].verdict(strict=True), "FAIL")
        self.assertEqual(hp.exit_code(accepted, 2, strict=True), 1)
        self.assertEqual(hp.exit_code(passing, 2, strict=True), 0)  # nothing to be strict about

        # Anything else that went wrong in an episode is exit 1 too.
        for key, value in (("failures", ["extra_root_call: ..."]), ("subcalls_unmatched", [{"chars": 1}]),
                           ("subcalls_unused", 1)):
            with self.subTest(key):
                self.assertEqual(hp.exit_code([result("a", **{key: value}), result("b")], 2, strict=False), 1)

        # Fewer kept results than asked is a setup failure, exit 2; a replaced (toolchain) result is not kept.
        self.assertEqual(hp.exit_code(passing[:1], 2, strict=False), 2)
        self.assertEqual(hp.exit_code([], 2, strict=False), 2)
        replaced = [result("a"), result("b", toolchain=True)]
        self.assertEqual(replaced[1].verdict(), "replaced(toolchain)")
        self.assertEqual(hp.exit_code(replaced, 2, strict=False), 2)
        self.assertEqual(hp.exit_code(replaced, 1, strict=False), 0)
        both_replaced = [result("a"), result("b", toolchain=True, failures=["x"]), result("c", toolchain=True,
                         differences=[diff()])]
        self.assertEqual(hp.exit_code(both_replaced, 1, strict=True), 0)  # a replaced episode's failures don't count
        self.assertEqual(hp.exit_code(both_replaced + [result("d")], 2, strict=False), 0)

        # The engine died: exit 1, whatever else holds.
        self.assertEqual(hp.exit_code(passing, 2, strict=False, engine_died=True), 1)
        self.assertEqual(hp.exit_code([], 2, strict=False, engine_died=True), 1)

    def test_counters_header_and_digest(self):
        mixed = [diff(), diff(accepted=True, classes=["timing", "python-version"])]
        results = self.results() + [result("e3", differences=mixed, requests=3, matched=1,
                                           subcalls_unmatched=[{"chars": 5}], subcalls_unused=2,
                                           failures=["extra_root_call: root call 4"]),
                                    result("e4", toolchain=True, requests=9, matched=0)]
        c = hp.counters(results, {"e4": "toolchain"}, collections.Counter({"events": 3, "limits": 1}), strict=False)
        self.assertEqual((c["episodes"], c["read"], c["edit"]), (3, 2, 1))  # the replaced episode isn't counted
        self.assertEqual((c["requests"], c["matched"]), (15, 12))
        self.assertEqual(c["accepted"], {"python-version": 1, "timing": 2})
        self.assertEqual(c["unexplained"], 1)
        self.assertEqual(c["failures"], {"extra_root_call": 1, "subcall_unmatched": 1, "subcall_unused": 2})
        self.assertEqual((c["replaced"], c["skipped"]), ({"e4": "toolchain"}, {"events": 3, "limits": 1}))

        lines = hp.header_lines(self.HEADER, c, "FAIL")
        for want in ("date: 2026-10-31", "harness commit: abc123 dirty", "RLM commit: def456",
                     "engine: 0.1.0, python 3.12.3", "OS: Linux-6", "git: git version 2.45.0", "shell: sh",
                     "permission mode: accept-edits", "episodes: 3 (read 2, edit 1)", "requests compared: 15",
                     "byte-for-byte requests: 12 of 15", "accepted differences: python-version: 1, timing: 2",
                     "unexplained differences: 1", "replaced: e4: toolchain", "skipped: events: 3, limits: 1",
                     "verdict: FAIL"):
            self.assertIn(want, lines)
        empty = hp.counters([], {}, collections.Counter(), strict=False)
        lines = hp.header_lines(self.HEADER, empty, "PASS")
        for want in ("episodes: 0 (read 0, edit 0)", "accepted differences: none", "replaced: none", "skipped: none",
                     "verdict: PASS"):
            self.assertIn(want, lines)

        # The report holds the diffs, with the class and whether it was accepted; the JSON form dumps.
        report = hp.report_md(self.HEADER, c, results, "FAIL", False)
        self.assertIn("### message 3 (user), first in request 2: timing, accepted", report)
        self.assertIn("-Ran 4 tests in 0.1s", report)
        self.assertIn("| timing | yes |", report)
        self.assertIn("| unexplained | no | not in the ACCEPTED table |", report)
        json.dumps([r.as_dict() for r in results])

        # The digest follows verdicts, requests compared, unexplained differences and failures: what two runs must
        # agree on. It leaves out times, the byte-for-byte count and the accepted classes, which can vary run to run.
        again = copy.deepcopy(results)
        again[0].seconds = 99.9
        again[0].matched -= 1
        again[0].differences[0].classes = ["timing", "python-version"]
        self.assertEqual(hp.digest(results, False), hp.digest(again, False))
        for change in (lambda r: setattr(r, "requests", r.requests - 1), lambda r: r.differences.append(diff()),
                       lambda r: r.failures.append("extra_root_call: 4"), lambda r: setattr(r, "subcalls_unused", 1),
                       lambda r: r.subcalls_unmatched.append({}), lambda r: setattr(r, "toolchain", True)):
            changed = copy.deepcopy(results)
            change(changed[0])
            self.assertNotEqual(hp.digest(results, False), hp.digest(changed, False))
        self.assertNotEqual(hp.digest(results, False), hp.digest(results, True))  # --strict fails the accepted timing

    def test_summary_guard(self):
        results = self.results()
        c = hp.counters(results, {}, collections.Counter(), strict=False)
        summary = hp.summary_md(self.HEADER, c, results, "PASS", False)
        first_line = CONTEXT.splitlines()[0]

        # The summary has counters, ids, families and class names; the diffs (which hold the question) stay out.
        self.assertIn("every other one differs only in the accepted classes below", summary)
        self.assertNotIn("this run failed", summary)
        failing = results + [result("e9", differences=[diff(recorded=QUESTION)])]
        failed = hp.summary_md(self.HEADER, hp.counters(failing, {}, collections.Counter(), strict=False), failing,
                               "FAIL", False)
        self.assertIn("1 differences are unexplained and the failures above stand, so this run failed", failed)
        self.assertNotIn("every other one differs only", failed)
        self.assertNotIn(QUESTION[:60], failed)  # an unexplained difference's text stays out of the summary too
        self.assertIn("11 of 12 root requests matched byte for byte", summary)
        self.assertIn("| e1 | prod-v2 | aggregation | read |", summary)
        self.assertIn("| timing | 1 | yes |", summary)
        self.assertIn("## Engine fixes", summary)
        self.assertNotIn(QUESTION[:60], summary)
        self.assertNotIn(first_line[:60], summary)
        self.assertEqual(hp.guard(summary, [(QUESTION, CONTEXT)]), [])
        # ... while the full report does hold it, and the guard sees that.
        report = hp.report_md(self.HEADER, c, results, "PASS", False)
        self.assertEqual(len(hp.guard(report, [(QUESTION, CONTEXT)])), 1)

        # A summary containing a question's first 60 characters is refused, as is a first context line's.
        leaks = hp.guard(summary + f"\nnote: {QUESTION[:60]}\n", [(QUESTION, CONTEXT)])
        self.assertEqual(len(leaks), 1)
        self.assertTrue(leaks[0].startswith("question:"), leaks)
        leaks = hp.guard(summary + f"\nnote: {first_line[:60]}\n", [(QUESTION, CONTEXT)])
        self.assertEqual(len(leaks), 1)
        self.assertTrue(leaks[0].startswith("first context line:"), leaks)
        both = hp.guard(f"{QUESTION[:60]} {first_line[:60]}", [(QUESTION, CONTEXT)])
        self.assertEqual(len(both), 2)

        # Any episode in the list counts; needles under 10 characters aren't used.
        other = ("What is the capital of the country named in the second paragraph?", "Paragraph one is about trade.\n")
        self.assertEqual(len(hp.guard(f"{summary}\n{other[0][:60]}", [(QUESTION, CONTEXT), other])), 1)
        self.assertEqual(hp.guard("Why? Why? {", [("Why?", "{\nx")]), [])
        self.assertEqual(hp.guard("anything", [("", ""), (None, "")]), [])
        # Leading blank lines in a context leave no first line to match.
        self.assertEqual(hp.guard(summary, [(QUESTION, "\nsecond")]), [])
        # Only the first 60 characters count: the rest of a long question is not part of the needle.
        self.assertEqual(hp.guard(f"{summary} {QUESTION[:59]}", [(QUESTION, CONTEXT)]), [])

    def test_check_end(self):
        rec = make_record("aggregation", n_turns=3, subcalls_used=2, answer="42")
        lim = rec["limits"]
        started = {"limits": {"max_turns": lim["max_turns"], "max_subcalls": lim["max_subcalls"], "extra": 1},
                   "input": {"chars": len(CONTEXT), "sha256": rec["context_sha256"],
                             "lines": len(CONTEXT.splitlines())}}
        finished = {"turns": 3, "submitted": True, "answer": "42", "subcalls": 2}
        self.assertEqual(hp.check_end(rec, CONTEXT, started, finished), [])

        self.assertEqual(hp.check_end(rec, CONTEXT, started, None), ["no_run_finished: the run never finished"])
        self.assertEqual(hp.check_end(rec, CONTEXT, None, finished)[0].split(":")[0], "no_run_started")
        out = hp.check_end(rec, CONTEXT, started, dict(finished, turns=4))
        self.assertEqual(out, ["end_mismatch: run.finished turns 4, training 3"])
        self.assertEqual(len(hp.check_end(rec, CONTEXT, started, dict(finished, answer="43", subcalls=3))), 2)
        self.assertEqual(len(hp.check_end(rec, CONTEXT, started, dict(finished, submitted=False))), 1)
        out = hp.check_end(rec, CONTEXT, started, dict(finished, infra_error="boom"))
        self.assertEqual(out, ["infra_error: boom"])
        bad_input = dict(started, input=dict(started["input"], lines=len(CONTEXT.splitlines()) + 1, chars=1))
        self.assertEqual(len(hp.check_end(rec, CONTEXT, bad_input, finished)), 2)
        bad_limits = dict(started, limits={"max_turns": lim["max_turns"] + 1})
        self.assertEqual(len(hp.check_end(rec, CONTEXT, bad_limits, finished)), 1)

    def test_subcall_params_and_cell_locators(self):
        read, edit = make_record("aggregation")["limits"], make_record("edit_bugs")["limits"]

        def request(lim, **over):
            params = {"temperature": bridge.SUBCALL_TEMPERATURE, "max_tokens": lim["subcall_max_tokens"],
                      "thinking": lim.get("subcall_thinking") or "off", "tools": [],
                      "messages": [{"role": "user", "content": "p"}]}
            params.update(over)
            return params

        self.assertEqual(hp.check_subcall_params([request(read), request(read)], read), [])
        self.assertEqual(hp.check_subcall_params([request(edit)], edit), [])  # edit sub-calls think "low"
        self.assertEqual(request(edit)["thinking"], "low")
        self.assertEqual(hp.check_subcall_params([request(read, thinking="low")], read),
                         ["subcall_param: thinking 'low', expected 'off' (x1)"])
        self.assertEqual(hp.check_subcall_params([request(read, temperature=1.0)] * 2, read),
                         ["subcall_param: temperature 1.0, expected 0.3 (x2)"])
        self.assertEqual(hp.check_subcall_params([request(read, messages=[{"role": "user", "content": "a"},
                                                                          {"role": "user", "content": "b"}])], read),
                         ["subcall_param: messages are not one user message (x1)"])
        self.assertEqual(hp.check_subcall_params([request(read, messages=[{"role": "system", "content": "a"}])], read),
                         ["subcall_param: messages are not one user message (x1)"])
        self.assertEqual(hp.check_subcall_params([request(read, tools=[{"name": "python"}])], read),
                         ["subcall_param: tools [{'name': 'python'}], expected [] (x1)"])

        turns = [{"code": "print(1)", "cell": {"error": None, "restarted": False}, "observation": "obs 1"},
                 {"code": None, "cell": None, "observation": "obs 2"}]
        same = {"turn": 1, "code": "print(1)", "error": None, "observation": "obs 1", "restarted": False}
        no_code = {"turn": 2, "code": None, "error": bridge.no_code_error("v4"), "observation": "obs 2",
                   "restarted": False}
        self.assertEqual(hp.cell_locators(turns, [same, no_code]), [])
        self.assertEqual(hp.cell_locators(turns, [dict(same, observation="x", restarted=True)]),
                         [{"turn": 1, "fields": ["observation", "restarted"]}])
        self.assertEqual(hp.cell_locators(turns, [dict(no_code, error=None)]), [{"turn": 2, "fields": ["error"]}])
        self.assertEqual(hp.cell_locators(turns, [dict(same, turn=3), dict(same, turn=None)]),
                         [{"turn": 3, "fields": ["no such recorded turn"]},
                          {"turn": None, "fields": ["no such recorded turn"]}])


class RunLogTests(unittest.TestCase):
    def test_runlog_info(self):
        rec = make_record("edit_bugs", diff="diff --git a/f b/f\nindex 1234567890abcdef..fedcba0987654321 100644\n+x\n")
        short = "diff --git a/f b/f\nindex 1234567..fedcba0 100644\n+x\n"
        self.assertEqual(hp.short_index("index 1234567890abcdef..fedcba0987654321 100644"),
                         "index 1234567..fedcba0 100644")

        def info(logged, record=rec, run_id="r1", finish=True):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "runs.jsonl"
                lines = [json.dumps({"run_id": "other", "messages": []}), json.dumps(dict(logged, run_id=run_id))]
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                return hp.runlog_info({"log_path": str(path)} if finish else None, run_id, record)

        out = info({"messages": rec["messages"][:5], "diff": rec["diff"]})
        self.assertEqual(out, {"runlog_messages": "prefix", "diff": "equal"})
        out = info({"messages": rec["messages"][:5], "diff": short})  # training graded a --full-index diff
        self.assertEqual(out["diff"], "equal apart from index abbreviation")
        out = info({"messages": [dict(rec["messages"][0], content="x")], "diff": rec["diff"] + "+y\n"})
        self.assertEqual(out, {"runlog_messages": "differs", "diff": "differs"})
        read = make_record("aggregation")  # a read episode has no diff to compare
        self.assertEqual(info({"messages": read["messages"]}, record=read), {"runlog_messages": "prefix"})
        self.assertTrue(info({}, finish=False)["runlog"].startswith("unreadable"))  # never gates; just says so


class WorkspaceTests(unittest.TestCase):
    def test_listed_files_and_fidelity(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "src").mkdir()
            (folder / "src" / "app.py").write_bytes(b"x" * 12)
            (folder / "README.md").write_bytes(b"y" * 3401)
            (folder / "src" / "naïve.py").write_bytes("héllo".encode("utf-8"))  # 5 characters, 6 bytes
            (folder / "src" / "odd  name.txt").write_bytes(b"zz")

            bytes_ctx = ("The issue.\n\nRepository files (path, size in bytes):\n"
                         "src/app.py  12\nREADME.md  3,401\nsrc/naïve.py  6\nsrc/odd  name.txt  2\n")
            listed, unit = hp.listed_files(bytes_ctx)
            self.assertEqual((listed, unit), ([("src/app.py", 12), ("README.md", 3401), ("src/naïve.py", 6),
                                               ("src/odd  name.txt", 2)], "bytes"))
            self.assertEqual(hp.fidelity(bytes_ctx, folder), (4, 4))

            chars_ctx = ("Another issue.\n\nRepository files (path, size in characters):\n"
                         "src/app.py  12\nsrc/naïve.py  5\n")
            self.assertEqual(hp.listed_files(chars_ctx)[1], "characters")
            self.assertEqual(hp.fidelity(chars_ctx, folder), (2, 2))  # 5 characters, though 6 bytes
            self.assertEqual(hp.fidelity(chars_ctx.replace("naïve.py  5", "naïve.py  6"), folder), (1, 2))
            self.assertEqual(hp.fidelity(bytes_ctx.replace("naïve.py  6", "naïve.py  5"), folder), (3, 4))

            # A wrong size, a missing file, a directory in a file's place and a symlink-free miss are not matches.
            wrong = ("x\n\nRepository files (path, size in bytes):\nsrc/app.py  13\nmissing.py  1\nsrc  4\n"
                     "README.md  3401\n")
            self.assertEqual(hp.fidelity(wrong, folder), (1, 4))

            # No listing, or one with no parsable line, lists nothing; the last listing is the one used.
            self.assertEqual(hp.listed_files("just an issue"), ([], "bytes"))
            self.assertEqual(hp.fidelity("just an issue", folder), (0, 0))
            self.assertEqual(hp.listed_files("Repository files (path, size in bytes):\nnot a listing line\n")[0], [])
            twice = bytes_ctx + "\n" + chars_ctx
            self.assertEqual(hp.listed_files(twice)[1], "characters")
            self.assertEqual(len(hp.listed_files(twice)[0]), 2)

    def test_materialise_checks_fidelity_before_git(self):
        names = [f"f{i:03d}.txt" for i in range(100)]
        context = "issue\n\nRepository files (path, size in bytes):\n" + "".join(f"{n}  3\n" for n in names)

        class Workspace:
            def __init__(self, write=100, boom=None, size=3):
                self.write, self.boom, self.size = write, boom, size

            def build(self, folder):
                if self.boom:
                    raise self.boom
                for name in names[:self.write]:
                    (folder / name).write_bytes(b"a" * self.size)

        def build(workspace):
            task = SimpleNamespace(context=context, workspace=workspace)
            with tempfile.TemporaryDirectory() as tmp, mock.patch.object(hp, "commit_base") as commit:
                folder = Path(tmp) / "ws" / "abcd1234"
                try:
                    hp.materialise(task, folder)
                except hp.Skip as skip:
                    return skip.code, skip.detail, commit.call_count
                return "built", "", commit.call_count

        self.assertEqual(build(Workspace()), ("built", "", 1))  # every file there: the base commit is made
        self.assertEqual(build(Workspace(write=99))[0], "built")  # 99% is enough
        code, detail, commits = build(Workspace(write=98))
        self.assertEqual((code, commits), ("workspace_mismatch", 0))
        self.assertIn("98 of 100", detail)
        self.assertEqual(build(Workspace(size=4))[0], "workspace_mismatch")  # every size wrong (CRLF, say)
        code, detail, commits = build(Workspace(boom=OSError("disk full")))  # build() raising is the same skip
        self.assertEqual((code, commits), ("workspace_mismatch", 0))
        self.assertIn("disk full", detail)

        # A context that lists nothing can't be checked, so the workspace is not trusted.
        task = SimpleNamespace(context="no listing here", workspace=Workspace())
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(hp, "commit_base") as commit:
            with self.assertRaises(hp.Skip) as cm:
                hp.materialise(task, Path(tmp) / "ws")
            self.assertEqual(cm.exception.code, "workspace_mismatch")
            commit.assert_not_called()

    def test_remove_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            tree = base / "ws"
            (tree / ".git" / "objects").mkdir(parents=True)
            locked = tree / ".git" / "objects" / "pack.idx"
            locked.write_bytes(b"x")
            os.chmod(locked, 0o444)  # read-only, as git writes its objects
            hp.remove_tree(tree)
            self.assertFalse(tree.exists())
            hp.remove_tree(base / "never-existed")  # nothing to do, and no error


class EngineLinkTests(unittest.TestCase):
    def test_split_command(self):
        split = hp.split_command
        self.assertEqual(split("uv run --project F:\\rlm\\engine python"), ["uv", "run", "--project", "F:\\rlm\\engine",
                                                                            "python"])  # backslashes are plain
        self.assertEqual(split('uv run --project "C:\\Program Files\\rlm\\engine" python -X utf8'),
                         ["uv", "run", "--project", "C:\\Program Files\\rlm\\engine", "python", "-X", "utf8"])
        self.assertEqual(split("a 'b c' d"), ["a", "b c", "d"])
        self.assertEqual(split("  a \t b\n"), ["a", "b"])
        self.assertEqual(split(""), [])
        self.assertEqual(split("   "), [])
        self.assertEqual(split('a "" b'), ["a", "", "b"])  # an empty quoted word is a word
        self.assertEqual(split('a"b c"d e'), ["ab cd", "e"])  # quotes join to the word around them
        self.assertEqual(split("""x "it's" 'say "hi"'"""), ["x", "it's", 'say "hi"'])
        for bad in ('a "b c', "a 'b", 'a "b\''):
            with self.subTest(bad):
                with self.assertRaises(hp.SetupError):
                    split(bad)

    def test_engine_env(self):
        base = {**FAKE_ENV, "OPENAI_API_KEY": "sk-FAKE-1", "openai_api_key": "sk-FAKE-2", "GITHUB_TOKEN": "ghp_FAKE",
                "ANTHROPIC_AUTH_TOKEN": "FAKE", "DB_PASSWORD": "FAKE", "AWS_SECRET_ACCESS_KEY": "FAKE",
                "GOOGLE_CREDENTIALS": "FAKE", "Hf_Token": "FAKE",
                "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "core.hooksPath", "GIT_CONFIG_VALUE_0": "/tmp/hooks",
                "GIT_CONFIG_COUNT_EXTRA": "kept", "GIT_TERMINAL_PROMPT": "kept", "RLMH_HOME": "/somewhere/else",
                "PYTHONUTF8": "0", "EDITOR": "vim"}
        before = dict(base)
        home = Path("/work/home")
        env = hp.engine_env(home, base)
        self.assertEqual(base, before)  # the parent's environment is not changed
        for name in ("OPENAI_API_KEY", "openai_api_key", "GITHUB_TOKEN", "ANTHROPIC_AUTH_TOKEN", "DB_PASSWORD",
                     "AWS_SECRET_ACCESS_KEY", "GOOGLE_CREDENTIALS", "Hf_Token", "GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0",
                     "GIT_CONFIG_VALUE_0"):
            self.assertNotIn(name, env, name)
        self.assertEqual(env["RLMH_HOME"], str(home))
        self.assertEqual(env["PYTHONUTF8"], "1")
        self.assertEqual(env["PYTHONUNBUFFERED"], "1")
        for name, value in FAKE_ENV.items():
            self.assertEqual(env[name], value)
        self.assertEqual((env["EDITOR"], env["GIT_TERMINAL_PROMPT"], env["GIT_CONFIG_COUNT_EXTRA"]),
                         ("vim", "kept", "kept"))
        # Nothing else changed: the keys are the kept ones plus the three the script sets.
        self.assertEqual(set(env) - set(base), {"PYTHONUNBUFFERED"})

        # With no base it reads the process environment.
        with mock.patch.dict(os.environ, {"MY_API_KEY": "FAKE", "GIT_CONFIG_COUNT": "2", "KEPT_VAR": "yes"}):
            env = hp.engine_env(home)
        self.assertNotIn("MY_API_KEY", env)
        self.assertNotIn("GIT_CONFIG_COUNT", env)
        self.assertEqual((env["KEPT_VAR"], env["RLMH_HOME"]), ("yes", str(home)))

    def test_check_info(self):
        th = hp.TRAINING_HASH
        self.assertEqual(th, bridge.PROMPT_HASHES["v4"])  # the literal in the script is training's
        profiles = (hp.PROFILE, hp.PROFILE_NOPACE)
        info = {"protocol": 2, "training_hash": th, "version": "0.1.0", "python": "3.12.3",
                "profiles": [{"name": "training-v4", "prompt_hash": th},
                             {"name": "training-v4-nopace", "prompt_hash": th},
                             {"name": "default", "prompt_hash": "0" * 64}]}
        self.assertIsNone(hp.check_info(info, profiles))
        bad = [dict(info, protocol=1), dict(info, protocol=None), dict(info, training_hash="0" * 64),
               {k: v for k, v in info.items() if k != "training_hash"},
               dict(info, profiles=info["profiles"][:1]), dict(info, profiles=info["profiles"][1:]),
               dict(info, profiles=None),
               dict(info, profiles=[info["profiles"][0], dict(info["profiles"][1], prompt_hash="f" * 64)]),
               dict(info, profiles=[dict(info["profiles"][0], prompt_hash=None), info["profiles"][1]])]
        for i, broken in enumerate(bad):
            with self.subTest(i):
                with self.assertRaises(hp.SetupError):
                    hp.check_info(broken, profiles)

    def test_write_nopace_profile(self):
        source_text = ('contract = "v4"\nguidance = ""\n\n[repo]\nmax_turns = 30\n'
                       "pace_nudge = true             # training's prod-v2 runs used the pacing line\n")
        with tempfile.TemporaryDirectory() as tmp:
            harness, home = Path(tmp) / "harness", Path(tmp) / "home"
            profile = harness / "engine" / "rlm_engine" / "profiles" / "training-v4" / "profile.toml"
            with self.assertRaises(hp.SetupError):  # no profile there
                hp.write_nopace_profile(harness, home)
            profile.parent.mkdir(parents=True)
            profile.write_bytes(source_text.encode("utf-8"))
            hp.write_nopace_profile(harness, home)
            out = (home / "profiles" / "training-v4-nopace" / "profile.toml").read_bytes().decode("utf-8")
            self.assertEqual(out, source_text.replace("pace_nudge = true", "pace_nudge = false"))
            self.assertEqual(out.count("pace_nudge"), 1)
            self.assertNotIn("\r", out)
            self.assertEqual(profile.read_bytes().decode("utf-8"), source_text)  # the built-in is not touched

            for broken in (source_text.replace("pace_nudge = true", "pace_nudge = false"),
                           source_text + "pace_nudge = true\n"):  # none, and two
                profile.write_bytes(broken.encode("utf-8"))
                with self.assertRaises(hp.SetupError):
                    hp.write_nopace_profile(harness, home)

    def test_parse_args(self):
        args = hp.parse_args(["--harness", "somewhere"])
        self.assertEqual((args.read, args.edit, args.permission_mode, args.strict, args.write_summary),
                         (3, 2, "accept-edits", False, False))
        self.assertEqual((args.runs, args.episodes, args.episode_timeout_s), ("prod-v2,prod-v1", "", 1800))
        args = hp.parse_args(["--harness", "h", "--read", "0", "--edit", "1", "--strict", "--date", "2026-10-28",
                              "--permission-mode", "bypass", "--episodes", "a,b"])
        self.assertEqual((args.read, args.edit, args.strict, args.date, args.permission_mode, args.episodes),
                         (0, 1, True, "2026-10-28", "bypass", "a,b"))
        for argv in ([], ["--harness", "h", "--date", "2026-02-30"], ["--harness", "h", "--date", "yesterday"],
                     ["--harness", "h", "--read", "-1"], ["--harness", "h", "--edit", "-2"],
                     ["--harness", "h", "--episode-timeout-s", "0"], ["--harness", "h", "--permission-mode", "yolo"]):
            with self.subTest(argv):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
                    hp.parse_args(argv)
                self.assertEqual(cm.exception.code, 2)  # bad flags are a setup failure


if __name__ == "__main__":
    unittest.main()
