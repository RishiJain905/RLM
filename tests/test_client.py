"""H6 and the request half of H5: one shared limiter, pacing, retries, and infra errors instead of fake answers.

Runs against tests/fake_model.py, a local OpenAI-compatible server that measures requests in
flight on its own side and answers 429 above 20 per model name, like W&B. No W&B call is made.

    python3 -m unittest -v tests.test_client
"""

import asyncio
import collections
import json
import random
import time
import unittest
from pathlib import Path

import openai

from rlm_rl import bridge, client as rclient
from rlm_rl.bridge import SubcallError
from rlm_rl.client import Limiter, ModelClient, Pacer, RequestFailed, RetryPolicy, Usage
from tests.fake_model import AsyncCase, FakeModelServer

FAST = RetryPolicy(tries=6, base_s=0.01, max_s=0.05, timeout_s=10)
ROOT, BASE = "fake-model:step3", "fake-model:step0"
MESSAGES = [{"role": "system", "content": bridge.SYSTEM_PROMPT}, {"role": "user", "content": "Question: q"}]


class ServerCase(AsyncCase):
    async def asyncSetUp(self):
        self.server = FakeModelServer()
        await self.server.start()
        self.openai = openai.AsyncOpenAI(base_url=self.server.base_url, api_key="test-key")

    async def asyncTearDown(self):
        await self.openai.close()
        await self.server.close()

    def client(self, retry=FAST, **kw) -> ModelClient:
        return ModelClient(self.openai, subcall_model=BASE, retry=retry, rng=random.Random(0), **kw)


class LimiterTest(AsyncCase):
    async def test_caps_in_total_and_per_model(self):
        limiter = Limiter(total=5, per_model=3)
        running = {"a": 0, "b": 0, "all": 0}
        peaks = {"a": 0, "b": 0, "all": 0}

        async def job(model):
            async with limiter.slot(model):
                running[model] += 1
                running["all"] += 1
                for k in (model, "all"):
                    peaks[k] = max(peaks[k], running[k])
                await asyncio.sleep(random.uniform(0, 0.01))
                running[model] -= 1
                running["all"] -= 1

        await asyncio.gather(*(job(m) for m in "ab" * 40))
        self.assertEqual(peaks, {"a": 3, "b": 3, "all": 5})
        self.assertEqual(limiter.peaks["total"], 5)
        self.assertEqual(limiter.peaks["by_model"], {"a": 3, "b": 3})
        self.assertEqual((limiter.in_flight, limiter.waiting, dict(limiter.by_model)), (0, 0, {}))

    async def test_one_model_at_its_cap_does_not_block_another(self):
        limiter = Limiter(total=4, per_model=2)
        for _ in range(2):
            await limiter.acquire("a")
        blocked = asyncio.ensure_future(limiter.acquire("a"))
        await asyncio.sleep(0)
        self.assertFalse(blocked.done())
        await asyncio.wait_for(limiter.acquire("b"), 1)  # gets in ahead of the waiting "a"
        limiter.release("a")
        await asyncio.wait_for(blocked, 1)
        self.assertEqual(dict(limiter.by_model), {"a": 2, "b": 1})

    async def test_first_come_first_served(self):
        limiter = Limiter(total=1, per_model=1)
        await limiter.acquire("a")
        order = []

        async def wait(i):
            await limiter.acquire("a")
            order.append(i)
            limiter.release("a")

        waiters = [asyncio.ensure_future(wait(i)) for i in range(5)]
        await asyncio.sleep(0)
        limiter.release("a")
        await asyncio.gather(*waiters)
        self.assertEqual(order, [0, 1, 2, 3, 4])

    async def test_cancelled_waiters_leak_no_slot(self):
        limiter = Limiter(total=1, per_model=1)
        await limiter.acquire("a")
        waiter = asyncio.ensure_future(limiter.acquire("a"))
        await asyncio.sleep(0)
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        # Granted and cancelled in the same tick: the slot goes back.
        granted = asyncio.ensure_future(limiter.acquire("a"))
        await asyncio.sleep(0)
        limiter.release("a")
        granted.cancel()
        await asyncio.gather(granted, return_exceptions=True)
        self.assertEqual((limiter.in_flight, limiter.waiting), (0, 0))
        await asyncio.wait_for(limiter.acquire("a"), 1)

    async def test_a_model_with_a_cap_of_its_own(self):
        """The sub-call model's cap holds while another model still gets the per-model cap (step 10 §6)."""
        limiter = Limiter(total=6, per_model=4, model_caps={"sub": 2})
        self.assertEqual((limiter.cap("sub"), limiter.cap("root")), (2, 4))
        running, peaks = collections.Counter(), collections.Counter()

        async def job(model):
            async with limiter.slot(model):
                running[model] += 1
                running["all"] += 1
                for k in (model, "all"):
                    peaks[k] = max(peaks[k], running[k])
                await asyncio.sleep(random.uniform(0, 0.01))
                running[model] -= 1
                running["all"] -= 1

        await asyncio.gather(*(job(m) for m in ("sub", "root") * 40))
        self.assertEqual(dict(peaks), {"sub": 2, "root": 4, "all": 6})
        self.assertEqual(limiter.peaks["by_model"], {"sub": 2, "root": 4})

        # A sub-call blocked only by its own cap doesn't hold up the other model's requests.
        limiter = Limiter(total=6, per_model=4, model_caps={"sub": 2})
        for _ in range(2):
            await limiter.acquire("sub")
        blocked = asyncio.ensure_future(limiter.acquire("sub"))
        await asyncio.sleep(0)
        self.assertFalse(blocked.done())
        for _ in range(4):
            await asyncio.wait_for(limiter.acquire("root"), 1)
        limiter.release("sub")
        await asyncio.wait_for(blocked, 1)
        self.assertEqual(dict(limiter.by_model), {"sub": 2, "root": 4})

    def test_a_model_cap_above_the_total_is_refused(self):
        with self.assertRaises(ValueError):
            Limiter(total=5, per_model=3, model_caps={"sub": 6})
        with self.assertRaises(ValueError):
            Limiter(total=5, per_model=3, model_caps={"sub": 0})
        self.assertEqual(Limiter(total=5, per_model=3, model_caps={"sub": 5}).cap("sub"), 5)

    def test_retry_delays_back_off_with_jitter(self):
        policy, rng = RetryPolicy(), random.Random(1)
        for attempt in range(7):
            base = min(120, 4 * 2 ** attempt)
            for _ in range(50):
                self.assertTrue(0.5 * base <= policy.delay(attempt, rng) <= 1.5 * base)
        self.assertGreaterEqual(policy.delay(0, rng, retry_after=30), 30)  # Retry-After is honoured
        self.assertGreaterEqual(policy.delay(0, rng, retry_after=999), 120)  # up to the cap
        self.assertLessEqual(policy.delay(9, rng, retry_after=999), 180)

    def test_spec_values(self):
        self.assertEqual((rclient.MAX_IN_FLIGHT, rclient.MAX_IN_FLIGHT_PER_MODEL, rclient.MAX_SUBCALLS_IN_FLIGHT),
                         (24, 20, 12))
        self.assertEqual(rclient.SUBCALL_START_GAP_S, 0.25)
        policy = RetryPolicy()
        self.assertEqual((policy.tries, policy.base_s, policy.max_s, policy.jitter), (12, 4.0, 120.0, 0.5))
        # The 11 waits between 12 tries come to about 14 minutes before jitter (prod-v1, 2026-10-18).
        self.assertEqual(sum(RetryPolicy(jitter=0).delay(a, random.Random(0)) for a in range(11)), 844)
        self.assertEqual(rclient.settings(), {"max_in_flight": 24, "per_model": 20, "subcall_in_flight": 12,
                                              "subcall_start_gap_s": 0.25,
                                              "retry": {"tries": 12, "base_s": 4.0, "max_s": 120.0, "jitter": 0.5,
                                                        "timeout_s": 600.0}})
        self.assertFalse(rclient.ENABLE_THINKING)


class PacerTest(AsyncCase):
    GAP = 0.05  # short, so the test is quick; production's is SUBCALL_START_GAP_S

    async def test_spreads_simultaneous_starts_at_least_the_gap_apart(self):
        pacer = Pacer({"sub": self.GAP})
        t0 = time.monotonic()

        async def start():
            await pacer.wait("sub")
            return time.monotonic() - t0

        starts = sorted(await asyncio.gather(*(start() for _ in range(10))))
        for k, at in enumerate(starts):
            self.assertGreaterEqual(at, k * self.GAP - 1e-3, starts)  # the k-th start waits k gaps
        self.assertLess(starts[-1], 9 * self.GAP + 0.3, starts)  # and not much longer: 10 go out over ~0.45 s

    async def test_late_wake_ups_do_not_bunch(self):
        """Starts booked while the event loop is blocked still go out a gap apart once it's free."""
        pacer = Pacer({"sub": self.GAP})

        async def start():
            await pacer.wait("sub")
            return time.monotonic()

        tasks = [asyncio.ensure_future(start()) for _ in range(4)]
        await asyncio.sleep(0)  # all four book their start times
        time.sleep(4 * self.GAP)  # a busy loop: every booked time passes before anyone wakes
        starts = sorted(await asyncio.gather(*tasks))
        gaps = [b - a for a, b in zip(starts, starts[1:])]
        self.assertGreaterEqual(min(gaps), self.GAP - 1e-3, gaps)

    async def test_other_models_are_not_delayed(self):
        pacer = Pacer({"sub": 0.5})
        for _ in range(10):
            pacer.reserve("sub")
        self.assertGreater(pacer.reserve("sub"), 4.9)  # the sub-call model is booked 5 s ahead
        t0 = time.monotonic()
        self.assertEqual(await asyncio.gather(*(pacer.wait("root") for _ in range(10))), [0.0] * 10)
        self.assertLess(time.monotonic() - t0, 0.05)
        self.assertEqual(Pacer().reserve("sub"), 0.0)  # no gaps given: nothing is paced
        with self.assertRaises(ValueError):
            Pacer({"sub": -1})


class RequestTest(ServerCase):
    async def test_root_turn_rebuilds_token_ids_and_pins_thinking_off(self):
        usage = Usage()
        choice = await self.client().root_turn(MESSAGES, model=ROOT, usage=usage)
        self.assertIn("```python", choice.message.content)
        self.assertEqual(choice.model_extra["prompt_token_ids"][0], rclient.IM_START_ID)
        self.assertTrue(rclient.thinking_off_prompt(choice.model_extra["prompt_token_ids"]))
        self.assertEqual(len(choice.model_extra["token_ids"]), len(choice.logprobs.content))
        body = self.server.bodies[-1]
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual((body["prompt_logprobs"], body["logprobs"], body["temperature"], body["max_tokens"]),
                         (0, True, 1.0, bridge.ROOT_MAX_TOKENS))
        self.assertEqual(usage.root_calls, 1)
        self.assertGreater(usage.cost_usd(), 0)

    async def test_subcall_settings(self):
        usage = Usage()
        out = await self.client().subcaller(usage)("Summarize this.")
        self.assertEqual(out, "ok")
        body = self.server.bodies[-1]
        self.assertEqual((body["model"], body["temperature"], body["max_tokens"], body["chat_template_kwargs"]),
                         (BASE, 0.3, 1024, {"enable_thinking": False}))
        self.assertNotIn("logprobs", body)
        self.assertEqual(usage.subcalls, 1)

    async def test_thinking_on_is_infra(self):
        self.server.thinking = True
        with self.assertRaises(RequestFailed) as root:
            await self.client().root_turn(MESSAGES, model=ROOT)
        with self.assertRaises(RequestFailed) as sub:
            await self.client().subcaller()("p")
        self.assertEqual((root.exception.kind, sub.exception.kind), ("thinking_on", "thinking_on"))

    def test_thinking_tail_check(self):
        from tests.fake_model import THINKING_OFF_TAIL, THINKING_ON_TAIL
        self.assertTrue(rclient.thinking_off_prompt([5, 6] + THINKING_OFF_TAIL))
        self.assertFalse(rclient.thinking_off_prompt([5, 6] + THINKING_ON_TAIL))
        self.assertFalse(rclient.thinking_off_prompt([5, 6, 7, 8, 9, 10]))

    def test_pinned_token_ids_match_the_tokenizer(self):
        found = sorted(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/tokenizer_config.json"))
        if not found:
            self.skipTest("Qwen3.8-27B tokenizer_config.json is not in the local Hugging Face cache")
        tokens = {v["content"]: int(k) for k, v in json.loads(found[-1].read_text("utf-8"))["added_tokens_decoder"].items()}
        self.assertEqual((tokens["<|im_start|>"], tokens["<think>"], tokens["</think>"]),
                         (rclient.IM_START_ID, rclient.THINK_START_ID, rclient.THINK_END_ID))


class FailureTest(ServerCase):
    async def test_persistent_429_becomes_infra_after_six_tries(self):
        self.server.fault = lambda kind, body: 429
        c = self.client()
        with self.assertRaises(RequestFailed) as caught:
            await c.subcaller()("p")
        exc = caught.exception
        self.assertEqual((exc.kind, exc.status, exc.attempts), ("rate_limited", 429, 6))
        self.assertTrue(exc.cf_ray.startswith("fake"))
        self.assertIsInstance(exc, bridge.InfraError)
        self.assertEqual(self.server.requests["subcall"], 6)  # the OpenAI client's own retries are off
        stats = c.reset_stats()
        self.assertEqual((stats["attempts"], stats["retries"], stats["rate_limited"], stats["failed"]),
                         (6, 5, 6, {"rate_limited": 1}))
        self.assertTrue(all(e["cf_ray"] for e in stats["events"]))

    async def test_transient_429_and_5xx_recover(self):
        failures = iter([429, 503, 500])
        self.server.fault = lambda kind, body: next(failures, None)
        c = self.client()
        choice = await c.root_turn(MESSAGES, model=ROOT)
        self.assertIsNotNone(choice)
        stats = c.reset_stats()
        self.assertEqual((stats["attempts"], stats["rate_limited"], stats["server_errors"], stats["failed"]),
                         (4, 1, 2, {}))

    async def test_persistent_5xx_is_infra(self):
        self.server.fault = lambda kind, body: 502
        with self.assertRaises(RequestFailed) as caught:
            await self.client().root_turn(MESSAGES, model=ROOT)
        self.assertEqual((caught.exception.kind, caught.exception.attempts), ("server_error", 6))

    async def test_auth_and_missing_model_fail_at_once(self):
        for status in (401, 403, 404):
            self.server.fault = lambda kind, body, s=status: s
            for call in (lambda c: c.root_turn(MESSAGES, model=ROOT), lambda c: c.subcaller()("p")):
                with self.assertRaises(RequestFailed) as caught:
                    await call(self.client())
                self.assertEqual((caught.exception.kind, caught.exception.attempts), ("rejected", 1))

    async def test_a_rejected_subcall_prompt_is_the_models_error(self):
        self.server.fault = lambda kind, body: 400
        with self.assertRaises(SubcallError):
            await self.client().subcaller()("p")
        with self.assertRaises(RequestFailed):  # a root call the API rejects is never the model's doing
            await self.client().root_turn(MESSAGES, model=ROOT)

    async def test_timeouts_and_dropped_connections_are_retried_then_infra(self):
        self.server.fault = lambda kind, body: "hang"
        with self.assertRaises(RequestFailed) as caught:
            await self.client(RetryPolicy(tries=2, base_s=0.01, timeout_s=0.3)).subcaller()("p")
        self.assertEqual((caught.exception.kind, caught.exception.attempts), ("timeout", 2))
        await self.server.close()
        with self.assertRaises(RequestFailed) as caught:
            await self.client(RetryPolicy(tries=2, base_s=0.01, timeout_s=2)).subcaller()("p")
        # Linux refuses the connection at once; Windows retries the connect until the timeout.
        self.assertIn(caught.exception.kind, ("connection", "timeout"))

    async def test_slot_is_free_while_backing_off(self):
        failures = iter([429])
        self.server.fault = lambda kind, body: next(failures, None)
        c = self.client(RetryPolicy(tries=2, base_s=2.0, jitter=0))
        task = asyncio.ensure_future(c.subcaller()("p"))
        for _ in range(100):  # wait until the client has had its 429 and started backing off
            if c.stats.retries:
                break
            await asyncio.sleep(0.02)
        self.assertEqual((c.stats.retries, c.limiter.in_flight, self.server.requests["subcall"]), (1, 0, 1))
        self.assertEqual(await task, "ok")


class PacingTest(ServerCase):
    """Sub-calls spread out (step 10 §6): the sub-call model's own cap and start gap, set by the client."""

    GAP = 0.1  # short, so the test is quick; production's is SUBCALL_START_GAP_S

    def arrivals(self) -> collections.defaultdict:
        """When each request reached the server, by model."""
        seen = collections.defaultdict(list)

        def fault(kind, body):
            seen[body["model"]].append(time.monotonic())

        self.server.fault = fault
        return seen

    async def test_the_client_caps_and_paces_its_subcall_model(self):
        c = ModelClient(self.openai, subcall_model=BASE)
        self.assertEqual((c.limiter.total, c.limiter.cap(BASE), c.limiter.cap(ROOT)), (24, 12, 20))
        self.assertEqual(c.pacer.gaps, {BASE: 0.25})
        limiter, pacer = Limiter(), Pacer()
        shared = ModelClient(self.openai, subcall_model=BASE, limiter=limiter, pacer=pacer)
        self.assertTrue(shared.limiter is limiter and shared.pacer is pacer)  # one given is used as it is

    async def test_subcall_starts_are_spread_and_root_calls_are_not(self):
        arrivals = self.arrivals()
        self.server.latency = (0.05, 0.05)
        c = self.client(pacer=Pacer({BASE: self.GAP}))
        t0 = time.monotonic()

        async def watch():  # while the burst is being paced, it holds no slot and isn't queued in the limiter
            await asyncio.sleep(3.5 * self.GAP)
            return c.limiter.waiting, c.limiter.by_model[BASE]

        *_, held = await asyncio.gather(*(c.subcaller()("p") for _ in range(10)),
                                        *(c.root_turn(MESSAGES, model=ROOT) for _ in range(10)), watch())
        subs = sorted(at - t0 for at in arrivals[BASE])
        for k, at in enumerate(subs):
            self.assertGreaterEqual(at, k * self.GAP - 0.02, subs)  # the k-th waits k gaps; 0.02 s for the wire
        self.assertLess(max(arrivals[ROOT]) - t0, 4 * self.GAP, subs)  # all 10 root calls went out at once
        self.assertEqual(held[0], 0)
        self.assertLessEqual(held[1], 2)
        paced = c.reset_stats()["paced_s"]  # 0 + 1 + ... + 9 gaps, summed over the 10 sub-calls
        self.assertTrue(0.5 * 45 * self.GAP < paced < 45 * self.GAP + 1.0, paced)

    async def test_retries_are_paced_too(self):
        arrivals = self.arrivals()
        self.server.fault = lambda kind, body, f=self.server.fault: f(kind, body) or 429
        c = self.client(RetryPolicy(tries=4, base_s=0.001, max_s=0.001, jitter=0, timeout_s=10),
                        pacer=Pacer({BASE: self.GAP}))
        with self.assertRaises(RequestFailed) as caught:
            await c.subcaller()("p")
        self.assertEqual(caught.exception.attempts, 4)
        tries = arrivals[BASE]
        self.assertEqual(len(tries), 4)
        self.assertGreaterEqual(tries[-1] - tries[0], 3 * self.GAP - 0.02)  # the backoff alone is 1 ms
        self.assertGreater(c.stats.paced_s, 3 * self.GAP * 0.5)


class LoadTest(ServerCase):
    """Done when: a load test never has more than the cap in flight (plan step 4). The caps are 24 in
    total, 20 per model and 12 for the sub-call model (step 10 §6). Pacing is off here, so the caps are
    what hold the load back."""

    LATENCY = (0.3, 0.6)  # long enough that the client can keep 24 requests open on a slow machine

    async def _load(self, models: list[str], n: int) -> ModelClient:
        self.server.latency = self.LATENCY
        c = self.client(pacer=Pacer())
        calls = [c.root_turn(MESSAGES, model=m) if i % 3 == 0 else c.subcaller(model=m)("p")
                 for i, m in zip(range(n), models * n)]
        await asyncio.gather(*calls)
        return c

    async def test_training_shape_reaches_24_and_never_more(self):
        c = await self._load([ROOT, BASE], 300)
        # The client holds 24 slots at its peak. The server sees a little less, since a slot is also
        # held while the request is sent and the reply parsed; it must never see more.
        self.assertEqual(c.limiter.peaks["total"], 24)
        self.assertEqual(c.limiter.peaks["by_model"][BASE], 12)
        self.assertTrue(12 < self.server.peak <= 24, self.server.peak)
        self.assertLessEqual(self.server.peak_by_model[BASE], 12)
        self.assertLessEqual(max(self.server.peak_by_model.values()), 20)
        self.assertEqual(self.server.statuses[429], 0)
        self.assertEqual(c.reset_stats()["attempts"], 300)

    async def test_one_model_alone(self):
        """A base-model eval sends everything to the sub-call model, so it runs at 12; any other model runs at 20."""
        c = await self._load([BASE], 100)
        self.assertEqual(c.limiter.peaks["total"], 12)
        self.assertTrue(self.server.peak <= 12 and self.server.statuses[429] == 0)
        self.server.reset_counts()
        c = await self._load([ROOT], 100)
        self.assertEqual(c.limiter.peaks["total"], 20)
        self.assertTrue(self.server.peak <= 20 and self.server.statuses[429] == 0)

    async def test_two_clients_sharing_one_limiter(self):
        limiter, pacer = Limiter(model_caps={BASE: rclient.MAX_SUBCALLS_IN_FLIGHT}), Pacer()
        clients = [self.client(limiter=limiter, pacer=pacer) for _ in range(3)]
        self.server.latency = self.LATENCY
        await asyncio.gather(*(c.subcaller(model=m)("p") for c in clients for m in (ROOT, BASE) for _ in range(60)))
        self.assertEqual(limiter.peaks["total"], 24)
        self.assertTrue(12 < self.server.peak <= 24, self.server.peak)
        self.assertLessEqual(self.server.peak_by_model[BASE], 12)
        self.assertEqual(self.server.statuses[429], 0)

    async def test_without_the_per_model_cap_w_and_b_would_refuse(self):
        """Why the cap of 20 per model exists: 30 to one model name gets 429s, as the step 2 probe found."""
        c = self.client(limiter=Limiter(total=30, per_model=30), pacer=Pacer(),
                        retry=RetryPolicy(tries=1, timeout_s=10))
        self.server.latency = (1.0, 1.0)
        results = await asyncio.gather(*(c.subcaller()("p") for _ in range(30)), return_exceptions=True)
        self.assertEqual(sum(isinstance(r, RequestFailed) for r in results), 10)


if __name__ == "__main__":
    unittest.main()
