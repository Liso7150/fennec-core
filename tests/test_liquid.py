"""The d1.2 client, driven against a scripted transport.

Nothing here reaches api.liquid.ai or spends money: every call goes through a fake
opener that replays canned responses, the same trick `test_outlets.py` uses for
OpenBot. The retry clock is injected too, so a test that asserts three attempts
never sleeps for real.
"""
from __future__ import annotations

import hermetic  # noqa: F401

import io
import json
import math
import unittest
import urllib.error

import liquid
from liquid import LiquidError


class FakeResponse:
    def __init__(self, payload, status=200):
        self._raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._raw


class FakeOpener:
    """Replays a script of responses and records what it was asked to send."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def open(self, request, timeout=None):
        self.calls.append({
            "url": request.full_url,
            "headers": dict(request.header_items()),
            "body": request.data,
            "timeout": timeout,
        })
        outcome = self.script.pop(0) if self.script else FakeResponse({"answers": {}})
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def http_error(code, body, headers=None):
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return urllib.error.HTTPError(liquid.ENDPOINT, code, "err", headers or {}, io.BytesIO(raw))


def answer(choice="b", probabilities=None, confidence=0.9):
    return {"type": "choice", "choice": choice,
            "probabilities": probabilities if probabilities is not None else {"a": 0.1, "b": 0.9},
            "confidence": confidence}


class Transport(unittest.TestCase):
    def test_sends_the_documented_request_over_the_documented_path(self):
        opener = FakeOpener([FakeResponse({"model": "d1.2", "answers": {"operation": answer()},
                                           "usage": {"input_tokens": 54, "output_tokens": 0}})])
        result = liquid.decide({"page": {"url": "https://example.com"}},
                               {"operation": {"type": "choice", "criteria": {"a": "one", "b": "two"},
                                              "instructions": "pick"}},
                               key="liquid_secret", opener=opener)
        self.assertEqual(len(opener.calls), 1)
        call = opener.calls[0]
        self.assertEqual(call["url"], "https://api.liquid.ai/decisions/v1/systemone")
        self.assertEqual(call["headers"]["Authorization"], "Bearer liquid_secret")
        self.assertEqual(call["headers"]["Content-type"], "application/json")
        body = json.loads(call["body"])
        self.assertEqual(body["model"], "d1.2")
        self.assertEqual(set(body), {"model", "state", "questions"})
        self.assertEqual(result["model"], "d1.2")
        self.assertEqual(result["usage"], {"input_tokens": 54, "output_tokens": 0})

    def test_cost_comes_from_billed_input_tokens_only(self):
        # Output is free on a decision model, so a stray output_tokens must not move the price.
        opener = FakeOpener([FakeResponse({"answers": {}, "usage": {"input_tokens": 1_000_000,
                                                                    "output_tokens": 9_999}})])
        result = liquid.decide("state", {"q": {"type": "noul"}}, key="k", opener=opener)
        self.assertEqual(result["cost_usd"], 0.02)
        opener = FakeOpener([FakeResponse({"answers": {}, "usage": {"input_tokens": 1_000_000}})])
        ultra = liquid.decide("state", {"q": {"type": "noul"}}, model="d1.2-ultra", key="k", opener=opener)
        self.assertEqual(ultra["cost_usd"], 0.042)
        self.assertEqual(liquid.cost_usd("d1.2", {}), 0.0)

    def test_request_bytes_are_counted_in_real_utf8_not_escaped_ascii(self):
        # CJK page text is billed in bytes; \\uXXXX escaping would inflate the same text sixfold.
        text = "搜索" * 50
        opener = FakeOpener([FakeResponse({"answers": {}})])
        result = liquid.decide(text, {"q": {"type": "noul", "instructions": text}}, key="k", opener=opener)
        self.assertEqual(result["request_bytes"], len(opener.calls[0]["body"]))
        self.assertIn(text.encode("utf-8"), opener.calls[0]["body"])

    def test_an_oversized_body_is_refused_before_it_is_sent(self):
        opener = FakeOpener([])
        huge = "字" * (liquid.MAX_BODY_BYTES // 3 + 10)
        with self.assertRaises(LiquidError) as caught:
            liquid.decide(huge, {"q": {"type": "noul"}}, key="k", opener=opener)
        self.assertIn("4,500,000", str(caught.exception))
        self.assertEqual(opener.calls, [])

    def test_a_429_retries_and_a_retry_after_header_is_honoured(self):
        slept = []
        opener = FakeOpener([http_error(429, {"error": {"message": "slow down", "type": "rate_limit",
                                                        "code": "overloaded"}},
                                        {"Retry-After": "3"}),
                            FakeResponse({"answers": {"q": answer()}})])
        result = liquid.decide("s", {"q": {"type": "choice", "criteria": {"a": "1", "b": "2"}}},
                               key="k", opener=opener, sleep=slept.append)
        self.assertEqual(len(opener.calls), 2)
        self.assertEqual(slept, [3.0])
        self.assertEqual(result["answers"]["q"]["choice"], "b")

    def test_retry_after_is_capped_so_one_header_cannot_stall_the_loop(self):
        slept = []
        opener = FakeOpener([http_error(503, {}, {"Retry-After": "86400"}),
                            FakeResponse({"answers": {}})])
        liquid.decide("s", {"q": {"type": "noul"}}, key="k", opener=opener, sleep=slept.append)
        self.assertEqual(slept, [liquid.RETRY_AFTER_CAP_S])

    def test_backoff_is_exponential_when_the_api_gives_no_hint(self):
        slept = []
        opener = FakeOpener([http_error(429, {}), http_error(429, {}), FakeResponse({"answers": {}})])
        liquid.decide("s", {"q": {"type": "noul"}}, key="k", opener=opener, sleep=slept.append)
        self.assertEqual(slept, [0.5, 1.0])

    def test_attempts_are_bounded_and_the_last_failure_is_reported(self):
        opener = FakeOpener([http_error(529, {"error": {"message": "overloaded"}})
                             for _ in range(liquid.ATTEMPTS)])
        with self.assertRaises(LiquidError) as caught:
            liquid.decide("s", {"q": {"type": "noul"}}, key="k", opener=opener, sleep=lambda _s: None)
        self.assertEqual(len(opener.calls), liquid.ATTEMPTS)
        self.assertIn("no decision was made", str(caught.exception))

    def test_a_payload_error_is_not_retried(self):
        # Repeating a 400 would bill the same mistake three times.
        opener = FakeOpener([http_error(400, {"error": {"type": "invalid_request_error",
                                                        "code": "context_length_exceeded",
                                                        "message": "too many tokens"}})])
        with self.assertRaises(LiquidError) as caught:
            liquid.decide("s", {"q": {"type": "noul"}}, key="k", opener=opener, sleep=lambda _s: None)
        self.assertEqual(len(opener.calls), 1)
        self.assertIn("context_length_exceeded", str(caught.exception))
        self.assertIn("too many tokens", str(caught.exception))

    def test_the_api_key_never_reaches_an_error_message(self):
        leak = "liquid_supersecret_abcdef"
        opener = FakeOpener([http_error(401, {"error": {"type": "authentication_error",
                                                        "message": "Invalid API key provided."}})])
        with self.assertRaises(LiquidError) as caught:
            liquid.decide("s", {"q": {"type": "noul"}}, key=leak, opener=opener)
        self.assertNotIn(leak, str(caught.exception))
        self.assertIn("authentication_error", str(caught.exception))

    def test_connection_and_garbage_failures_say_no_decision_was_made(self):
        with self.assertRaises(LiquidError) as caught:
            liquid.decide("s", {"q": {"type": "noul"}}, key="k",
                          opener=FakeOpener([urllib.error.URLError("no route")]))
        self.assertIn("could not be reached", str(caught.exception))
        with self.assertRaises(LiquidError):
            liquid.decide("s", {"q": {"type": "noul"}}, key="k",
                          opener=FakeOpener([FakeResponse(b"<html>not json</html>")]))
        with self.assertRaises(LiquidError):
            liquid.decide("s", {"q": {"type": "noul"}}, key="k",
                          opener=FakeOpener([FakeResponse({"model": "d1.2"})]))

    def test_error_body_that_cannot_be_read_still_reports_the_status(self):
        class Unreadable(io.BytesIO):
            def read(self, *_a):
                raise OSError("connection reset")

        error = urllib.error.HTTPError(liquid.ENDPOINT, 502, "bad gateway", {}, Unreadable())
        with self.assertRaises(LiquidError) as caught:
            liquid.post_json({"model": "d1.2"}, key="k", opener=FakeOpener([error]))
        self.assertIn("502", str(caught.exception))


class Questions(unittest.TestCase):
    def test_the_question_budget_is_enforced_locally(self):
        questions = {f"q{i}": {"type": "noul"} for i in range(liquid.MAX_QUESTIONS + 1)}
        with self.assertRaises(LiquidError) as caught:
            liquid.check_questions(questions)
        self.assertIn("128", str(caught.exception))

    def test_criteria_shape_is_per_question_type(self):
        opener = FakeOpener([])
        for question, phrase in (
                ({"type": "choice", "criteria": ["a", "b"]}, "option->description"),
                ({"type": "score", "criteria": {"low": "1"}}, "ordered list"),
                ({"type": "noul", "criteria": {"a": "1"}}, "cannot carry criteria"),
                ({"type": "choice"}, "needs criteria"),
                ({}, "has no type")):
            with self.subTest(question=question):
                with self.assertRaises(LiquidError) as caught:
                    liquid.check_questions({"q": question})
                self.assertIn(phrase, str(caught.exception))
        self.assertEqual(opener.calls, [])

    def test_an_empty_or_non_mapping_question_set_is_refused(self):
        for questions in ({}, [], None):
            with self.assertRaises(LiquidError):
                liquid.check_questions(questions)

    def test_an_oversized_question_is_refused_in_bytes(self):
        with self.assertRaises(LiquidError) as caught:
            liquid.check_questions({"q": {"type": "noul", "instructions": "x" * (liquid.MAX_QUESTION_BYTES + 1)}})
        self.assertIn("250,000", str(caught.exception))

    def test_an_unknown_model_is_refused_before_the_call(self):
        with self.assertRaises(LiquidError) as caught:
            liquid.decide("s", {"q": {"type": "noul"}}, model="d1.1", key="k", opener=FakeOpener([]))
        self.assertIn("d1.2-ultra", str(caught.exception))

    def test_the_model_comes_from_the_environment_by_default(self):
        self.assertEqual(liquid.model_name({}), "d1.2")
        self.assertEqual(liquid.model_name({"LIQUID_MODEL": "d1.2-ultra"}), "d1.2-ultra")
        self.assertEqual(liquid.model_name({"LIQUID_MODEL": ""}), "d1.2")

    def test_a_missing_key_explains_how_to_get_one(self):
        with self.assertRaises(LiquidError) as caught:
            liquid.api_key({"LIQUID_API_KEY": "   "})
        self.assertIn("LIQUID_API_KEY", str(caught.exception))
        self.assertIn("0.020", str(caught.exception))
        self.assertEqual(liquid.api_key({"LIQUID_API_KEY": "liquid_x"}), "liquid_x")


class LocalLane(unittest.TestCase):
    """The free path: the open-weight d1 checkpoint served on this machine.

    These assert the shape of the seam, not that llama.cpp answers - `probe.py` under
    .local-docs/web-probe does that against a real server and a real DOM.
    """

    LOCAL = "http://127.0.0.1:8199/v1/systemone"

    def test_loopback_means_this_machine_and_nothing_else(self):
        for url in (self.LOCAL, "http://localhost:9931/v1/systemone",
                    "http://[::1]:8199/v1/systemone", "http://127.9.9.9:1/v1/systemone"):
            self.assertTrue(liquid.is_local(url), url)
        for url in (liquid.ENDPOINT, "http://127.0.0.1.evil.test/v1/systemone",
                    "http://10.0.0.5:8199/v1/systemone", "not a url"):
            self.assertFalse(liquid.is_local(url), url)

    def test_the_default_model_follows_the_endpoint(self):
        self.assertEqual(liquid.model_name({"LIQUID_ENDPOINT": self.LOCAL}), "d1-2B")
        self.assertEqual(liquid.model_name({"LIQUID_ENDPOINT": self.LOCAL,
                                            "LIQUID_MODEL": "d1-omni-600M"}), "d1-omni-600M")
        self.assertEqual(liquid.model_name({"LIQUID_ENDPOINT": "  "}), "d1.2")

    def test_a_local_decision_needs_no_key_and_costs_nothing(self):
        opener = FakeOpener([FakeResponse({"model": "d1-2B", "answers": {"q": answer()},
                                           "usage": {"input_tokens": 1500, "output_tokens": 0}})])
        result = liquid.decide("s", {"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}},
                               environ={"LIQUID_ENDPOINT": self.LOCAL}, opener=opener)
        self.assertEqual(opener.calls[0]["url"], self.LOCAL)
        # Not "the header is empty" - a request that carries no credential at all.
        self.assertNotIn("authorization", {k.lower() for k in opener.calls[0]["headers"]})
        self.assertEqual(result["cost_usd"], 0.0)
        self.assertEqual(result["model"], "d1-2B")

    def test_an_open_weight_model_is_refused_on_the_paid_endpoint(self):
        # The zero price lives in this table, not in a server. Sending the name to
        # api.liquid.ai and reporting $0 for the tokens it billed would be a lie.
        with self.assertRaises(LiquidError) as caught:
            liquid.decide("s", {"q": {"type": "noul"}}, model="d1-2B", key="k",
                          opener=FakeOpener([]))
        self.assertIn("not a hosted decision model", str(caught.exception))

    def test_a_hosted_model_name_is_refused_on_a_local_server(self):
        with self.assertRaises(LiquidError) as caught:
            liquid.decide("s", {"q": {"type": "noul"}}, model="d1.2",
                          endpoint=self.LOCAL, opener=FakeOpener([]))
        self.assertIn("d1-omni-600M", str(caught.exception))


class Answers(unittest.TestCase):
    def test_a_distribution_over_exactly_the_offered_options_passes(self):
        checked = liquid.validate_choice(answer(), ["a", "b"])
        self.assertEqual(checked["choice"], "b")

    def test_a_choice_the_model_invented_is_refused(self):
        for bad in (answer(choice="c"),
                    answer(probabilities={"a": 0.1, "b": 0.9, "c": 0.0}),      # extra option
                    answer(probabilities={"a": 0.1}),                            # missing option
                    answer(probabilities={"a": 0.55, "b": 0.25}),                # does not sum to 1
                    answer(probabilities={"a": 0.9, "b": 0.1}),                  # not the argmax
                    answer(probabilities={"a": 0.1, "b": 1.9}),                  # out of range
                    answer(probabilities={"a": 0.1, "b": float("nan")}),         # not finite
                    answer(probabilities={"a": 0.1, "b": math.inf}),
                    answer(probabilities={"a": "0.1", "b": "0.9"}),              # not numbers
                    answer(probabilities="b"),                                   # not a mapping
                    answer(confidence=float("nan")),
                    {"type": "choice", "choice": "b", "probabilities": {"a": 0.1, "b": 0.9}}):
            with self.subTest(answer=bad):
                with self.assertRaises(LiquidError) as caught:
                    liquid.validate_choice(bad, ["a", "b"])
                self.assertIn("no action executed", str(caught.exception))

    def test_a_sums_to_one_tolerance_of_two_percent(self):
        # Real distributions are rounded, so a sum of 0.99 must still be a usable answer.
        liquid.validate_choice(answer(probabilities={"a": 0.03, "b": 0.96}), ["a", "b"])
        with self.assertRaises(LiquidError):
            liquid.validate_choice(answer(probabilities={"a": 0.025, "b": 0.95}), ["a", "b"])

    def test_noul_answers_are_probabilities(self):
        self.assertEqual(liquid.validate_noul({"type": "noul", "noul": 0.989}), 0.989)
        self.assertEqual(liquid.validate_noul({"noul": 0}), 0.0)
        for bad in ({}, {"noul": None}, {"noul": 1.5}, {"noul": -0.1}, {"noul": float("nan")},
                    {"noul": "0.5"}):
            with self.subTest(answer=bad):
                with self.assertRaises(LiquidError):
                    liquid.validate_noul(bad)


if __name__ == "__main__":
    unittest.main()
