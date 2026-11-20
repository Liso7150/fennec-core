"""Cancelling a stream while the engine is silent.

The opener is faked, so nothing here reaches an engine, a port or the network.
The stall is an Event the fake response waits on, which is what a real engine
that stops emitting looks like to a blocking readline.
"""
from __future__ import annotations

import hermetic  # noqa: F401

import json
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

import engine as engine_mod
from config import load_config
from engine import Engine, EngineError


def sse(payload):
    return b"data: " + json.dumps(payload).encode() + b"\n"


def delta(**fields):
    return sse({"choices": [{"delta": fields}]})


class FakeResponse:
    """An SSE response that can go silent after the lines it was given."""

    def __init__(self, lines, stall=None, error=None, content_type="text/event-stream"):
        self.lines = list(lines)
        self.stall = stall
        self.error = error
        self.headers = {"content-type": content_type}
        self.closed = False
        self.iterated = threading.Event()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def __iter__(self):
        for line in self.lines:
            yield line
        if self.stall is not None:
            self.iterated.set()
            self.stall.wait(5)
        if self.error is not None:
            raise self.error

    def read(self, limit=None):
        raw = b"".join(self.lines)
        return raw if limit is None else raw[:limit]


class FakeOpener:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def open(self, request, timeout=None):
        self.requests.append((request.full_url, timeout))
        return self.response


class StreamCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = load_config(environ={}, home=Path(self.tmp.name))
        self.engine = Engine(self.config)
        patcher = mock.patch.object(self.engine, "model_id", return_value="qwen36-35b")
        patcher.start()
        self.addCleanup(patcher.stop)

    def stream(self, response, abort=None):
        opener = FakeOpener(response)
        with mock.patch.object(urllib.request, "build_opener", return_value=opener):
            events = list(self.engine.chat_stream([{"role": "user", "content": "hi"}], abort=abort))
        return events, opener

    def result(self, events):
        self.assertEqual(events[-1]["type"], "result")
        return events[-1]


class NormalStreamingTests(StreamCase):
    def test_a_full_stream_is_unchanged(self):
        lines = [delta(content="Hello"), delta(reasoning_content="thinking"),
                 delta(tool_calls=[{"index": 0, "id": "c1", "function": {"name": "read_file", "arguments": '{"p":'}}]),
                 delta(tool_calls=[{"index": 0, "function": {"arguments": '"a"}'}}]),
                 sse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
                 sse({"usage": {"prompt_tokens": 7, "completion_tokens": 3}}),
                 b"data: [DONE]\n"]
        for abort in (None, lambda: False):
            events, opener = self.stream(FakeResponse(lines), abort=abort)
            self.assertEqual([e["text"] for e in events if e["type"] == "text"], ["Hello"])
            self.assertEqual([e["text"] for e in events if e["type"] == "reasoning"], ["thinking"])
            self.assertEqual([e["delta"] for e in events if e["type"] == "tool_delta"], ['{"p":', '"a"}'])
            self.assertEqual([e["finish_reason"] for e in events if e["type"] == "finish"], ["tool_calls"])
            result = self.result(events)
            self.assertFalse(result["cancelled"])
            self.assertEqual(result["finish_reason"], "tool_calls")
            self.assertEqual(result["usage"], {"prompt_tokens": 7, "completion_tokens": 3})
            self.assertEqual(result["message"]["content"], "Hello")
            self.assertEqual(result["message"]["tool_calls"][0]["function"]["arguments"], '{"p":"a"}')
            self.assertEqual(opener.requests[0][1], 600, "the caller's read budget is still passed through")

    def test_lines_after_a_pause_are_still_delivered(self):
        stall = threading.Event()
        threading.Timer(0.2, stall.set).start()
        self.addCleanup(stall.set)
        lines = [delta(content="before"), delta(content="after"), b"data: [DONE]\n"]
        events, _ = self.stream(FakeResponse(lines, stall=stall), abort=lambda: False)
        self.assertEqual("".join(e["text"] for e in events if e["type"] == "text"), "beforeafter")
        self.assertFalse(self.result(events)["cancelled"])

    def test_a_non_sse_reply_is_still_read_whole(self):
        body = json.dumps({"choices": [{"message": {"role": "assistant", "content": "plain"},
                                       "finish_reason": "stop"}],
                           "usage": {"total_tokens": 2}}).encode()
        events, _ = self.stream(FakeResponse([body], content_type="application/json"))
        self.assertEqual(self.result(events)["message"]["content"], "plain")
        self.assertEqual(self.result(events)["finish_reason"], "stop")

    def test_a_stream_failure_is_still_an_engine_error(self):
        response = FakeResponse([delta(content="x")], error=OSError("connection reset"))
        with self.assertRaises(EngineError):
            self.stream(response, abort=lambda: False)

    def test_an_engine_error_chunk_is_still_reported_verbatim(self):
        response = FakeResponse([sse({"error": {"message": "expert page failed"}})])
        with self.assertRaisesRegex(EngineError, "expert page failed"):
            self.stream(response, abort=lambda: False)


class CancellationTests(StreamCase):
    def test_abort_between_lines_keeps_the_partial_message(self):
        lines = [delta(content="partial"), delta(content="dropped"), b"data: [DONE]\n"]
        stopped = threading.Event()
        opener = FakeOpener(FakeResponse(lines))
        with mock.patch.object(urllib.request, "build_opener", return_value=opener):
            stream = self.engine.chat_stream([{"role": "user", "content": "hi"}], abort=stopped.is_set)
            self.assertEqual(next(stream)["text"], "partial")
            stopped.set()
            events = list(stream)
        result = self.result(events)
        self.assertTrue(result["cancelled"])
        self.assertEqual(result["finish_reason"], "cancelled")
        self.assertEqual(result["message"]["content"], "partial")

    def test_abort_is_noticed_while_the_engine_is_silent(self):
        stall = threading.Event()
        self.addCleanup(stall.set)
        response = FakeResponse([delta(content="partial")], stall=stall)
        opener = FakeOpener(response)
        stopped = threading.Event()
        with mock.patch.object(urllib.request, "build_opener", return_value=opener):
            stream = self.engine.chat_stream([{"role": "user", "content": "hi"}], abort=stopped.is_set)
            self.assertEqual(next(stream)["text"], "partial")
            response.iterated.wait(2)
            stopped.set()
            began = time.monotonic()
            events = list(stream)
            elapsed = time.monotonic() - began
        self.assertLess(elapsed, 2.0, f"a stalled stream took {elapsed:.2f}s to notice the stop")
        result = self.result(events)
        self.assertTrue(result["cancelled"])
        self.assertEqual(result["finish_reason"], "cancelled")
        self.assertEqual(result["message"]["content"], "partial", "the partial answer is kept")
        self.assertTrue(response.closed, "the response is released when the turn is cancelled")

    def test_a_stalled_stream_is_polled_not_spun(self):
        stall = threading.Event()
        self.addCleanup(stall.set)
        polls = []
        release = threading.Event()

        def abort():
            polls.append(time.monotonic())
            return release.is_set()

        lines = engine_mod.abortable_lines(FakeResponse([b"data: one\n"], stall=stall), abort, poll_s=0.05)
        self.assertEqual(next(lines), b"data: one\n")
        threading.Timer(0.4, release.set).start()
        self.assertEqual(list(lines), [])
        self.assertGreater(len(polls), 3, "the abort flag must be re-checked while nothing arrives")
        self.assertLess(len(polls), 40, "polling must not become a busy loop")

    def test_abortable_lines_without_abort_is_a_plain_iteration(self):
        lines = [b"data: one\n", b"data: [DONE]\n"]
        self.assertEqual(list(engine_mod.abortable_lines(FakeResponse(lines), None)), lines)


if __name__ == "__main__":
    unittest.main()
