"""The Rust port has to be indistinguishable from the Python it replaces.

`rust/goldens/` pins the ported logic against CPython byte for byte; this is the
other half - the live pipe. It compacts a transcript through both paths, walks
several steps over one shared `seen` map, and redacts a payload that carries a
credential, so a wrong order, a stale environment or a dropped key shows up here
rather than in someone's session.

Skipped when the binary is not built: the Python path stands on its own, which is
the whole point of the fallback.
"""
import hermetic  # noqa: F401

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fennec_core import context, outlets, rust_bridge

EXE = rust_bridge.binary_path()
BUDGET = context.DEFAULT_BUDGET_CHARS
RESULT = 8_192


def transcript(steps=7, size=9_000):
    """Long, repeating tool results plus the calls that produced them."""
    messages = [{"role": "system", "content": "You are Fennec."},
                {"role": "user", "content": "read the artifacts"}]
    for step in range(steps):
        body = "".join(chr(97 + (i * 7 + step) % 26) for i in range(size))
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": [{"id": f"c{step}", "type": "function",
                                         "function": {"name": "read_file",
                                                     "arguments": json.dumps(
                                                         {"path": f"artifact-{step}.txt"})}}]})
        messages.append({"role": "tool", "tool_call_id": f"c{step}", "content": body})
        if step % 3 == 0:
            # the same bytes a second time, under a different call id
            messages.append({"role": "tool", "tool_call_id": f"d{step}", "content": body})
    return messages


@unittest.skipIf(not EXE, "fennec-rs is not built - cargo build --release -p fennec-rs")
class BridgeEqualityTests(unittest.TestCase):
    def setUp(self):
        # these tests ask the pipe directly, and the pipe is opt-in
        env = mock.patch.dict(os.environ, {"FENNEC_RUST": "1"})
        env.start()
        self.addCleanup(env.stop)

    def test_one_call_agrees_field_for_field(self):
        messages = transcript()
        through, stats_through = rust_bridge.compact(messages, budget_chars=BUDGET,
                                                     result_chars=RESULT, floor_chars=600,
                                                     seen={})
        self.assertIsNotNone(through, "the bridge returned nothing with a binary present")
        with mock.patch.dict(os.environ, {"FENNEC_RUST": "0"}):
            plain, stats_plain = context.compact(messages, budget_chars=BUDGET,
                                                 result_chars=RESULT)
        self.assertEqual(json.dumps(through, ensure_ascii=False),
                         json.dumps(plain, ensure_ascii=False))
        self.assertEqual(stats_through, stats_plain)

    def test_a_run_of_steps_stays_agreeable_across_the_shared_map(self):
        """Duplicates are caught across steps, so `seen` is the real state."""
        steps = [transcript(count) for count in (3, 5, 7)]
        shared, carried = {}, {}
        for messages in steps:
            through, stats_through = rust_bridge.compact(messages, budget_chars=BUDGET,
                                                         result_chars=RESULT, floor_chars=600,
                                                         seen=carried)
            with mock.patch.dict(os.environ, {"FENNEC_RUST": "0"}):
                plain, stats_plain = context.compact(messages, budget_chars=BUDGET,
                                                     result_chars=RESULT, seen=shared)
            self.assertEqual(json.dumps(through, ensure_ascii=False),
                             json.dumps(plain, ensure_ascii=False))
            self.assertEqual(stats_through, stats_plain)
            self.assertEqual(carried, shared, "the digest map drifted")
        self.assertGreater(len(shared), 3, "nothing was carried across steps")

    def test_the_three_conditions_that_open_the_pipe(self):
        """Opted in, nothing seen yet, big - or it stays in CPython."""
        big = transcript()
        asked = []
        real_compact = rust_bridge.compact

        def spy(*args, **kwargs):
            asked.append(kwargs["seen"])
            return real_compact(*args, **kwargs)

        gates = mock.patch.dict(os.environ, {"FENNEC_RUST": "1", "FENNEC_RUST_MIN_CHARS": "1"})
        with gates, mock.patch.object(rust_bridge, "compact", spy):
            delegated, delegated_stats = context.compact(big, budget_chars=BUDGET,
                                                          result_chars=RESULT)
        self.assertEqual(len(asked), 1, "a cold 60 KB transcript never reached the port")
        self.assertLess(delegated_stats["chars_after"], delegated_stats["chars_before"])
        self.assertGreater(delegated_stats["elided"] + delegated_stats["deduped"], 0)

        # warm: the port already knows these results, so the pipe would only
        # re-render them
        with gates, mock.patch.object(rust_bridge, "compact", spy):
            context.compact(big, budget_chars=BUDGET, result_chars=RESULT,
                            seen={digest: "c0" for digest in ("a" * 64, "b" * 64)})
        self.assertEqual(len(asked), 1, "a warm transcript went down the pipe")

        # opted out, which is the default
        with mock.patch.dict(os.environ, {"FENNEC_RUST": "0"}), \
                mock.patch.object(rust_bridge, "compact", spy):
            context.compact(big, budget_chars=BUDGET, result_chars=RESULT)
        self.assertEqual(len(asked), 1, "delegation is not opt-in")

        # below the gate, which is the measured break-even and 60 KB is nowhere
        # near it
        with mock.patch.dict(os.environ, {"FENNEC_RUST": "1"}), \
                mock.patch.object(rust_bridge, "compact", spy):
            context.compact(big, budget_chars=BUDGET, result_chars=RESULT)
        self.assertEqual(len(asked), 1, "a payload under the gate paid for a pipe")

        # all three conditions met on a payload this small only because the gate
        # was lowered to one character, which is what opting in means
        small = [{"role": "tool", "tool_call_id": "c", "content": "short"}]
        with gates, mock.patch.object(rust_bridge, "compact", spy):
            through, through_stats = context.compact(small, budget_chars=BUDGET,
                                                     result_chars=RESULT)
        self.assertEqual(len(asked), 2, "the conditions did not compose")
        self.assertEqual(through, small, "the port changed a short payload")
        self.assertEqual(through_stats["chars_after"], through_stats["chars_before"])

        # and left alone by the default, the same payload on the same call
        with mock.patch.object(rust_bridge, "compact", spy):
            plain, plain_stats = context.compact(small, budget_chars=BUDGET,
                                                 result_chars=RESULT)
        self.assertEqual(len(asked), 2, "a 5 byte payload went down a pipe for nothing")
        self.assertEqual(plain, small)
        self.assertEqual(plain_stats, through_stats)

    def test_the_seen_map_is_updated_in_place_for_the_caller(self):
        messages = transcript(3)
        seen = {}
        rust_bridge.compact(messages, budget_chars=BUDGET, result_chars=RESULT,
                            floor_chars=600, seen=seen)
        self.assertTrue(seen, "the caller's map stayed empty")
        self.assertTrue(all(len(digest) == 64 for digest in seen), list(seen)[:1])

    def test_a_missing_binary_leaves_the_result_alone(self):
        messages = transcript(4)
        dead = rust_bridge.Bridge(exe=str(Path(tempfile.gettempdir()) / "no-such-fennec-rs"))
        # every condition says "go", the pipe says "no such file": the caller must
        # still get the Python answer, and the bridge must stop being asked
        with mock.patch.object(rust_bridge, "_BRIDGE", dead), \
                mock.patch.dict(os.environ, {"FENNEC_RUST": "1", "FENNEC_RUST_MIN_CHARS": "1"}):
            through, through_stats = context.compact(messages, budget_chars=BUDGET,
                                                     result_chars=RESULT)
        plain, plain_stats = context.compact(messages, budget_chars=BUDGET,
                                             result_chars=RESULT)
        self.assertEqual(json.dumps(through, ensure_ascii=False),
                         json.dumps(plain, ensure_ascii=False))
        self.assertEqual(through_stats, plain_stats)
        self.assertTrue(dead._dead, "a dead bridge must stop being asked")


@unittest.skipIf(not EXE, "fennec-rs is not built - cargo build --release -p fennec-rs")
class RedactionStaysInProcessTests(unittest.TestCase):
    """The ported redactor is faster than CPython; the pipe to it is not.

    Measured at every size from 2 000 to 8 000 000 characters, redaction through
    the bridge costs 2-3x what the Python loop does, so `outlets.redact` must not
    delegate. It also must not hand a live key to a long-lived child, whose view
    of the environment is the snapshot taken when it was spawned.
    """
    KEY = "hf_RustBridgeChecksAreNotRealKeys_000"

    def payload(self):
        # big enough that a size-based gate would have opened the pipe: redaction
        # has no gate left, and this is what proves it
        return ("artifacts:\n" + ("a row of listing output\n" * 10_000)
                + f"Authorization: Bearer {self.KEY}\n"
                + "curl -H 'x-key: " + self.KEY + "'\n"
                + ("tail of the transcript\n" * 400))

    def test_a_large_payload_is_redacted_without_opening_the_pipe(self):
        asked = []
        real = rust_bridge.Bridge.call

        def spy(self, request):
            asked.append(request.get("op"))
            return real(self, request)

        text = self.payload()
        self.assertGreater(len(text), 100_000, "the payload is not the big case")
        with mock.patch.object(rust_bridge.Bridge, "call", spy), \
                mock.patch.dict(os.environ, {"FENNEC_API_KEY": self.KEY}):
            out = outlets.redact(text)
        self.assertEqual(asked, [], "redaction paid for a round trip")
        self.assertNotIn(self.KEY, out, "the key survived")
        self.assertIn("bearer [redacted]", out.lower())


class GateTests(unittest.TestCase):
    def test_the_binary_is_found_next_to_the_sources_or_not_at_all(self):
        with mock.patch.dict(os.environ, {"FENNEC_RS": str(Path(tempfile.gettempdir()) / "nope.exe")}):
            self.assertIsNone(rust_bridge.binary_path())

    def test_delegation_is_opt_in(self):
        for off in ("0", "", "auto", "maybe"):
            with mock.patch.dict(os.environ, {"FENNEC_RUST": off}):
                self.assertFalse(rust_bridge.enabled(), repr(off))
                self.assertIsNone(rust_bridge.Bridge().call({"op": "ping"}),
                                  "an opted-out bridge must not even spawn")
        for on in ("1", "on", "yes", "true", "TRUE"):
            with mock.patch.dict(os.environ, {"FENNEC_RUST": on}):
                self.assertTrue(rust_bridge.enabled(), on)

    def test_the_threshold_is_positive_even_when_the_environment_is_not(self):
        for junk in ("nope", "-4", "0", ""):
            with mock.patch.dict(os.environ, {"FENNEC_RUST_MIN_CHARS": junk}):
                self.assertEqual(rust_bridge.min_chars(), rust_bridge.DEFAULT_MIN_CHARS, junk)

    def test_the_bridge_answers_and_exits_cleanly(self):
        if not EXE:
            self.skipTest("fennec-rs is not built")
        bridge = rust_bridge.Bridge()
        with mock.patch.dict(os.environ, {"FENNEC_RUST": "1"}):
            try:
                answer = bridge.call({"op": "ping"})
                self.assertEqual(answer.get("impl"), "fennec-rs")
                self.assertIsNone(bridge.call({"op": "nonsense"}),
                                  "an unknown op must fall through, not raise")
            finally:
                proc = bridge._proc
                bridge.close()
        self.assertIsNotNone(proc)
        self.assertEqual(proc.returncode, 0, "the child outlived the exit frame")


if __name__ == "__main__":
    unittest.main()
