import hermetic  # noqa: F401

import json
import tempfile
import unittest
from pathlib import Path

from fennec_core import context, loop


class ElideTests(unittest.TestCase):
    def test_short_text_is_unchanged(self):
        self.assertEqual(context.elide("abc", 10), "abc")

    def test_long_text_keeps_head_and_tail(self):
        text = "A" * 5000 + "MIDDLE" + "Z" * 5000
        out = context.elide(text, 1000)
        self.assertTrue(out.startswith("A" * 100))
        self.assertIn("chars elided", out)
        self.assertTrue(out.endswith("Z" * 100))
        self.assertLess(len(out), len(text))


class LimitTests(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(context.limits({}),
                         {"budget_chars": context.DEFAULT_BUDGET_CHARS,
                          "result_chars": context.DEFAULT_RESULT_CHARS})

    def test_environment_overrides(self):
        got = context.limits({"FENNEC_AGENT_RESULT_CHARS": "1200",
                              "FENNEC_AGENT_CONTEXT_BUDGET": "9000"})
        self.assertEqual(got, {"budget_chars": 9000, "result_chars": 1200})

    def test_invalid_values_fall_back_and_are_floored(self):
        bad = context.limits({"FENNEC_AGENT_RESULT_CHARS": "nope",
                              "FENNEC_AGENT_CONTEXT_BUDGET": "-3"})
        self.assertEqual(bad, {"budget_chars": context.DEFAULT_BUDGET_CHARS,
                               "result_chars": context.DEFAULT_RESULT_CHARS})
        floored = context.limits({"FENNEC_AGENT_RESULT_CHARS": "10"})
        self.assertEqual(floored["result_chars"], context.MIN_RESULT_CHARS)

    def test_budget_is_never_below_the_result_cap(self):
        got = context.limits({"FENNEC_AGENT_RESULT_CHARS": "8000",
                              "FENNEC_AGENT_CONTEXT_BUDGET": "3000"})
        self.assertEqual(got["budget_chars"], 8000)


class CompactTests(unittest.TestCase):
    def test_duplicate_tool_results_become_a_reference(self):
        messages = [{"role": "tool", "tool_call_id": "call_1", "content": "x" * 500},
                    {"role": "tool", "tool_call_id": "call_2", "content": "x" * 500}]
        payload, stats = context.compact(messages, result_chars=6000)
        self.assertIn("identical to the result of call_1", payload[1]["content"])
        self.assertEqual(stats["deduped"], 1)
        self.assertLess(stats["chars_after"], stats["chars_before"])

    def test_large_results_keep_head_and_tail(self):
        body = "head" + "m" * 20000 + "tail"
        payload, stats = context.compact([{"role": "tool", "tool_call_id": "c", "content": body}],
                                         result_chars=600)
        content = payload[0]["content"]
        self.assertTrue(content.startswith("head"))
        self.assertTrue(content.endswith("tail"))
        self.assertIn("chars elided", content)
        self.assertGreaterEqual(stats["elided"], 1)

    def test_budget_shrinks_oldest_results_only(self):
        messages = [{"role": "system", "content": "S" * 200},
                    {"role": "user", "content": "task"},
                    {"role": "tool", "tool_call_id": "a", "content": "A" * 4000},
                    {"role": "tool", "tool_call_id": "b", "content": "B" * 4000}]
        payload, stats = context.compact(messages, budget_chars=5000, result_chars=6000)
        self.assertEqual(payload[0]["content"], "S" * 200)
        self.assertEqual(payload[1]["content"], "task")
        self.assertGreater(len(payload[3]["content"]), len(payload[2]["content"]))
        self.assertLess(stats["chars_after"], stats["chars_before"])

    def test_seen_carries_across_steps(self):
        seen = {}
        context.compact([{"role": "tool", "tool_call_id": "c1", "content": "same"}], seen=seen)
        payload, stats = context.compact([{"role": "tool", "tool_call_id": "c2", "content": "same"}],
                                         seen=seen)
        self.assertEqual(stats["deduped"], 1)
        self.assertIn("identical to the result of c1", payload[0]["content"])

    def test_messages_are_not_mutated(self):
        original = {"role": "tool", "tool_call_id": "c", "content": "z" * 9000}
        context.compact([original], result_chars=600)
        self.assertEqual(original["content"], "z" * 9000)


class ScriptedEngine:
    """Two identical file reads, then a final answer."""

    def __init__(self):
        self.payloads = []

    def chat(self, messages, tools=None, **kwargs):
        self.payloads.append([dict(message) for message in messages])
        step = len(self.payloads)

        def read(call_id):
            return {"role": "assistant", "content": "", "reasoning_content": "think",
                    "tool_calls": [{"id": call_id, "type": "function",
                                    "function": {"name": "read_file",
                                                 "arguments": json.dumps({"path": "big.txt"})}}]}

        if step == 1:
            return read("c1"), {"total_tokens": 5}
        if step == 2:
            return read("c2"), {"total_tokens": 5}
        return {"role": "assistant", "content": "done"}, {"total_tokens": 5}


class LoopContextTests(unittest.TestCase):
    def setUp(self):
        loop.SESSIONS.clear()
        self.addCleanup(loop.SESSIONS.clear)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "big.txt").write_text("line of text\n" * 2000, encoding="utf-8")

    def test_repeated_read_is_referenced_and_savings_are_recorded(self):
        engine = ScriptedEngine()
        sid = loop.new_session("read big.txt twice", str(self.root), engine_client=engine)
        loop.run_agent(sid, max_steps=5)
        state = loop.get_steps(sid)
        self.assertEqual(state["status"], "done")
        tool_messages = [m for m in engine.payloads[2] if m.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 2)
        self.assertIn("identical to the result of c1", tool_messages[1]["content"])
        self.assertGreater(state["usage"]["context_chars_before"], state["usage"]["context_chars_after"])
        self.assertGreaterEqual(state["usage"]["context_deduped"], 1)

    def test_system_and_task_are_never_compacted(self):
        engine = ScriptedEngine()
        sid = loop.new_session("keep my words", str(self.root), engine_client=engine)
        loop.run_agent(sid, max_steps=2)
        last = engine.payloads[-1]
        self.assertTrue(last[0]["content"].startswith("You are Fennec"))
        self.assertEqual(last[1]["content"], "keep my words")


if __name__ == "__main__":
    unittest.main()
