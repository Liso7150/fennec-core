"""Tests for the typed event envelope and domain models."""
import unittest

from fennec_core.events import (
    PlanEvent, PlanEntry, PartialEvent, TextEvent, ReasoningEvent,
    AssistantEvent, ToolStartEvent, ToolEvent, ToolEndEvent,
    FinalEvent, ErrorEvent, CancelledEvent, ApprovalRequiredEvent,
    UsageEvent, to_dict, from_dict
)


class TypedEventTests(unittest.TestCase):
    """Every event the loop produces has a typed envelope now."""

    def test_plan_event_carries_entries(self):
        entries = [
            PlanEntry(content="Inspect", status="completed", priority="medium"),
            PlanEntry(content="Implement", status="in_progress", priority="high"),
        ]
        event = PlanEvent(entries=entries, i=5, t=1.23)
        self.assertEqual(event.type, "plan")
        self.assertEqual(len(event.entries), 2)
        self.assertEqual(event.entries[0].status, "completed")
        self.assertEqual(event.entries[1].priority, "high")
        self.assertEqual(event.i, 5)
        self.assertAlmostEqual(event.t, 1.23)

    def test_partial_event_carries_text(self):
        event = PartialEvent(content="Hello", i=1, t=0.5)
        self.assertEqual(event.type, "partial")
        self.assertEqual(event.content, "Hello")

    def test_text_and_reasoning_events(self):
        text = TextEvent(text="Some text", i=2, t=1.0)
        reasoning = ReasoningEvent(text="Thinking...", i=3, t=1.5)
        self.assertEqual(text.type, "text")
        self.assertEqual(reasoning.type, "reasoning")
        self.assertEqual(text.text, "Some text")
        self.assertEqual(reasoning.text, "Thinking...")

    def test_assistant_event_with_tool_calls(self):
        calls = [{"name": "read_file", "args": {"path": "test.py"}}]
        event = AssistantEvent(
            content="Reading the file",
            reasoning="Let me check this",
            tool_calls=calls,
            i=10,
            t=5.0
        )
        self.assertEqual(event.type, "assistant")
        self.assertEqual(len(event.tool_calls), 1)
        self.assertEqual(event.tool_calls[0]["name"], "read_file")

    def test_tool_lifecycle_events(self):
        start = ToolStartEvent(name="write_file", args={"path": "out.py"}, call_id="abc", i=11, t=5.5)
        tool = ToolEvent(name="write_file", args={"path": "out.py"}, result={"ok": True}, ok=True, call_id="abc", i=12, t=6.0)
        end = ToolEndEvent(name="write_file", args={"path": "out.py"}, result={"ok": True}, ok=True, call_id="abc", i=13, t=6.5)
        
        self.assertEqual(start.type, "tool_start")
        self.assertEqual(tool.type, "tool")
        self.assertEqual(end.type, "tool_end")
        self.assertTrue(tool.ok)
        self.assertEqual(start.call_id, "abc")

    def test_terminal_events(self):
        final = FinalEvent(content="Done!", i=20, t=10.0)
        error = ErrorEvent(content="Something broke", i=21, t=10.5)
        cancelled = CancelledEvent(i=22, t=11.0)
        
        self.assertEqual(final.type, "final")
        self.assertEqual(error.type, "error")
        self.assertEqual(cancelled.type, "cancelled")
        self.assertEqual(cancelled.content, "Cancelled by the user.")

    def test_approval_required_event(self):
        event = ApprovalRequiredEvent(
            approval_id="req123",
            name="run_shell",
            args={"command": "rm -rf /"},
            risk="high",
            reason="Destructive command",
            files=["/etc/passwd"],
            i=15,
            t=7.5
        )
        self.assertEqual(event.type, "approval_required")
        self.assertEqual(event.risk, "high")
        self.assertEqual(len(event.files), 1)

    def test_usage_event(self):
        event = UsageEvent(
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            context_chars=5000,
            i=25,
            t=12.0
        )
        self.assertEqual(event.type, "usage")
        self.assertEqual(event.usage["prompt_tokens"], 100)
        self.assertEqual(event.context_chars, 5000)

    def test_to_dict_serialization(self):
        event = PlanEvent(
            entries=[PlanEntry(content="Test", status="in_progress")],
            i=1,
            t=0.5
        )
        data = to_dict(event)
        self.assertIsInstance(data, dict)
        self.assertEqual(data["type"], "plan")
        self.assertEqual(len(data["entries"]), 1)
        self.assertEqual(data["entries"][0]["content"], "Test")

    def test_from_dict_deserialization(self):
        data = {
            "type": "tool_start",
            "name": "read_file",
            "args": {"path": "test.py"},
            "call_id": "xyz",
            "i": 5,
            "t": 2.5
        }
        event = from_dict(data)
        self.assertIsInstance(event, ToolStartEvent)
        self.assertEqual(event.name, "read_file")
        self.assertEqual(event.call_id, "xyz")

    def test_from_dict_unknown_type_returns_dict(self):
        data = {"type": "future_event", "field": "value"}
        result = from_dict(data)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["type"], "future_event")

    def test_round_trip_serialization(self):
        original = AssistantEvent(
            content="Hello",
            reasoning="Thinking",
            tool_calls=[{"name": "test", "args": {}}],
            i=10,
            t=5.0
        )
        data = to_dict(original)
        restored = from_dict(data)
        self.assertIsInstance(restored, AssistantEvent)
        self.assertEqual(restored.content, original.content)
        self.assertEqual(restored.reasoning, original.reasoning)
        self.assertEqual(len(restored.tool_calls), 1)

    def test_events_are_frozen(self):
        event = PlanEvent(entries=[], i=0, t=0.0)
        with self.assertRaises(AttributeError):
            event.i = 5  # type: ignore

    def test_plan_entry_defaults(self):
        entry = PlanEntry(content="Test", status="pending")
        self.assertEqual(entry.priority, "medium")
        self.assertEqual(entry.status, "pending")


if __name__ == "__main__":
    unittest.main()
