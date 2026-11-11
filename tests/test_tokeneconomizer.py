import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fennec_core import tokeneconomizer as te


class StubHook:
    def __init__(self):
        self.payloads = []

    def run(self, payload, env, host_hint=None):
        self.payloads.append(dict(payload))
        return {}


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.TemporaryDirectory()
        self.addCleanup(self.home.cleanup)
        self.env = {k: os.environ.get(k) for k in ("USERPROFILE", "HOME")}
        os.environ["USERPROFILE"] = self.home.name
        os.environ["HOME"] = self.home.name
        self.addCleanup(self._restore_env)
        self.saved = (te._LOADED, te._HOOK)
        self.addCleanup(self._restore)

    def _restore_env(self):
        for key, value in self.env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _restore(self):
        te._LOADED, te._HOOK = self.saved

    def test_bridge_is_a_noop_without_tokeneconomizer(self):
        te._LOADED, te._HOOK = True, None
        bridge = te.Bridge("/work", "sid-1", "task")
        self.assertEqual(bridge.start(), "")
        self.assertEqual(bridge.after_tool("read_file", {}, "ok", True), "")
        bridge.assistant("text", [("c1", "read_file", {})])
        bridge.tool_result("c1", "ok", True)
        bridge.end("exit")
        self.assertFalse(Path(bridge.transcript).exists())

    def test_session_transcript_uses_the_claude_shape(self):
        hook = StubHook()
        te._LOADED, te._HOOK = True, hook
        bridge = te.Bridge("/work", "sid-2", "fix it")
        bridge._enabled = True
        bridge._hook = hook
        bridge.start()
        bridge.assistant("checking", [("c1", "read_file", {"path": "a.py"})],
                         {"prompt_tokens": 10, "completion_tokens": 4})
        bridge.tool_result("c1", '{"ok": true}', True)
        bridge.end("exit")

        entries = [json.loads(line) for line in
                   Path(bridge.transcript).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(entries[0]["type"], "user")
        self.assertEqual(entries[0]["message"]["content"][0]["text"], "fix it")
        blocks = entries[1]["message"]["content"]
        self.assertEqual(blocks[0]["type"], "text")
        self.assertEqual(blocks[1]["type"], "tool_use")
        self.assertEqual(blocks[1]["name"], "read_file")
        self.assertEqual(entries[1]["message"]["usage"]["input_tokens"], 10)
        self.assertEqual(entries[2]["message"]["content"][0]["type"], "tool_result")
        self.assertFalse(entries[2]["message"]["content"][0]["is_error"])

        events = [p["hook_event_name"] for p in hook.payloads]
        self.assertEqual(events, ["SessionStart", "UserPromptSubmit", "Stop", "SessionEnd"])

    def test_after_tool_names_the_event_that_answers_it(self):
        hook = StubHook()
        te._LOADED, te._HOOK = True, hook
        bridge = te.Bridge("/work", "sid-3", "task")
        bridge._enabled = True
        bridge._hook = hook
        bridge.after_tool("run_python", {}, "boom", False)
        self.assertEqual(hook.payloads[-1]["hook_event_name"], "PostToolUseFailure")
        self.assertEqual(hook.payloads[-1]["tool_response"]["exit_code"], 1)


if __name__ == "__main__":
    unittest.main()
