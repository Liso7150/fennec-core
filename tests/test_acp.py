"""The ACP v1 agent, driven by an in-process client over the real stdio framing.

No engine and no paid API is ever contacted: turns run against a scripted
streaming fake, and the client side answers permission requests from a policy
the test chooses.
"""
from __future__ import annotations

import hermetic  # noqa: F401

import io
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from fennec_core import acp, loop, mcp

MCP_SERVER = """
import json, sys
TOOLS = [{"name": "echo", "description": "Echo text back",
          "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}},
                          "required": ["text"]}}]
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    message = json.loads(line)
    method, request_id = message.get("method"), message.get("id")
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                  "serverInfo": {"name": "demo", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        result = {"content": [{"type": "text",
                               "text": "echo:" + message["params"]["arguments"]["text"]}]}
    else:
        result = {}
    if request_id is not None:
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}) + "\\n")
        sys.stdout.flush()
"""


MCP_HANG = """
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    message = json.loads(line)
    if message.get("method") == "initialize":
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": message["id"],
                                     "result": {"protocolVersion": "2024-11-05",
                                                "capabilities": {"tools": {}},
                                                "serverInfo": {"name": "sloth", "version": "1"}}})
                         + "\\n")
        sys.stdout.flush()
        continue
    # Accepts the connection and never answers tools/list: the pipe only closes
    # when the client gives up on it.
    while sys.stdin.readline():
        pass
"""


class QuietBridge:
    """The TokenEconomizer hook stays out of these tests."""

    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        return ""

    def assistant(self, text, calls, usage=None):
        pass

    def tool_result(self, call_id, text, ok=True):
        pass

    def after_tool(self, name, args, text, ok):
        return ""

    def end(self, reason="exit"):
        pass


class ScriptEngine:
    """Answers each turn from a scripted reply; streams like the real engine does."""

    def __init__(self, *replies, running=True):
        self.replies = list(replies)
        self.calls = []
        self.running = running
        self.stopped = False
        self.started = 0

    def is_running(self):
        return self.running

    def start(self, wait_s=180):
        self.started += 1
        self.running = True
        return {"running": True, "started": True, "owned": True}

    def stop(self):
        self.stopped = True
        self.running = False
        return {"stopped": True, "owned": True}

    def chat_stream(self, messages, tools=None, abort=None, **kwargs):
        self.calls.append({"messages": list(messages), "tools": list(tools or [])})
        reply = self.replies.pop(0) if self.replies else {"text": "(nothing scripted)"}
        for piece in reply.get("chunks") or ([reply["text"]] if reply.get("text") else []):
            yield {"type": "text", "text": piece}
        if reply.get("reasoning"):
            yield {"type": "reasoning", "text": reply["reasoning"]}
        if reply.get("wait"):
            while not (abort and abort()):
                time.sleep(0.01)
            yield {"type": "result", "usage": {}, "finish_reason": "cancelled", "cancelled": True,
                   "message": {"role": "assistant", "content": reply.get("text") or ""}}
            return
        message = {"role": "assistant", "content": reply.get("content", reply.get("text") or "")}
        if reply.get("reasoning"):
            message["reasoning_content"] = reply["reasoning"]
        if reply.get("tool_calls"):
            message["tool_calls"] = [{"id": f"call-{index}", "type": "function", "function":
                                      {"name": name, "arguments": json.dumps(args)}}
                                     for index, (name, args) in enumerate(reply["tool_calls"])]
        yield {"type": "result", "message": message,
               "usage": reply.get("usage") or {"prompt_tokens": 900, "completion_tokens": 20},
               "finish_reason": reply.get("finish", "stop"), "cancelled": False}


class FailingEngine(ScriptEngine):
    """A stream that dies mid-turn, the way a crashed engine does."""

    def chat_stream(self, messages, tools=None, abort=None, **kwargs):
        from fennec_core.engine import EngineError
        self.calls.append({"messages": list(messages), "tools": list(tools or [])})
        raise EngineError("Engine stream failed; check engine health and configuration")


class BootingEngine(ScriptEngine):
    """Answers ``start`` with "still spawning" and comes up ``up_after`` seconds later."""

    def __init__(self, *replies, up_after=0.0, **kwargs):
        super().__init__(*replies, **kwargs)
        self.up_after = up_after
        self.spawned = None

    def start(self, wait_s=180):
        self.started += 1
        self.spawned = time.monotonic()
        return {"running": False, "starting": True, "owned": True}

    def is_running(self):
        if self.running:
            return True
        if self.up_after and self.spawned is not None \
                and time.monotonic() - self.spawned >= self.up_after:
            self.running = True
        return self.running


class SlowStartEngine(ScriptEngine):
    """A spawn that blocks, so a cancel can land inside it."""

    def __init__(self, *replies, delay=1.5, **kwargs):
        super().__init__(*replies, **kwargs)
        self.delay = delay

    def start(self, wait_s=180):
        self.started += 1
        time.sleep(self.delay)
        self.running = True
        return {"running": True, "started": True, "owned": True}


class NoOutlets:
    def __init__(self, *available):
        self.enabled = set(available)
        self.calls = []

    def available(self, name):
        return name in self.enabled

    def web_task(self, goal, url="", **kwargs):
        self.calls.append(("web_task", goal, url))
        return {"ok": True, "outlet": "jev-ultrafast", "status": "done", "text": "the page says hi"}

    def coworker_task(self, goal, agent="", **kwargs):
        self.calls.append(("coworker_task", goal, agent))
        return {"ok": True, "outlet": "openbot", "answer": "done"}


class _Reader:
    def __init__(self, source):
        self.source = source

    def __iter__(self):
        while True:
            line = self.source.get()
            if line is None:
                return
            yield line


class Client:
    """The editor side: one JSON object per line, both directions."""

    def __init__(self, engine, *, store, outlets=None, permission="allow_once", environ=None):
        self.source = queue.Queue()
        self.lock = threading.Lock()
        self.messages = []
        self.permissions = []
        self.permission = permission
        self.permission_result = "selected"
        self._id = 0
        self.engine = engine
        env = {"FENNEC_ACP_APPROVE_TIMEOUT": "5", "FENNEC_ACP_MAX_STEPS": "6",
               "FENNEC_ACP_IDLE_TIMEOUT": "3600"}
        env.update(environ or {})
        self.agent = acp.Agent(reader=_Reader(self.source), writer=_Recorder(self),
                               store_dir=store, engine_client=engine,
                               outlets=outlets or NoOutlets(), environ=env)
        self.thread = threading.Thread(target=self.agent.run, daemon=True)
        self.thread.start()

    def receive(self, text):
        for line in text.splitlines():
            if not line.strip():
                continue
            message = json.loads(line)
            with self.lock:
                self.messages.append(message)
            if "method" in message and "id" in message:
                self._answer(message)

    def _answer(self, message):
        if message["method"] != "session/request_permission":
            return
        self.permissions.append(message)
        if self.permission is None:
            return
        if self.permission_result == "cancelled":
            result = {"outcome": {"outcome": "cancelled"}}
        else:
            result = {"outcome": {"outcome": "selected", "optionId": self.permission}}
        self.source.put(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}) + "\n")

    def put(self, payload):
        self.source.put(json.dumps(payload, separators=(",", ":")) + "\n")

    def raw(self, line):
        self.source.put(line if line.endswith("\n") else line + "\n")

    def send(self, method, params=None):
        self._id += 1
        request_id = self._id
        self.put({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})
        return request_id

    def request(self, method, params=None, timeout=15):
        request_id = self.send(method, params)
        return self.response(request_id, timeout)

    def notify(self, method, params=None):
        self.put({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def _find(self, predicate):
        with self.lock:
            for message in self.messages:
                if predicate(message):
                    return message

    def response(self, request_id, timeout=15):
        deadline = time.time() + timeout
        while time.time() < deadline:
            found = self._find(lambda m: m.get("id") == request_id and "method" not in m)
            if found is not None:
                return found
            time.sleep(0.01)
        raise AssertionError(f"no response to request {request_id}")

    def result(self, request_id, timeout=15):
        found = self.response(request_id, timeout)
        assert "result" in found, f"request {request_id} failed: {found.get('error')}"
        return found["result"]

    def error(self, request_id, timeout=15):
        found = self.response(request_id, timeout)
        assert "error" in found, f"request {request_id} succeeded: {found.get('result')}"
        return found["error"]

    def updates(self, session_id=None, kind=None, timeout=15):
        deadline = time.time() + timeout
        while True:
            with self.lock:
                found = [message["params"]["update"] for message in self.messages
                         if message.get("method") == "session/update"
                         and (session_id is None or message["params"]["sessionId"] == session_id)
                         and (kind is None or message["params"]["update"].get("sessionUpdate") == kind)]
            if found:
                return found
            if time.time() >= deadline:
                return found
            time.sleep(0.01)

    def seen(self, kind, session_id=None, timeout=15):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.updates(session_id, kind, timeout=0):
                return True
            time.sleep(0.01)
        return False

    def text(self, session_id=None):
        return "".join(update["content"]["text"] for update in self.updates(session_id)
                       if update["sessionUpdate"] == "agent_message_chunk")

    def start_session(self, cwd, mcp_servers=()):
        self.negotiated = self.request("initialize", {"protocolVersion": 1,
                                                     "clientInfo": {"name": "test",
                                                                   "version": "1"}})
        created = self.result(self.send("session/new", {"cwd": cwd,
                                                        "mcpServers": list(mcp_servers)}))
        self.session_id = created["sessionId"]
        return self.session_id

    def prompt(self, text, timeout=20):
        request_id = self.send("session/prompt", {"sessionId": self.session_id, "prompt":
                                                  [{"type": "text", "text": text}]})
        return self.response(request_id, timeout)

    def close(self):
        self.source.put(None)
        self.thread.join(10)
        self.agent.shutdown()


class _Recorder:
    def __init__(self, client):
        self.client = client

    def write(self, text):
        self.client.receive(text)
        return len(text)

    def flush(self):
        pass


class AcpCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
        self.store = self.root / "acp-store"
        self.store.mkdir()
        patcher = mock.patch("fennec_core.tokeneconomizer.Bridge", QuietBridge)
        patcher.start()
        self.addCleanup(patcher.stop)
        loop.SESSIONS.clear()
        self.addCleanup(loop.SESSIONS.clear)

    def client(self, engine, **kwargs):
        client = Client(engine, store=self.store, **kwargs)
        self.addCleanup(client.close)
        return client


class HandshakeTests(AcpCase):
    def test_initialize_answers_the_v1_capability_block(self):
        client = self.client(ScriptEngine())
        result = client.result(client.send("initialize", {"protocolVersion": 1,
                                                         "clientCapabilities": {"fs": {}}}))
        self.assertEqual(result["protocolVersion"], acp.PROTOCOL_VERSION)
        self.assertEqual(result["agentInfo"]["name"], "fennec")
        from fennec_core import __version__
        self.assertEqual(result["agentInfo"]["version"], __version__)
        capabilities = result["agentCapabilities"]
        self.assertTrue(capabilities["loadSession"])
        self.assertEqual(capabilities["promptCapabilities"],
                         {"image": False, "audio": False, "embeddedContext": True})
        self.assertEqual(capabilities["mcpCapabilities"], {"http": False, "sse": False})
        # sessionCapabilities markers are objects, never booleans.
        self.assertEqual(sorted(capabilities["sessionCapabilities"]), ["close", "list", "resume"])
        self.assertEqual(capabilities["sessionCapabilities"]["list"], {})
        self.assertEqual(result["authMethods"], [])

    def test_methods_before_initialize_are_invalid_requests(self):
        client = self.client(ScriptEngine())
        error = client.error(client.send("session/new", {"cwd": str(self.root), "mcpServers": []}))
        self.assertEqual(error["code"], acp.INVALID_REQUEST)

    def test_bad_json_and_unknown_method(self):
        client = self.client(ScriptEngine())
        client.raw("{not json at all")
        client.put([1, 2, 3])
        client.request("initialize", {"protocolVersion": 1})
        self.assertEqual(client.error(client.send("session/delete", {"sessionId": "x"}))["code"],
                         acp.METHOD_NOT_FOUND)
        codes = [message["error"]["code"] for message in client.messages if "error" in message]
        self.assertEqual(codes, [acp.PARSE_ERROR, acp.INVALID_REQUEST, acp.METHOD_NOT_FOUND])
        self.assertIsNone(client.messages[0]["id"])
        self.assertNotIn("result", client.messages[0])

    def test_notifications_are_never_answered(self):
        client = self.client(ScriptEngine())
        client.request("initialize", {"protocolVersion": 1})
        before = len(client.messages)
        client.notify("session/cancel", {"sessionId": "nope"})
        client.notify("some/client/notification", {})
        time.sleep(0.1)
        self.assertEqual(len(client.messages), before)

    def test_unadvertised_session_options_are_refused_not_ignored(self):
        client = self.client(ScriptEngine())
        client.start_session(str(self.root))
        error = client.error(client.send("session/new", {
            "cwd": str(self.root), "mcpServers": [],
            "systemPrompt": {"mode": "append", "content": [{"type": "text", "text": "be brief"}]}}))
        self.assertEqual(error["code"], acp.INVALID_PARAMS)
        self.assertIn("systemPrompt", error["message"])
        error = client.error(client.send("session/new", {
            "cwd": str(self.root), "mcpServers": [], "additionalDirectories": [str(self.root / "other")]}))
        self.assertIn("additionalDirectories", error["message"])


class SessionTests(AcpCase):
    def test_new_persists_and_lists(self):
        client = self.client(ScriptEngine({"text": "read it"}))
        session_id = client.start_session(str(self.root))
        self.assertTrue((self.store / f"acp-{session_id}.json").is_file())
        listed = client.result(client.send("session/list", {}))
        self.assertEqual([item["sessionId"] for item in listed["sessions"]], [session_id])
        self.assertEqual(Path(listed["sessions"][0]["cwd"]).resolve(), self.root.resolve())
        self.assertIsNone(listed["sessions"][0]["title"])

    def test_prompt_populates_the_title_used_by_list(self):
        client = self.client(ScriptEngine({"text": "the answer"}))
        session_id = client.start_session(str(self.root))
        self.assertEqual(client.prompt("Summarise src/app.py")["result"], {"stopReason": "end_turn"})
        listed = client.result(client.send("session/list", {"cwd": str(self.root)}))
        self.assertEqual(listed["sessions"][0]["title"], "Summarise src/app.py")

    def test_unknown_session_is_resource_not_found_with_the_id(self):
        client = self.client(ScriptEngine())
        client.start_session(str(self.root))
        for method in ("session/prompt", "session/set_mode", "session/close"):
            params = {"sessionId": "missing"}
            if method == "session/prompt":
                params["prompt"] = [{"type": "text", "text": "x"}]
            if method == "session/set_mode":
                params["modeId"] = "ask"
            error = client.error(client.send(method, params))
            self.assertEqual(error["code"], acp.SESSION_NOT_FOUND, method)
            self.assertEqual(error["data"]["sessionId"], "missing")

    def test_close_frees_the_session_and_load_refuses_a_foreign_id(self):
        client = self.client(ScriptEngine({"text": "done"}))
        session_id = client.start_session(str(self.root))
        client.prompt("hello")
        self.assertEqual(client.result(client.send("session/close", {"sessionId": session_id})), {})
        self.assertEqual(client.error(client.send("session/prompt", {
            "sessionId": session_id, "prompt": [{"type": "text", "text": "again"}]}))["code"],
            acp.SESSION_NOT_FOUND)
        self.assertEqual(client.error(client.send("session/load", {
            "sessionId": "../escape", "cwd": str(self.root), "mcpServers": []}))["code"],
            acp.SESSION_NOT_FOUND)

    def test_set_mode_switches_and_rejects_the_impossible(self):
        client = self.client(ScriptEngine())
        session_id = client.start_session(str(self.root))
        self.assertEqual(client.result(client.send("session/set_mode",
                                                   {"sessionId": session_id, "modeId": "ask"})), {})
        kinds = [update for update in client.updates(session_id, "current_mode_update")]
        self.assertEqual(kinds[0]["currentModeId"], "ask")
        self.assertEqual(client.error(client.send("session/set_mode",
                                                  {"sessionId": session_id, "modeId": "yolo"}))["code"],
                         acp.INVALID_PARAMS)
        self.assertEqual(client.error(client.send("session/set_config_option",
                                                  {"sessionId": session_id, "configId": "model",
                                                   "value": "x"}))["code"], acp.METHOD_NOT_FOUND)

    def test_ask_mode_offers_no_write_or_outlet_tools(self):
        engine = ScriptEngine({"text": "only reading here"})
        client = self.client(engine, outlets=NoOutlets("web_task"))
        session_id = client.start_session(str(self.root))
        client.result(client.send("session/set_mode", {"sessionId": session_id, "modeId": "ask"}))
        client.prompt("what does app.py do?")
        names = {call["function"]["name"] for call in engine.calls[0]["tools"]}
        self.assertIn("read_file", names)
        self.assertNotIn("write_file", names)
        self.assertNotIn("run_shell", names)
        self.assertNotIn("web_task", names)
        self.assertEqual(engine.calls[0]["tools"][0]["type"], "function")

    def test_two_prompts_on_one_session_continue_the_transcript(self):
        engine = ScriptEngine({"text": "first"}, {"text": "second"})
        client = self.client(engine)
        client.start_session(str(self.root))
        self.assertEqual(client.prompt("read src/app.py")["result"]["stopReason"], "end_turn")
        self.assertEqual(client.prompt("and now?")["result"]["stopReason"], "end_turn")
        second = engine.calls[-1]["messages"]
        roles = [message["role"] for message in second]
        self.assertEqual(roles[0], "system")
        self.assertIn("read src/app.py", [message.get("content") for message in second])
        self.assertEqual(second[-1]["content"], "and now?")


class TurnTests(AcpCase):
    def test_a_tool_turn_streams_every_update(self):
        engine = ScriptEngine(
            {"reasoning": "let me look", "tool_calls": [("read_file", {"path": "src/app.py"})]},
            {"text": "It prints hi."})
        client = self.client(engine)
        session_id = client.start_session(str(self.root))
        response = client.prompt("what is in app.py?")
        self.assertEqual(response["result"], {"stopReason": "end_turn"})
        updates = client.updates(session_id, timeout=0)
        kinds = [update["sessionUpdate"] for update in updates]
        # The client already showed what it typed; the user chunk is stored for
        # session/load replay instead of streamed back.
        self.assertNotIn("user_message_chunk", kinds)
        self.assertIn("agent_thought_chunk", kinds)
        opened = next(update for update in updates if update["sessionUpdate"] == "tool_call")
        self.assertEqual(opened["kind"], "read")
        self.assertEqual(opened["status"], "in_progress")
        self.assertEqual(opened["rawInput"], {"path": "src/app.py"})
        self.assertEqual(opened["title"], "read_file: src/app.py")
        self.assertEqual(Path(opened["locations"][0]["path"]).name, "app.py")
        closed = next(update for update in updates
                      if update["sessionUpdate"] == "tool_call_update"
                      and update["toolCallId"] == opened["toolCallId"])
        self.assertEqual(closed["status"], "completed")
        self.assertIn("print('hi')", closed["content"][0]["content"]["text"])
        self.assertEqual(client.text(session_id), "It prints hi.")
        plans = [update for update in updates if update["sessionUpdate"] == "plan"]
        self.assertEqual([entry["status"] for entry in plans[-1]["entries"]][0], "completed")
        usage = next(update for update in updates if update["sessionUpdate"] == "usage_update")
        self.assertEqual(usage["used"], 900)
        self.assertGreaterEqual(usage["size"], 4096)
        message_ids = {update["messageId"] for update in updates
                       if update["sessionUpdate"] == "agent_message_chunk"}
        self.assertEqual(len(message_ids), 1)

    def test_chunk_ids_open_a_new_message_after_a_tool(self):
        engine = ScriptEngine({"text": "opening ", "tool_calls": [("list_dir", {"path": "."})]},
                              {"text": "closing"})
        client = self.client(engine)
        session_id = client.start_session(str(self.root))
        client.prompt("walk the tree")
        chunks = [update for update in client.updates(session_id, timeout=0)
                  if update["sessionUpdate"] == "agent_message_chunk"]
        self.assertEqual(len({chunk["messageId"] for chunk in chunks}), 2)
        self.assertEqual(client.text(session_id), "opening closing")

    def test_an_absolutely_long_delta_is_split_not_dropped(self):
        engine = ScriptEngine({"text": "x" * 9000})
        client = self.client(engine)
        session_id = client.start_session(str(self.root))
        client.prompt("say a lot")
        chunks = client.updates(session_id, "agent_message_chunk", timeout=0)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(len(client.text(session_id)), 9000)

    def test_permission_is_requested_before_a_write_and_the_answer_is_honoured(self):
        for decision, expected in (("allow_once", "completed"), ("reject_once", "failed")):
            with self.subTest(decision=decision):
                engine = ScriptEngine({"tool_calls": [("write_file", {"path": f"{decision}.md",
                                                                     "content": "hello"})]},
                                      {"text": "written" if decision == "allow_once" else "refused"})
                client = self.client(engine, permission=decision)
                session_id = client.start_session(str(self.root))
                self.assertEqual(client.prompt("make a note")["result"]["stopReason"], "end_turn")
                self.assertEqual(len(client.permissions), 1)
                request = client.permissions[0]["params"]
                self.assertEqual(request["sessionId"], session_id)
                self.assertEqual(request["toolCall"]["kind"], "edit")
                self.assertEqual(request["toolCall"]["status"], "pending")
                self.assertEqual({option["kind"] for option in request["options"]},
                                 {"allow_once", "allow_always", "reject_once", "reject_always"})
                update = [item for item in client.updates(session_id, "tool_call_update", timeout=0)
                          if item["toolCallId"] == request["toolCall"]["toolCallId"]]
                self.assertEqual(update[-1]["status"], expected)
                self.assertEqual((self.root / f"{decision}.md").is_file(),
                                 decision == "allow_once")
                if decision == "reject_once":
                    tool_message = [message for message in engine.calls[-1]["messages"]
                                    if message["role"] == "tool"][-1]
                    self.assertIn("rejected by the user", tool_message["content"])

    def test_allow_always_stops_asking_for_that_tool(self):
        engine = ScriptEngine({"tool_calls": [("write_file", {"path": "a.md", "content": "1"})]},
                              {"tool_calls": [("write_file", {"path": "b.md", "content": "2"})]},
                              {"text": "both"})
        client = self.client(engine, permission="allow_always")
        client.start_session(str(self.root))
        client.prompt("write two files")
        self.assertEqual(len(client.permissions), 1)
        self.assertTrue((self.root / "a.md").is_file() and (self.root / "b.md").is_file())

    def test_a_silent_client_means_rejection_not_a_hang(self):
        engine = ScriptEngine({"tool_calls": [("run_shell", {"command": "echo hi"})]}, {"text": "no"})
        client = self.client(engine, permission=None)
        client.agent.approve_timeout = 0.3
        client.start_session(str(self.root))
        self.assertEqual(client.prompt("run something")["result"]["stopReason"], "end_turn")
        self.assertEqual(len(client.permissions), 1)
        failed = [update for update in client.updates(timeout=0)
                  if update["sessionUpdate"] == "tool_call_update" and update["status"] == "failed"]
        self.assertTrue(failed)

    def test_outlet_tools_appear_only_when_the_outlet_exists(self):
        engine = ScriptEngine({"tool_calls": [("web_task", {"goal": "open the cart",
                                                            "url": "https://example.com"})]},
                              {"text": "the cart is empty"})
        client = self.client(engine, outlets=NoOutlets("web_task"), permission="allow_once")
        session_id = client.start_session(str(self.root))
        client.prompt("check my cart")
        names = {call["function"]["name"] for call in engine.calls[0]["tools"]}
        self.assertIn("web_task", names)
        self.assertNotIn("coworker_task", names)
        opened = next(update for update in client.updates(session_id, "tool_call", timeout=0)
                      if update["title"].startswith("web_task"))
        self.assertEqual(opened["kind"], "fetch")
        self.assertEqual(client.agent._outlets().calls[0][0], "web_task")

    def test_cancelling_a_turn_answers_cancelled_never_an_error(self):
        engine = ScriptEngine({"text": "half a sentence", "wait": True})
        client = self.client(engine)
        client.start_session(str(self.root))
        request_id = client.send("session/prompt", {"sessionId": client.session_id,
                                                    "prompt": [{"type": "text", "text": "talk"}]})
        self.assertTrue(client.seen("agent_message_chunk", timeout=5))
        client.notify("session/cancel", {"sessionId": client.session_id})
        self.assertEqual(client.response(request_id, 10)["result"], {"stopReason": "cancelled"})
        self.assertNotIn("error", client.response(request_id, 1))

    def test_cancel_request_stops_the_matching_turn(self):
        engine = ScriptEngine({"text": "working", "wait": True})
        client = self.client(engine)
        client.start_session(str(self.root))
        request_id = client.send("session/prompt", {"sessionId": client.session_id,
                                                    "prompt": [{"type": "text", "text": "long job"}]})
        self.assertTrue(client.seen("agent_message_chunk", timeout=5))
        client.notify("$/cancel_request", {"requestId": request_id})
        self.assertEqual(client.response(request_id, 10)["result"], {"stopReason": "cancelled"})

    def test_cancelling_a_client_request_does_not_answer_our_own(self):
        # Fennec's session/request_permission carries id "fennec-1" while the
        # client's own request 1 is still in flight. A $/cancel_request for that
        # number means "drop my request", not "your permission question is answered".
        engine = ScriptEngine({"tool_calls": [("write_file", {"path": "kept.md",
                                                              "content": "1"})]},
                              {"text": "written"})
        client = self.client(engine, permission=None)
        client.agent.approve_timeout = 10
        session_id = client.start_session(str(self.root))
        request_id = client.send("session/prompt", {"sessionId": session_id,
                                                    "prompt": [{"type": "text", "text": "write"}]})
        deadline = time.time() + 10
        while not client.permissions and time.time() < deadline:
            time.sleep(0.02)
        self.assertTrue(client.permissions, "no permission request reached the client")
        client.notify("$/cancel_request", {"requestId": 1})
        time.sleep(0.3)
        self.assertEqual([message for message in client.messages if message.get("id") == request_id],
                         [], "the turn answered out from under the pending permission")
        client.put({"jsonrpc": "2.0", "id": client.permissions[0]["id"],
                    "result": {"outcome": {"outcome": "selected", "optionId": "allow_once"}}})
        self.assertEqual(client.response(request_id, 15)["result"], {"stopReason": "end_turn"})
        self.assertTrue((self.root / "kept.md").is_file())

    def test_a_request_id_already_in_flight_is_refused(self):
        engine = ScriptEngine({"text": "working", "wait": True})
        client = self.client(engine)
        session_id = client.start_session(str(self.root))
        prompt_id = client.send("session/prompt", {"sessionId": session_id,
                                                   "prompt": [{"type": "text", "text": "long job"}]})
        self.assertTrue(client.seen("agent_message_chunk", timeout=5))
        # A client that reuses its own id must get an error, never a second answer
        # the original caller then mistakes for its own.
        client.put({"jsonrpc": "2.0", "id": prompt_id, "method": "session/list", "params": {}})
        refused = []
        deadline = time.time() + 5
        while time.time() < deadline and not refused:
            refused = [message for message in client.messages
                       if message.get("id") == prompt_id and "error" in message]
            time.sleep(0.02)
        self.assertTrue(refused, "reusing an in-flight request id was not refused")
        self.assertEqual(refused[0]["error"]["code"], acp.INVALID_REQUEST)
        self.assertIn("already has a turn in flight", refused[0]["error"]["message"])
        self.assertEqual([message for message in client.messages
                          if message.get("id") == prompt_id and "sessions" in (message.get("result") or {})],
                         [], "the second request ran instead of being refused")
        client.notify("session/cancel", {"sessionId": session_id})
        answered = []
        deadline = time.time() + 10
        while time.time() < deadline and not answered:
            answered = [message for message in client.messages
                        if message.get("id") == prompt_id and "result" in message]
            time.sleep(0.02)
        self.assertEqual([message["result"] for message in answered], [{"stopReason": "cancelled"}])

    def test_a_second_prompt_on_a_busy_session_is_refused(self):
        engine = ScriptEngine({"text": "still going", "wait": True}, {"text": "never"})
        client = self.client(engine)
        client.start_session(str(self.root))
        first = client.send("session/prompt", {"sessionId": client.session_id,
                                               "prompt": [{"type": "text", "text": "one"}]})
        self.assertTrue(client.seen("agent_message_chunk", timeout=5))
        self.assertEqual(client.error(client.send("session/prompt", {
            "sessionId": client.session_id,
            "prompt": [{"type": "text", "text": "two"}]}))["code"], acp.INVALID_REQUEST)
        client.notify("session/cancel", {"sessionId": client.session_id})
        self.assertEqual(client.response(first, 10)["result"]["stopReason"], "cancelled")

    def test_step_limit_reports_max_turn_requests(self):
        engine = ScriptEngine(*[{"tool_calls": [("list_dir", {"path": "."})]}] * 8)
        client = self.client(engine)
        client.agent.max_steps = 2
        client.start_session(str(self.root))
        self.assertEqual(client.prompt("over and over")["result"]["stopReason"], "max_turn_requests")

    def test_a_cut_off_answer_reports_max_tokens(self):
        engine = ScriptEngine({"text": "the answer was trunca", "finish": "length"})
        client = self.client(engine)
        client.start_session(str(self.root))
        self.assertEqual(client.prompt("answer long")["result"]["stopReason"], "max_tokens")

    def test_engine_failure_is_an_internal_error(self):
        client = self.client(FailingEngine())
        session_id = client.start_session(str(self.root))
        error = client.error(client.send("session/prompt", {"sessionId": session_id,
                                                            "prompt": [{"type": "text",
                                                                        "text": "go"}]}))
        self.assertEqual(error["code"], acp.INTERNAL_ERROR)
        self.assertIn("Engine stream failed", error["message"])
        self.assertIn(session_id, error["data"]["sessionId"])

    def test_a_failing_tool_is_reported_and_the_turn_continues(self):
        engine = ScriptEngine({"tool_calls": [("read_file", {"path": "missing.py"})]},
                              {"text": "it is not there"})
        client = self.client(engine)
        session_id = client.start_session(str(self.root))
        self.assertEqual(client.prompt("read missing")["result"]["stopReason"], "end_turn")
        failed = [update for update in client.updates(session_id, "tool_call_update", timeout=0)
                  if update["status"] == "failed"]
        self.assertIn("not a file", failed[0]["content"][0]["content"]["text"])

    def test_paths_outside_the_workspace_are_never_offered_as_locations(self):
        engine = ScriptEngine({"tool_calls": [("read_file", {"path": "../secrets.env"})]},
                              {"text": "no"})
        client = self.client(engine, permission="reject_once")
        session_id = client.start_session(str(self.root))
        client.prompt("read outside")
        opened = client.updates(session_id, "tool_call", timeout=0)
        self.assertTrue(all("locations" not in update for update in opened))

    def test_image_only_prompts_are_invalid_params(self):
        client = self.client(ScriptEngine())
        client.start_session(str(self.root))
        error = client.error(client.send("session/prompt", {
            "sessionId": client.session_id,
            "prompt": [{"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"}]}))
        self.assertEqual(error["code"], acp.INVALID_PARAMS)
        self.assertEqual(client.error(client.send("session/prompt",
                                                  {"sessionId": client.session_id, "prompt": []}))["code"],
                         acp.INVALID_PARAMS)

    def test_resources_arrive_as_text_and_media_is_named(self):
        engine = ScriptEngine({"text": "read both"})
        client = self.client(engine)
        client.start_session(str(self.root))
        client.result(client.send("session/prompt", {"sessionId": client.session_id, "prompt": [
            {"type": "text", "text": "compare"},
            {"type": "resource", "resource": {"uri": "file:///a.txt", "text": "BODY"}},
            {"type": "resource_link", "name": "notes", "uri": "file:///b.txt"},
            {"type": "image", "data": "aGk=", "mimeType": "image/png"}]}))
        user = [message for message in engine.calls[0]["messages"] if message["role"] == "user"][-1]
        self.assertIn("BODY", user["content"])
        self.assertIn("notes", user["content"])
        self.assertIn("image content (not supported by Fennec)", user["content"])


class TranscriptTests(AcpCase):
    def test_load_replays_and_resume_does_not(self):
        engine = ScriptEngine({"reasoning": "thinking",
                               "tool_calls": [("read_file", {"path": "src/app.py"})]},
                              {"text": "done reading"})
        client = self.client(engine)
        session_id = client.start_session(str(self.root))
        client.prompt("read the file")
        client.close()
        expected = len(json.loads((self.store / f"acp-{session_id}.json")
                                  .read_text(encoding="utf-8"))["updates"])

        second = self.client(ScriptEngine())
        second.request("initialize", {"protocolVersion": 1})
        loaded = second.result(second.send("session/load", {"sessionId": session_id,
                                                            "cwd": str(self.root), "mcpServers": []}))
        self.assertEqual(loaded["modes"]["currentModeId"], "agent")
        replayed = second.updates(session_id, timeout=0)
        self.assertEqual(len(replayed), expected)
        self.assertIn("user_message_chunk", [item["sessionUpdate"] for item in replayed])
        self.assertEqual(second.text(session_id), "done reading")

        third = self.client(ScriptEngine())
        third.request("initialize", {"protocolVersion": 1})
        resumed = third.result(third.send("session/resume", {"sessionId": session_id,
                                                             "cwd": str(self.root), "mcpServers": []}))
        self.assertIn("modes", resumed)
        time.sleep(0.2)
        self.assertEqual([message for message in third.messages
                          if message.get("method") == "session/update"], [])

    def test_a_resumed_session_continues_the_old_transcript(self):
        engine = ScriptEngine({"text": "first answer"})
        client = self.client(engine)
        session_id = client.start_session(str(self.root))
        client.prompt("the first question")
        client.close()

        again = ScriptEngine({"text": "second answer"})
        second = self.client(again)
        second.request("initialize", {"protocolVersion": 1})
        second.result(second.send("session/load", {"sessionId": session_id, "cwd": str(self.root),
                                                   "mcpServers": []}))
        second.session_id = session_id
        self.assertEqual(second.prompt("the second question")["result"]["stopReason"], "end_turn")
        roles = [message["role"] for message in again.calls[0]["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])
        self.assertEqual(again.calls[0]["messages"][2]["content"], "first answer")

    def test_load_of_a_session_that_never_existed(self):
        client = self.client(ScriptEngine())
        client.request("initialize", {"protocolVersion": 1})
        error = client.error(client.send("session/load", {"sessionId": "ghost", "cwd": str(self.root),
                                                           "mcpServers": []}))
        self.assertEqual(error["code"], acp.SESSION_NOT_FOUND)

    def test_the_stored_transcript_holds_the_tools_and_results(self):
        engine = ScriptEngine({"tool_calls": [("grep", {"pattern": "hi", "path": "src"})]},
                              {"text": "one hit"})
        client = self.client(engine)
        session_id = client.start_session(str(self.root))
        client.prompt("find hi")
        record = json.loads((self.store / f"acp-{session_id}.json").read_text(encoding="utf-8"))
        self.assertEqual(record["sessionId"], session_id)
        self.assertEqual(record["cwd"], str(self.root))
        self.assertEqual([message["role"] for message in record["history"]],
                         ["system", "user", "assistant", "tool", "assistant"])
        self.assertEqual(record["history"][3]["tool_call_id"], "call-0")
        stored = json.dumps(record["updates"])
        self.assertIn('"sessionUpdate": "plan"', stored)
        self.assertIn("grep: src", stored)
        # usage_update is live state, so it is streamed but not replayed later.
        self.assertNotIn("usage_update", stored)
        self.assertTrue(client.updates(session_id, "usage_update", timeout=0))


class AdoptTests(AcpCase):
    """``session/load`` and ``session/resume`` on a session this process still holds."""

    def spec(self, name):
        script = self.root / f"{name}_mcp_server.py"
        script.write_text(MCP_SERVER, encoding="utf-8")
        return {"name": name, "type": "stdio", "command": sys.executable,
                "args": [str(script)], "env": []}

    def wait_idle(self, client, session_id):
        deadline = time.time() + 10
        while client.agent.sessions[session_id].busy and time.time() < deadline:
            time.sleep(0.02)

    def test_reloading_a_live_session_on_another_workspace_is_refused(self):
        client = self.client(ScriptEngine({"text": "hi"}))
        session_id = client.start_session(str(self.root))
        client.prompt("hello")
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        error = client.error(client.send("session/load", {"sessionId": session_id,
                                                          "cwd": str(elsewhere), "mcpServers": []}))
        self.assertEqual(error["code"], acp.INVALID_PARAMS)
        self.assertIn("another workspace", error["message"])
        # A refusal must leave the running session exactly as it was.
        self.assertEqual(client.agent.sessions[session_id].cwd, str(self.root.resolve()))
        self.assertEqual(client.prompt("and again")["result"]["stopReason"], "end_turn")

    def test_the_same_servers_are_a_no_op(self):
        client = self.client(ScriptEngine())
        session_id = client.start_session(str(self.root), mcp_servers=[self.spec("same")])
        pool = client.agent.sessions[session_id].mcp
        result = client.result(client.send("session/load", {
            "sessionId": session_id, "cwd": str(self.root), "mcpServers": [self.spec("same")]}))
        self.assertEqual(result["modes"]["currentModeId"], "agent")
        self.assertIs(client.agent.sessions[session_id].mcp, pool)
        self.assertEqual(list(pool.servers), ["same"])

    def test_different_servers_wait_for_the_turn_to_end(self):
        client = self.client(ScriptEngine({"text": "working", "wait": True}))
        session_id = client.start_session(str(self.root), mcp_servers=[self.spec("first")])
        pool = client.agent.sessions[session_id].mcp
        request_id = client.send("session/prompt", {"sessionId": session_id,
                                                    "prompt": [{"type": "text", "text": "go"}]})
        self.assertTrue(client.seen("agent_message_chunk", timeout=5))
        error = client.error(client.send("session/load", {
            "sessionId": session_id, "cwd": str(self.root), "mcpServers": [self.spec("second")]}))
        self.assertEqual(error["code"], acp.INVALID_REQUEST)
        self.assertIn("mid-turn", error["message"])
        self.assertIs(client.agent.sessions[session_id].mcp, pool)
        client.notify("session/cancel", {"sessionId": session_id})
        self.assertEqual(client.response(request_id, 10)["result"]["stopReason"], "cancelled")
        self.wait_idle(client, session_id)
        client.result(client.send("session/resume", {"sessionId": session_id, "cwd": str(self.root),
                                                     "mcpServers": [self.spec("second")]}))
        self.assertEqual(list(client.agent.sessions[session_id].mcp.servers), ["second"])
        self.assertIsNot(client.agent.sessions[session_id].mcp, pool)
        self.assertEqual(pool.servers, {}, "the replaced pool was never shut down")


class McpTests(AcpCase):
    def server_spec(self, name="demo"):
        script = self.root / "demo_mcp_server.py"
        script.write_text(MCP_SERVER, encoding="utf-8")
        return {"name": name, "type": "stdio", "command": sys.executable,
                "args": [str(script)], "env": []}

    def test_a_stdio_server_becomes_a_tool_and_answers(self):
        engine = ScriptEngine({"tool_calls": [("mcp__demo__echo", {"text": "hello"})]},
                              {"text": "the server said hello"})
        client = self.client(engine)
        session_id = client.start_session(str(self.root), mcp_servers=[self.server_spec()])
        self.assertEqual(client.negotiated["result"]
                         ["agentCapabilities"]["mcpCapabilities"], {"http": False, "sse": False})
        self.assertEqual(client.prompt("echo hello")["result"]["stopReason"], "end_turn")
        names = {call["function"]["name"] for call in engine.calls[0]["tools"]}
        self.assertIn("mcp__demo__echo", names)
        tool = next(update for update in client.updates(session_id, "tool_call", timeout=0)
                    if update["toolCallId"] == "call-0")
        self.assertEqual(tool["kind"], "other")
        result = [message for message in engine.calls[-1]["messages"] if message["role"] == "tool"][-1]
        self.assertIn("echo:hello", result["content"])

    def test_a_server_that_stops_answering_is_given_up_on(self):
        script = self.root / "hang_mcp_server.py"
        script.write_text(MCP_HANG, encoding="utf-8")
        engine = ScriptEngine({"text": "moving on"})
        client = self.client(engine)
        session_id = client.start_session(str(self.root), mcp_servers=[
            {"name": "sloth", "type": "stdio", "command": sys.executable,
             "args": [str(script)], "env": []}])
        with mock.patch.object(mcp, "CALL_TIMEOUT_S", 1):
            # Before the reader thread existed this blocked the whole turn, and
            # after it the lock, until the client gave up on the editor.
            self.assertEqual(client.prompt("hello", timeout=30)["result"],
                             {"stopReason": "end_turn"})
        thoughts = [update["content"]["text"] for update in
                    client.updates(session_id, "agent_thought_chunk", timeout=0)]
        self.assertTrue(any("sloth" in text and "timed out" in text for text in thoughts),
                        thoughts)
        self.assertEqual(client.agent.sessions[session_id].mcp.servers, {})
        names = {call["function"]["name"] for call in engine.calls[0]["tools"]}
        self.assertNotIn("mcp__sloth__echo", names)
        self.assertEqual(client.text(session_id), "moving on")

    def test_an_http_server_is_reported_not_connected(self):
        engine = ScriptEngine({"text": "moving on"})
        client = self.client(engine)
        session_id = client.start_session(str(self.root), mcp_servers=[
            {"name": "remote", "type": "http", "url": "https://example.com/mcp", "headers": []}])
        client.prompt("hello")
        thoughts = [update["content"]["text"] for update in
                    client.updates(session_id, "agent_thought_chunk", timeout=0)]
        self.assertTrue(any("stdio only" in thought for thought in thoughts))


class EngineOwnershipTests(AcpCase):
    def test_the_engine_is_started_once_and_stopped_on_exit(self):
        engine = ScriptEngine({"text": "one"}, {"text": "two"}, running=False)
        client = self.client(engine)
        session_id = client.start_session(str(self.root))
        client.prompt("first")
        client.prompt("second")
        self.assertEqual(engine.started, 1)
        self.assertFalse(engine.stopped)
        client.close()
        self.assertTrue(engine.stopped)

    def test_a_turn_waits_for_an_engine_that_is_still_spawning(self):
        engine = BootingEngine({"text": "late but here"}, running=False, up_after=0.8)
        client = self.client(engine)
        session_id = client.start_session(str(self.root))
        self.assertEqual(client.prompt("hello", timeout=20)["result"], {"stopReason": "end_turn"})
        self.assertEqual(engine.started, 1)
        self.assertEqual(client.text(session_id), "late but here")
        # The wait must not lose the ownership flag, or the engine we spawned is
        # the one process nobody ever stops.
        client.close()
        self.assertTrue(engine.stopped)

    def test_an_engine_that_never_finishes_spawning_is_an_error(self):
        engine = BootingEngine({"text": "never"}, running=False, up_after=9999)
        client = self.client(engine)
        client.agent.max_steps = 2
        session_id = client.start_session(str(self.root))
        with mock.patch.object(acp, "ENGINE_START_WAIT_S", 0.4):
            error = client.error(client.send("session/prompt", {"sessionId": session_id,
                                                                "prompt": [{"type": "text",
                                                                            "text": "go"}]}))
        self.assertEqual(error["code"], acp.INTERNAL_ERROR)
        self.assertIn("not answering", error["message"])

    def test_a_cancel_landing_inside_the_engine_spawn_is_honoured(self):
        engine = SlowStartEngine({"text": "never reached"}, running=False, delay=1.5)
        client = self.client(engine)
        session_id = client.start_session(str(self.root))
        request_id = client.send("session/prompt", {"sessionId": session_id,
                                                    "prompt": [{"type": "text", "text": "go"}]})
        deadline = time.time() + 5
        while not engine.started and time.time() < deadline:
            time.sleep(0.01)
        client.notify("session/cancel", {"sessionId": session_id})
        self.assertEqual(client.response(request_id, 10)["result"], {"stopReason": "cancelled"})
        self.assertEqual(engine.calls, [])

    def test_exit_ends_the_read_loop_without_another_stdin_line(self):
        engine = ScriptEngine({"text": "hi"}, running=False)
        client = self.client(engine)
        client.start_session(str(self.root))
        client.prompt("hello")
        client.notify("exit")
        # The pipe stays open here: an agent that only noticed the notification on
        # the next line would hold the process - and the engine it started - forever.
        client.thread.join(5)
        self.assertFalse(client.thread.is_alive())
        self.assertTrue(engine.stopped)

    def test_an_attach_only_engine_is_never_stopped_by_us(self):
        engine = ScriptEngine({"text": "hi"})
        engine.start = lambda wait_s=180: {"running": True, "already": True, "owned": False}
        client = self.client(engine)
        client.start_session(str(self.root))
        client.prompt("hello")
        client.close()
        self.assertFalse(engine.stopped)

    def test_a_dead_engine_is_an_error_the_editor_can_show(self):
        engine = ScriptEngine(running=False)
        engine.start = lambda wait_s=180: {"running": False, "error": "Engine exited during startup"}
        client = self.client(engine)
        client.agent.engine_autostart = True
        session_id = client.start_session(str(self.root))
        error = client.error(client.send("session/prompt", {"sessionId": session_id,
                                                            "prompt": [{"type": "text",
                                                                        "text": "go"}]}))
        self.assertEqual(error["code"], acp.INTERNAL_ERROR)
        self.assertIn("Engine exited during startup", error["message"])

    def test_an_engine_we_started_is_stopped_when_the_agent_goes_idle(self):
        engine = ScriptEngine({"text": "hi"}, running=False)
        client = self.client(engine, environ={"FENNEC_ACP_IDLE_TIMEOUT": "0.3"})
        client.start_session(str(self.root))
        client.prompt("hello")
        deadline = time.time() + 5
        while time.time() < deadline and not engine.stopped:
            time.sleep(0.05)
        self.assertTrue(engine.stopped)
        self.assertEqual(engine.running, False)

    def test_a_wedged_mcp_shutdown_still_stops_the_engine(self):
        engine = ScriptEngine({"text": "hi"}, running=False)
        client = self.client(engine)
        session_id = client.start_session(str(self.root))
        client.prompt("hello")

        def wedge():
            raise OSError("mcp server will not die")

        client.agent.sessions[session_id].mcp.stop = wedge
        client.close()
        # One unkillable server must not be able to keep an engine we spawned alive.
        self.assertTrue(engine.stopped)

    def test_autostart_can_be_left_to_the_host(self):
        engine = ScriptEngine({"text": "hi"})
        client = self.client(engine, environ={"FENNEC_ACP_ENGINE_START": "0"})
        client.start_session(str(self.root))
        client.prompt("hello")
        self.assertEqual(engine.started, 0)


class DiagnosticTests(AcpCase):
    def test_check_describes_the_agent_without_a_client(self):
        report = acp.check()
        self.assertEqual(report["protocolVersion"], 1)
        self.assertEqual(report["transport"], "stdio")
        self.assertEqual(report["modes"], ["agent", "ask"])
        self.assertIn("session/prompt", report["methods"])
        self.assertIn("session/cancel", report["methods"])
        self.assertTrue(Path(report["store"]).is_absolute())
        self.assertIn("jev", report["outlets"])


class StdioEncodingTests(AcpCase):
    """ACP is UTF-8 JSON-RPC, whatever the code page a spawned pipe starts on."""

    def _pipes(self, inbound=b""):
        raw_in, raw_out = io.BytesIO(inbound), io.BytesIO()
        # cp1250 is a real Windows ANSI code page: no emoji, no CJK.
        stdin = io.TextIOWrapper(raw_in, encoding="cp1250", newline="\n")
        stdout = io.TextIOWrapper(raw_out, encoding="cp1250", newline="\n")
        return raw_in, raw_out, stdin, stdout

    def test_a_code_page_pipe_still_carries_what_the_agent_answers(self):
        _, raw_out, stdin, stdout = self._pipes()
        text = "pong \U0001F600 你好"
        with mock.patch.object(sys, "stdin", stdin), mock.patch.object(sys, "stdout", stdout):
            agent = acp.Agent(store_dir=self.store)
            agent._write({"jsonrpc": "2.0", "id": 1, "result": {"text": text}})
        self.assertFalse(agent._stop.is_set(), "an emoji is not a dead client")
        self.assertEqual(json.loads(raw_out.getvalue().decode("utf-8"))["result"]["text"], text)

    def test_a_code_page_pipe_reads_what_the_editor_sent(self):
        payload = json.dumps({"jsonrpc": "2.0", "method": "session/prompt",
                              "params": {"prompt": [{"text": "daj przykład 漢字"}]}},
                             ensure_ascii=False).encode("utf-8") + b"\n"
        _, _, stdin, stdout = self._pipes(payload)
        with mock.patch.object(sys, "stdin", stdin), mock.patch.object(sys, "stdout", stdout):
            agent = acp.Agent(store_dir=self.store)
            line = next(iter(agent.reader))
        self.assertEqual(json.loads(line)["params"]["prompt"][0]["text"], "daj przykład 漢字")

    def test_the_check_report_prints_even_when_the_console_cannot(self):
        _, raw_out, stdin, stdout = self._pipes()
        with mock.patch.object(sys, "stdin", stdin), mock.patch.object(sys, "stdout", stdout), \
                mock.patch.object(acp, "check", lambda: {"model": "D:/漢字/qwen36-35b"}):
            self.assertEqual(acp.main(["--check"]), 0)
            stdout.flush()
        self.assertEqual(json.loads(raw_out.getvalue().decode("utf-8"))["model"],
                         "D:/漢字/qwen36-35b")

    def test_a_stream_that_cannot_be_reconfigured_is_left_alone(self):
        buffer = io.StringIO()
        self.assertIs(acp._utf8(buffer), buffer)
        self.assertEqual(buffer.write("ok\n"), 3)


if __name__ == "__main__":
    unittest.main()
