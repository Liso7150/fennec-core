import hermetic  # noqa: F401

import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient

from fennec_core import loop
from fennec_core import server as server_mod
from fennec_core.config import ConfigError, load_config
from fennec_core.server import main as agent_main
from fennec_core.server import make_app, serve

HEADERS = {"X-Fennec-Client": "1"}


class FakeEngine:
    def __init__(self, running=True):
        self.running = running
        self.calls = []
        self.reply = {"role": "assistant", "content": "done"}
        self.gate = None

    def is_running(self):
        return self.running

    def chat(self, messages, tools=None, tool_choice="auto", max_tokens=2048, temperature=0.2,
             timeout=600, reasoning_effort=None, enable_thinking=None):
        self.calls.append({"messages": list(messages), "reasoning_effort": reasoning_effort,
                           "enable_thinking": enable_thinking})
        if self.gate is not None:
            self.gate.wait(5)
        return self.reply, {"total_tokens": 3}

    def ensure_stt(self, wait_s=300):
        return {"running": True, "model": "test", "device": "cpu"}

    def transcribe(self, audio, content_type="audio/webm", timeout=180):
        self.audio = audio
        return {"text": "hello from audio", "language": "en", "duration_s": 1.0}

    def telemetry(self):
        return {"running": True, "backend": "vulkan"}

    def status(self):
        return {"running": True}

    def logs(self, n=60):
        return {"lines": []}

    def plan(self):
        return {"exit": 0, "out": "plan"}

    def doctor(self):
        return {"exit": 0, "out": "doctor"}

    def start(self, wait_s=180):
        return {"running": True}

    def stop(self):
        return {"stopped": False}

    def hardware(self):
        return {"platform": "test"}

    def ensure_npu(self, enabled=False):
        return {"running": False}

    def npu_health(self, timeout=2):
        return {"status": "down"}

    def embed(self, texts, timeout=120):
        return [[0.0]]


def make_config(root: Path, web: Path | None = None):
    env = {"HOME": str(root), "USERPROFILE": str(root)}
    with mock.patch.dict(os.environ, env, clear=True):
        config = load_config(environ=env, home=root)
    from dataclasses import replace
    return replace(config, workspace=root, models_root=root / "models",
                   cache_root=root / "cache", web_root=web or root / "missing-web")


OUTLET_STATUS = {"jev": {"present": False, "root": None, "importable": False, "launcher": None,
                         "policy_key": False, "paid_api": True, "note": "stubbed for tests"},
                 "openbot": {"present": False, "url": "http://127.0.0.1:3001", "loopback": True,
                             "agents": [], "error": "no OpenBot workspace answering"}}


class StubOutlets:
    """Stands in for the real outlets so no test can reach a browser or a port."""

    def __init__(self, config=None):
        self.config = config
        self.calls = []
        self.available_tools = set()

    def status(self, refresh=False):
        return OUTLET_STATUS

    def available(self, name):
        return name in self.available_tools

    def web_task(self, goal, url="", **kwargs):
        self.calls.append(("web_task", goal, url, kwargs))
        return {"ok": True, "outlet": "jev-ultrafast", "steps": 1}

    def coworker_task(self, goal, agent="", **kwargs):
        self.calls.append(("coworker_task", goal, agent, kwargs))
        return {"ok": False, "outlet": "openbot", "error": "stubbed"}


class ClientCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
        self.web = self.root / "web"
        self.web.mkdir()
        (self.web / "index.html").write_text("<!doctype html><title>Fennec</title>", encoding="utf-8")
        self.engine = FakeEngine()
        self.config = make_config(self.root, self.web)
        loop.SESSIONS.clear()
        # The real bridge opens a socket to 127.0.0.1:3001 and scans the disk for a
        # Jev checkout on the first turn of every session. On a machine that has
        # either installed that is a unit test reaching the network - and offering
        # the model a tool that spends money.
        self.outlets_stub = StubOutlets(self.config)
        patcher = mock.patch.object(server_mod, "Outlets", lambda *a, **kw: self.outlets_stub)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(make_app(self.config, self.engine, allowed_hosts=["127.0.0.1"]),
                                 base_url="http://127.0.0.1")

    def tearDown(self):
        loop.SESSIONS.clear()


class SecurityTests(ClientCase):
    def test_health_needs_no_header(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"app": "Fennec", "engine_running": True,
                                           "workspace": str(self.root)})
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_sensitive_routes_require_client_header(self):
        self.assertEqual(self.client.get("/api/files").status_code, 403)
        self.assertEqual(self.client.get("/api/files", headers=HEADERS).status_code, 200)

    def test_cross_origin_origin_is_refused(self):
        bad = {**HEADERS, "Origin": "http://evil.example"}
        self.assertEqual(self.client.get("/api/files", headers=bad).status_code, 403)
        same = {**HEADERS, "Origin": "http://127.0.0.1"}
        self.assertEqual(self.client.get("/api/files", headers=same).status_code, 200)

    def test_foreign_host_is_refused(self):
        response = self.client.get("/api/health", headers={"Host": "evil.example"})
        self.assertEqual(response.status_code, 400)

    def test_app_token_is_enforced_when_configured(self):
        with mock.patch.dict(os.environ, {"FENNEC_APP_TOKEN": "s3cret"}):
            self.assertEqual(self.client.get("/api/files", headers=HEADERS).status_code, 401)
            wrong = {**HEADERS, "Authorization": "Bearer nope"}
            self.assertEqual(self.client.get("/api/files", headers=wrong).status_code, 401)
            good = {**HEADERS, "Authorization": "Bearer s3cret"}
            self.assertEqual(self.client.get("/api/files", headers=good).status_code, 200)

    def test_non_loopback_bind_requires_token_and_tls(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit):
                serve("0.0.0.0", 8787)

    def test_service_shutdown_closes_the_engine_client(self):
        engine = FakeEngine()
        closed = []
        engine.close = lambda: closed.append(True)
        app = make_app(self.config, engine, allowed_hosts=["127.0.0.1"])
        with TestClient(app, base_url="http://127.0.0.1") as lifecycle:
            self.assertEqual(lifecycle.get("/api/health").status_code, 200)
        self.assertEqual(closed, [True])


class FileTests(ClientCase):
    def test_listing_contract(self):
        data = self.client.get("/api/files", headers=HEADERS).json()
        self.assertEqual(data["path"], "")
        names = {entry["name"]: entry for entry in data["entries"]}
        self.assertTrue(names["src"]["is_dir"])
        self.assertEqual(names["src"]["path"], "src")
        subdir = self.client.get("/api/files?path=src", headers=HEADERS).json()
        self.assertEqual(subdir["path"], "src")
        self.assertEqual(subdir["entries"][0]["path"], "src/app.py")
        self.assertFalse(subdir["entries"][0]["is_dir"])

    def test_file_content_is_raw(self):
        data = self.client.get("/api/file?path=src/app.py", headers=HEADERS).json()
        self.assertEqual(data["content"], "print('hi')\n")

    def test_traversal_is_blocked(self):
        outside = self.root.parent / "outside-secret.txt"
        outside.write_text("secret", encoding="utf-8")
        self.addCleanup(outside.unlink)
        for path in ("../outside-secret.txt", str(outside)):
            response = self.client.get("/api/file", params={"path": path}, headers=HEADERS)
            self.assertIn(response.status_code, (403, 404))

    def test_absolute_path_is_blocked(self):
        response = self.client.get("/api/files", params={"path": str(self.root.parent)}, headers=HEADERS)
        self.assertEqual(response.status_code, 403)

    def test_symlink_escape_is_blocked(self):
        target = self.root.parent / "outside-dir"
        target.mkdir(exist_ok=True)
        (target / "secret.txt").write_text("secret", encoding="utf-8")
        link = self.root / "link"
        try:
            link.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlink creation is not permitted here")
        self.addCleanup(lambda: target.exists() and __import__("shutil").rmtree(target, ignore_errors=True))
        response = self.client.get("/api/file", params={"path": "link/secret.txt"}, headers=HEADERS)
        self.assertEqual(response.status_code, 403)


class ChatAndAgentTests(ClientCase):
    def test_chat_contract(self):
        response = self.client.post("/api/chat", headers=HEADERS,
                                    json={"messages": [{"role": "user", "content": "hi"}],
                                          "max_tokens": 8, "temperature": 0})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": {"role": "assistant", "content": "done"},
                                           "usage": {"total_tokens": 3}})

    def test_chat_passes_content_parts_reasoning_and_image(self):
        image = "data:image/png;base64,iVBORw0KGgo="
        response = self.client.post("/api/chat", headers=HEADERS, json={
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": image}}]}],
            "reasoning_effort": "high", "enable_thinking": False})
        self.assertEqual(response.status_code, 200)
        sent = self.engine.calls[-1]
        self.assertEqual(sent["messages"][0]["content"][1]["image_url"]["url"], image)
        self.assertEqual(sent["reasoning_effort"], "high")
        self.assertIs(sent["enable_thinking"], False)

    def test_chat_rejects_more_than_one_image(self):
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
        response = self.client.post("/api/chat", headers=HEADERS, json={
            "messages": [{"role": "user", "content": [image, image]}]})
        self.assertEqual(response.status_code, 422)

    def test_chat_rejects_non_data_image_url(self):
        response = self.client.post("/api/chat", headers=HEADERS, json={
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}]}]})
        self.assertEqual(response.status_code, 422)

    def test_stt_returns_transcript(self):
        response = self.client.post("/api/stt", headers={**HEADERS, "Content-Type": "audio/wav"},
                                    content=b"RIFFfake-audio")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text"], "hello from audio")
        self.assertEqual(self.engine.audio, b"RIFFfake-audio")

    def test_stt_rejects_empty_audio(self):
        response = self.client.post("/api/stt", headers=HEADERS, content=b"")
        self.assertEqual(response.status_code, 400)

    def test_chat_refuses_when_engine_down(self):
        self.engine.running = False
        response = self.client.post("/api/chat", headers=HEADERS,
                                    json={"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(response.status_code, 503)

    def test_agent_lifecycle_completes(self):
        started = self.client.post("/api/agent/start", headers=HEADERS,
                                   json={"task": "say done", "max_steps": 2})
        self.assertEqual(started.status_code, 200)
        sid = started.json()["id"]
        for _ in range(100):
            state = self.client.get("/api/agent/poll", params={"id": sid}, headers=HEADERS).json()
            if state["status"] != "running":
                break
            threading.Event().wait(0.02)
        self.assertEqual(state["status"], "done")
        # The loop publishes its plan twice: at turn start, before any tool has
        # run, and again when the turn ends without one. Poll carries both, so a
        # shell never has to re-derive phases from tool names of its own.
        self.assertEqual([step["type"] for step in state["steps"]],
                         ["plan", "assistant", "plan", "final"])
        plans = [step for step in state["steps"] if step["type"] == "plan"]
        # Nothing has happened yet, so the first list is only the next stage --
        # and not marked complete, which is the whole point of the rule.
        self.assertEqual([(e["content"], e["status"]) for e in plans[0]["entries"]],
                         [(loop._PHASE_LABEL["inspect"], "in_progress")])
        self.assertEqual([e["content"] for e in plans[-1]["entries"]],
                         [loop._PHASE_LABEL["summarise"]])
        self.assertEqual(plans[-1]["entries"][0]["status"], "completed")
        self.assertEqual(state["usage"]["total_tokens"], 3)

    def test_poll_steps_stay_inside_the_declared_vocabulary(self):
        # The PWA and the desktop each mirror `loop.STEP_TYPES`, and a step
        # outside that list reaches a shell as an unrecognized row, so the
        # declaration has to be complete. The tool path is where the newer types
        # come from at all, so drive it rather than reading the source.
        self.engine.reply = {"role": "assistant", "content": "looking", "tool_calls": [
            {"id": "call-1", "type": "function",
             "function": {"name": "list_dir", "arguments": "{}"}}]}
        sid = self.client.post("/api/agent/start", headers=HEADERS,
                               json={"task": "list the files", "max_steps": 1}).json()["id"]
        for _ in range(150):
            state = self.client.get("/api/agent/poll", params={"id": sid}, headers=HEADERS).json()
            if state["status"] != "running":
                break
            threading.Event().wait(0.02)
        types = [step["type"] for step in state["steps"]]
        self.assertTrue(set(types) <= set(loop.STEP_TYPES), types)
        self.assertEqual(types, ["plan", "assistant", "tool_start", "tool", "plan", "final"])
        started = next(step for step in state["steps"] if step["type"] == "tool_start")
        done = next(step for step in state["steps"] if step["type"] == "tool")
        # The pair key a shell closes a running row with, in both steps.
        self.assertEqual(started["call_id"], done["call_id"])
        plans = [step for step in state["steps"] if step["type"] == "plan"]
        self.assertEqual([entry["status"] for entry in plans[-1]["entries"]],
                         ["completed", "in_progress"])
        # A shell walks the list with `after`; the new steps must not shift it.
        tail = self.client.get("/api/agent/poll", params={"id": sid, "after": 2},
                               headers=HEADERS).json()
        self.assertEqual([step["type"] for step in tail["steps"]], types[2:])
        self.assertEqual(tail["total"], len(types))

    def test_agent_rejects_workspace_outside_root(self):
        response = self.client.post("/api/agent/start", headers=HEADERS,
                                    json={"task": "x", "workspace": str(self.root.parent)})
        self.assertEqual(response.status_code, 403)

    def test_agent_cancel_marks_terminal_and_stops_emitting(self):
        self.engine.gate = threading.Event()
        started = self.client.post("/api/agent/start", headers=HEADERS,
                                   json={"task": "wait", "max_steps": 4}).json()
        cancel = self.client.post("/api/agent/cancel", headers=HEADERS, json={"id": started["id"]})
        self.assertEqual(cancel.json()["status"], "cancelled")
        self.engine.gate.set()
        for _ in range(150):
            state = self.client.get("/api/agent/poll", params={"id": started["id"]}, headers=HEADERS).json()
            if state["status"] != "running":
                break
            threading.Event().wait(0.02)
        self.assertEqual(state["status"], "cancelled")
        self.assertNotIn("final", [step["type"] for step in state["steps"]])
        self.assertEqual(self.client.post("/api/agent/cancel", headers=HEADERS,
                                          json={"id": "missing"}).status_code, 404)

    def test_unknown_poll_is_404(self):
        self.assertEqual(self.client.get("/api/agent/poll", params={"id": "missing"},
                                         headers=HEADERS).status_code, 404)


def sse_events(text):
    events = []
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


class OutletAndAgUiTests(ClientCase):
    def setUp(self):
        super().setUp()
        # This file is where a Jev checkout exists on the machine under test, so
        # the routes that depend on one declare it instead of inheriting it.
        self.outlets_stub.available_tools.add("web_task")

    def run_ag_ui(self, **overrides):
        body = {"threadId": "t-1", "runId": "r-1", "state": {}, "tools": [], "context": [],
                "forwardedProps": {}, "messages": [{"id": "m1", "role": "user",
                                                    "content": "say done"}]}
        body.update(overrides)
        return self.client.post("/api/ag-ui", headers=HEADERS, json=body)

    def test_outlet_status_is_reported_and_guarded(self):
        self.assertEqual(self.client.get("/api/outlets").status_code, 403)
        response = self.client.get("/api/outlets", headers=HEADERS)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), OUTLET_STATUS)
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_outlet_run_drives_the_installed_jev_outlet(self):
        response = self.client.post("/api/outlets/run", headers=HEADERS,
                                    json={"outlet": "jev", "goal": "cart shows one item",
                                          "url": "https://example.com", "max_steps": 3})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True, "outlet": "jev-ultrafast", "steps": 1})
        self.assertEqual(self.outlets_stub.calls,
                         [("web_task", "cart shows one item", "https://example.com",
                           {"max_steps": 3})])

    def test_outlet_run_refuses_an_absent_outlet_and_names_the_fix(self):
        response = self.client.post("/api/outlets/run", headers=HEADERS,
                                    json={"outlet": "openbot", "goal": "ask the researcher"})
        self.assertEqual(response.status_code, 409)
        self.assertIn("fennec outlets", response.json()["detail"])
        self.assertEqual(self.outlets_stub.calls, [])

    def test_outlet_run_takes_the_coworker_path_when_a_workspace_answers(self):
        self.outlets_stub.available_tools.add("coworker_task")
        response = self.client.post("/api/outlets/run", headers=HEADERS,
                                    json={"outlet": "openbot", "goal": "summarise the changelog",
                                          "agent": "researcher"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.outlets_stub.calls,
                         [("coworker_task", "summarise the changelog", "researcher", {})])

    def test_outlet_run_validates_the_outlet_name_and_goal(self):
        unknown = self.client.post("/api/outlets/run", headers=HEADERS,
                                   json={"outlet": "browser", "goal": "x"})
        empty = self.client.post("/api/outlets/run", headers=HEADERS,
                                 json={"outlet": "jev", "goal": ""})
        self.assertEqual(unknown.status_code, 422)
        self.assertIn("jev", unknown.json()["detail"])
        self.assertEqual(empty.status_code, 422)
        self.assertEqual(self.client.post("/api/outlets/run",
                                          json={"outlet": "jev", "goal": "x"}).status_code, 403)

    def test_agent_sessions_are_handed_the_outlet_bridge(self):
        started = self.client.post("/api/agent/start", headers=HEADERS,
                                   json={"task": "open a page", "allow_web": True,
                                         "max_steps": 2}).json()
        ctx = loop.SESSIONS[started["id"]]["ctx"]
        self.assertIs(ctx.outlets, self.outlets_stub)
        self.assertTrue(ctx.allow_web)
        self.assertFalse(ctx.allow_shell)

    def test_ag_ui_run_streams_the_documented_event_names(self):
        response = self.run_ag_ui()
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        events = sse_events(response.text)
        self.assertEqual(events[0], {"type": "RUN_STARTED", "threadId": "t-1", "runId": "r-1"})
        self.assertEqual(events[-1]["type"], "RUN_FINISHED")
        self.assertEqual(events[-1]["result"]["stopReason"], "end_turn")
        self.assertEqual(events[-1]["result"]["usage"]["total_tokens"], 3)
        self.assertGreater(events[-1]["result"]["usage"]["context_chars_after"], 0)
        deltas = [event["delta"] for event in events
                  if event["type"] == "TEXT_MESSAGE_CONTENT"]
        self.assertEqual(deltas, ["done"])
        self.assertEqual([event["role"] for event in events
                          if event["type"] == "TEXT_MESSAGE_START"], ["assistant"])
        self.assertEqual(len([event for event in events if event["type"] == "TEXT_MESSAGE_END"]),
                         len([event for event in events if event["type"] == "TEXT_MESSAGE_START"]))

    def test_ag_ui_reports_tool_calls_and_their_results(self):
        self.engine.reply = {"role": "assistant", "content": "looking", "tool_calls": [
            {"id": "call-1", "type": "function",
             "function": {"name": "list_dir", "arguments": "{}"}}]}
        events = sse_events(self.run_ag_ui(forwardedProps={"max_steps": 1}).text)
        types = [event["type"] for event in events]
        self.assertIn("TOOL_CALL_START", types)
        self.assertIn("TOOL_CALL_RESULT", types)
        start = next(event for event in events if event["type"] == "TOOL_CALL_START")
        self.assertEqual(start["toolCallName"], "list_dir")
        self.assertEqual(start["toolCallId"], "call-1")
        args = next(event for event in events if event["type"] == "TOOL_CALL_ARGS")
        self.assertEqual(args["toolCallId"], start["toolCallId"])
        self.assertEqual(json.loads(args["delta"]), {})
        self.assertEqual([event["type"] for event in events].count("TOOL_CALL_END"), 1)
        result = next(event for event in events if event["type"] == "TOOL_CALL_RESULT")
        self.assertEqual(result["toolCallId"], "call-1")
        self.assertTrue(json.loads(result["content"])["ok"])
        self.assertEqual(events[-1]["result"]["stopReason"], "max_turn_requests")

    def test_ag_ui_reasoning_becomes_reasoning_events(self):
        self.engine.reply = {"role": "assistant", "content": "", "reasoning_content": "thinking"}
        types = [event["type"] for event in sse_events(self.run_ag_ui().text)]
        self.assertIn("REASONING_MESSAGE_CONTENT", types)
        self.assertNotIn("THINKING_TEXT_MESSAGE_CONTENT", types)

    def test_ag_ui_rejects_a_run_it_cannot_start(self):
        self.assertEqual(self.client.post("/api/ag-ui", json={
            "messages": [{"role": "user", "content": "hi"}]}).status_code, 403)
        self.assertEqual(self.run_ag_ui(messages=[]).status_code, 400)
        self.assertEqual(self.run_ag_ui(messages=[{"role": "assistant", "content": "hi"}])
                         .status_code, 400)
        self.assertEqual(self.run_ag_ui(messages=[{"role": "user", "content": "   "}])
                         .status_code, 400)
        self.assertEqual(self.run_ag_ui(forwardedProps={"workspace": str(self.root.parent)})
                         .status_code, 403)
        self.engine.running = False
        self.assertEqual(self.run_ag_ui().status_code, 503)

    def test_ag_ui_refuses_forwarded_props_it_cannot_use(self):
        cases = [{"max_steps": "many"}, {"max_steps": 2.5}, {"max_steps": 1000},
                 {"max_steps": 0}, {"workspace": 5}]
        for props in cases:
            response = self.run_ag_ui(forwardedProps=props)
            self.assertEqual(response.status_code, 422, props)
        # A number carried as a string is still a number of steps, and a directory
        # that simply is not there is a 400 about the path, not about its type.
        self.assertEqual(self.run_ag_ui(forwardedProps={"max_steps": "7"}).status_code, 200)
        self.assertEqual(self.run_ag_ui(forwardedProps={"workspace": "no/such/here"}).status_code, 400)
        self.assertEqual([session["status"] for session in loop.SESSIONS.values()], ["done"])

    def test_an_unanswered_tool_call_is_closed_before_the_run_ends(self):
        stream = server_mod.AgUiStream("t-1", "r-1")
        events = stream.feed({"type": "assistant", "content": "", "tool_calls": [
            {"call_id": "c-1", "name": "web_task", "args": {}}]})
        self.assertEqual([event["type"] for event in events],
                         ["TOOL_CALL_START", "TOOL_CALL_ARGS", "TOOL_CALL_END"])
        closed = stream.close_tools()
        self.assertEqual([event["type"] for event in closed], ["TOOL_CALL_RESULT"])
        self.assertEqual(closed[0]["toolCallId"], "c-1")
        self.assertEqual(json.loads(closed[0]["content"]), server_mod.AGUI_UNFINISHED)
        self.assertEqual(stream.close_tools(), [])

        stream = server_mod.AgUiStream("t-1", "r-1")
        stream.feed({"type": "assistant", "content": "", "tool_calls": [
            {"call_id": "c-1", "name": "web_task", "args": {}}]})
        stream.feed({"type": "tool", "call_id": "c-1", "result": {"ok": True}})
        self.assertEqual(stream.close_tools(), [])
        self.assertEqual([event["type"] for event in stream.close_tools()], [])

    def test_the_terminal_event_names_a_cancellation_as_one(self):
        stream = server_mod.AgUiStream("t-1", "r-1")
        finished = stream.finish({"id": "s-1", "status": "cancelled", "stop": None,
                                  "usage": {}})[0]
        self.assertEqual(finished["result"]["stopReason"], "cancelled")
        stream = server_mod.AgUiStream("t-1", "r-1")
        finished = stream.finish({"id": "s-1", "status": "finished", "usage": {}})[0]
        self.assertEqual(finished["result"]["stopReason"], "end_turn")
        stream = server_mod.AgUiStream("t-1", "r-1")
        stream.feed({"type": "assistant", "content": "", "tool_calls": [
            {"call_id": "c-9", "name": "list_dir", "args": {}}]})
        self.assertEqual([event["type"] for event in
                          stream.finish({"id": "s-1", "status": "cancelled", "usage": {}})],
                         ["TOOL_CALL_RESULT", "RUN_FINISHED"])

    def test_a_long_tool_result_is_cut_as_json_not_through_json(self):
        text = "x" * (server_mod.AGUI_RESULT_CHARS * 2)
        clipped = json.loads(server_mod._json_clip({"ok": True, "text": text}))
        self.assertTrue(clipped["truncated"])
        self.assertLessEqual(len(clipped["excerpt"]), server_mod.AGUI_RESULT_CHARS)
        self.assertEqual(json.loads(server_mod._json_clip({"ok": True})), {"ok": True})

    def test_a_client_that_hangs_up_ends_the_run_it_was_watching(self):
        self.engine.gate = threading.Event()
        self.addCleanup(self.engine.gate.set)
        body = {"threadId": "t-9", "runId": "r-9", "messages": [{"role": "user",
                                                                "content": "keep going"}]}
        with self.client.stream("POST", "/api/ag-ui", headers=HEADERS, json=body) as response:
            self.assertIn("RUN_STARTED", next(response.iter_lines()))
            sid = next(iter(loop.SESSIONS))
        for _ in range(60):
            if loop.get_steps(sid)["status"] != "running":
                break
            time.sleep(0.05)
        self.assertEqual(loop.get_steps(sid)["status"], "cancelled")

    def test_ag_ui_cancel_surfaces_as_a_cancelled_run(self):
        self.engine.gate = threading.Event()
        self.addCleanup(self.engine.gate.set)
        started = self.client.post("/api/agent/start", headers=HEADERS,
                                   json={"task": "wait", "max_steps": 3}).json()
        self.client.post("/api/agent/cancel", headers=HEADERS, json={"id": started["id"]})
        for _ in range(150):
            if loop.get_steps(started["id"])["status"] != "running":
                break
            threading.Event().wait(0.02)
        self.assertEqual(loop.get_steps(started["id"])["status"], "cancelled")
        self.engine.gate.set()

    def test_agent_cli_reports_outlets_without_touching_the_network(self):
        with mock.patch.object(server_mod, "load_config", return_value=self.config), \
             mock.patch.object(sys, "stdout", new_callable=io.StringIO) as out:
            self.assertEqual(agent_main(["outlets"]), 0)
        self.assertEqual(json.loads(out.getvalue()), OUTLET_STATUS)

    def test_agent_cli_exposes_the_acp_entrypoint(self):
        from fennec_core import acp as acp_mod
        report = {"protocolVersion": 1, "agent": "fennec", "methods": ["initialize"]}
        with mock.patch.object(acp_mod, "check", return_value=report), \
             mock.patch.object(sys, "stdout", new_callable=io.StringIO) as out:
            self.assertEqual(agent_main(["acp", "--check"]), 0)
        self.assertEqual(json.loads(out.getvalue()), report)
        with mock.patch.object(acp_mod, "main", return_value=0) as main:
            self.assertEqual(agent_main(["acp"]), 0)
        main.assert_called_once_with([])


class StaticTests(ClientCase):
    def test_pwa_is_served_same_origin(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Fennec", response.text)


class ConfigTests(unittest.TestCase):
    def test_secrets_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"api_key": "x"}), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path, environ={}, home=Path(tmp))

    def test_engine_url_must_be_a_bare_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"engine_url": "http://127.0.0.1:8184/v1"}), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path, environ={}, home=Path(tmp))

    def test_backend_and_policy_enums_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            for bad in ({"backend": "rocm"}, {"policy": "speed"}, {"ctx": -1}, {"ctx": 3.5}):
                path.write_text(json.dumps(bad), encoding="utf-8")
                with self.assertRaises(ConfigError, msg=bad):
                    load_config(path, environ={}, home=Path(tmp))

    def test_every_wizard_backend_loads(self):
        """Whatever the wizard offers has to load, so the two sets cannot drift."""
        from fennec_core import setup_wizard as wizard
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            for backend in wizard.BACKENDS:
                path.write_text(json.dumps({"backend": backend}), encoding="utf-8")
                self.assertEqual(load_config(path, environ={}, home=Path(tmp)).backend, backend)

    def test_env_overrides_and_ctx_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"ctx": 0}), encoding="utf-8")
            env = {"FENNEC_ENGINE_URL": "http://localhost:9999", "FENNEC_MODELS": str(path.parent / "m")}
            config = load_config(path, environ=env, home=Path(tmp))
            self.assertEqual(config.engine_url, "http://localhost:9999")
            self.assertEqual(str(config.models_root), str((path.parent / "m").resolve()))
            self.assertEqual(config.ctx, 0)

    def test_non_loopback_engine_requires_a_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"engine_url": "http://10.0.0.5:8184"}), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path, environ={}, home=Path(tmp))
            config = load_config(path, environ={"FENNEC_API_KEY": "k"}, home=Path(tmp))
            self.assertEqual(config.api_key, "k")

    def test_missing_explicit_config_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ConfigError):
                load_config(Path(tmp) / "missing.json", environ={}, home=Path(tmp))

    def test_engine_root_defaults_to_fennec(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            fennec = home / "Fennec"
            self.assertEqual(load_config(environ={}, home=home).engine_root, fennec.resolve())
            # A Colibri checkout on disk is no longer an engine root.
            (home / "Colibri").mkdir()
            self.assertEqual(load_config(environ={}, home=home).engine_root, fennec.resolve())
            fennec.mkdir()
            self.assertEqual(load_config(environ={}, home=home).engine_root, fennec.resolve())

    def test_an_explicit_engine_root_beats_the_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            path = home / "config.json"
            path.write_text(json.dumps({"engine_root": str(home / "elsewhere")}), encoding="utf-8")
            self.assertEqual(load_config(path, environ={}, home=home).engine_root,
                             (home / "elsewhere").resolve())
            env = {"FENNEC_HOME": str(home / "from-env")}
            self.assertEqual(load_config(environ=env, home=home).engine_root,
                             (home / "from-env").resolve())


class EngineEnvironmentTests(unittest.TestCase):
    def test_explicit_backends_pin_their_flag_and_auto_pins_nothing(self):
        from types import SimpleNamespace
        from fennec_core import engine
        stub = SimpleNamespace(config=SimpleNamespace(backend="auto", cache_root=Path("C:/cache"),
                                                      model="m", api_key="local"))
        env = engine.Engine._engine_env(stub)
        # "auto" is the engine's own vendor policy (NVIDIA=cuda, Apple=metal,
        # else vulkan); a pin would outrank it.
        for name in ("FENNEC_VULKAN", "FENNEC_METAL", "FENNEC_CUDA"):
            self.assertNotIn(name, env, f"{name} would outrank vendor policy")
        stub.config.backend = "cpu"
        env = engine.Engine._engine_env(stub)
        self.assertEqual(env.get("FENNEC_VULKAN"), "0")
        self.assertEqual(env.get("FENNEC_CUDA"), "0")
        stub.config.backend = "vulkan"
        env = engine.Engine._engine_env(stub)
        self.assertEqual(env.get("FENNEC_VULKAN"), "1")
        self.assertEqual(env.get("FENNEC_CUDA"), "0")
        stub.config.backend = "cuda"
        env = engine.Engine._engine_env(stub)
        self.assertEqual(env.get("FENNEC_CUDA"), "1")
        self.assertEqual(env.get("FENNEC_VULKAN"), "0")


class EngineBinaryTests(unittest.TestCase):
    """The suite spawns the native Fennec binary and nothing else."""

    def setUp(self):
        from fennec_core import engine
        self.engine = engine
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.config = SimpleNamespace(engine_root=self.root, python="python", model=str(self.root))
        self.stub = SimpleNamespace(config=self.config)
        # _command dispatches through _engine_binary; bind it onto the stub the
        # same way the other tests call unbound Engine methods.
        self.stub._engine_binary = lambda: engine.Engine._engine_binary(self.stub)

    def write(self, rel):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")
        return p

    def test_the_release_binary_is_found_despite_unrelated_files(self):
        binary = self.write("target/release/fennec.exe" if os.name == "nt" else "target/release/fennec")
        self.write("c/coli")  # leftover from the retired engine; it is not an engine
        self.assertEqual(self.engine.Engine._engine_binary(self.stub), binary)

    def test_a_debug_build_is_found_when_release_is_absent(self):
        binary = self.write("target/debug/fennec.exe" if os.name == "nt" else "target/debug/fennec")
        self.assertEqual(self.engine.Engine._engine_binary(self.stub), binary)

    def test_a_colibri_launcher_is_not_an_engine(self):
        self.write("c/coli")
        with self.assertRaises(self.engine.EngineError):
            self.engine.Engine._engine_binary(self.stub)

    def test_no_engine_is_an_error_not_a_silent_success(self):
        with self.assertRaises(self.engine.EngineError):
            self.engine.Engine._engine_binary(self.stub)

    def test_the_native_binary_is_not_wrapped_in_python(self):
        binary = self.write("target/release/fennec.exe" if os.name == "nt" else "target/release/fennec")
        command, _ = self.engine.Engine._command(self.stub, ["serve", "--auto-tier"])
        self.assertEqual(command[0], str(binary))
        self.assertNotIn("python", command[0].lower())
        self.assertEqual(command[1:], ["serve", "--auto-tier"])


class _Scripted(FakeEngine):
    """FakeEngine that answers each turn from a list, like the ACP tests' ScriptEngine."""

    def __init__(self, replies):
        super().__init__()
        self.replies = list(replies)

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": list(messages)})
        reply = self.replies.pop(0) if self.replies else {"role": "assistant", "content": "done"}
        return reply, {"total_tokens": 3}


def _call(name, args, call_id="call-1"):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}]}


class ApprovalTests(ClientCase):
    """Per-call approvals on the polling path — the question, the answer, and
    what happens when nobody answers. The ACP stdio path has asked for a long
    time; these cover `/api/agent/start` + `/api/agent/approve`, which is what
    the desktop and the PWA watch."""

    def start(self, replies, **flags):
        self.engine = _Scripted(replies)
        self.client = TestClient(make_app(self.config, self.engine, allowed_hosts=["127.0.0.1"]),
                                 base_url="http://127.0.0.1")
        body = {"task": "make a note", "max_steps": 3}
        body.update(flags)
        started = self.client.post("/api/agent/start", headers=HEADERS, json=body)
        self.assertEqual(started.status_code, 200)
        return started.json()["id"]

    def poll_until(self, sid, done):
        for _ in range(150):
            state = self.client.get("/api/agent/poll", params={"id": sid}, headers=HEADERS).json()
            if done(state):
                return state
            threading.Event().wait(0.02)
        self.fail(f"the run never got there: last state {state['status']} / {state['pending']}")

    def ask(self, sid):
        state = self.poll_until(sid, lambda s: s["pending"])
        steps = [step for step in state["steps"] if step["type"] == "approval_required"]
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["approval_id"], state["pending"][0])
        return steps[0]

    def answer(self, sid, approval_id, decision):
        return self.client.post("/api/agent/approve", headers=HEADERS,
                               json={"id": sid, "approval_id": approval_id,
                                     "decision": decision})

    def settle(self, sid):
        return self.poll_until(sid, lambda s: s["status"] != "running")

    def test_a_gated_write_asks_the_polling_client_before_it_touches_the_disk(self):
        sid = self.start([_call("write_file", {"path": "note.md", "content": "hi"}),
                          {"role": "assistant", "content": "done"}],
                         require_approval=True)
        ask = self.ask(sid)
        self.assertEqual(ask["name"], "write_file")
        self.assertEqual(ask["args"], {"path": "note.md", "content": "hi"})
        self.assertFalse((self.root / "note.md").exists(),
                         "nothing may be written before the answer arrives")
        self.assertEqual(self.answer(sid, ask["approval_id"], "allow_once").status_code, 200)
        state = self.settle(sid)
        self.assertEqual(state["status"], "done")
        self.assertEqual((self.root / "note.md").read_text(encoding="utf-8"), "hi")
        self.assertTrue([step for step in state["steps"]
                         if step["type"] == "tool" and step["ok"]])

    def test_the_question_carries_the_risk_and_reason_the_gate_table_owns(self):
        for name, args, risk in (("write_file", {"path": "a.md", "content": "x"}, "medium"),
                                 ("run_shell", {"command": "echo hi"}, "high"),
                                 ("git_diff", {}, "low")):
            with self.subTest(name=name):
                sid = self.start([_call(name, args), {"role": "assistant", "content": "done"}],
                                 require_approval=True)
                ask = self.ask(sid)
                self.assertEqual(ask["risk"], risk)
                self.assertTrue(ask["reason"])
                self.answer(sid, ask["approval_id"], "reject_once")
                self.settle(sid)

    def test_rejecting_leaves_the_disk_alone_and_says_so_to_the_model(self):
        sid = self.start([_call("write_file", {"path": "note.md", "content": "hi"}),
                          {"role": "assistant", "content": "done"}],
                         require_approval=True)
        ask = self.ask(sid)
        self.assertEqual(self.answer(sid, ask["approval_id"], "reject_once").status_code, 200)
        state = self.settle(sid)
        self.assertFalse((self.root / "note.md").exists())
        refused = next(step for step in state["steps"] if step["type"] == "tool")
        self.assertFalse(refused["ok"])
        self.assertIn("rejected by the user", json.dumps(refused["result"]))

    def test_nobody_answering_is_a_rejection_and_not_a_hang(self):
        with mock.patch.object(loop, "APPROVAL_TIMEOUT", 0.3):
            sid = self.start([_call("write_file", {"path": "note.md", "content": "hi"}),
                              {"role": "assistant", "content": "done"}],
                             require_approval=True)
            ask = self.ask(sid)
            state = self.settle(sid)
        self.assertFalse((self.root / "note.md").exists())
        self.assertEqual(state["pending"], [])
        refused = next(step for step in state["steps"] if step["type"] == "tool")
        self.assertFalse(refused["ok"])

    def test_cancelling_while_a_question_is_open_does_not_wait_out_the_timeout(self):
        started = time.monotonic()
        with mock.patch.object(loop, "APPROVAL_TIMEOUT", 60):
            sid = self.start([_call("write_file", {"path": "note.md", "content": "hi"}),
                              {"role": "assistant", "content": "done"}],
                             require_approval=True)
            self.ask(sid)
            self.client.post("/api/agent/cancel", headers=HEADERS, json={"id": sid})
            state = self.settle(sid)
        self.assertEqual(state["status"], "cancelled")
        self.assertLess(time.monotonic() - started, 10,
                        "a cancel should stop the wait, not extend it")
        self.assertFalse((self.root / "note.md").exists())

    def test_a_client_that_never_asks_for_approvals_is_still_denied_without_waiting(self):
        started = time.monotonic()
        sid = self.start([_call("write_file", {"path": "note.md", "content": "hi"}),
                          {"role": "assistant", "content": "done"}])
        state = self.settle(sid)
        self.assertEqual(state["pending"], [])
        self.assertNotIn("approval_required", [step["type"] for step in state["steps"]])
        self.assertFalse((self.root / "note.md").exists())
        self.assertLess(time.monotonic() - started, 10)

    def test_answering_twice_or_answering_nothing_is_a_404(self):
        sid = self.start([_call("write_file", {"path": "note.md", "content": "hi"}),
                          {"role": "assistant", "content": "done"}],
                         require_approval=True)
        ask = self.ask(sid)
        self.assertEqual(self.answer(sid, ask["approval_id"], "allow_once").status_code, 200)
        self.assertEqual(self.answer(sid, ask["approval_id"], "allow_once").status_code, 404)
        self.assertEqual(self.answer(sid, "not-a-real-id", "allow_once").status_code, 404)
        self.assertEqual(self.client.post("/api/agent/approve", headers=HEADERS,
                                          json={"id": "nope", "approval_id": ask["approval_id"],
                                                "decision": "allow_once"}).status_code, 404)
        self.settle(sid)

    def test_an_option_outside_the_acp_vocabulary_is_refused(self):
        sid = self.start([_call("write_file", {"path": "note.md", "content": "hi"}),
                          {"role": "assistant", "content": "done"}],
                         require_approval=True)
        ask = self.ask(sid)
        self.assertEqual(self.answer(sid, ask["approval_id"], "yes_please").status_code, 404)
        self.assertFalse((self.root / "note.md").exists())
        self.assertEqual(self.answer(sid, ask["approval_id"], "allow_once").status_code, 200)
        self.settle(sid)

    def test_every_gated_tool_is_a_real_tool_and_every_real_tool_has_a_risk(self):
        """The gate table is what an approval card is built from, so a name in it
        that is not a tool — or a gated tool missing from it — is a card that
        lies about what the agent is about to do."""
        from fennec_core.tools import GATES, TOOLS, approval_context
        self.assertTrue(set(GATES) <= set(TOOLS), set(GATES) - set(TOOLS))
        for name in TOOLS:
            self.assertIn(approval_context(name)["risk"],
                          {"low", "medium", "high", "unknown"})
        # A call the table does gate must never be described as though its
        # effects were unknown, and the reason must not claim a suite tool is
        # something outside it.
        for name, gate in GATES.items():
            context = approval_context(name)
            self.assertEqual(context["risk"], gate.risk)
            self.assertEqual(context["reason"], gate.reason)
        for name in set(TOOLS) - set(GATES):
            self.assertNotIn("outside", approval_context(name)["reason"])

    def test_the_sessions_list_counts_what_each_run_is_waiting_on(self):
        """The rails list tasks from `/api/agent/sessions`, which is the only
        place a shell that is not watching this run can learn it is blocked on a
        question rather than still thinking."""
        sid = self.start([_call("write_file", {"path": "note.md", "content": "hi"}),
                          {"role": "assistant", "content": "done"}],
                         require_approval=True)
        ask = self.ask(sid)

        def row():
            sessions = self.client.get("/api/agent/sessions", headers=HEADERS).json()["sessions"]
            return next(s for s in sessions if s["id"] == sid)

        self.assertEqual(row()["pending"], 1)
        self.assertEqual(row()["status"], "running")
        self.assertEqual(self.answer(sid, ask["approval_id"], "allow_once").status_code, 200)
        self.settle(sid)
        self.assertEqual(row()["pending"], 0)


class TextToolCallTests(ApprovalTests):
    """The engine has no function calling, so a compliant turn arrives as a JSON
    object written into the content. The loop must honor the same shape the real
    model emitted (tool/source/content, ``source`` aliasing ``path``) and keep
    quoted or foreign JSON as prose."""

    def test_the_json_the_model_actually_writes_executes(self):
        reply = json.dumps({"tool": "write_file",
                            "source": str(self.root / "note.md"),
                            "content": "hi"}, indent=2)
        sid = self.start([{"role": "assistant", "content": reply},
                          {"role": "assistant", "content": "done"}], allow_write=True)
        state = self.settle(sid)
        self.assertEqual(state["status"], "done")
        self.assertEqual((self.root / "note.md").read_text(encoding="utf-8"), "hi")
        self.assertTrue([step for step in state["steps"]
                         if step["type"] == "tool" and step["ok"]])

    def test_a_text_call_asks_for_approval_like_a_native_one(self):
        sid = self.start([{"role": "assistant",
                           "content": json.dumps({"tool": "write_file", "path": "note.md",
                                                  "content": "hi"})},
                          {"role": "assistant", "content": "done"}],
                         require_approval=True)
        ask = self.ask(sid)
        self.assertEqual(ask["name"], "write_file")
        self.assertEqual(ask["args"], {"path": "note.md", "content": "hi"})
        self.answer(sid, ask["approval_id"], "allow_once")
        state = self.settle(sid)
        self.assertEqual((self.root / "note.md").read_text(encoding="utf-8"), "hi")

    def test_prose_json_and_foreign_keys_stay_prose(self):
        cases = [{"tool": "deploy", "target": "prod"},
                 {"tool": "write_file", "path": "note.md", "content": "hi", "force": True},
                 "I would use {\"tool\": \"write_file\"} here but this is an example"]
        for content in cases:
            with self.subTest(content=str(content)[:40]):
                sid = self.start([{"role": "assistant",
                                   "content": content if isinstance(content, str)
                                   else json.dumps(content)},
                                  {"role": "assistant", "content": "done"}],
                                 allow_write=True)
                state = self.settle(sid)
                self.assertEqual(state["status"], "done")
                self.assertFalse((self.root / "note.md").exists())
                self.assertEqual([s["type"] for s in state["steps"]].count("tool"), 0)

    def test_a_fenced_call_inside_prose_still_executes(self):
        reply = ("Reading the workspace first.\n```json\n"
                 + json.dumps({"name": "write_file", "path": "note.md", "content": "hi"})
                 + "\n```\nThen writing the note.")
        sid = self.start([{"role": "assistant", "content": reply},
                          {"role": "assistant", "content": "done"}], allow_write=True)
        state = self.settle(sid)
        self.assertEqual(state["status"], "done")
        self.assertEqual((self.root / "note.md").read_text(encoding="utf-8"), "hi")


if __name__ == "__main__":
    unittest.main()
