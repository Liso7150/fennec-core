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
        self.assertEqual([step["type"] for step in state["steps"]], ["assistant", "final"])
        self.assertEqual(state["usage"]["total_tokens"], 3)

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
            for bad in ({"backend": "cuda"}, {"policy": "speed"}, {"ctx": -1}, {"ctx": 3.5}):
                path.write_text(json.dumps(bad), encoding="utf-8")
                with self.assertRaises(ConfigError, msg=bad):
                    load_config(path, environ={}, home=Path(tmp))

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


class EngineEnvironmentTests(unittest.TestCase):
    def test_vulkan_loader_detection_per_platform(self):
        from fennec_core import engine
        self.assertTrue(engine.vulkan_loader_present(exists=lambda p: p.endswith("vulkan-1.dll"),
                                                     system="Windows", environ={"SystemRoot": "C:/Windows"}))
        self.assertFalse(engine.vulkan_loader_present(exists=lambda p: False, system="Windows", environ={}))
        self.assertFalse(engine.vulkan_loader_present(exists=lambda p: True, system="Darwin", environ={}))
        self.assertTrue(engine.vulkan_loader_present(exists=lambda p: p.endswith("libvulkan.so.1"),
                                                     system="Linux", environ={}))

    def test_auto_requests_the_gpu_and_cpu_switches_it_off(self):
        from types import SimpleNamespace
        from fennec_core import engine
        stub = SimpleNamespace(config=SimpleNamespace(backend="auto", cache_root=Path("C:/cache"),
                                                      model="m", api_key="local"))
        with mock.patch.object(engine, "vulkan_loader_present", return_value=True):
            env = engine.Engine._engine_env(stub)
            self.assertEqual(env.get("COLI_VULKAN"), "1")
            self.assertEqual(env.get("COLI_VK_TIER_HOST_GB"), "4")
        with mock.patch.object(engine, "vulkan_loader_present", return_value=True), \
                mock.patch.dict(os.environ, {"COLI_VK_TIER_HOST_GB": "9"}):
            self.assertEqual(engine.Engine._engine_env(stub).get("COLI_VK_TIER_HOST_GB"), "9")
        with mock.patch.object(engine, "vulkan_loader_present", return_value=False):
            self.assertNotIn("COLI_VULKAN", engine.Engine._engine_env(stub))
        stub.config.backend = "cpu"
        self.assertEqual(engine.Engine._engine_env(stub).get("COLI_VULKAN"), "0")


if __name__ == "__main__":
    unittest.main()
