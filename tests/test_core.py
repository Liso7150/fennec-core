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

import liquid
import loop
import server as server_mod
import tools
from config import ConfigError, load_config
from server import main as agent_main
from server import make_app, serve

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
                           "enable_thinking": enable_thinking, "max_tokens": max_tokens})
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


OUTLET_STATUS = {"fennec": {"present": False, "interpreter": None, "transport": False,
                            "transport_from": "none", "decision_key": False,
                            "model": liquid.DEFAULT_MODEL, "note": "stubbed for tests"},
                 "openbot": {"present": False, "url": "http://127.0.0.1:3001", "loopback": True,
                             "agents": [], "error": "no OpenBot workspace answering"}}


class StubOutlets:
    """Stands in for the real outlets so no test can reach a browser or a port."""

    def __init__(self, config=None):
        self.config = config
        self.calls = []
        self.asks = []
        self.available_tools = set()

    def status(self, refresh=False):
        return OUTLET_STATUS

    def available(self, name):
        # Recorded, because the route must gate on exactly the tool it then calls: a shell
        # clicking a row gets a refusal about that row, never about some other lane.
        self.asks.append(name)
        return name in self.available_tools

    def web_task(self, goal, url="", **kwargs):
        self.calls.append(("web_task", goal, url, kwargs))
        return {"ok": True, "engine": "liquid-d1.2", "outlet": "fennec-webagent", "steps": 1}

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
        # The real bridge opens a socket to 127.0.0.1:3001 and probes every interpreter
        # on the first turn of every session. On a machine that has either installed that
        # is a unit test reaching the network - and offering the model a tool that spends
        # money on a page.
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

    def test_bytes_is_the_size_on_disk_not_of_the_decoded_text(self):
        # Windows turns "\n" into "\r\n" on write, so a file can hold more bytes
        # than the text read back out of it. The card, the desktop pane and the
        # PWA pane all quote one number for one file, and it is this one.
        crlf = self.root / "src" / "crlf.md"
        crlf.write_bytes(b"- [ ] a\r\n- [ ] b\r\n")
        self.addCleanup(crlf.unlink)
        data = self.client.get("/api/file?path=src/crlf.md", headers=HEADERS).json()
        self.assertEqual(data["content"], "- [ ] a\n- [ ] b\n")
        self.assertEqual(data["lines"], 2)
        self.assertEqual(data["bytes"], crlf.stat().st_size)

    def test_write_reports_the_size_left_on_disk(self):
        ctx = tools.ToolContext(workspace=str(self.root), allow_write=True)
        made = self.root / "made.md"
        self.addCleanup(lambda: made.unlink(missing_ok=True))
        self.assertEqual(tools.write_file(ctx, "made.md", "a\nb\n")["bytes"], made.stat().st_size)

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

    def test_a_tool_that_reports_says_so_between_its_open_and_its_close(self):
        """A browser decision takes half a minute, and a chat with nothing on screen reads as frozen.

        The note is a step of its own keyed to the call it reports on, so a shell updates the
        row it already drew rather than adding one, and it arrives while the call is still
        running - after the result it would be news of a call that already finished.
        """
        self.engine.reply = {"role": "assistant", "content": "looking", "tool_calls": [
            {"id": "call-7", "type": "function",
             "function": {"name": "list_dir", "arguments": "{}"}}]}

        def working(name, args, ctx, **kwargs):
            ctx.progress({"text": "Deciding step 1 of 3", "step": 1, "of": 3})
            ctx.progress({"text": "Clicking Sign in"})
            return {"ok": True, "result": "empty"}

        with mock.patch.object(loop, "execute", working):
            sid = self.client.post("/api/agent/start", headers=HEADERS,
                                   json={"task": "sign in", "max_steps": 1}).json()["id"]
            for _ in range(150):
                state = self.client.get("/api/agent/poll", params={"id": sid},
                                        headers=HEADERS).json()
                if state["status"] != "running":
                    break
                threading.Event().wait(0.02)
        steps = state["steps"]
        types = [step["type"] for step in steps]
        self.assertTrue(set(types) <= set(loop.STEP_TYPES), types)
        self.assertEqual(types, ["plan", "assistant", "tool_start", "progress", "progress",
                                 "tool", "plan", "final"])
        notes = [step for step in steps if step["type"] == "progress"]
        started = next(step for step in steps if step["type"] == "tool_start")
        self.assertEqual([note["call_id"] for note in notes], [started["call_id"]] * 2,
                         "a note belongs to the row a shell already drew")
        self.assertEqual({note["name"] for note in notes}, {"list_dir"})
        self.assertEqual(notes[0]["text"], "Deciding step 1 of 3")
        self.assertEqual([notes[0]["step"], notes[0]["of"]], [1, 3])
        self.assertEqual([notes[1]["step"], notes[1]["of"]], [0, 0],
                         "a note with no meter arrives with zeros, not nulls")
        self.assertLess(steps.index(notes[0]),
                        steps.index(next(step for step in steps if step["type"] == "tool")))

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


class SubtaskQueueTests(ClientCase):
    """A complex run queues the follow-ups it left behind; a short one asks nothing."""

    def start(self, replies, task, **flags):
        self.engine = _Scripted(replies)
        self.client = TestClient(make_app(self.config, self.engine, allowed_hosts=["127.0.0.1"]),
                                 base_url="http://127.0.0.1")
        started = self.client.post("/api/agent/start", headers=HEADERS,
                                   json={"task": task, "max_steps": 6, **flags})
        self.assertEqual(started.status_code, 200)
        sid = started.json()["id"]
        for _ in range(200):
            state = self.client.get("/api/agent/poll", params={"id": sid},
                                    headers=HEADERS).json()
            if state["status"] != "running":
                return state
            threading.Event().wait(0.02)
        self.fail(f"the run never finished: {state}")

    def test_a_complex_run_queues_what_it_left_undone(self):
        state = self.start([_call("read_file", {"path": "src/app.py"}),
                            _call("write_file", {"path": "note.md", "content": "hi\n"}),
                            {"role": "assistant", "content": "fixed the drain and wrote a note"},
                            {"role": "assistant",
                             "content": '["Add a test for the drain path", "Refresh the docs"]'}],
                           "fix the drain bug in loop.py and run the tests after it",
                           allow_write=True)
        types = [step["type"] for step in state["steps"]]
        self.assertLess(types.index("queue"), types.index("final"),
                        "a shell drains on completion, so the items have to arrive first")
        items = next(step["items"] for step in state["steps"] if step["type"] == "queue")
        self.assertEqual(items, ["Add a test for the drain path", "Refresh the docs"])
        self.assertEqual(state["status"], "done")

    def test_a_run_that_admits_nothing_still_queues_the_phase_it_skipped(self):
        """The guarantee the ask alone cannot give.

        Measured on the live engine over a run's own transcript — which had read
        the file and written the checklist and never verified either — the answer
        was 'NONE' both to the plain ask and to the ask with the missing phase
        named in it. So the queue reads the phases the run really ran instead of
        believing it when it says it is finished.
        """
        loop._PROPOSED.clear()
        self.addCleanup(loop._PROPOSED.clear)
        state = self.start([_call("read_file", {"path": "src/app.py"}),
                            _call("write_file", {"path": "note.md", "content": "hi\n"}),
                            {"role": "assistant", "content": "fixed the drain and wrote a note"},
                            {"role": "assistant", "content": "NONE"}],
                           "fix the drain bug in loop.py and run the tests after it",
                           allow_write=True)
        types = [step["type"] for step in state["steps"]]
        self.assertLess(types.index("queue"), types.index("final"))
        items = next(step["items"] for step in state["steps"] if step["type"] == "queue")
        self.assertEqual(items, [loop._UNDONE_TASK["verify"]])
        self.assertEqual(state["status"], "done")

    def test_an_admitted_item_that_names_the_skipped_phase_is_not_doubled(self):
        """The run's own follow-up already covers the phase, so no second item."""
        state = self.start([_call("read_file", {"path": "src/app.py"}),
                            _call("write_file", {"path": "note.md", "content": "hi\n"}),
                            {"role": "assistant", "content": "fixed the drain and wrote a note"},
                            {"role": "assistant", "content": '["Add a test for the drain path"]'}],
                           "fix the drain bug in loop.py and run the tests after it",
                           allow_write=True)
        items = next(step["items"] for step in state["steps"] if step["type"] == "queue")
        self.assertEqual(items, ["Add a test for the drain path"])

    def test_a_run_that_did_every_kind_of_work_queues_nothing(self):
        """Inspected, changed, verified: there is no phase left to hand on, and an
        invented item would only make the queue lie about what is unfinished."""
        state = self.start([_call("read_file", {"path": "src/app.py"}),
                            _call("write_file", {"path": "note.md", "content": "hi\n"}),
                            _call("run_shell", {"command": "printf ok"}, call_id="call-2"),
                            {"role": "assistant", "content": "fixed it and ran the check"},
                            {"role": "assistant", "content": "NONE"}],
                           "fix the drain bug in loop.py and run the tests after it",
                           allow_write=True, allow_shell=True)
        self.assertNotIn("queue", [step["type"] for step in state["steps"]])

    def test_the_ask_is_one_plain_completion_with_no_tools(self):
        state = self.start([_call("read_file", {"path": "src/app.py"}),
                            _call("write_file", {"path": "note.md", "content": "hi\n"}),
                            {"role": "assistant", "content": "fixed the drain and wrote a note"},
                            {"role": "assistant", "content": "1. Add a test for the drain path"}],
                           "fix the drain bug in loop.py and run the tests after it",
                           allow_write=True)
        self.assertEqual([step["type"] for step in state["steps"]][-1], "final")
        ask = self.engine.calls[-1]
        self.assertIsNone(ask["tools"])
        self.assertEqual(ask["max_tokens"], loop.MAX_SUBTASK_TOKENS)
        self.assertIsNone(ask.get("reasoning_effort"))
        self.assertIsNone(ask.get("enable_thinking"))
        self.assertIn("follow-up tasks", ask["messages"][-1]["content"])

    def test_a_short_run_is_never_asked(self):
        state = self.start([_call("read_file", {"path": "src/app.py"}),
                            {"role": "assistant", "content": "it prints hi"}],
                           "what does app.py print")
        self.assertNotIn("queue", [step["type"] for step in state["steps"]])
        self.assertEqual(len(self.engine.calls), 2)

    def test_one_phase_of_work_is_not_complex_enough_to_continue(self):
        state = self.start([_call("read_file", {"path": "src/app.py"}),
                            _call("grep", {"pattern": "hi"}),
                            {"role": "assistant", "content": "it prints hi and greps clean"}],
                           "check the file at src/app.py and report what it prints")
        self.assertNotIn("queue", [step["type"] for step in state["steps"]])

    def test_a_run_that_skipped_part_of_its_own_prompt_still_queues(self):
        """The shape session 20a8f838a133 arrived in on the live engine.

        It was asked to read a file and then write a checklist, wrote the
        checklist, and never made the reading into a call — one executed phase,
        which the two-executed-phases gate left undecomposed. The run that
        silently drops part of its own plan is the one a follow-up is for, so what
        the prompt asked counts as evidence of complexity too.
        """
        loop._PROPOSED.clear()
        self.addCleanup(loop._PROPOSED.clear)
        state = self.start([_call("write_file", {"path": "checklist.md", "content": "1. a\n"}),
                            {"role": "assistant", "content": "the checklist is written"},
                            {"role": "assistant", "content": "NONE"}],
                           "Read the notes file and write a checklist file with three items",
                           allow_write=True, reasoning_effort="medium")
        items = next(step["items"] for step in state["steps"] if step["type"] == "queue")
        self.assertEqual(items, [loop._UNDONE_TASK["verify"]])
        self.assertNotIn(items[0], [loop._UNDONE_TASK["inspect"]],
                         "the reading it skipped is behind the writing, not ahead of it")

    def test_a_turn_that_touched_nothing_queues_nothing(self):
        """A prompt can ask for three kinds of work and get none of them done.

        Then every phase is unfinished, and handing all three on describes a run
        that finished nothing rather than one with a next step.
        """
        state = self.start([{"role": "assistant", "content": "you could read it and write one"}],
                           "Read the notes file, write a checklist and test it",
                           reasoning_effort="medium")
        self.assertNotIn("queue", [step["type"] for step in state["steps"]])
        self.assertNotIn("follow-up tasks", str(self.engine.calls),
                         "the ask is not even paid for")

    def test_a_queued_subtask_is_never_decomposed_again(self):
        loop._PROPOSED.clear()
        self.addCleanup(loop._PROPOSED.clear)
        state = self.start([_call("read_file", {"path": "src/app.py"}),
                            _call("write_file", {"path": "note.md", "content": "hi\n"}),
                            {"role": "assistant", "content": "fixed the drain and wrote a note"},
                            {"role": "assistant",
                             "content": '["Add a test for the drain path", "Refresh the docs"]'}],
                           "fix the drain bug in loop.py and run the tests after it",
                           allow_write=True)
        items = next(step["items"] for step in state["steps"] if step["type"] == "queue")
        self.assertTrue(loop.is_complex("Review the drain path and update the docs today",
                                        {"inspect", "implement"}, "medium"))
        self.assertFalse(loop.is_complex(items[0], {"inspect", "implement"}, "medium"),
                             "same work, same level: what stops it is that we queued it")
        drained = self.start([{"role": "assistant", "content": "the test is in place"}], items[0],
                            allow_write=True)
        self.assertNotIn("queue", [step["type"] for step in drained["steps"]])
        self.assertEqual(len(self.engine.calls), 1)

    def test_a_dead_ask_leaves_the_run_alone(self):
        class _DyingOnTheAsk(_Scripted):
            def chat(self, messages, **kwargs):
                if "follow-up tasks" in (messages[-1].get("content") or ""):
                    raise RuntimeError("the engine died")
                return super().chat(messages, **kwargs)

        self.engine = _DyingOnTheAsk([_call("read_file", {"path": "src/app.py"}),
                                      _call("write_file", {"path": "note.md", "content": "hi\n"}),
                                      {"role": "assistant", "content": "fixed the drain"}])
        self.client = TestClient(make_app(self.config, self.engine, allowed_hosts=["127.0.0.1"]),
                                 base_url="http://127.0.0.1")
        started = self.client.post("/api/agent/start", headers=HEADERS,
                                   json={"task": "fix the drain bug in loop.py and run it",
                                         "max_steps": 6, "allow_write": True})
        sid = started.json()["id"]
        for _ in range(200):
            state = self.client.get("/api/agent/poll", params={"id": sid},
                                    headers=HEADERS).json()
            if state["status"] != "running":
                break
            threading.Event().wait(0.02)
        types = [step["type"] for step in state["steps"]]
        self.assertEqual(state["status"], "done")
        self.assertNotIn("error", types)
        self.assertEqual(state["steps"][-1]["content"], "fixed the drain")
        self.assertEqual(next(step["items"] for step in state["steps"] if step["type"] == "queue"),
                         [loop._UNDONE_TASK["verify"]],
                         "the skipped phases are read off the run, so a dead ask still fills the queue")


class SubtaskComplexityTests(unittest.TestCase):
    """What the queue calls complex: the work a run did, and what the prompt asked."""

    def test_two_kinds_of_work_run_is_complex(self):
        self.assertTrue(loop.is_complex("read the file", {"inspect", "implement"}, "medium"))

    def test_one_kind_run_needs_the_prompt_to_ask_for_two(self):
        self.assertTrue(loop.is_complex("read the file and write the checklist",
                                        {"implement"}, "medium"))
        self.assertFalse(loop.is_complex("read the file", {"implement"}, "medium"),
                         "a prompt that asked for one kind of work has no phase left to name")

    def test_no_work_run_is_not_complex_however_the_prompt_reads(self):
        self.assertFalse(loop.is_complex("read the file and write and test it", set(), "medium"),
                         "a turn that touched nothing has no next step, it has no step")

    def test_a_low_level_is_never_decomposed(self):
        self.assertFalse(loop.is_complex("read the file and write and test it",
                                        {"inspect", "implement"}, "low"))

    def test_a_phase_is_named_by_a_word_not_by_a_substring(self):
        self.assertEqual(loop._task_phases("count the runtime of the threading code"),
                         {"inspect"}, "runtime is not a request to run, threading not to read")


class SubtaskParserTests(unittest.TestCase):
    """The shapes a decomposition answer actually arrives in, and what must not queue."""

    def test_a_json_array_is_read(self):
        self.assertEqual(loop.parse_subtasks('["Add the drain test", "Refresh the docs"]'),
                         ["Add the drain test", "Refresh the docs"])

    def test_a_numbered_list_is_read(self):
        self.assertEqual(loop.parse_subtasks("Follow-ups:\n1. Add the drain test\n"
                                             "2. Refresh the docs\n"),
                         ["Add the drain test", "Refresh the docs"])

    def test_bullets_and_checkboxes_are_read(self):
        self.assertEqual(loop.parse_subtasks("- Add the drain test\n* [x] Refresh the docs"),
                         ["Add the drain test", "Refresh the docs"])

    def test_unmarked_lines_count_only_when_there_are_two(self):
        self.assertEqual(loop.parse_subtasks("Add the drain test\nRefresh the docs"),
                         ["Add the drain test", "Refresh the docs"])
        self.assertEqual(loop.parse_subtasks("Add the drain test"), [],
                         "one bare line is as likely to be an answer as a task")

    def test_a_third_item_and_nothing_after_it_is_kept(self):
        self.assertEqual(loop.parse_subtasks("1. Add the drain test\n2. Refresh the docs\n"
                                             "3. Delete the cache\n4. Something later"),
                         ["Add the drain test", "Refresh the docs", "Delete the cache"])

    def test_code_and_the_goal_are_not_tasks(self):
        self.assertEqual(loop.parse_subtasks("```python\ndef drain():\n    pass\n```\n"
                                             "1. Add the test"), ["Add the test"])
        self.assertEqual(loop.parse_subtasks("Fix the drain bug", goal="fix the drain bug"), [])

    def test_an_answer_that_is_not_a_list_queues_nothing(self):
        for text in ("", "NONE", "The work is finished.", "   ", "[]", "42"):
            self.assertEqual(loop.parse_subtasks(text), [], text)


def sse_events(text):
    events = []
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


class ReasoningEffortTests(ClientCase):
    """The four dropdown levels reach the engine, and thinking reaches the shells."""

    def _run(self, body):
        started = self.client.post("/api/agent/start", headers=HEADERS, json=body)
        self.assertEqual(started.status_code, 200)
        sid = started.json()["id"]
        for _ in range(150):
            state = self.client.get("/api/agent/poll", params={"id": sid}, headers=HEADERS).json()
            if state["status"] != "running":
                break
            threading.Event().wait(0.02)
        return state

    def test_named_level_reaches_engine_and_prompt(self):
        state = self._run({"task": "say done", "max_steps": 2, "reasoning_effort": "high"})
        self.assertEqual(state["status"], "done")
        sent = self.engine.calls[-1]
        self.assertEqual(sent["reasoning_effort"], "high")
        # Measured on the live engine: `reasoning_effort` alone returns no think
        # block at all, so the flag that does open it has to go with it.
        self.assertIs(sent["enable_thinking"], True)
        # The engine has no thinking budget to size, so the level is also said to
        # the model -- otherwise the middle rows of the dropdown would do nothing.
        self.assertIn("Reasoning effort: high", sent["messages"][0]["content"])

    def test_auto_spends_little_on_a_bare_question(self):
        self._run({"task": "say done", "max_steps": 2})
        sent = self.engine.calls[-1]
        self.assertEqual(sent["reasoning_effort"], "low")
        self.assertIn("Reasoning effort: low", sent["messages"][0]["content"])

    def test_auto_spends_more_when_the_task_names_work(self):
        for task in ("fix the parser and run the tests",
                     "what does src/loop.py do here"):
            self._run({"task": task, "max_steps": 2})
            self.assertEqual(self.engine.calls[-1]["reasoning_effort"], "medium", task)

    def test_a_level_change_applies_to_the_next_turn(self):
        # The stored history carries the system message, so a run that kept the
        # previous turn's line would keep the previous turn's thinking length too.
        first = loop.new_session(task="say done", workspace=str(self.root),
                                 engine_client=self.engine, outlets=self.outlets_stub,
                                 reasoning_effort="low")
        loop.run_agent(first, max_steps=2)
        self.assertIn("Reasoning effort: low", self.engine.calls[-1]["messages"][0]["content"])
        loop.SESSIONS[first]["reasoning_effort"] = "high"
        self.engine.calls.clear()
        loop.run_agent(first, max_steps=2)
        sent = self.engine.calls[-1]
        self.assertEqual(sent["reasoning_effort"], "high")
        self.assertIn("Reasoning effort: high", sent["messages"][0]["content"])

    def test_unknown_level_is_refused(self):
        response = self.client.post("/api/agent/start", headers=HEADERS,
                                    json={"task": "say done", "reasoning_effort": "sideways"})
        self.assertEqual(response.status_code, 422)

    def test_thinking_reaches_the_poll_step(self):
        # This is the whole point of the feature: the shells read the assistant
        # step's `reasoning`, which is what the "Thoughts" block shows.
        self.engine.reply = {"role": "assistant", "content": "done",
                             "reasoning_content": "Multiply first, then check the carry."}
        state = self._run({"task": "say done", "max_steps": 2})
        step = next(s for s in state["steps"] if s["type"] == "assistant")
        self.assertEqual(step["reasoning"], "Multiply first, then check the carry.")

    def test_a_long_think_block_does_not_swallow_the_answer(self):
        # Measured: with a 900-token budget, `medium` spent all 900 on reasoning
        # and returned no content at all. The level's own tokens are what keeps
        # that from being the answer the user sees.
        self._run({"task": "say done", "max_steps": 2, "reasoning_effort": "high"})
        sent = self.engine.calls[-1]
        self.assertEqual(sent["max_tokens"], 2048 + loop.EFFORT_THINK_TOKENS["high"])
        self._run({"task": "say done", "max_steps": 2, "reasoning_effort": "low"})
        self.assertEqual(self.engine.calls[-1]["max_tokens"], 2048)

    def test_reasoning_only_reply_still_answers(self):
        self.engine.reply = {"role": "assistant", "content": "",
                             "reasoning_content": "It catches the first at 11:45."}
        state = self._run({"task": "say done", "max_steps": 2})
        self.assertEqual(state["status"], "done")
        self.assertEqual([s for s in state["steps"] if s["type"] == "final"][0]["content"],
                         "It catches the first at 11:45.")

    def test_resolve_effort(self):
        for level in loop.EFFORT_LINE:
            self.assertEqual(loop.resolve_effort(level, "whatever the task says"), level)
        # Anything unrecognised is `auto`, not a level nobody picked.
        for given in ("sideways", "", None):
            self.assertEqual(loop.resolve_effort(given, "what is 12 times 9"), "low")
        self.assertEqual(loop.resolve_effort("auto", "what is 12 times 9"), "low")
        self.assertEqual(loop.resolve_effort("auto", "rename the field across"
                                             " " + "many " * 12 + "words"), "medium")


class OutletAndAgUiTests(ClientCase):
    def setUp(self):
        super().setUp()
        # The browser lane is stubbed as installed here, so a route that needs it says
        # so rather than inheriting the answer from this machine, and the refusal tests
        # take it away again.
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

    def test_outlet_run_answers_a_fennec_click_with_fennecs_own_loop(self):
        response = self.client.post("/api/outlets/run", headers=HEADERS,
                                    json={"outlet": "fennec", "goal": "cart shows one item",
                                          "url": "https://example.com", "max_steps": 3})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True, "engine": "liquid-d1.2",
                                           "outlet": "fennec-webagent", "steps": 1})
        self.assertEqual(self.outlets_stub.calls,
                         [("web_task", "cart shows one item", "https://example.com",
                           {"max_steps": 3})])
        # There is one browser lane, so the route calls exactly what it gated on. No engine
        # is threaded through, which means no second agent can answer a click aimed at this one.
        self.assertEqual(self.outlets_stub.asks, ["web_task"])

    def test_outlet_run_refuses_an_absent_outlet_and_names_the_fix(self):
        self.outlets_stub.available_tools.clear()
        for outlet, tool in (("fennec", "web_task"), ("openbot", "coworker_task")):
            with self.subTest(outlet=outlet):
                self.outlets_stub.calls.clear()
                self.outlets_stub.asks.clear()
                response = self.client.post("/api/outlets/run", headers=HEADERS,
                                            json={"outlet": outlet, "goal": "do the thing"})
                self.assertEqual(response.status_code, 409)
                self.assertIn(f"{outlet} is not available", response.json()["detail"])
                self.assertIn("fennec outlets", response.json()["detail"])
                self.assertEqual(self.outlets_stub.calls, [])
                self.assertEqual(self.outlets_stub.asks, [tool])

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
        # A shell still spelling the agent this suite deleted lands on the same refusal:
        # the name is unknown now, and the answer tells it what the route does take.
        retired = self.client.post("/api/outlets/run", headers=HEADERS,
                                   json={"outlet": "jev", "goal": "x"})
        empty = self.client.post("/api/outlets/run", headers=HEADERS,
                                 json={"outlet": "fennec", "goal": ""})
        for response in (unknown, retired):
            self.assertEqual(response.status_code, 422, response.json())
            self.assertNotIn("jev", response.json()["detail"])
        self.assertEqual(self.outlets_stub.calls, [], "a refused name never reaches a lane")
        self.assertEqual(self.outlets_stub.asks, [], "and is never gated on either tool")
        # The refusal has to name every id the route takes, or a shell reading it
        # cannot discover the lane it should have asked for.
        for name in ("fennec", "openbot"):
            self.assertIn(f"'{name}'", unknown.json()["detail"])
        self.assertEqual(empty.status_code, 422)
        self.assertEqual(self.client.post("/api/outlets/run",
                                          json={"outlet": "fennec", "goal": "x"}).status_code, 403)

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
        import acp as acp_mod
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
        import setup_wizard as wizard
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            for backend in wizard.BACKENDS:
                path.write_text(json.dumps({"backend": backend}), encoding="utf-8")
                self.assertEqual(load_config(path, environ={}, home=Path(tmp)).backend, backend)

    def test_the_browser_interpreter_loads_from_json_and_env(self):
        """The interpreter the loop runs in is a configured choice, not only an env var."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"web_python": "/opt/browser/python"}), encoding="utf-8")
            self.assertEqual(load_config(path, environ={}, home=Path(tmp)).web_python,
                             "/opt/browser/python")
            path.write_text(json.dumps({"web_python": "   "}), encoding="utf-8")
            self.assertEqual(load_config(path, environ={}, home=Path(tmp)).web_python, "")
            self.assertEqual(load_config(path, environ={"FENNEC_WEB_PYTHON": "/env/python"},
                                         home=Path(tmp)).web_python, "/env/python")
            for bad in ({"web_python": 5}, {"web_python": None}):
                path.write_text(json.dumps(bad), encoding="utf-8")
                with self.assertRaises(ConfigError, msg=bad):
                    load_config(path, environ={}, home=Path(tmp))

    def test_a_config_written_before_the_lane_was_collapsed_still_loads(self):
        """Retired keys are ignored on read, so an upgrade never bricks an existing install."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"jev_root": str(Path(tmp) / "jev-ultrafast"),
                                        "web_engine": "jev",
                                        "web_python": "/opt/browser/python"}), encoding="utf-8")
            config = load_config(path, environ={}, home=Path(tmp))
            self.assertEqual(config.web_python, "/opt/browser/python")
            for retired in ("jev_root", "web_engine"):
                self.assertFalse(hasattr(config, retired), f"{retired} is back")

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
        import engine
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
        import engine
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
        self.calls.append({"messages": list(messages), **kwargs})
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
        from tools import GATES, TOOLS, approval_context
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


# The closing tags, built in two pieces: a literal markup close inside a test
# string trips the tooling that edits this file, so they stay assembled here.
_PARAM_CLOSE = "<" + "/parameter>"
_FUNCTION_CLOSE = "<" + "/function>"


def _param(key, value):
    return f'<parameter name="{key}">{value}{_PARAM_CLOSE}'


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

    def test_a_call_expression_executes(self):
        for content in ('toolwrite_file(path="note.md", content="hi")',
                        "write_file(path='note.md', content='hi')"):
            with self.subTest(content=content[:20]):
                sid = self.start([{"role": "assistant", "content": content},
                                  {"role": "assistant", "content": "done"}],
                                 allow_write=True)
                state = self.settle(sid)
                self.assertEqual(state["status"], "done")
                self.assertEqual((self.root / "note.md").read_text(encoding="utf-8"), "hi")

    def test_a_fence_named_call_the_model_wrote_instead_of_making_is_run(self):
        """The live miss: a `read_file` fence the loop read as an answer.

        Session 298eb3e772ae ended with the file still unread, so the queue had
        no phase evidence to work from and handed on nothing.
        """
        (self.root / "notes.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
        sid = self.start([{"role": "assistant",
                           "content": "\n\n```read_file\npath=notes.txt\n```"},
                          {"role": "assistant", "content": "notes.txt has three items"}],
                         task="Read notes.txt, then write checklist.md with one item",
                         allow_write=True, max_steps=6)
        state = self.settle(sid)
        self.assertEqual(state["status"], "done")
        read = [s for s in state["steps"] if s["type"] == "tool"]
        self.assertEqual([s["name"] for s in read], ["read_file"])
        self.assertTrue(read[0]["ok"], "the block the model wrote is the call it meant")

    def test_a_call_quoted_inside_a_sentence_stays_prose(self):
        sid = self.start([{"role": "assistant",
                           "content": 'I could call write_file(path="note.md", '
                                      'content="hi") here, or not.'},
                          {"role": "assistant", "content": "done"}],
                         allow_write=True)
        state = self.settle(sid)
        self.assertEqual(state["status"], "done")
        self.assertFalse((self.root / "note.md").exists())
        self.assertEqual([s["type"] for s in state["steps"]].count("tool"), 0)

    def test_a_command_spelling_asks_for_approval_too(self):
        sid = self.start([{"role": "assistant",
                           "content": json.dumps({"command": "write_file",
                                                  "path": "note.md", "content": "hi"})},
                          {"role": "assistant", "content": "done"}],
                         require_approval=True)
        ask = self.ask(sid)
        self.assertEqual(ask["name"], "write_file")
        self.assertEqual(ask["args"], {"path": "note.md", "content": "hi"})

    def test_the_parameter_bag_spelled_arguments_counts_too(self):
        for bag in ("args", "arguments"):
            with self.subTest(bag=bag):
                sid = self.start([{"role": "assistant",
                                   "content": json.dumps({"name": "write_file",
                                                          bag: {"path": "note.md",
                                                                "content": "hi"}})},
                                  {"role": "assistant", "content": "done"}],
                                 require_approval=True)
                ask = self.ask(sid)
                self.assertEqual(ask["name"], "write_file")
                self.assertEqual(ask["args"], {"path": "note.md", "content": "hi"})

    def test_a_bag_that_is_not_an_object_falls_back_to_the_flat_keys(self):
        sid = self.start([{"role": "assistant",
                           "content": json.dumps({"tool": "write_file",
                                                  "arguments": "not an object",
                                                  "path": "note.md", "content": "hi"})},
                          {"role": "assistant", "content": "done"}],
                         require_approval=True)
        ask = self.ask(sid)
        self.assertEqual(ask["name"], "write_file")
        self.assertEqual(ask["args"], {"path": "note.md", "content": "hi"})
        self.assertNotIn("arguments", ask["args"])

    def test_a_bare_shell_command_is_not_read_as_a_tool_name(self):
        sid = self.start([{"role": "assistant",
                           "content": json.dumps({"command": "ls -la"})},
                          {"role": "assistant", "content": "done"}],
                         require_approval=True)
        state = self.settle(sid)
        self.assertEqual(state["status"], "done")
        self.assertEqual([s["type"] for s in state["steps"]].count("tool"), 0)

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

    def test_a_call_written_as_a_bare_line_still_executes(self):
        for content in ("read_file path = src/app.py",
                        "read_file file_path = src/app.py",
                        "grep pattern = hi"):
            with self.subTest(content=content):
                sid = self.start([{"role": "assistant", "content": content},
                                  {"role": "assistant", "content": "done"}])
                state = self.settle(sid)
                self.assertEqual(state["status"], "done")
                ran = [s for s in state["steps"] if s["type"] == "tool"]
                self.assertEqual([s["name"] for s in ran], [content.split()[0]])
                self.assertTrue(ran[0]["ok"], ran[0])

    def test_a_line_that_is_not_a_whole_call_stays_prose(self):
        for content in ("read_file path = src/app.py and then tell me what it says",
                        "write_file path = note.md content = hi",
                        "I could read_file path = src/app.py here, or not."):
            with self.subTest(content=content[:32]):
                sid = self.start([{"role": "assistant", "content": content},
                                  {"role": "assistant", "content": "done"}], allow_write=True)
                state = self.settle(sid)
                self.assertEqual(state["status"], "done")
                self.assertEqual([s["type"] for s in state["steps"]].count("tool"), 0)
                self.assertFalse((self.root / "note.md").exists())

    def test_a_how_to_reply_is_pushed_back_once_and_then_runs(self):
        how_to = ("Sure.\n```bash\nwrite_file path = note.md content = \"hi\"\n```")
        sid = self.start([{"role": "assistant", "content": how_to},
                          _call("write_file", {"path": "note.md", "content": "hi"}),
                          {"role": "assistant", "content": "done"}], allow_write=True)
        state = self.settle(sid)
        self.assertEqual(state["status"], "done")
        self.assertEqual((self.root / "note.md").read_text(encoding="utf-8"), "hi")
        # One push-back, one call, one answer: the loop asked exactly once.
        self.assertEqual(len(self.engine.calls), 3)

    def test_the_loop_does_not_ask_twice_in_one_turn(self):
        how_to = "```bash\nwrite_file path = note.md content = \"hi\"\n```"
        sid = self.start([{"role": "assistant", "content": how_to},
                          {"role": "assistant", "content": how_to},
                          {"role": "assistant", "content": "done"}], allow_write=True)
        state = self.settle(sid)
        self.assertEqual(state["status"], "done")
        self.assertEqual([s["type"] for s in state["steps"]].count("tool"), 0)
        self.assertFalse((self.root / "note.md").exists())
        self.assertEqual(len(self.engine.calls), 2)

    def test_a_second_prose_regression_after_a_real_call_is_asked_again(self):
        """The ask is per stretch of prose, so a long task gets held to it twice.

        The live engine read the file, then wrote the next step down as a bash
        block instead of making it, and the run closed one phase short of the
        work it was asked for. A model that has already proved it can call has
        no excuse the second time.
        """
        fence = "```bash\nwrite_file path = b.md content = \"two\"\n```"
        sid = self.start([_call("write_file", {"path": "a.md", "content": "one"}),
                          {"role": "assistant", "content": fence},
                          _call("write_file", {"path": "b.md", "content": "two"}),
                          {"role": "assistant", "content": "done"}],
                         allow_write=True, max_steps=6)
        state = self.settle(sid)
        self.assertEqual(state["status"], "done")
        self.assertEqual(len(self.engine.calls), 4)
        self.assertEqual((self.root / "a.md").read_text(encoding="utf-8"), "one")
        self.assertEqual((self.root / "b.md").read_text(encoding="utf-8"), "two")

    def test_an_answer_that_shows_code_is_not_pushed_back(self):
        answer = "Here is the helper:\n```python\ndef helper():\n    return 1\n```"
        sid = self.start([{"role": "assistant", "content": answer}])
        state = self.settle(sid)
        self.assertEqual(state["status"], "done")
        self.assertEqual(len(self.engine.calls), 1)
        self.assertIn("def helper", state["steps"][-1]["content"])

    def test_a_bash_fence_with_no_answer_is_pushed_back(self):
        """The shape the live engine sent when it skipped the call format.

        A shell info string names a language, not a tool, so the tool-name test
        cannot see it. With no prose around it the reply holds no answer, so the
        loop asks for the real call rather than closing on a command nobody ran.
        """
        content = "\n\n```bash\nprintf ok\n```"
        sid = self.start([{"role": "assistant", "content": content},
                          _call("run_shell", {"command": "printf ok"}),
                          {"role": "assistant", "content": "done"}], allow_shell=True)
        state = self.settle(sid)
        self.assertEqual(state["status"], "done")
        self.assertEqual(len(self.engine.calls), 3)
        self.assertIn("tool", [s["type"] for s in state["steps"]])

    def test_an_answer_that_quotes_a_command_is_not_pushed_back(self):
        """The same fence inside a sentence is documentation, and stays one."""
        answer = "To list everything, run:\n```bash\nls -R\n```"
        sid = self.start([{"role": "assistant", "content": answer}], allow_shell=True)
        state = self.settle(sid)
        self.assertEqual(state["status"], "done")
        self.assertEqual(len(self.engine.calls), 1)
        self.assertIn("ls -R", state["steps"][-1]["content"])

    def test_the_tagged_call_the_live_engine_sends_executes(self):
        """The shape a qwen36 run actually answered with, executed not argued with.

        The chat template parks the tool name in a fence and the parameters
        outside it, so the loop used to read that as prose, hand it back to the
        user as a how-to, and never do the work it had just described.
        """
        content = ("```tools\nwrite_file\n```\n" + _param("path", "note.md") + "\n"
                   + _param("content", "\nhi\n") + "\n```")
        sid = self.start([{"role": "assistant", "content": content},
                          {"role": "assistant", "content": "done"}], allow_write=True)
        state = self.settle(sid)
        self.assertEqual(state["status"], "done")
        self.assertEqual((self.root / "note.md").read_text(encoding="utf-8"), "hi")
        # It ran on the first turn, so the loop never spent a push-back asking.
        self.assertEqual(len(self.engine.calls), 2)

    def test_a_tagged_call_missing_a_required_parameter_stays_prose(self):
        content = "```tools\nwrite_file\n```\n" + _param("content", "hi")
        sid = self.start([{"role": "assistant", "content": content},
                          {"role": "assistant", "content": "done"}], allow_write=True)
        state = self.settle(sid)
        self.assertEqual(state["status"], "done")
        self.assertFalse((self.root / "note.md").exists())

    def test_a_tool_code_block_the_engine_sends_executes(self):
        """Qwen3-Coder's own call wrapper, executed rather than shown.

        The body is a call expression, so the wrapper has to come off before the
        parsers run: left on, the closing tag reads as trailing prose after the
        bracket and the call becomes the how-to the loop used to accept.
        """
        content = '<tool_code>\nwrite_file(path="note.md", content="hi")\n</tool_code>'
        sid = self.start([{"role": "assistant", "content": content},
                          {"role": "assistant", "content": "done"}], allow_write=True)
        state = self.settle(sid)
        self.assertEqual(state["status"], "done")
        self.assertEqual((self.root / "note.md").read_text(encoding="utf-8"), "hi")
        self.assertEqual(len(self.engine.calls), 2)

    def test_a_tool_code_block_we_cannot_read_is_pushed_back_once(self):
        """A call the model wrapped in the tool tag but spelled as raw Python.

        The tag is the model declaring it is making a call, so the reply is not
        an answer even when no tool name is in it: the loop asks for the real
        call instead of closing the run on a block that did nothing.
        """
        block = "<tool_code>\nprint(open('note.md').read())\n</tool_code>"
        sid = self.start([{"role": "assistant", "content": block},
                          _call("write_file", {"path": "note.md", "content": "hi"}),
                          {"role": "assistant", "content": "done"}], allow_write=True)
        state = self.settle(sid)
        self.assertEqual(state["status"], "done")
        self.assertEqual((self.root / "note.md").read_text(encoding="utf-8"), "hi")
        self.assertEqual(len(self.engine.calls), 3)

    def test_a_pasted_file_the_task_asked_for_is_written_not_answered(self):
        """The fifth shape: the deliverable, pasted into the chat instead of disk.

        Live on the PWA the run read the notes file and then closed with the
        checklist it had been told to write sitting in a fence. No test could see
        it -- `markdown` is a language, not a tool, and nothing in the block named
        a tool -- so the file never existed and the task ended one phase short of
        the queue ask.
        """
        body = "- [ ] verify the queue strip\n- [ ] write the regression test\n"
        sid = self.start([{"role": "assistant", "content": "```markdown\n" + body + "```"},
                          _call("write_file", {"path": "checklist.md", "content": body}),
                          {"role": "assistant", "content": "done"}],
                         task="Read notes.txt, then write checklist.md with one "
                              "checkbox line per item you found.",
                         allow_write=True, max_steps=6)
        state = self.settle(sid)
        self.assertEqual(state["status"], "done")
        self.assertEqual((self.root / "checklist.md").read_text(encoding="utf-8"), body)
        asks = [call for call in self.engine.calls
                if "follow-up tasks" in (call["messages"][-1].get("content") or "")]
        self.assertEqual(len(self.engine.calls) - len(asks), 3)
        self.assertEqual(len(asks), 1,
                         "asked to read and to write, and it only wrote: that is the run "
                         "whose skipped phases the queue has to hand on")
        self.assertEqual([s["items"] for s in state["steps"] if s["type"] == "queue"],
                         [[loop._UNDONE_TASK["verify"]]])
        self.assertIn("tool", [s["type"] for s in state["steps"]])

    def test_the_same_block_after_the_file_is_written_is_a_summary(self):
        """Restating a file the run really wrote is the closing summary, not a
        second deliverable, so the loop leaves the answer alone."""
        body = "- [ ] verify the queue strip\n"
        sid = self.start([_call("write_file", {"path": "checklist.md", "content": body}),
                          {"role": "assistant", "content": "Wrote it:\n```markdown\n" + body + "```"}],
                         task="write checklist.md with one item",
                         allow_write=True, max_steps=6)
        state = self.settle(sid)
        self.assertEqual(state["status"], "done")
        self.assertEqual(len(self.engine.calls), 2)
        self.assertEqual((self.root / "checklist.md").read_text(encoding="utf-8"), body)

    def test_a_pasted_file_is_not_demanded_when_writing_is_switched_off(self):
        sid = self.start([{"role": "assistant", "content": "```markdown\n- [ ] a\n```"}],
                         task="write checklist.md with one item", max_steps=3)
        state = self.settle(sid)
        self.assertEqual(state["status"], "done")
        self.assertEqual(len(self.engine.calls), 1)
        self.assertIn("- [ ] a", state["steps"][-1]["content"])

    def test_a_task_that_only_reads_a_file_is_not_pushed_to_write_it(self):
        sid = self.start([{"role": "assistant",
                           "content": "The file says:\n```text\nall green\n```"}],
                         task="read notes.txt and tell me what it says",
                         allow_write=True, max_steps=3)
        state = self.settle(sid)
        self.assertEqual(state["status"], "done")
        self.assertEqual(len(self.engine.calls), 1)
        self.assertFalse((self.root / "notes.txt").exists())

    def test_a_call_cut_off_mid_json_is_asked_for_and_then_written(self):
        """The sixth shape, caught live on the PWA at 06:57.

        The run pushed back a bash fence, the model answered with a real JSON
        call, and the completion ran out inside it. Nothing could parse the
        fragment, so the loop was about to end the task with the call as its
        final answer and no file on disk.
        """
        cut = ('{"name": "write_file", "args": {"path": "checklist.md", '
               '"content": "- [ ] one\\n- [ ] two\\n')
        sid = self.start([{"role": "assistant", "content": cut},
                          _call("write_file", {"path": "checklist.md",
                                               "content": "- [ ] one\n- [ ] two\n"}),
                          {"role": "assistant", "content": "done"}],
                         task="create a file checklist.md with two checkbox lines",
                         allow_write=True, max_steps=6)
        state = self.settle(sid)
        self.assertEqual(state["status"], "done")
        self.assertEqual(len(self.engine.calls), 3)
        self.assertEqual((self.root / "checklist.md").read_text(encoding="utf-8"),
                         "- [ ] one\n- [ ] two\n")


class XmlToolCallTests(unittest.TestCase):
    """Reading the XML call shape back, where the template split it apart."""

    @classmethod
    def setUpClass(cls):
        from tools import ToolContext, openai_tools
        cls.tools = openai_tools(ToolContext(workspace=".", allow_write=True))

    def call(self, text):
        return loop._text_tool_call(text, self.tools)

    def test_the_name_and_its_tags_are_joined_in_both_spellings(self):
        with self.subTest("name on a line of its own"):
            self.assertEqual(self.call("read_file\n" + _param("path", "notes.txt")),
                             ("read_file", {"path": "notes.txt"}))
        with self.subTest("name inside a function tag"):
            self.assertEqual(self.call('<function name="write_file">'
                                       + _param("path", "note.md")
                                       + _param("content", "hi") + _FUNCTION_CLOSE),
                             ("write_file", {"path": "note.md", "content": "hi"}))

    def test_the_template_noise_falls_out_and_the_payload_stays(self):
        # What the live run wrote: an empty body tag the template fills in, an
        # encoding no tool declares, and the same call repeated after it.
        text = ("```tools\nread_file\n```\n" + _param("path", "notes.txt") + "\n"
                + _param("content", "\n") + "\n" + _param("encoding", "utf-8")
                + "\n```\n\n" + "```tools\nread_file\n```\n" + _param("path", "notes.txt"))
        self.assertEqual(self.call(text), ("read_file", {"path": "notes.txt"}))
        # Only the newline the template wraps a value in goes; the model's own
        # blank line between items survives.
        self.assertEqual(
            self.call("write_file\n" + _param("path", "x.md") + "\n"
                      + _param("content", "\n\nbody\n\n"))[1]["content"], "\nbody\n")

    def test_the_last_call_named_is_the_one_that_runs(self):
        text = ("read_file\n" + _param("path", "notes.txt") + "\n"
                + "write_file\n" + _param("path", "note.md") + "\n"
                + _param("content", "hi"))
        self.assertEqual(self.call(text), ("write_file", {"path": "note.md",
                                                          "content": "hi"}))

    def test_prose_that_names_a_tool_without_tags_is_not_a_call(self):
        self.assertIsNone(self.call("I would call read_file on notes.txt, "
                                    "then write_file."))
        self.assertIsNone(self.call("```tools\nread_file\n```"))


class FenceNamedCallTests(unittest.TestCase):
    """The call the model parks in a fence: name as the language, parameters below.

    Live on 2026-11-19, twice (sessions 298eb3e772ae and 20a8f838a133), a reading
    prompt came back as a `read_file` fence with `path=` under it. The loop read
    only the last line for that shape, so the name and its argument never joined
    up: it pushed back once, the model repeated the block, and the run ended with
    the file still unread and nothing for the queue to hand on. A third run
    (068a4d2ea0c4) qualified the same name as `tool:read_file`.
    """

    @classmethod
    def setUpClass(cls):
        from tools import ToolContext, openai_tools
        cls.tools = openai_tools(ToolContext(workspace=".", allow_write=True))

    def call(self, text):
        return loop._text_tool_call(text, self.tools)

    def test_the_fence_language_and_its_parameters_are_one_call(self):
        self.assertEqual(self.call("\n\n```read_file\n"
                                   "path=.local-docs/queue-probe/notes.txt\n```"),
                         ("read_file", {"path": ".local-docs/queue-probe/notes.txt"}))
        self.assertEqual(self.call("```write_file\npath=note.md\ncontent=hi\n```"),
                         ("write_file", {"path": "note.md", "content": "hi"}))

    def test_a_qualified_tool_name_is_still_the_tool(self):
        with self.subTest("the live block, name qualified"):
            self.assertEqual(self.call("\n\n```tool:read_file\npath=notes.txt\n```"),
                             ("read_file", {"path": "notes.txt"}))
        with self.subTest("other spellings of the same qualification"):
            self.assertEqual(self.call("```call:write_file\npath=note.md\ncontent=hi\n```"),
                             ("write_file", {"path": "note.md", "content": "hi"}))
            self.assertEqual(self.call("```tool.read_file\npath=notes.txt\n```"),
                             ("read_file", {"path": "notes.txt"}))
        with self.subTest("a qualified name that is no tool stays prose"):
            self.assertIsNone(self.call("```tool:get_weather\npath=notes.txt\n```"))

    def test_the_last_block_is_the_one_that_runs(self):
        text = ("```read_file\npath=notes.txt\n```\nNow the note itself.\n"
                "```write_file\npath=note.md\ncontent=hi\n```")
        self.assertEqual(self.call(text), ("write_file", {"path": "note.md", "content": "hi"}))

    def test_a_block_that_trails_into_prose_is_not_a_call(self):
        self.assertIsNone(self.call("```write_file\npath=note.md\nthen think about it\n```"),
                          "half a call would leave a file with no body")
        self.assertIsNone(self.call("```read_file\n```"), "a name with no parameter is prose")

    def test_a_language_that_is_no_tool_is_still_an_answer(self):
        self.assertIsNone(self.call("```python\nimport os\nprint(os.listdir('.'))\n```"))


class ToolCodeBlockTests(unittest.TestCase):
    """The wrapper a Qwen3-Coder checkpoint puts its call in."""

    @classmethod
    def setUpClass(cls):
        from tools import ToolContext, openai_tools
        cls.tools = openai_tools(ToolContext(workspace=".", allow_write=True))

    def call(self, text):
        return loop._text_tool_call(text, self.tools)

    def test_the_call_inside_the_wrapper_is_read(self):
        self.assertEqual(self.call("<tool_code>\nread_file(path='notes.txt')\n</tool_code>"),
                         ("read_file", {"path": "notes.txt"}))
        # A lead-in sentence does not move the call out of the last position.
        self.assertEqual(self.call('Let me write it.\n<tool_code>'
                                   'write_file(path="note.md", content="hi")</tool_code>'),
                         ("write_file", {"path": "note.md", "content": "hi"}))

    def test_a_quoted_example_followed_by_prose_stays_prose(self):
        self.assertIsNone(self.call('<tool_code>read_file(path="note.md")</tool_code>\n'
                                    "is the shape you want."))

    def test_a_block_naming_no_tool_is_still_a_call_the_model_meant(self):
        # The tag says "make this call", so a body the parsers cannot read is a
        # call that did not happen, not an answer to be settled on.
        self.assertTrue(loop._looks_like_how_to(
            "<tool_code>\nprint(open('notes.txt').read())\n</tool_code>", self.tools))


class CommandFenceTests(unittest.TestCase):
    """A reply that is nothing but a shell block the loop could have run."""

    @classmethod
    def setUpClass(cls):
        from tools import ToolContext, openai_tools
        cls.tools = openai_tools(ToolContext(workspace=".", allow_write=True,
                                             allow_shell=True))
        cls.noshell = openai_tools(ToolContext(workspace=".", allow_write=True))

    def test_a_bare_shell_block_is_a_call_the_model_meant(self):
        self.assertTrue(loop._looks_like_how_to("\n```bash\nprintf ok\n```", self.tools))
        self.assertTrue(loop._looks_like_how_to("```sh\ncat notes.txt\n```", self.tools))
        self.assertTrue(loop._looks_like_how_to("```powershell\nGet-ChildItem\n```", self.tools))

    def test_prose_outside_the_block_makes_it_an_answer(self):
        self.assertFalse(loop._looks_like_how_to(
            "To list the files, run:\n```bash\nls -R\n```", self.tools))
        self.assertFalse(loop._looks_like_how_to(
            "```bash\nls -R\n``` shows the tree.", self.tools))

    def test_a_shell_block_is_left_alone_when_no_shell_tool_is_offered(self):
        # Pushing back here would demand a call the toolset cannot make.
        self.assertFalse(loop._looks_like_how_to("```bash\nls -R\n```", self.noshell))

    def test_a_non_shell_language_is_not_a_command(self):
        self.assertFalse(loop._looks_like_how_to(
            "```python\ndef helper():\n    return 1\n```", self.tools))


class UnreadableCallTests(unittest.TestCase):
    """A reply that opened as a call and ran out of room before it could be read."""

    @classmethod
    def setUpClass(cls):
        from tools import ToolContext, openai_tools
        cls.tools = openai_tools(ToolContext(workspace=".", allow_write=True,
                                             allow_shell=True))

    def test_a_cut_off_call_names_a_tool_the_loop_offered(self):
        cut = '{"name": "write_file", "args": {"path": "a.md", "content": "- x'
        self.assertTrue(loop._opens_as_an_unreadable_call(cut, {"write_file"}))
        self.assertTrue(loop._looks_like_how_to(cut, self.tools))

    def test_the_same_fragment_naming_no_advertised_tool_is_an_answer(self):
        cut = '{"name": "get_weather", "args": {"city": "Os'
        self.assertFalse(loop._opens_as_an_unreadable_call(cut, {"write_file"}))
        self.assertFalse(loop._looks_like_how_to(cut, self.tools))

    def test_a_complete_call_the_schema_rejects_is_still_a_call_attempt(self):
        # No `path`, so `_text_tool_call` refuses it; the reply is still the
        # call, and the one ask that follows is what the loop can do.
        odd = '{"tool": "read_file", "args": {"offset": 5}}'
        self.assertIsNone(loop._text_tool_call(odd, self.tools))
        self.assertTrue(loop._looks_like_how_to(odd, self.tools))

    def test_prose_and_json_that_does_not_open_a_call_are_left_alone(self):
        self.assertFalse(loop._looks_like_how_to('The answer is {"name": "x"}.', self.tools))
        self.assertFalse(loop._looks_like_how_to("", self.tools))
        self.assertFalse(loop._looks_like_how_to(None, self.tools))


class PastedFileTests(unittest.TestCase):
    """Which paths a prompt asks for, and when a pasted block owes the disk one."""

    def test_only_a_write_ask_in_the_same_clause_counts(self):
        task = ("Read .local-docs/queue-probe/notes.txt, then write "
                ".local-docs/queue-probe/checklist8.md with one checkbox line per "
                "item. Afterwards run python to count the lines in checklist8.md.")
        self.assertEqual(loop._paths_asked_to_write(task),
                         [".local-docs/queue-probe/checklist8.md"])

    def test_a_verb_from_the_previous_clause_does_not_reach_the_next_path(self):
        self.assertEqual(loop._paths_asked_to_write("write nothing; then read notes.txt"), [])
        self.assertEqual(loop._paths_asked_to_write("read notes.txt and summarize it"), [])

    def test_a_bare_name_and_a_quoted_path_both_count(self):
        self.assertEqual(loop._paths_asked_to_write("Create a README.md for this project"),
                         ["README.md"])
        self.assertEqual(loop._paths_asked_to_write('save the log to "logs/day.txt"'),
                         ["logs/day.txt"])

    def test_a_block_owes_the_disk_only_for_a_path_still_unwritten(self):
        names = {"write_file", "read_file"}
        self.assertTrue(loop._pasted_an_unwritten_file("```md\n- a\n```", ["/w/a.md"], names))
        self.assertFalse(loop._pasted_an_unwritten_file("```md\n- a\n```", [], names))
        self.assertFalse(loop._pasted_an_unwritten_file("all done", ["/w/a.md"], names))
        self.assertFalse(loop._pasted_an_unwritten_file("```md\n- a\n```", ["/w/a.md"],
                                                        {"read_file"}))

    def test_the_same_file_spelled_two_ways_compares_equal(self):
        from tools import ToolContext
        ctx = ToolContext(workspace=".")
        self.assertEqual(loop._resolve_path(ctx, "./a.md"), loop._resolve_path(ctx, "a.md"))
        self.assertEqual(loop._resolve_path(ctx, "sub/../a.md"), loop._resolve_path(ctx, "a.md"))


if __name__ == "__main__":
    unittest.main()
