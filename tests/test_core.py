import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from fennec_core import loop
from fennec_core.config import ConfigError, load_config
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
