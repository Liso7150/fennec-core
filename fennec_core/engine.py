"""Lightweight engine client/supervisor. Importing never loads models or starts services."""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from .config import Config, is_loopback, load_config


class EngineError(RuntimeError):
    pass


class EngineCancelled(EngineError):
    """The turn was aborted while the engine was still answering."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward credentials to another origin.


VULKAN_LINUX = ("/usr/lib/x86_64-linux-gnu/libvulkan.so.1", "/usr/lib64/libvulkan.so.1",
                 "/usr/lib/libvulkan.so.1")


def vulkan_loader_present(exists=None, system=None, environ=None):
    """A Vulkan loader on disk is what makes COLI_VULKAN=1 startable."""
    probe = os.path.exists if exists is None else exists
    name = platform.system() if system is None else system
    env = os.environ if environ is None else environ
    if name == "Darwin":
        return False
    if name == "Windows":
        root = env.get("SystemRoot") or env.get("SYSTEMROOT") or "C:\\Windows"
        return probe(os.path.join(root, "System32", "vulkan-1.dll"))
    return any(probe(candidate) for candidate in VULKAN_LINUX)


def spawn_options(python):
    if os.name == "nt":
        windowless = Path(python).with_name("pythonw.exe")
        if windowless.is_file():
            python = str(windowless)
        return python, {"creationflags": subprocess.CREATE_NO_WINDOW}
    return python, {"start_new_session": True}


class Engine:
    def __init__(self, config: Config):
        self.config = config
        self._process = None
        self._npu_process = None
        self._stt_process = None
        self._lock = threading.Lock()
        self.log_path = config.cache_root / "fennec" / "engine.log"
        self.npu_log_path = config.cache_root / "fennec" / "npu.log"
        self.stt_log_path = config.cache_root / "fennec" / "stt.log"
        self.npu_url = "http://127.0.0.1:8182"
        self.stt_url = "http://127.0.0.1:8183"

    def _redact(self, text):
        for secret in (self.config.api_key, os.environ.get("FENNEC_APP_TOKEN")):
            if secret:
                text = text.replace(secret, "[redacted]")
        return text

    def _request(self, path, body=None, timeout=4, *, npu=False):
        base = self.npu_url if npu else self.config.engine_url
        headers = {"Content-Type": "application/json"}
        if not npu and self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib.request.Request(base + path, headers=headers,
                                         data=None if body is None else json.dumps(body).encode())
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        try:
            with opener.open(request, timeout=timeout) as response:
                payload = response.read(2_000_001)
                if len(payload) > 2_000_000:
                    raise EngineError("Engine response exceeded the size limit")
                return json.loads(payload)
        except EngineError:
            raise
        except Exception as exc:
            raise EngineError("Engine request failed; check engine health and configuration") from exc

    def is_running(self):
        try:
            self._request("/health", timeout=2)
            return True
        except EngineError:
            return False

    def model_id(self):
        # An attached server can switch models; do not retain a stale ID.
        try:
            return self._request("/v1/models")["data"][0]["id"]
        except (KeyError, IndexError, TypeError) as exc:
            raise EngineError("Engine did not advertise a model") from exc

    def _engine_env(self):
        env = dict(os.environ)
        for key in ("FENNEC_APP_TOKEN", "FENNEC_API_KEY", "COLI_API_KEY",
                    "COLI_VULKAN", "COLI_METAL", "COLI_CUDA", "COLI_VK_DEV2"):
            env.pop(key, None)
        env.update(HF_HOME=str(self.config.cache_root / "huggingface"),
                   COLI_MODEL=self.config.model, COLI_API_KEY=self.config.api_key)
        # serve has no --backend flag. `auto` still needs the request: without it the
        # engine logs "COLI_VULKAN is not set: the Vulkan device stays unused" and runs
        # CPU + RAM + SSD only. --auto-tier then plans GPU, host pool and disk stream.
        if self.config.backend == "auto":
            if vulkan_loader_present():
                env["COLI_VULKAN"] = "1"
                # 4 GB of host expert pool costs ~6 % decode against the engine's
                # 13 GB default (13.95 vs 14.88 tok/s, 2026-11-10) and gives back
                # ~9 GB of RAM; 2 GB collapses to 6.27 tok/s.
                if "COLI_VK_TIER_HOST_GB" not in env:
                    env["COLI_VK_TIER_HOST_GB"] = "4"
        else:
            env.update(COLI_VULKAN="0", COLI_METAL="0", COLI_CUDA="0")
            if self.config.backend != "cpu":
                env["COLI_" + self.config.backend.upper()] = "1"
        return env

    def _command(self, args):
        launcher = self.config.engine_root / "c" / "coli"
        if not launcher.is_file():
            raise EngineError("Fennec launcher missing; configure engine_root with an existing installation")
        python, options = spawn_options(self.config.python)
        return [python, str(launcher), *args], options

    def _model_args(self):
        if not self.config.model or not Path(self.config.model).is_dir():
            raise EngineError("Choose an installed model with FENNEC_MODEL or the model configuration key")
        return ["--model", self.config.model, "--policy", self.config.policy, "--ctx", str(self.config.ctx)]

    def _owns_engine(self):
        return self._process is not None and self._process.poll() is None

    def start(self, wait_s=180):
        with self._lock:
            if self.is_running():
                return {"running": True, "already": True, "owned": self._owns_engine()}
            if self._owns_engine():
                return {"running": False, "starting": True, "owned": True}
            url = urlsplit(self.config.engine_url)
            if not is_loopback(url.hostname) or url.scheme != "http":
                raise EngineError("Only a local HTTP engine can be spawned; remote/TLS engines are attach-only")
            command, options = self._command([
                "serve", "--auto-tier", *self._model_args(), "--host", url.hostname,
                "--port", str(url.port or 80), "--ngen", "4096"])
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("w", encoding="utf-8") as log:
                self._process = subprocess.Popen(command, cwd=self.config.engine_root,
                    env=self._engine_env(), stdin=subprocess.DEVNULL, stdout=log, stderr=log, **options)
            process = self._process
        began = time.monotonic()
        while time.monotonic() - began < wait_s:
            if process.poll() is not None:
                return {"running": False, "error": "Engine exited during startup", "log": str(self.log_path)}
            if self.is_running():
                return {"running": True, "started": True, "owned": True,
                        "warmup_s": round(time.monotonic() - began, 1)}
            time.sleep(0.25)
        return {"running": False, "error": "Engine startup timed out; inspect logs or stop the owned process",
                "log": str(self.log_path), "owned": self._owns_engine()}

    @staticmethod
    def _stop_owned(process):
        if process is None or process.poll() is not None:
            return False
        import psutil
        # Only descendants of the live Popen we own; never discover a process by port/name.
        try:
            children = psutil.Process(process.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
        for child in reversed(children):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        _, alive = psutil.wait_procs(children, timeout=3)
        for child in alive:
            try:
                child.kill()  # psutil guards against PID reuse.
            except psutil.NoSuchProcess:
                pass
        return True

    def stop(self):
        with self._lock:
            if not self._owns_engine():
                return {"stopped": False, "owned": False, "reason": "No owned engine; attached servers are never stopped"}
            stopped = self._stop_owned(self._process)
            self._process = None
            return {"stopped": stopped, "owned": True}

    def close(self):
        self.stop()
        with self._lock:
            self._stop_owned(self._npu_process)
            self._npu_process = None
            self._stop_owned(self._stt_process)
            self._stt_process = None

    def _coli(self, args, timeout=120):
        command, options = self._command(args)
        with tempfile.TemporaryFile() as output:
            try:
                result = subprocess.run(command, cwd=self.config.engine_root, env=self._engine_env(),
                    stdin=subprocess.DEVNULL, stdout=output, stderr=output, timeout=timeout, **options)
            except subprocess.TimeoutExpired as exc:
                raise EngineError("Fennec diagnostic timed out") from exc
            output.seek(0, 2)
            output.seek(max(0, output.tell() - 8000))
            return {"exit": result.returncode, "out": self._redact(output.read().decode("utf-8", "replace"))}

    def plan(self):
        return self._coli(["plan", *self._model_args()])

    def doctor(self):
        return self._coli(["doctor"] + (self._model_args() if self.config.model else []))

    def status(self):
        return {"running": self.is_running(), "owned": self._owns_engine(), "engine_url": self.config.engine_url}

    def logs(self, n=60):
        try:
            with self.log_path.open("rb") as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 32_000))
                lines = self._redact(stream.read().decode("utf-8", "replace")).splitlines(True)
            return {"lines": lines[-max(1, min(n, 200)):], "path": str(self.log_path)}
        except OSError:
            return {"lines": [], "path": str(self.log_path)}

    def telemetry(self):
        url = urlsplit(self.config.engine_url)
        out = {"running": self.is_running(), "owned": self._owns_engine(),
               "port": url.port or (443 if url.scheme == "https" else 80),
               "model_dir": self.config.model, "backend": self.config.backend}
        if out["running"]:
            try:
                out["model"] = self.model_id()
            except EngineError:
                out["model"] = None
            for key in ("health", "metrics", "experts", "profile"):
                try:
                    out[key] = self._request("/" + key, timeout=5)
                except EngineError:
                    out[key] = {"error": "unavailable"}
        return out

    def chat(self, messages, tools=None, tool_choice="auto", max_tokens=2048,
             temperature=0.2, timeout=600, reasoning_effort=None, enable_thinking=None):
        body = {"model": self.model_id(), "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "stream": False}
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        if enable_thinking is not None:
            body["enable_thinking"] = bool(enable_thinking)
        if tools:
            body.update(tools=tools, tool_choice=tool_choice)
        try:
            data = self._request("/v1/chat/completions", body, timeout=timeout)
            return data["choices"][0]["message"], data.get("usage", {})
        except (KeyError, IndexError, TypeError) as exc:
            raise EngineError("Invalid chat response from engine") from exc

    def chat_stream(self, messages, tools=None, tool_choice="auto", max_tokens=2048,
                    temperature=0.2, timeout=600, reasoning_effort=None, enable_thinking=None,
                    abort=None):
        """Yield stream events for one completion, then a final ``result`` event.

        Events: ``text`` / ``reasoning`` deltas, ``tool_delta`` per fragment,
        ``finish`` when the engine names a stop reason, and
        ``result`` = {"message", "usage", "finish_reason"}. ``abort`` is polled
        between SSE lines, so a cancelled turn keeps the partial message.
        """
        body = {"model": self.model_id(), "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "stream": True}
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        if enable_thinking is not None:
            body["enable_thinking"] = bool(enable_thinking)
        if tools:
            body.update(tools=tools, tool_choice=tool_choice)
        request = urllib.request.Request(
            self.config.engine_url + "/v1/chat/completions",
            headers={"Content-Type": "application/json", "Accept": "text/event-stream",
                     "Authorization": f"Bearer {self.config.api_key}"},
            data=json.dumps(body).encode())
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        content, reasoning, usage, finish = "", "", {}, None
        calls: dict = {}
        cancelled = False
        try:
            with opener.open(request, timeout=timeout) as response:
                if "text/event-stream" not in (response.headers.get("content-type") or ""):
                    payload = json.loads(response.read(4_000_001))
                    choice = payload["choices"][0]
                    message, plain_usage = self._result(choice.get("message") or {}, payload)
                    yield {"type": "text", "text": message.get("content") or ""}
                    yield {"type": "result", "message": message, "usage": plain_usage,
                           "finish_reason": choice.get("finish_reason") or "stop"}
                    return
                for raw in response:
                    if abort is not None and abort():
                        cancelled = True
                        break
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    fragment = line[5:].strip()
                    if fragment == "[DONE]":
                        break
                    try:
                        chunk = json.loads(fragment)
                    except ValueError:
                        continue
                    if isinstance(chunk.get("usage"), dict):
                        usage = chunk["usage"]
                    error = chunk.get("error")
                    if isinstance(error, dict):
                        raise EngineError(str(error.get("message") or "engine stream error"))
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        piece = delta.get("content")
                        if piece:
                            content += piece
                            yield {"type": "text", "text": piece}
                        thought = delta.get("reasoning_content") or delta.get("reasoning")
                        if thought:
                            reasoning += thought
                            yield {"type": "reasoning", "text": thought}
                        for item in delta.get("tool_calls") or []:
                            slot = calls.setdefault(item.get("index", len(calls)), {
                                "id": "", "type": "function",
                                "function": {"name": "", "arguments": ""}})
                            if item.get("id"):
                                slot["id"] = item["id"]
                            spec = item.get("function") or {}
                            if spec.get("name"):
                                slot["function"]["name"] += spec["name"]
                            if spec.get("arguments"):
                                slot["function"]["arguments"] += spec["arguments"]
                                yield {"type": "tool_delta", "call": slot,
                                       "delta": spec["arguments"]}
                        if choice.get("finish_reason"):
                            finish = choice["finish_reason"]
                            yield {"type": "finish", "finish_reason": finish}
        except EngineError:
            raise
        except Exception as exc:
            raise EngineError(self._redact("Engine stream failed; check engine health") +
                              f": {type(exc).__name__}") from exc
        if cancelled:
            finish = "cancelled"
        message, usage = self._result({"content": content, "reasoning_content": reasoning,
                                       "tool_calls": [calls[key] for key in sorted(calls)]},
                                      {"usage": usage})
        yield {"type": "result", "message": message, "usage": usage, "finish_reason": finish,
               "cancelled": cancelled}

    @staticmethod
    def _result(message, payload):
        out = {"role": "assistant", "content": message.get("content") or ""}
        thought = message.get("reasoning_content")
        if isinstance(thought, str) and thought:
            out["reasoning_content"] = thought
        calls = [call for call in (message.get("tool_calls") or [])
                 if isinstance(call, dict) and (call.get("function") or {}).get("name")]
        if calls:
            out["tool_calls"] = calls
        usage = payload.get("usage") or {}
        return out, usage

    def npu_health(self, timeout=2):
        try:
            return self._request("/health", timeout=timeout, npu=True)
        except EngineError:
            return {"status": "down"}

    def ensure_npu(self, wait_s=60, *, enabled=False):
        if not enabled:
            return {"running": False, "error": "NPU startup requires explicit enablement"}
        with self._lock:
            health = self.npu_health()
            if health.get("status") == "ok":
                return {"running": True, "already": True, **health}
            if self._npu_process is not None and self._npu_process.poll() is None:
                return {"starting": True}
            script = self.config.engine_root / "scripts" / "npu_embed_server.py"
            if not script.is_file():
                raise EngineError("Optional NPU sidecar is not installed")
            python, options = spawn_options(self.config.python)
            self.npu_log_path.parent.mkdir(parents=True, exist_ok=True)
            env = self._engine_env()
            env.pop("COLI_API_KEY", None)
            with self.npu_log_path.open("w", encoding="utf-8") as log:
                self._npu_process = subprocess.Popen(
                    [python, str(script), "--port", "8182", "--device", "NPU"],
                    cwd=self.config.engine_root, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=log, **options)
            process = self._npu_process
        began = time.monotonic()
        while time.monotonic() - began < wait_s:
            health = self.npu_health()
            if health.get("status") == "ok":
                return {"running": True, "started": True, **health}
            if process.poll() is not None:
                break
            time.sleep(0.25)
        return {"running": False, "error": "NPU sidecar did not become healthy"}

    def embed(self, texts, timeout=120):
        if isinstance(texts, str):
            texts = [texts]
        data = self._request("/v1/embeddings", {"input": texts}, timeout=timeout, npu=True)
        return [row["embedding"] for row in data.get("data", [])]

    # ---- speech-to-text sidecar (faster-whisper, on demand) ----

    def stt_health(self, timeout=1):
        try:
            return self._request_path(self.stt_url, "/health", timeout=timeout)
        except EngineError:
            return {"status": "down"}

    def _request_path(self, base, path, body=None, timeout=4):
        request = urllib.request.Request(base + path, headers={"Content-Type": "application/json"},
                                         data=None if body is None else json.dumps(body).encode())
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        try:
            with opener.open(request, timeout=timeout) as response:
                return json.loads(response.read(1_000_001))
        except Exception as exc:
            raise EngineError("Speech sidecar request failed") from exc

    def ensure_stt(self, wait_s=300):
        with self._lock:
            health = self.stt_health()
            if health.get("status") == "ok":
                return {"running": True, "already": True, **health}
            if self._stt_process is not None and self._stt_process.poll() is None:
                process = self._stt_process
            else:
                script = Path(__file__).with_name("stt.py")
                if not script.is_file():
                    raise EngineError("Speech sidecar is not installed")
                python, options = spawn_options(self.config.python)
                self.stt_log_path.parent.mkdir(parents=True, exist_ok=True)
                env = self._engine_env()  # carries HF_HOME so the model lands in cache_root
                env.pop("COLI_API_KEY", None)
                with self.stt_log_path.open("w", encoding="utf-8") as log:
                    self._stt_process = subprocess.Popen(
                        [python, str(script), "--port", "8183"],
                        cwd=self.config.engine_root, env=env, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=log, **options)
                process = self._stt_process
        began = time.monotonic()
        while time.monotonic() - began < wait_s:
            health = self.stt_health()
            if health.get("status") == "ok":
                return {"running": True, "started": True, **health}
            if process.poll() is not None:
                break
            time.sleep(0.5)
        return {"running": False, "error": "Speech model did not become ready; inspect the sidecar log",
                "log": str(self.stt_log_path)}

    def transcribe(self, audio, content_type="audio/webm", timeout=180):
        request = urllib.request.Request(
            self.stt_url + "/transcribe", data=audio, method="POST",
            headers={"Content-Type": content_type or "application/octet-stream"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        try:
            with opener.open(request, timeout=timeout) as response:
                data = json.loads(response.read(1_000_001))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = str(json.loads(exc.read(1_000_001)).get("error") or "")
            except Exception:
                detail = ""
            message = f"Speech transcription failed: {detail}" if detail else \
                "Speech transcription failed; check the sidecar log"
            raise EngineError(message) from exc
        except Exception as exc:
            raise EngineError("Speech transcription failed; check the sidecar log") from exc
        if not isinstance(data.get("text"), str):
            raise EngineError("Speech sidecar returned no text")
        return data

    def hardware(self):
        return hardware(self)


_default = None
_default_lock = threading.Lock()


def get_engine():
    global _default
    with _default_lock:
        if _default is None:
            _default = Engine(load_config())
        return _default


def __getattr__(name):
    if name in {"is_running", "model_id", "start", "stop", "telemetry", "plan", "doctor",
                "status", "logs", "chat", "chat_stream", "npu_health", "ensure_npu", "embed"}:
        return getattr(get_engine(), name)
    raise AttributeError(name)


def _run(cmd, timeout=10):
    try:
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **options)
        return (p.stdout or "")[:16_000].strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def hardware(client=None):
    import psutil
    client = client or get_engine()
    vm = psutil.virtual_memory()
    disk_path = client.config.workspace
    while not disk_path.exists() and disk_path != disk_path.parent:
        disk_path = disk_path.parent
    disk = shutil.disk_usage(disk_path)
    out = {"platform": platform.system(), "cpu": {"name": platform.processor(), "cores": psutil.cpu_count()},
           "ram": {"total_gb": round(vm.total / 2**30, 1), "used_gb": round(vm.used / 2**30, 1),
                   "available_gb": round(vm.available / 2**30, 1)},
           "disk": {"path": str(disk_path), "total_gb": round(disk.total / 2**30, 1),
                    "free_gb": round(disk.free / 2**30, 1)},
           "npu": client.npu_health(), "engine_running": client.is_running(), "gpu": []}
    smi = _run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits"])
    for line in smi.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4:
            try:
                out["gpu"].append({"name": parts[0], "mem_used_mb": int(parts[1]),
                                   "mem_total_mb": int(parts[2]), "util_pct": int(parts[3])})
            except ValueError:
                continue
    out["gpu_adapters"] = list(out["gpu"])
    if os.name == "nt":
        names = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                      "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }"])
        out["gpu_adapters"] = [{"name": name.strip()} for name in names.splitlines() if name.strip()]
    out["tier"] = [line.strip() for line in client.logs(200)["lines"] if "[VK] tier" in line][-3:]
    return out