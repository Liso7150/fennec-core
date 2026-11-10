"""Local Fennec service: engine proxy, supervised agent runs, workspace files, PWA host.

Security model: loopback HTTP by default. API calls other than health require the
`X-Fennec-Client: 1` header, a loopback Host, and a same-origin Origin when one is
sent (DNS-rebinding/CSRF guard). A non-loopback bind is refused unless an app token
and TLS are configured. File access is jailed to the configured workspace.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from . import loop
from .config import Config, is_loopback, load_config
from .engine import Engine, EngineError, get_engine
from .tools import ToolContext, list_dir, read_file

CLIENT_HEADER = "X-Fennec-Client"
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


MAX_TEXT_PART = 200_000
MAX_MESSAGE_BYTES = 12_000_000
MAX_AUDIO_BYTES = 16 << 20


class ChatMessage(BaseModel):
    """OpenAI-style message: plain text, or content parts (text + one image)."""
    role: str = Field(pattern="^(system|user|assistant|tool)$")
    content: str | list[dict]


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=200)
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    temperature: float = Field(default=0.2, ge=0, le=2)
    reasoning_effort: str | None = Field(
        default=None, pattern="^(none|minimal|low|medium|high|xhigh)$")
    enable_thinking: bool | None = None

    @model_validator(mode="after")
    def _check_content(self):
        images = 0
        total = 0
        for index, message in enumerate(self.messages):
            content = message.content
            if isinstance(content, str):
                if len(content) > MAX_TEXT_PART:
                    raise ValueError(f"messages.{index}.content is too long")
                total += len(content)
                continue
            if len(content) > 20:
                raise ValueError(f"messages.{index}.content has too many parts")
            for position, part in enumerate(content):
                where = f"messages.{index}.content.{position}"
                kind = part.get("type") if isinstance(part, dict) else None
                if kind in ("text", "input_text"):
                    text = part.get("text")
                    if not isinstance(text, str):
                        raise ValueError(f"{where}.text must be a string")
                    if len(text) > MAX_TEXT_PART:
                        raise ValueError(f"{where}.text is too long")
                    total += len(text)
                elif kind in ("image_url", "input_image"):
                    url = part.get("image_url") if kind == "image_url" else part.get("url")
                    if isinstance(url, dict):
                        url = url.get("url")
                    if (not isinstance(url, str) or not url.startswith("data:image/")
                            or ";base64," not in url):
                        raise ValueError(f"{where}: images must be base64 data: URLs")
                    total += len(url)
                    images += 1
                else:
                    raise ValueError(f"{where}: unsupported content part {kind!r}")
        if images > 1:
            raise ValueError("one image per request: the engine holds a single pending image")
        if total > MAX_MESSAGE_BYTES:
            raise ValueError("the request is too large")
        return self


class AgentRequest(BaseModel):
    task: str = Field(min_length=1, max_length=8000)
    workspace: str | None = None
    allow_write: bool = False
    allow_shell: bool = False
    max_steps: int = Field(default=14, ge=1, le=40)


class CancelRequest(BaseModel):
    id: str = Field(min_length=1, max_length=64)


class EmbedRequest(BaseModel):
    input: str | list[str]


def _host_name(value: str) -> str:
    value = (value or "").strip().lower()
    if value.startswith("["):
        return value[1:value.find("]")] if "]" in value else value
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


def _norm(path) -> str:
    return os.path.normcase(os.path.realpath(str(path)))


def workspace_within(root: Path, requested: str | None) -> Path:
    base = Path(_norm(root))
    candidate = Path(_norm(Path(requested).expanduser())) if requested else base
    if not candidate.is_dir():
        raise HTTPException(400, "workspace must be an existing directory")
    try:
        inside = os.path.commonpath([str(base), str(candidate)]) == str(base)
    except ValueError:
        inside = False
    if not inside:
        raise HTTPException(403, "workspace must be inside the configured workspace root")
    return candidate


def _listing(ctx: ToolContext, path: str):
    try:
        result = list_dir(ctx, path)
    except (PermissionError, ValueError, OSError) as exc:
        raise HTTPException(403, str(exc)) from exc
    if not result.get("ok"):
        raise HTTPException(404, result.get("error") or "not a directory")
    base = "" if result.get("path") in (None, ".") else str(result["path"]).replace("\\", "/")
    entries = []
    for entry in result.get("entries", []):
        name = entry.get("name", "")
        is_dir = name.endswith("/")
        name = name.rstrip("/")
        if not name:
            continue
        entries.append({"name": name, "path": f"{base}/{name}" if base else name,
                        "is_dir": is_dir, "size": entry.get("size")})
    return {"path": base, "entries": entries}


def make_app(config: Config, engine_client=None, allowed_hosts=None) -> FastAPI:
    client = engine_client or get_engine()
    hosts = set(LOOPBACK_HOSTS) | {h.strip().lower() for h in
                                   os.environ.get("COLI_ALLOWED_HOSTS", "").split(",") if h.strip()}
    if allowed_hosts:
        hosts |= {h.strip().lower() for h in allowed_hosts if h.strip()}

    @asynccontextmanager
    async def lifespan(_app):
        yield
        # Stop an engine this service spawned; attached engines are never stopped.
        close = getattr(client, "close", None)
        if close is not None:
            close()

    app = FastAPI(title="Fennec local service", docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=lifespan)

    @app.middleware("http")
    async def guard(request, call_next):
        name = _host_name(request.headers.get("host", ""))
        if name not in hosts:
            return PlainTextResponse("Host not allowed", status_code=400)
        origin = request.headers.get("origin")
        if origin and origin not in {f"http://{request.headers.get('host')}",
                                     f"https://{request.headers.get('host')}"}:
            return PlainTextResponse("Cross-origin request refused", status_code=403)
        path = request.url.path
        if path.startswith("/api/") and path != "/api/health":
            if request.headers.get(CLIENT_HEADER) != "1":
                return JSONResponse({"detail": "Missing X-Fennec-Client header"}, status_code=403)
            token = os.environ.get("FENNEC_APP_TOKEN", "")
            if token:
                auth = request.headers.get("authorization", "")
                supplied = auth[7:] if auth.lower().startswith("bearer ") else ""
                if not hmac.compare_digest(supplied, token):
                    return JSONResponse({"detail": "Invalid or missing app token"}, status_code=401)
        response = await call_next(request)
        if path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def engine_or_503():
        try:
            if not client.is_running():
                raise HTTPException(503, "Colibri engine is not running")
        except EngineError as exc:
            raise HTTPException(503, str(exc)) from exc
        return client

    @app.get("/api/health")
    def health():
        try:
            running = client.is_running()
        except EngineError:
            running = False
        return {"app": "Fennec", "engine_running": running, "workspace": str(config.workspace)}

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        engine_or_503()
        try:
            message, usage = client.chat([m.model_dump() for m in req.messages],
                                         max_tokens=req.max_tokens, temperature=req.temperature,
                                         reasoning_effort=req.reasoning_effort,
                                         enable_thinking=req.enable_thinking)
        except EngineError as exc:
            raise HTTPException(503, str(exc)) from exc
        return {"message": message, "usage": usage}

    @app.post("/api/stt")
    async def stt(request: Request):
        """Local speech-to-text: raw audio body -> {text}. The sidecar starts on demand."""
        body = await request.body()
        if not body:
            raise HTTPException(400, "send raw audio bytes")
        if len(body) > MAX_AUDIO_BYTES:
            raise HTTPException(413, "audio must be at most 16 MiB")
        try:
            ready = client.ensure_stt()
        except EngineError as exc:
            raise HTTPException(503, str(exc)) from exc
        if not ready.get("running"):
            raise HTTPException(503, ready.get("error") or "speech sidecar did not start")
        try:
            return client.transcribe(body, request.headers.get("content-type", "audio/webm"))
        except EngineError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/api/engine")
    def engine_telemetry():
        return client.telemetry()

    @app.get("/api/engine/status")
    def engine_status():
        return client.status()

    @app.get("/api/engine/logs")
    def engine_logs():
        return client.logs(80)

    @app.get("/api/engine/plan")
    def engine_plan():
        return client.plan()

    @app.get("/api/engine/doctor")
    def engine_doctor():
        return client.doctor()

    @app.post("/api/engine/start")
    def engine_start():
        try:
            if client.is_running():
                return {"running": True, "already": True}
        except EngineError:
            pass
        import threading
        threading.Thread(target=client.start, kwargs={"wait_s": 240}, daemon=True).start()
        return {"starting": True}

    @app.post("/api/engine/stop")
    def engine_stop():
        return client.stop()

    @app.get("/api/hardware")
    def hardware():
        return client.hardware()

    @app.post("/api/npu/start")
    def npu_start():
        try:
            return client.ensure_npu(enabled=True)
        except EngineError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.post("/api/npu/embed")
    def npu_embed(req: EmbedRequest):
        try:
            return {"device": client.npu_health(timeout=1).get("device", "?"),
                    "embeddings": client.embed(req.input)}
        except EngineError as exc:
            raise HTTPException(503, f"NPU embedding service unavailable: {exc}") from exc

    @app.post("/api/agent/start")
    def agent_start(req: AgentRequest):
        engine_or_503()
        workspace = workspace_within(config.workspace, req.workspace)
        sid = loop.new_session(req.task, str(workspace), allow_write=req.allow_write,
                               allow_shell=req.allow_shell, engine_client=client)
        loop.start_thread(sid, req.max_steps)
        return {"id": sid, "workspace": str(workspace)}

    @app.get("/api/agent/poll")
    def agent_poll(id: str, after: int = 0):
        state = loop.get_steps(id, max(0, after))
        if state is None:
            raise HTTPException(404, "unknown session")
        return state

    @app.post("/api/agent/cancel")
    def agent_cancel(req: CancelRequest):
        if req.id not in loop.SESSIONS:
            raise HTTPException(404, "unknown session")
        cancelled = loop.cancel(req.id)
        state = loop.get_steps(req.id, 0)
        return {"cancelled": cancelled, "status": state["status"] if state else "unknown"}

    @app.get("/api/agent/sessions")
    def agent_sessions():
        return {"sessions": [{"id": sid, "task": s["task"], "status": s["status"],
                              "steps": len(s["steps"])} for sid, s in list(loop.SESSIONS.items())]}

    @app.get("/api/files")
    def files(path: str = "."):
        return _listing(ToolContext(workspace=str(config.workspace)), path)

    @app.get("/api/file")
    def file_read(path: str):
        try:
            result = read_file(ToolContext(workspace=str(config.workspace)), path, numbered=False)
        except (PermissionError, ValueError, OSError) as exc:
            raise HTTPException(403, str(exc)) from exc
        if not result.get("ok"):
            raise HTTPException(404, result.get("error") or "not a file")
        return {"path": result["path"], "lines": result["lines"], "content": result["content"]}

    web_root = Path(config.web_root)
    if web_root.is_dir() and (web_root / "index.html").is_file():
        app.mount("/", StaticFiles(directory=str(web_root), html=True), name="web")
    else:
        @app.get("/")
        def no_web():
            return {"app": "Fennec", "error": "web_root does not contain index.html",
                    "web_root": str(web_root)}

    return app


def _http(method: str, url: str, body=None, timeout=30):
    headers = {CLIENT_HEADER: "1", "Content-Type": "application/json"}
    token = os.environ.get("FENNEC_APP_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, method=method, headers=headers,
                                     data=None if body is None else json.dumps(body).encode())
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read(4_000_001))


def serve(host="127.0.0.1", port=8787, *, web_root=None, tls_cert=None, tls_key=None):
    tls_cert = tls_cert or os.environ.get("FENNEC_TLS_CERTFILE")
    tls_key = tls_key or os.environ.get("FENNEC_TLS_KEYFILE")
    if not is_loopback(host):
        if not os.environ.get("FENNEC_APP_TOKEN") or not tls_cert or not tls_key:
            raise SystemExit("Refusing a non-loopback bind without FENNEC_APP_TOKEN, "
                             "FENNEC_TLS_CERTFILE and FENNEC_TLS_KEYFILE")
    config = load_config()
    if web_root:
        config = replace(config, web_root=Path(web_root).expanduser().resolve())
    import uvicorn
    uvicorn.run(make_app(config, allowed_hosts=[host]), host=host, port=port,
                ssl_certfile=tls_cert or None, ssl_keyfile=tls_key or None, log_level="info")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="fennec-agent",
        description="Local Fennec service and CLI. Engine base idea: Colibri "
                    "(https://github.com/JustVugg/colibri); fine-tuning: AirLLM "
                    "(https://github.com/lyogavin/airllm).")
    sub = parser.add_subparsers(dest="command", required=True)
    p_serve = sub.add_parser("serve", help="run the local service (default http://127.0.0.1:8787)")
    p_serve.add_argument("--host", default=os.environ.get("FENNEC_BIND", "127.0.0.1"))
    p_serve.add_argument("--port", type=int, default=int(os.environ.get("FENNEC_PORT", "8787")))
    p_serve.add_argument("--web-root", default=None)
    p_status = sub.add_parser("status", help="query a running service")
    p_status.add_argument("--url", default=os.environ.get("FENNEC_SERVICE_URL",
                                                          "http://127.0.0.1:8787"))
    p_chat = sub.add_parser("chat", help="one-shot chat straight to the engine (no service needed)")
    p_chat.add_argument("prompt")
    p_chat.add_argument("--max-tokens", type=int, default=512)
    p_chat.add_argument("--temperature", type=float, default=0.2)
    p_agent = sub.add_parser("agent", help="run a supervised task against a running service")
    p_agent.add_argument("task")
    p_agent.add_argument("--workspace", default=None)
    p_agent.add_argument("--allow-write", action="store_true")
    p_agent.add_argument("--allow-shell", action="store_true")
    p_agent.add_argument("--max-steps", type=int, default=14)
    p_agent.add_argument("--url", default=os.environ.get("FENNEC_SERVICE_URL",
                                                         "http://127.0.0.1:8787"))
    args = parser.parse_args(argv)

    if args.command == "serve":
        serve(args.host, args.port, web_root=args.web_root)
        return 0
    if args.command == "status":
        print(json.dumps(_http("GET", args.url.rstrip("/") + "/api/health")))
        return 0
    if args.command == "chat":
        engine = Engine(load_config())
        try:
            message, usage = engine.chat([{"role": "user", "content": args.prompt}],
                                         max_tokens=args.max_tokens, temperature=args.temperature)
        except EngineError as exc:
            print(f"fennec-agent: {exc}", file=sys.stderr)
            return 2
        print(message.get("content", ""))
        if usage:
            print(json.dumps({"usage": usage}), file=sys.stderr)
        return 0
    if args.command == "agent":
        base = args.url.rstrip("/")
        try:
            started = _http("POST", base + "/api/agent/start", {
                "task": args.task, "workspace": args.workspace,
                "allow_write": args.allow_write, "allow_shell": args.allow_shell,
                "max_steps": args.max_steps})
        except Exception as exc:
            print(f"fennec-agent: cannot start task: {exc}", file=sys.stderr)
            return 2
        sid, after = started["id"], 0
        import time
        while True:
            state = _http("GET", f"{base}/api/agent/poll?id={sid}&after={after}")
            for step in state["steps"]:
                kind = step.get("type")
                if kind == "tool":
                    print(f"[tool] {step.get('name')} -> {'ok' if step.get('ok') else 'failed'}")
                elif kind in ("assistant", "final", "error", "cancelled"):
                    print(f"[{kind}] {step.get('content', '')}")
            after = state["total"]
            if state["status"] != "running":
                return 0 if state["status"] == "done" else 1
            time.sleep(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
