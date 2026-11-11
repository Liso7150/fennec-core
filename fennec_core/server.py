"""Local Fennec service: engine proxy, supervised agent runs, workspace files, PWA host.

Security model: loopback HTTP by default. API calls other than health require the
`X-Fennec-Client: 1` header, a loopback Host, and a same-origin Origin when one is
sent (DNS-rebinding/CSRF guard). A non-loopback bind is refused unless an app token
and TLS are configured. File access is jailed to the configured workspace.
"""
from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import os
import sys
import time
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from . import loop
from .config import Config, is_loopback, load_config
from .engine import Engine, EngineError, get_engine
from .outlets import Outlets
from .tools import ToolContext, list_dir, read_file

CLIENT_HEADER = "X-Fennec-Client"
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


MAX_TEXT_PART = 200_000
MAX_MESSAGE_BYTES = 12_000_000
MAX_AUDIO_BYTES = 16 << 20
AGUI_TIMEOUT_S = int(os.environ.get("FENNEC_AGUI_TIMEOUT", "900"))
AGUI_POLL_S = 0.2


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
    allow_web: bool = False
    max_steps: int = Field(default=14, ge=1, le=40)


class CancelRequest(BaseModel):
    id: str = Field(min_length=1, max_length=64)


class EmbedRequest(BaseModel):
    input: str | list[str]


class OutletRequest(BaseModel):
    """One task for a local web-agent outlet, from any front-end but this one."""
    outlet: str = Field(min_length=1, max_length=20)
    goal: str = Field(min_length=1, max_length=4000)
    url: str = Field(default="", max_length=2000)
    agent: str = Field(default="", max_length=120)
    max_steps: int = Field(default=12, ge=1, le=40)


def _host_name(value: str) -> str:
    value = (value or "").strip().lower()
    if value.startswith("["):
        return value[1:value.find("]")] if "]" in value else value
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


def _norm(path) -> str:
    return os.path.normcase(os.path.realpath(str(path)))


def workspace_within(root: Path, requested) -> Path:
    base = Path(_norm(root))
    if requested is not None and not isinstance(requested, str):
        raise HTTPException(422, "workspace must be a path string")
    try:
        candidate = Path(_norm(Path(requested).expanduser())) if requested else base
    except (OSError, ValueError):
        raise HTTPException(400, "workspace is not a usable path") from None
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


class RunAgentInput(BaseModel):
    """The AG-UI run body that CopilotKit-style clients post.

    Only the last user message drives Fennec; everything before it is the
    client's own transcript. AG-UI has no field for permissions, so the
    Fennec-specific knobs (workspace, allow_write, allow_shell, allow_web,
    max_steps) travel in forwardedProps.
    """
    thread_id: str = Field(default="", alias="threadId", max_length=128)
    run_id: str = Field(default="", alias="runId", max_length=128)
    state: dict = Field(default_factory=dict)
    messages: list[dict] = Field(default_factory=list, max_length=400)
    tools: list = Field(default_factory=list)
    context: list = Field(default_factory=list)
    forwarded_props: dict = Field(default_factory=dict, alias="forwardedProps")

    model_config = {"populate_by_name": True}


AGUI_RESULT_CHARS = 8000
AGUI_UNFINISHED = {"ok": False, "error": "the run ended before this tool answered"}


def _sse(event: dict) -> str:
    return "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"


def _json_clip(value) -> str:
    """A tool result as JSON, and never with the cut running through the JSON.

    CopilotKit parses this field, so a slice that lands mid-object is a client-side
    crash rather than a long result.
    """
    text = json.dumps(value, ensure_ascii=False)
    if len(text) <= AGUI_RESULT_CHARS:
        return text
    return json.dumps({"truncated": True, "excerpt": text[:AGUI_RESULT_CHARS - 200]},
                      ensure_ascii=False)


class AgUiStream:
    """Translates the loop's step log into AG-UI events, oldest step first."""

    def __init__(self, thread_id: str, run_id: str):
        self.thread_id = thread_id
        self.run_id = run_id
        self.after = 0
        self._counter = 0
        self._text = ""
        self._open: list = []

    def _id(self, prefix):
        self._counter += 1
        return f"{prefix}-{self.run_id}-{self._counter}"

    def _result(self, call_id, payload):
        return {"type": "TOOL_CALL_RESULT", "messageId": self._id("result"),
                "toolCallId": call_id, "role": "tool",
                "content": json.dumps(payload, ensure_ascii=False)}

    def close_tools(self):
        """Every AG-UI tool call has to be answered, including a hung one.

        A TOOL_CALL_END with no TOOL_CALL_RESULT leaves the client spinning on a
        card that never resolves, and it never learns the run is over.
        """
        pending, self._open = self._open, []
        return [self._result(call_id, AGUI_UNFINISHED) for call_id in pending]

    def _message(self, text):
        if not text:
            return []
        message_id = self._id("msg")
        return [{"type": "TEXT_MESSAGE_START", "messageId": message_id, "role": "assistant"},
                {"type": "TEXT_MESSAGE_CONTENT", "messageId": message_id, "delta": text},
                {"type": "TEXT_MESSAGE_END", "messageId": message_id}]

    def start(self):
        return [{"type": "RUN_STARTED", "threadId": self.thread_id, "runId": self.run_id}]

    def finish(self, state):
        events = self.close_tools()
        stop = state.get("stop")
        if not isinstance(stop, str):
            # cancel() marks the session cancelled before the worker can name a
            # reason, so the status is all that is left to report honestly.
            stop = "cancelled" if state.get("status") == "cancelled" else "end_turn"
        return events + [{"type": "RUN_FINISHED",
                          "result": {"sessionId": state["id"], "stopReason": stop,
                                     "usage": state.get("usage") or {}}}]

    def feed(self, step) -> list:
        kind = step.get("type")
        if kind == "assistant":
            events = []
            reasoning = step.get("reasoning") or ""
            if reasoning.strip():
                message_id = self._id("reasoning")
                events += [{"type": "REASONING_START"},
                           {"type": "REASONING_MESSAGE_START", "messageId": message_id,
                            "role": "assistant"},
                           {"type": "REASONING_MESSAGE_CONTENT", "messageId": message_id,
                            "delta": reasoning},
                           {"type": "REASONING_MESSAGE_END", "messageId": message_id},
                           {"type": "REASONING_END"}]
            content = step.get("content") or ""
            self._text = content
            events += self._message(content)
            for call in step.get("tool_calls") or []:
                call_id = str(call.get("call_id") or self._id("call"))
                self._open.append(call_id)
                events += [{"type": "TOOL_CALL_START", "toolCallId": call_id,
                            "toolCallName": call.get("name") or "tool"},
                           {"type": "TOOL_CALL_ARGS", "toolCallId": call_id,
                            "delta": json.dumps(call.get("args") or {}, ensure_ascii=False)},
                           {"type": "TOOL_CALL_END", "toolCallId": call_id}]
            return events
        if kind == "tool":
            call_id = step.get("call_id") or f"call-{step.get('i', 0)}"
            self._open = [open_id for open_id in self._open if open_id != call_id]
            return [{"type": "TOOL_CALL_RESULT", "messageId": self._id("result"),
                     "toolCallId": call_id, "role": "tool",
                     "content": _json_clip(step.get("result"))}]
        if kind in ("final", "partial", "cancelled"):
            content = step.get("content") or ""
            if kind == "final" and content == self._text:
                return []          # the loop emits the answer twice; the client sees it once
            return self._message(content)
        if kind == "error":
            return [{"type": "RUN_ERROR", "message": str(step.get("content") or "agent failed")}]
        return []


def _run_agent_input_goal(body: RunAgentInput) -> str:
    for message in reversed(body.messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()[:8000]
        if isinstance(content, list):
            parts = [part.get("text", "") for part in content
                     if isinstance(part, dict) and part.get("type") == "text"]
            text = "\n".join(part for part in parts if part).strip()
            if text:
                return text[:8000]
    raise HTTPException(400, "RunAgentInput carries no user message to run")


def _ag_ui_max_steps(value) -> int:
    """forwardedProps.max_steps, defaulted and bounded exactly like AgentRequest.

    forwardedProps is free-form JSON, so a word, a fraction or a five-digit number
    are all things a client can send. Refusing them here is the difference between
    a 422 the caller reads and a 500 inside an event stream nobody can parse.
    """
    if value is None:
        return 14
    try:
        steps = int(str(value).strip())
    except (TypeError, ValueError):
        raise HTTPException(422, "forwardedProps.max_steps must be a whole number of steps")
    if not 1 <= steps <= 40:
        raise HTTPException(422, "forwardedProps.max_steps must be between 1 and 40")
    return steps


def make_app(config: Config, engine_client=None, allowed_hosts=None) -> FastAPI:
    client = engine_client or get_engine()
    agent_outlets = Outlets(config)
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
                               allow_shell=req.allow_shell, allow_web=req.allow_web,
                               engine_client=client, outlets=agent_outlets)
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

    @app.get("/api/outlets")
    def outlet_status(refresh: bool = False):
        """What the optional web-agent outlets are, as this machine sees them."""
        return agent_outlets.status(refresh=refresh)

    @app.post("/api/outlets/run")
    def outlet_run(req: OutletRequest):
        """Run one task on the local browser outlet installed on this machine.

        Fennec Desktop drives Jev / OpenBot through here rather than keeping a
        second implementation in JavaScript: the outlet contract lives in
        outlets.py once. Approval belongs to the caller's UI; this route only
        reaches loopback outlets the user already installed, and is behind the
        same client-header, Host and app-token guard as /api/agent/start.
        """
        tool = {"jev": "web_task", "openbot": "coworker_task"}.get(req.outlet)
        if tool is None:
            raise HTTPException(422, "outlet must be 'jev' or 'openbot'")
        if not agent_outlets.available(tool):
            raise HTTPException(409, f"{req.outlet} is not available on this machine; "
                                     "run `fennec outlets` to see what is missing")
        if tool == "web_task":
            return agent_outlets.web_task(req.goal, req.url, max_steps=req.max_steps)
        return agent_outlets.coworker_task(req.goal, req.agent)

    @app.post("/api/ag-ui")
    async def ag_ui_run(req: RunAgentInput):
        """Fennec as an AG-UI server: one RunAgentInput in, one event stream out.

        The run is a normal supervised agent task; the stream is the AG-UI event
        names CopilotKit clients already understand. Permissions stay with the
        client header, Host and app-token guard, so this route is no more open
        than /api/agent/start. Hanging up ends the run: the open stream is the only
        evidence that the caller still cares, and a browser left clicking through a
        page nobody is watching spends money on no one's behalf.
        """
        goal = _run_agent_input_goal(req)
        props = req.forwarded_props if isinstance(req.forwarded_props, dict) else {}
        engine_or_503()
        workspace = workspace_within(config.workspace, props.get("workspace"))
        max_steps = _ag_ui_max_steps(props.get("max_steps"))
        sid = loop.new_session(goal, str(workspace),
                               allow_write=bool(props.get("allow_write", False)),
                               allow_shell=bool(props.get("allow_shell", False)),
                               allow_web=bool(props.get("allow_web", False)),
                               engine_client=client, outlets=agent_outlets)
        loop.start_thread(sid, max_steps)
        thread_id = req.thread_id or f"agui-{sid}"
        run_id = req.run_id or f"run-{sid}"
        translator = AgUiStream(thread_id, run_id)

        async def events():
            for event in translator.start():
                yield _sse(event)
            began = time.monotonic()
            terminal = False
            try:
                while True:
                    state = loop.get_steps(sid, translator.after)
                    if state is None:
                        terminal = True
                        for event in translator.close_tools():
                            yield _sse(event)
                        yield _sse({"type": "RUN_ERROR", "message": "the session is gone"})
                        return
                    translator.after = state["total"]
                    for step in state["steps"]:
                        for event in translator.feed(step):
                            yield _sse(event)
                    if state["status"] != "running":
                        terminal = True
                        # An error step already carried the terminal RUN_ERROR; the
                        # only thing owed on that path is the tool card it left open.
                        remaining = (translator.close_tools() if state["status"] == "error"
                                     else translator.finish(state))
                        for event in remaining:
                            yield _sse(event)
                        return
                    if time.monotonic() - began > AGUI_TIMEOUT_S:
                        terminal = True
                        loop.cancel(sid)
                        for event in translator.close_tools():
                            yield _sse(event)
                        yield _sse({"type": "RUN_ERROR", "message": f"run exceeded {AGUI_TIMEOUT_S}s"})
                        return
                    # Awaiting, not sleeping: this generator runs on the service's
                    # event loop, and a sleeping one stalls every other session.
                    await asyncio.sleep(AGUI_POLL_S)
            finally:
                if not terminal:
                    loop.cancel(sid)

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"cache-control": "no-store",
                                          "x-accel-buffering": "no"})

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
    p_agent.add_argument("--allow-web", action="store_true",
                         help="let the task use the optional Jev/OpenBot web outlets")
    p_agent.add_argument("--max-steps", type=int, default=14)
    p_agent.add_argument("--url", default=os.environ.get("FENNEC_SERVICE_URL",
                                                         "http://127.0.0.1:8787"))
    sub.add_parser("outlets", help="show the optional web-agent outlets on this machine")
    p_acp = sub.add_parser("acp", help="run the ACP v1 agent on stdio, for editors and IDEs")
    p_acp.add_argument("--check", action="store_true",
                       help="print the advertised protocol capabilities and exit")
    args = parser.parse_args(argv)

    if args.command == "serve":
        serve(args.host, args.port, web_root=args.web_root)
        return 0
    if args.command == "acp":
        from . import acp as acp_agent
        return acp_agent.main(["--check"] if args.check else [])
    if args.command == "outlets":
        print(json.dumps(Outlets(load_config()).status(), indent=2, ensure_ascii=False))
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
                "allow_web": args.allow_web, "max_steps": args.max_steps})
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
