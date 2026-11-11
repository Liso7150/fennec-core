"""ACP v1 over stdio: Fennec as an agent inside someone else's editor.

One newline-delimited JSON-RPC 2.0 message per line on stdin and stdout is the
only transport ACP v1 defines, and this module is the whole agent side of it:
capability negotiation, the session lifecycle (new / load / resume / list /
close), modes, streamed message and thought chunks, tool-call reporting, plans,
usage, per-call permission requests and cancellation that answers with
``stopReason: "cancelled"`` rather than an error.

The turn itself is not re-implemented here. ``fennec_core.loop`` runs it and
this file translates the events it already produces into ``session/update``
notifications, so an editor, the desktop app and the PWA all drive one agent.
Jev and OpenBot arrive as ordinary tools through ``fennec_core.outlets``; an
MCP server the client hands us arrives as ordinary tools through
``fennec_core.mcp``.

Transcripts live in ``~/.fennec/sessions/acp-<id>.json`` (override with
``FENNEC_ACP_HOME``) so ``session/load`` can replay them in another process.
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import uuid

from . import engine, loop
from . import mcp as mcp_bridge
from . import outlets as outlets_bridge
from .context import limits

PROTOCOL_VERSION = 1

# JSON-RPC codes, with the two ACP adds.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
SESSION_NOT_FOUND = -32002

# engine.start(wait_s=...) default; a second turn waits out the same deadline
# rather than reporting an engine that is still spawning.
ENGINE_START_WAIT_S = 180.0
MCP_SHUTDOWN_WAIT_S = 5.0

MODES = ({"id": "agent", "name": "Agent",
          "description": "Read the workspace, then edit and run, asking before anything that changes state."},
         {"id": "ask", "name": "Ask",
          "description": "Read-only: inspect and answer, with no edits, no processes and no outlets."})

OPTIONS = ({"optionId": "allow_once", "name": "Allow once", "kind": "allow_once"},
           {"optionId": "allow_always", "name": "Always allow in this session", "kind": "allow_always"},
           {"optionId": "reject_once", "name": "Reject once", "kind": "reject_once"},
           {"optionId": "reject_always", "name": "Always reject this tool", "kind": "reject_always"})

_KIND = {"read_file": "read", "list_dir": "read", "grep": "search", "semantic_search": "search",
         "write_file": "edit", "edit_file": "edit", "run_python": "execute", "run_shell": "execute",
         "git_status": "execute", "git_diff": "execute", "web_task": "fetch", "coworker_task": "other"}

_FOCUS = ("path", "command", "query", "pattern", "goal", "code", "url")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_PAGE = 50
_MAX_UPDATES = 2000
_ASYNC = object()


def _kind(name: str) -> str:
    return _KIND.get(name, "other")


def _title(name: str, args: dict) -> str:
    for key in _FOCUS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return f"{name}: {value.strip().splitlines()[0][:120]}"
    return name


def _locations(session, name: str, args: dict) -> list:
    rel = args.get("path")
    if not isinstance(rel, str) or name not in {"read_file", "write_file", "edit_file", "list_dir", "grep"}:
        return []
    if not rel or rel in {".", "./"} or os.path.isabs(rel) or ":" in rel:
        return []
    root = os.path.realpath(session.cwd)
    full = os.path.realpath(os.path.join(root, rel))
    if os.path.commonpath([os.path.normcase(root), os.path.normcase(full)]) != os.path.normcase(root):
        return []
    return [{"path": full}]


def _clip(value, limit=4000):
    """Bound what a stored or streamed update can carry."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + f"…[{len(value) - limit} chars omitted]"
    if isinstance(value, dict):
        return {key: _clip(item, max(200, limit // 2)) for key, item in value.items()}
    if isinstance(value, list):
        return [_clip(item, max(200, limit // 2)) for item in value[:40]]
    return value


def _redact(text: str) -> str:
    return outlets_bridge.redact(text)


def _iso(stamp=None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stamp if stamp is not None else time.time()))


def _number(env_value, default):
    try:
        return float(str(env_value).strip())
    except (TypeError, ValueError):
        return float(default)


def _prompt_text(blocks, limit=60_000) -> str:
    """Flatten ACP content blocks into the text the loop is fed.

    Fennec advertises ``embeddedContext`` only: resources are read as text,
    images and audio are named but not decoded, and a prompt with no text at
    all is an invalid request rather than a silent no-op.
    """
    parts, skipped = [], []
    for block in blocks:
        if not isinstance(block, dict):
            raise AcpError(INVALID_PARAMS, "every prompt block must be an object")
        kind = block.get("type")
        if kind == "text":
            parts.append(str(block.get("text") or ""))
        elif kind == "resource":
            inner = block.get("resource") if isinstance(block.get("resource"), dict) else {}
            if isinstance(inner.get("text"), str):
                parts.append(f"[{inner.get('uri', 'resource')}]\n{inner['text']}")
            else:
                skipped.append(f"binary resource {inner.get('uri', '')}".strip())
        elif kind == "resource_link":
            parts.append(f"[linked resource {block.get('name') or block.get('uri')} "
                         f"at {block.get('uri')}; Fennec reads it with its own tools]")
        elif kind in ("image", "audio"):
            skipped.append(f"{kind} content (not supported by Fennec)")
        else:
            raise AcpError(INVALID_PARAMS, f"unsupported prompt content block {kind!r}")
    if not [part for part in parts if part.strip()]:
        raise AcpError(INVALID_PARAMS,
                       "Fennec needs a textual prompt; it cannot answer an image or audio only request"
                       + (f" (dropped: {', '.join(sorted(set(skipped)))})" if skipped else ""))
    if skipped:
        parts.append("[omitted from this prompt: " + ", ".join(sorted(set(skipped))) + "]")
    text = "\n\n".join(part for part in parts if part.strip()).strip()
    return text[:limit] + (f"…[{len(text) - limit} chars dropped]" if len(text) > limit else "")


class AcpError(Exception):
    def __init__(self, code: int, message: str, data=None):
        super().__init__(message)
        self.code, self.message, self.data = code, message, data

    def to_json(self) -> dict:
        out = {"code": self.code, "message": _redact(self.message)[:500]}
        if self.data is not None:
            out["data"] = self.data
        return out


class Session:
    """One ACP session: its mode, its transcript, its own MCP servers."""

    def __init__(self, session_id: str, cwd: str, *, mode="agent", title="",
                 outlets=None, mcp=None, system_extra=""):
        self.id = session_id
        self.cwd = cwd
        self.mode = mode if mode in {item["id"] for item in MODES} else "agent"
        self.title = title
        self.outlets = outlets
        self.mcp = mcp or mcp_bridge.McpPool()
        self.system_extra = system_extra
        self.history: list = []
        self.updates: list = []
        self.usage: dict = {}
        self.created = _iso()
        self.updated = self.created
        self.loop_sid: str | None = None
        self.current: dict | None = None
        self.message_id: str | None = None
        self.thought_id: str | None = None
        self.cancel = threading.Event()
        self.busy = False
        self.closed = False
        self.noted = False

    def record(self, update: dict) -> None:
        self.updates.append(update)
        if len(self.updates) > _MAX_UPDATES:
            del self.updates[:len(self.updates) - _MAX_UPDATES]

    def to_json(self) -> dict:
        return {"sessionId": self.id, "cwd": self.cwd, "mode": self.mode, "title": self.title,
                "createdAt": self.created, "updatedAt": self.updated, "usage": self.usage,
                "history": [_clip(message, 200_000) for message in self.history],
                "updates": self.updates}

    @classmethod
    def from_json(cls, raw: dict, outlets=None) -> "Session":
        session = cls(str(raw.get("sessionId")), str(raw.get("cwd") or ""),
                      mode=str(raw.get("mode") or "agent"), title=str(raw.get("title") or ""),
                      outlets=outlets)
        history = raw.get("history")
        session.history = [dict(item) for item in history if isinstance(item, dict)] \
            if isinstance(history, list) else []
        updates = raw.get("updates")
        session.updates = [item for item in updates if isinstance(item, dict)] \
            if isinstance(updates, list) else []
        usage = raw.get("usage")
        session.usage = dict(usage) if isinstance(usage, dict) else {}
        session.created = str(raw.get("createdAt") or session.created)
        session.updated = str(raw.get("updatedAt") or session.updated)
        return session


def _utf8(stream):
    """Make a real stdio pipe speak UTF-8, which is what ACP promises.

    A spawned interpreter on Windows defaults its pipes to the ANSI code page, so
    an emoji or a CJK path in a tool result raises UnicodeEncodeError - a
    ValueError, which _write reads as a dead client and answers by stopping the
    session. The same holds inbound: Air sends UTF-8, and a code page either
    mangles it or refuses to decode it. Reconfiguring in place keeps the object
    callers already hold, and a stream without reconfigure is left alone.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="strict")
        except (ValueError, OSError):
            pass
    return stream


class Agent:
    """ACP v1 agent serving one stdio client."""

    def __init__(self, *, config=None, environ=None, reader=None, writer=None,
                 store_dir=None, engine_client=None, outlets=None):
        self.env = dict(os.environ if environ is None else environ)
        self._config = config
        self.engine_client = engine_client
        self.outlets = outlets
        self.reader = _utf8(sys.stdin) if reader is None else reader
        self.writer = _utf8(sys.stdout) if writer is None else writer
        override = store_dir or self.env.get("FENNEC_ACP_HOME")
        self.store = Path(override).expanduser() if override else Path.home() / ".fennec" / "sessions"
        self.max_steps = int(_number(self.env.get("FENNEC_ACP_MAX_STEPS"), 24))
        self.approve_timeout = max(5.0, _number(self.env.get("FENNEC_ACP_APPROVE_TIMEOUT"), 300))
        self.idle_engine_timeout = _number(self.env.get("FENNEC_ACP_IDLE_TIMEOUT"), 300)
        self.engine_autostart = self.env.get("FENNEC_ACP_ENGINE_START", "1").strip().lower() \
            not in {"0", "false", "no", "off"}
        self.allow_npu = self.env.get("FENNEC_ACP_NPU", "").strip().lower() in {"1", "true", "yes"}
        self.sessions: dict[str, Session] = {}
        self.initialized = False
        self.client_info: dict = {}
        self.client_capabilities: dict = {}
        self._awaiting: dict = {}
        self._inflight: dict = {}
        self._next_id = 0
        self._write_lock = threading.Lock()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._last_activity = time.monotonic()
        self._started_engine = False

    # --- configuration ----------------------------------------------------
    def settings(self):
        if self._config is None:
            try:
                from .config import load_config
                self._config = load_config(environ=self.env)
            except Exception:
                self._config = False
        return None if self._config is False else self._config

    def _outlets(self):
        if self.outlets is None:
            self.outlets = outlets_bridge.Outlets(self.settings(), self.env)
        return self.outlets

    def _client(self):
        return self.engine_client or engine

    def _window(self) -> int:
        config = self.settings()
        try:
            ctx = int(getattr(config, "ctx", 0) or 0) if config else 0
        except (TypeError, ValueError):
            ctx = 0
        return ctx if ctx > 0 else max(4096, limits()["budget_chars"] // 4)

    # --- transport --------------------------------------------------------
    def _write(self, message: dict) -> None:
        try:
            payload = json.dumps(message, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError):
            payload = json.dumps({"jsonrpc": "2.0", "id": None,
                                  "error": {"code": INTERNAL_ERROR,
                                            "message": "response could not be serialized"}})
        with self._write_lock:
            try:
                self.writer.write(payload + "\n")
                self.writer.flush()
            except (OSError, ValueError):
                self._stop.set()

    def _respond(self, request_id, result=None, error=None) -> None:
        # A response carries result or error, never both; id stays null only when
        # the request itself could not be read.
        if request_id is None and error is None:
            return
        message = {"jsonrpc": "2.0", "id": request_id}
        if error is not None:
            message["error"] = error.to_json() if isinstance(error, AcpError) else error
        else:
            message["result"] = result if result is not None else {}
        self._write(message)

    def _update(self, session: Session, update: dict, *, record=True, send=True) -> None:
        if record:
            session.record(update)
        if send and not session.closed:
            self._write({"jsonrpc": "2.0", "method": "session/update",
                         "params": {"sessionId": session.id, "update": update}})

    def run(self) -> int:
        threading.Thread(target=self._watch_idle, daemon=True).start()
        try:
            for line in self.reader:
                self._line(line)
                # Honour an exit notification without waiting for another line:
                # a client that keeps the pipe open must not keep an agent alive.
                if self._stop.is_set():
                    break
        except (OSError, ValueError):
            pass
        finally:
            self.shutdown()
        return 0

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            sessions = list(self.sessions.values())
            self.sessions.clear()
        try:
            for session in sessions:
                session.closed = True
                session.cancel.set()
                try:
                    session.mcp.stop()
                except Exception:
                    # One wedged MCP server must not keep the others, or the
                    # engine we started, alive past the end of the session.
                    pass
        finally:
            self._engine_stop()

    def _line(self, raw: str) -> None:
        line = raw.strip()
        if not line:
            return
        try:
            message = json.loads(line)
        except ValueError:
            self._respond(None, None, AcpError(PARSE_ERROR, "invalid JSON on stdin"))
            return
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            self._respond(None, None, AcpError(INVALID_REQUEST, "expected a JSON-RPC 2.0 object"))
            return
        method = message.get("method")
        if method is None:
            self._resolve(message)
            return
        request_id = message.get("id")
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        if request_id is None:
            self._notification(method, params)
            return
        if not self.initialized and method != "initialize":
            self._respond(request_id, None,
                          AcpError(INVALID_REQUEST, "initialize must be called before any other method"))
            return
        if method != "initialize":
            with self._lock:
                replayed = request_id in self._inflight
            if replayed:
                self._respond(request_id, None, AcpError(INVALID_REQUEST,
                                                         "this request id already has a turn in flight"))
                return
        handler = self._METHODS.get(method)
        if handler is None:
            self._respond(request_id, None, AcpError(METHOD_NOT_FOUND, f"unknown method {method}"))
            return
        try:
            result = handler(self, params, request_id)
        except AcpError as exc:
            self._respond(request_id, None, exc)
        except Exception as exc:
            self._respond(request_id, None,
                          AcpError(INTERNAL_ERROR, f"Fennec failed: {type(exc).__name__}: {exc}"))
        else:
            if result is not _ASYNC:
                self._respond(request_id, result)

    def _notification(self, method: str, params: dict) -> None:
        # Unknown notifications are never answered, per JSON-RPC.
        if method == "session/cancel":
            self._cancel_session(params.get("sessionId"))
        elif method == "$/cancel_request":
            # Cancelling the request that started a turn means cancelling the turn;
            # the one legal answer is the prompt response with stopReason cancelled.
            with self._lock:
                session_id = self._inflight.get(params.get("requestId"))
            if session_id is None:
                self._resolve_pending(params.get("requestId"))
                return
            self._cancel_session(session_id)
        elif method == "exit":
            self._stop.set()

    # --- agent -> client requests ----------------------------------------
    def _request(self, method: str, params: dict, timeout: float, *, abort=None):
        with self._lock:
            self._next_id += 1
            # Our own request ids live in a separate namespace from the client's:
            # $/cancel_request carries a client id, and an unnumbered collision
            # would silently cancel an unrelated pending permission request.
            request_id = f"fennec-{self._next_id}"
            slot = {"event": threading.Event(), "message": None}
            self._awaiting[request_id] = slot
        self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self._stop.is_set():
            if slot["event"].wait(min(0.5, max(0.05, deadline - time.monotonic()))):
                break
            if abort is not None and abort():
                break
        with self._lock:
            self._awaiting.pop(request_id, None)
        return slot["message"]

    def _resolve(self, message: dict) -> None:
        request_id = message.get("id")
        with self._lock:
            slot = self._awaiting.get(request_id)
        if slot is None:
            return
        slot["message"] = message
        slot["event"].set()

    def _resolve_pending(self, request_id) -> None:
        """A $/cancel_request aimed at one of our own requests: stop waiting."""
        with self._lock:
            slot = self._awaiting.pop(request_id, None)
        if slot is not None:
            slot["message"] = None
            slot["event"].set()

    # --- initialize --------------------------------------------------------
    def initialize(self, params: dict, request_id) -> dict:
        if params.get("protocolVersion") is None:
            raise AcpError(INVALID_PARAMS, "initialize needs protocolVersion")
        self.initialized = True
        self.client_info = params.get("clientInfo") if isinstance(params.get("clientInfo"), dict) else {}
        self.client_capabilities = (params.get("clientCapabilities")
                                    if isinstance(params.get("clientCapabilities"), dict) else {})
        return {"protocolVersion": PROTOCOL_VERSION,
                "agentInfo": {"name": "fennec", "title": "Fennec", "version": _version()},
                "authMethods": [],
                "agentCapabilities": {
                    "loadSession": True,
                    "promptCapabilities": {"image": False, "audio": False, "embeddedContext": True},
                    "mcpCapabilities": {"http": False, "sse": False},
                    "sessionCapabilities": {"list": {}, "resume": {}, "close": {}}}}

    # --- sessions ----------------------------------------------------------
    def session_new(self, params: dict, request_id) -> dict:
        self._require_initialize()
        cwd = self._cwd(params.get("cwd"))
        servers = params.get("mcpServers")
        if not isinstance(servers, list):
            raise AcpError(INVALID_PARAMS, "session/new needs an mcpServers array")
        self._reject_unadvertised(params)
        session = Session(_new_id(), cwd, outlets=self._outlets(),
                          mcp=mcp_bridge.McpPool.from_acp(servers))
        with self._lock:
            self.sessions[session.id] = session
        self._persist(session)
        self._update(session, {"sessionUpdate": "available_commands_update", "availableCommands": []},
                     record=False)
        return {"sessionId": session.id, "modes": _modes(session.mode)}

    def session_load(self, params: dict, request_id) -> dict:
        return self._adopt(params, replay=True)

    def session_resume(self, params: dict, request_id) -> dict:
        return self._adopt(params, replay=False)

    def _adopt(self, params: dict, *, replay: bool) -> dict:
        self._require_initialize()
        session_id = params.get("sessionId")
        record = self._read(session_id)
        if record is None:
            raise AcpError(SESSION_NOT_FOUND, "Fennec has no transcript for this session id",
                           {"sessionId": session_id})
        cwd = self._cwd(params.get("cwd"))
        servers = params.get("mcpServers")
        if servers is not None and not isinstance(servers, list):
            raise AcpError(INVALID_PARAMS, "mcpServers must be an array")
        self._reject_unadvertised(params)
        with self._lock:
            existing = self.sessions.get(str(session_id))
        if existing is not None:
            if os.path.normcase(os.path.realpath(existing.cwd)) != \
                    os.path.normcase(os.path.realpath(cwd)):
                raise AcpError(INVALID_PARAMS,
                               "this session is already open on another workspace; close it first",
                               {"sessionId": existing.id})
            if servers:
                self._reattach_mcp(existing, servers)
            if replay:
                for update in list(existing.updates):
                    self._update(existing, update, record=False)
            return {"modes": _modes(existing.mode)}
        session = Session.from_json({**record, "sessionId": str(session_id), "cwd": cwd},
                                    outlets=self._outlets())
        session.mcp = mcp_bridge.McpPool.from_acp(servers or [])
        with self._lock:
            self.sessions[session.id] = session
        if replay:
            for update in list(session.updates):
                self._update(session, update, record=False)
        return {"modes": _modes(session.mode)}

    def _reattach_mcp(self, session: Session, servers: list) -> None:
        """``session/load`` on a live session re-declares its MCP servers.

        Silently dropping them would leave the client believing tools it never
        got, so an identical list is a no-op and a different one replaces the
        pool — unless a turn is running on the servers it would shut down.
        """
        wanted = mcp_bridge.McpPool.from_acp(servers)
        if _pool_spec(wanted) == _pool_spec(session.mcp):
            return
        if session.busy:
            raise AcpError(INVALID_REQUEST,
                           "this session is mid-turn; close it to change its mcpServers",
                           {"sessionId": session.id})
        session.mcp.stop()
        session.mcp = wanted

    def session_list(self, params: dict, request_id) -> dict:
        self._require_initialize()
        wanted = params.get("cwd")
        try:
            cursor = max(0, int(params.get("cursor") or 0))
        except (TypeError, ValueError):
            cursor = 0
        records = self._records()
        if isinstance(wanted, str) and wanted:
            root = os.path.realpath(wanted)
            records = [item for item in records
                       if os.path.normcase(os.path.realpath(str(item.get("cwd") or "")))
                       == os.path.normcase(root)]
        records.sort(key=lambda item: str(item.get("updatedAt") or ""), reverse=True)
        page = records[cursor:cursor + _PAGE]
        out = {"sessions": [{"sessionId": str(item.get("sessionId")), "cwd": str(item.get("cwd") or ""),
                             "title": str(item.get("title") or "") or None,
                             "updatedAt": str(item.get("updatedAt") or _iso())} for item in page]}
        if cursor + _PAGE < len(records):
            out["nextCursor"] = str(cursor + _PAGE)
        return out

    def session_close(self, params: dict, request_id) -> dict:
        self._require_initialize()
        session = self._known(params.get("sessionId"))
        self._cancel_session(session.id)
        with self._lock:
            self.sessions.pop(session.id, None)
        session.closed = True
        session.mcp.stop()
        self._persist(session)
        return {}

    def session_set_mode(self, params: dict, request_id) -> dict:
        session = self._known(params.get("sessionId"))
        mode = params.get("modeId")
        if mode not in {item["id"] for item in MODES}:
            raise AcpError(INVALID_PARAMS, f"unknown mode {mode!r}; Fennec has agent and ask",
                           {"sessionId": session.id})
        if mode == session.mode:
            return {}
        if session.busy:
            raise AcpError(INVALID_REQUEST, "session/set_mode cannot switch while a turn is running",
                           {"sessionId": session.id})
        session.mode = mode
        self._persist(session)
        self._update(session, {"sessionUpdate": "current_mode_update", "currentModeId": mode},
                     record=False)
        return {}

    # --- prompt ------------------------------------------------------------
    def session_prompt(self, params: dict, request_id):
        session = self._known(params.get("sessionId"))
        blocks = params.get("prompt")
        if not isinstance(blocks, list) or not blocks:
            raise AcpError(INVALID_PARAMS, "session/prompt needs a non-empty prompt array",
                           {"sessionId": session.id})
        with self._lock:
            if session.busy:
                raise AcpError(INVALID_REQUEST, "this session already has a prompt turn in flight",
                               {"sessionId": session.id})
            session.busy = True
            session.cancel.clear()
            self._inflight[request_id] = session.id
        threading.Thread(target=self._prompt_worker, args=(request_id, session, blocks),
                         daemon=True).start()
        return _ASYNC

    def _prompt_worker(self, request_id, session: Session, blocks: list) -> None:
        try:
            text = _prompt_text(blocks)
            stop = self._turn(session, text)
        except AcpError as exc:
            self._respond(request_id, None, exc)
        except Exception as exc:
            self._respond(request_id, None,
                          AcpError(INTERNAL_ERROR, f"Fennec failed: {type(exc).__name__}: {exc}"))
        else:
            self._respond(request_id, {"stopReason": stop})
        finally:
            with self._lock:
                session.busy = False
                self._inflight.pop(request_id, None)

    def _cancelled(self, session: Session) -> bool:
        return session.cancel.is_set() or session.closed or self._stop.is_set()

    def _turn(self, session: Session, text: str) -> str:
        self._last_activity = time.monotonic()
        self._update(session, {"sessionUpdate": "user_message_chunk",
                               "content": {"type": "text", "text": _clip(text, 20_000)}},
                     send=False)
        # Engine start and MCP handshake can take minutes. A cancel that lands in
        # that window has no loop session to stop yet, so each step re-checks the
        # flag instead of letting the turn run to completion.
        if self._cancelled(session):
            return "cancelled"
        self._ensure_engine()
        tools = self._mcp_tools(session)
        if self._cancelled(session):
            return "cancelled"
        if session.mcp.failures and not session.noted:
            session.noted = True
            self._update(session, {"sessionUpdate": "agent_thought_chunk", "content": {
                "type": "text", "text": f"[MCP not connected: {session.mcp.note()}]"}}, record=False)
        readonly = session.mode == "ask"
        sid = loop.new_session(text, session.cwd,
                               allow_write=not readonly, allow_shell=not readonly,
                               allow_web=not readonly, require_approval=not readonly,
                               allow_npu=self.allow_npu, engine_client=self._client(),
                               outlets=session.outlets, extra=tools,
                               system_extra=session.system_extra)
        session.current = None
        session.message_id = session.thought_id = None
        with self._lock:
            # Publishing the loop id and reading the cancel flag are one atomic
            # step with _cancel_session, so a cancel that lands in between cannot
            # be missed by either side.
            session.loop_sid = sid
            cancel_now = session.cancel.is_set()
        if cancel_now:
            loop.cancel(sid)
        try:
            loop.run_agent(sid, max_steps=self.max_steps, sink=self._sink(session),
                           approve=None if readonly else self._approver(session),
                           stream=True, history=session.history)
            state = loop.get_steps(sid) or {}
            session.history = loop.get_history(sid)
        finally:
            session.loop_sid = None
        if not session.title:
            session.title = text.splitlines()[0][:80]
        usage = state.get("usage") if isinstance(state.get("usage"), dict) else {}
        session.usage = usage
        if not session.closed:
            self._persist(session)
        self._last_activity = time.monotonic()
        status = state.get("status")
        if status == "error":
            raise AcpError(INTERNAL_ERROR,
                           _failure(state) or "Fennec could not finish the turn",
                           {"sessionId": session.id})
        if status == "cancelled" or state.get("stop") == "cancelled":
            return "cancelled"
        return state.get("stop") or "end_turn"

    def _sink(self, session: Session):
        def emit(event: dict) -> None:
            kind = event.get("type")
            if kind in ("text", "reasoning"):
                self._chunk(session, kind, event.get("text") or "")
            elif kind == "tool_start":
                session.current = event
                self._tool_open(session, event)
            elif kind == "tool_end":
                self._tool_close(session, event)
            elif kind == "plan":
                self._update(session, {"sessionUpdate": "plan", "entries": event.get("entries") or []})
            elif kind == "usage":
                used = (event.get("usage") or {}).get("prompt_tokens")
                self._update(session, {"sessionUpdate": "usage_update",
                                       "used": int(used or (event.get("context_chars") or 0) // 4),
                                       "size": self._window()}, record=False)
        return emit

    def _chunk(self, session: Session, kind: str, text: str) -> None:
        if not text:
            return
        stream = "agent_message_chunk" if kind == "text" else "agent_thought_chunk"
        field = "message_id" if kind == "text" else "thought_id"
        for piece in _pieces(text):
            if getattr(session, field) is None:
                setattr(session, field, uuid.uuid4().hex[:12])
            self._update(session, {"sessionUpdate": stream, "messageId": getattr(session, field),
                                   "content": {"type": "text", "text": piece}})

    def _tool_open(self, session: Session, event: dict) -> None:
        name = str(event.get("name") or "tool")
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        session.message_id = session.thought_id = None
        update = {"sessionUpdate": "tool_call", "toolCallId": _call_id(session, event),
                  "title": _title(name, args), "kind": _kind(name), "status": "in_progress",
                  "rawInput": _clip(args)}
        locations = _locations(session, name, args)
        if locations:
            update["locations"] = locations
        self._update(session, update)

    def _tool_close(self, session: Session, event: dict) -> None:
        result = event.get("result") if isinstance(event.get("result"), dict) else {}
        ok = bool(event.get("ok")) and "error" not in result
        name = str(event.get("name") or "tool")
        update = {"sessionUpdate": "tool_call_update",
                  "toolCallId": _call_id(session, event),
                  "status": "completed" if ok else "failed",
                  "rawOutput": _clip(result)}
        text = result.get("error") if not ok else json.dumps(result, ensure_ascii=False)
        update["content"] = [{"type": "content",
                              "content": {"type": "text", "text": _clip(str(text), 2000)}}]
        self._update(session, update)
        session.current = None

    def _mcp_tools(self, session: Session) -> dict:
        """Client-supplied MCP servers become ordinary tools, refreshed each turn."""
        if not session.mcp.servers:
            return {}
        try:
            return session.mcp.tools()
        except Exception as exc:
            session.mcp.failures["mcp"] = f"{type(exc).__name__}"
            return {}

    def _approver(self, session: Session):
        def approve(name: str, args: dict) -> str:
            if session.closed or self._stop.is_set():
                # Nobody is left to answer: asking would wait out the whole
                # approval timeout on a session the client has already dropped.
                return "reject_once"
            current = session.current if (session.current or {}).get("name") == name else None
            call_id = (current or {}).get("call_id") or f"permission-{uuid.uuid4().hex[:8]}"
            args = args if isinstance(args, dict) else {}
            tool_call = {"toolCallId": call_id, "title": _title(name, args), "kind": _kind(name),
                         "status": "pending", "rawInput": _clip(args)}
            locations = _locations(session, name, args)
            if locations:
                tool_call["locations"] = locations
            if current is not None:
                self._update(session, {"sessionUpdate": "tool_call_update", "toolCallId": call_id,
                                       "status": "pending"}, record=False)
            reply = self._request("session/request_permission",
                                  {"sessionId": session.id, "toolCall": tool_call,
                                   "options": [dict(option) for option in OPTIONS]},
                                  self.approve_timeout, abort=session.cancel.is_set)
            outcome = ((reply or {}).get("result") or {}).get("outcome") or {}
            option = outcome.get("optionId") if outcome.get("outcome") == "selected" else None
            if option not in {"allow_once", "allow_always", "reject_once", "reject_always"}:
                option = "reject_once"
            if option in ("allow_once", "allow_always") and current is not None:
                self._update(session, {"sessionUpdate": "tool_call_update", "toolCallId": call_id,
                                       "status": "in_progress"}, record=False)
            return option
        return approve

    # --- engine lifetime ---------------------------------------------------
    def _ensure_engine(self) -> None:
        if not self.engine_autostart:
            return
        client = self._client()
        try:
            if client.is_running():
                return
        except Exception as exc:
            raise AcpError(INTERNAL_ERROR, f"Fennec could not reach its engine: {type(exc).__name__}")
        try:
            report = client.start()
        except Exception as exc:
            raise AcpError(INTERNAL_ERROR, f"Fennec could not start its engine: {exc}")
        report = report or {}
        if not report.get("running") and report.get("starting"):
            # Another turn (or another app) is mid-spawn and answered without
            # waiting; failing here would make the second session of a cold start
            # always error. Wait for that spawn instead, up to the same deadline.
            deadline = time.monotonic() + ENGINE_START_WAIT_S
            while not report.get("running") and time.monotonic() < deadline:
                if self._stop.wait(0.5):
                    break
                try:
                    running = client.is_running()
                except Exception:
                    running = False
                # owned/error come from the start() answer, not the poll: losing
                # them here would leave an engine we spawned running forever.
                report = {"running": running, "owned": report.get("owned"),
                          "error": report.get("error")}
        if not report.get("running"):
            raise AcpError(INTERNAL_ERROR, _redact(str(report.get("error")
                                                   or "the engine is not answering")))
        self._started_engine = bool(report.get("owned"))

    def _engine_stop(self) -> None:
        if not self._started_engine:
            return
        self._started_engine = False
        try:
            self._client().stop()
        except Exception:
            pass

    def _watch_idle(self) -> None:
        pause = max(0.2, min(5.0, self.idle_engine_timeout / 3))
        while not self._stop.wait(pause):
            if not self._started_engine:
                continue
            if time.monotonic() - self._last_activity < self.idle_engine_timeout:
                continue
            if any(session.busy for session in list(self.sessions.values())):
                continue
            self._engine_stop()

    # --- storage -----------------------------------------------------------
    def _path(self, session_id: str) -> Path:
        return self.store / f"acp-{session_id}.json"

    def _persist(self, session: Session) -> None:
        temporary = None
        try:
            self.store.mkdir(parents=True, exist_ok=True)
            target = self._path(session.id)
            temporary = target.with_name(f"{target.name}.{uuid.uuid4().hex[:8]}.tmp")
            temporary.write_text(json.dumps(session.to_json(), ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, target)
        except OSError:
            if temporary is not None:
                with contextlib.suppress(OSError):
                    temporary.unlink()

    def _read(self, session_id) -> dict | None:
        if not isinstance(session_id, str) or not _SAFE_ID.fullmatch(session_id):
            return None
        try:
            raw = json.loads(self._path(session_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return raw if isinstance(raw, dict) else None

    def _records(self) -> list:
        out = []
        for path in list(self.store.glob("acp-*.json")) if self.store.is_dir() else []:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(raw, dict) and raw.get("sessionId"):
                out.append(raw)
        with self._lock:
            for session in self.sessions.values():
                out = [item for item in out if item.get("sessionId") != session.id] + [session.to_json()]
        return out

    # --- helpers -----------------------------------------------------------
    def _require_initialize(self) -> None:
        if not self.initialized:
            raise AcpError(INVALID_REQUEST, "initialize must be called first")

    def _cwd(self, value) -> str:
        if not isinstance(value, str) or not value.strip():
            raise AcpError(INVALID_PARAMS, "cwd must be a non-empty absolute path")
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise AcpError(INVALID_PARAMS, "cwd must be an absolute path")
        if not path.is_dir():
            raise AcpError(INVALID_PARAMS, f"cwd does not exist: {value}")
        return str(path.resolve())

    def _reject_unadvertised(self, params: dict) -> None:
        if params.get("systemPrompt"):
            raise AcpError(INVALID_PARAMS,
                           "Fennec does not advertise sessionCapabilities.systemPrompt")
        extra = params.get("additionalDirectories")
        if isinstance(extra, list) and extra:
            raise AcpError(INVALID_PARAMS,
                           "Fennec reaches only the session cwd; additionalDirectories is not advertised")

    def _known(self, session_id) -> Session:
        self._require_initialize()
        with self._lock:
            session = self.sessions.get(session_id) if isinstance(session_id, str) else None
        if session is None:
            raise AcpError(SESSION_NOT_FOUND, "unknown or closed session", {"sessionId": session_id})
        return session

    def _cancel_session(self, session_id) -> None:
        sid = None
        with self._lock:
            session = self.sessions.get(session_id) if isinstance(session_id, str) else None
            if session is not None:
                session.cancel.set()
                sid = session.loop_sid
        if sid:
            loop.cancel(sid)

    _METHODS = {"initialize": initialize, "session/new": session_new,
                "session/load": session_load, "session/resume": session_resume,
                "session/list": session_list, "session/close": session_close,
                "session/set_mode": session_set_mode, "session/prompt": session_prompt}

    _NOTIFIES = ("session/cancel", "$/cancel_request", "exit")


def _call_id(session: Session, event: dict) -> str:
    return str(event.get("call_id") or f"call-{uuid.uuid4().hex[:8]}")


def _pool_spec(pool) -> tuple:
    """Comparable identity of an MCP pool: name + command + args + env."""
    return tuple(sorted((name, server.command, tuple(server.args),
                        tuple(sorted(server.env.items())))
                        for name, server in pool.servers.items()))


def _modes(current: str) -> dict:
    return {"currentModeId": current, "availableModes": [dict(mode) for mode in MODES]}


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _pieces(text: str, size: int = 4000):
    if len(text) <= size:
        yield text
        return
    for start in range(0, len(text), size):
        yield text[start:start + size]


def _failure(state: dict) -> str:
    for event in reversed(state.get("steps") or []):
        if isinstance(event, dict) and event.get("type") == "error":
            return _redact(str(event.get("content") or ""))[:400]
    return ""


def _version() -> str:
    from . import __version__
    return __version__


def serve(**kwargs) -> int:
    """Speak ACP on the current stdin/stdout until the client closes the pipe."""
    agent = Agent(**kwargs)
    _utf8(sys.stderr)  # anything the loop prints is diverted here, and it is model text
    with contextlib.redirect_stdout(sys.stderr):
        return agent.run()


def check() -> dict:
    agent = Agent()
    config = agent.settings()
    try:
        outlets_state = agent._outlets().status(refresh=True)
    except Exception as exc:
        outlets_state = {"error": f"{type(exc).__name__}"}
    try:
        running = agent._client().is_running()
    except Exception:
        running = None
    return {"protocolVersion": PROTOCOL_VERSION, "version": _version(),
            "agent": "fennec", "transport": "stdio",
            "methods": sorted([*agent._METHODS, *agent._NOTIFIES]),
            "store": str(agent.store), "maxSteps": agent.max_steps,
            "modes": [mode["id"] for mode in MODES],
            "engineUrl": getattr(config, "engine_url", None),
            "model": getattr(config, "model", None) or None,
            "engineRunning": running,
            "engineAutostart": agent.engine_autostart,
            "outlets": outlets_state}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--check" in argv:
        # A model path is user data: it can hold anything the code page cannot.
        _utf8(sys.stdout)
        print(json.dumps(check(), indent=2, ensure_ascii=False))
        return 0
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
