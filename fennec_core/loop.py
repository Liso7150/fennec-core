from __future__ import annotations

import json
import threading
import time
import uuid

from . import engine, tokeneconomizer
from .context import compact, limits
from .tools import ToolContext, execute, openai_tools

SYSTEM = """You are Fennec Desktopr, a local coding agent running on the Fennec inference engine \
(a device-tiered MoE decoder). You solve coding tasks in the user's workspace by calling tools.

Workflow:
1. Inspect first: list_dir, read_file, grep — never guess a file's contents.
2. Plan briefly, then implement with write_file / edit_file (small, correct changes).
3. Verify your work by running code with run_python when process execution is enabled. Fix failures.
4. When done, reply with a concise summary and NO tool call.

Rules: All paths are relative to the workspace. Prefer editing over rewriting. Keep answers tight. \
Use tools rather than asking the user for information you can read yourself. run_python and run_shell \
are NOT sandboxed: they run with the server user's privileges and are only offered when the user \
enabled process execution for this task."""

MAX_STEPS = 40
MAX_SESSIONS = 50
SESSIONS: dict = {}
_LOCK = threading.Lock()


def new_session(task: str, workspace: str, *, allow_write: bool = False,
                allow_shell: bool = False, allow_npu: bool = False,
                engine_client=None) -> str:
    sid = uuid.uuid4().hex[:12]
    ctx = ToolContext(workspace=workspace, allow_write=allow_write, allow_shell=allow_shell,
                      allow_npu=allow_npu, engine_client=engine_client)
    with _LOCK:
        _evict()
        SESSIONS[sid] = {"id": sid, "task": task, "steps": [], "status": "running",
                         "ctx": ctx, "created": time.time(), "usage": {},
                         "cancel": threading.Event()}
    return sid


def _evict():
    if len(SESSIONS) < MAX_SESSIONS:
        return
    finished = [(session["created"], sid) for sid, session in SESSIONS.items()
                if session["status"] != "running"]
    for _, sid in sorted(finished)[:max(1, len(SESSIONS) - MAX_SESSIONS + 1)]:
        SESSIONS.pop(sid, None)


def _emit(sid, event):
    with _LOCK:
        session = SESSIONS.get(sid)
        if session is None or len(session["steps"]) >= 1000:
            return
        event["i"] = len(session["steps"])
        event["t"] = round(time.time() - session["created"], 2)
        session["steps"].append(event)


def get_steps(sid, after=0):
    with _LOCK:
        session = SESSIONS.get(sid)
        if not session:
            return None
        return {"id": sid, "status": session["status"], "task": session["task"],
                "steps": session["steps"][max(0, after):], "total": len(session["steps"]),
                "usage": session["usage"]}


def cancel(sid) -> bool:
    """Mark the run cancelled immediately; the worker stops at its next checkpoint."""
    with _LOCK:
        session = SESSIONS.get(sid)
        if session is None:
            return False
        if session["status"] == "running":
            session["status"] = "cancelled"
            session["cancel"].set()
            session["steps"].append({"type": "cancelled", "content": "Cancelled by the user.",
                                     "i": len(session["steps"]),
                                     "t": round(time.time() - session["created"], 2)})
            return True
        return session["status"] == "cancelled"


def _finish(sid, status):
    with _LOCK:
        session = SESSIONS.get(sid)
        if session is not None and session["status"] == "running":
            session["status"] = status


def _cancelled(sid):
    with _LOCK:
        session = SESSIONS.get(sid)
        return session is None or session["cancel"].is_set()


def run_agent(sid: str, max_steps: int = 14):
    with _LOCK:
        session = SESSIONS.get(sid)
    if session is None:
        return
    ctx = session["ctx"]
    client = ctx.engine_client or engine
    bridge = tokeneconomizer.Bridge(ctx.workspace, sid, session["task"])
    learned = bridge.start()
    system = SYSTEM + f"\n\nWorkspace: {ctx.workspace}"
    if learned:
        system += "\n\n" + learned
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": session["task"]}]
    tools = openai_tools(ctx)
    budget = limits()
    seen: dict[str, str] = {}
    try:
        for _ in range(max(1, min(max_steps, MAX_STEPS))):
            if _cancelled(sid):
                _finish(sid, "cancelled")
                return
            payload, context_stats = compact(messages, seen=seen, **budget)
            msg, usage = client.chat(payload, tools=tools)
            if _cancelled(sid):
                _finish(sid, "cancelled")
                return
            with _LOCK:
                current = SESSIONS.get(sid)
                if current is not None:
                    for key, value in (usage or {}).items():
                        if isinstance(value, int):
                            current["usage"][key] = current["usage"].get(key, 0) + value
                    for key in ("chars_before", "chars_after", "elided", "deduped"):
                        current["usage"]["context_" + key] = (current["usage"].get("context_" + key, 0)
                                                              + context_stats[key])
            assistant = {"role": "assistant", "content": msg.get("content") or ""}
            reasoning = msg.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                assistant["reasoning_content"] = reasoning
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                assistant["tool_calls"] = tool_calls
            messages.append(assistant)
            calls = [{"name": tc.get("function", {}).get("name"), "args": _args(tc)}
                     for tc in tool_calls]
            bridge.assistant(assistant["content"], [
                (tc.get("id", call["name"]), call["name"], call["args"])
                for tc, call in zip(tool_calls, calls)], usage)
            _emit(sid, {"type": "assistant", "content": assistant["content"],
                        "reasoning": reasoning or "", "tool_calls": calls})
            if not tool_calls:
                _emit(sid, {"type": "final", "content": assistant["content"]})
                _finish(sid, "done")
                return
            notes = []
            for tc, call in zip(tool_calls, calls):
                if _cancelled(sid):
                    _finish(sid, "cancelled")
                    return
                result = execute(call["name"], call["args"], ctx)
                _emit(sid, {"type": "tool", "name": call["name"], "args": call["args"],
                            "result": result, "ok": result.get("ok", False)})
                rendered = json.dumps(result, ensure_ascii=False)
                bridge.tool_result(tc.get("id", call["name"]), rendered,
                                   bool(result.get("ok", False)))
                note = bridge.after_tool(call["name"], call["args"], rendered,
                                         bool(result.get("ok", False)))
                if note:
                    notes.append(note)
                messages.append({"role": "tool", "tool_call_id": tc.get("id", call["name"]),
                                 "content": rendered})
            if notes:
                messages.append({"role": "user", "content": "\n\n".join(notes)})
        _emit(sid, {"type": "final", "content": "(reached the step limit without a final answer)"})
        _finish(sid, "done")
    except Exception as exc:
        _emit(sid, {"type": "error", "content": repr(exc)})
        _finish(sid, "error")
    finally:
        bridge.end()


def _args(tc):
    raw = tc.get("function", {}).get("arguments", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}


def start_thread(sid, max_steps=14):
    th = threading.Thread(target=run_agent, args=(sid, max_steps), daemon=True)
    th.start()
    return th