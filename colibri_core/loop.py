from __future__ import annotations

import json
import threading
import time
import uuid

from . import engine
from .tools import ToolContext, execute, openai_tools

SYSTEM = """You are NCM Coder, a local coding agent running on the Colibri inference engine \
(Qwen3-Coder-30B) inside the Next-Concept Modeling desktop app. You solve coding tasks in the \
user's workspace by calling tools.

Workflow:
1. Inspect first: list_dir, read_file, grep — never guess a file's contents.
2. Plan briefly, then implement with write_file / edit_file (small, correct changes).
3. VERIFY your work by running code with run_python (isolated sandbox). Fix failures.
4. When done, reply with a concise summary and NO tool call.

Rules: All paths are relative to the workspace. Prefer editing over rewriting. Keep answers tight. \
Use tools rather than asking the user for information you can read yourself. Do not call run_shell \
unless it is enabled."""

SESSIONS: dict = {}
_LOCK = threading.Lock()


def new_session(task: str, workspace: str, allow_shell: bool = False) -> str:
    sid = uuid.uuid4().hex[:12]
    session_dir = ""
    ctx = ToolContext(workspace=workspace, session_dir=session_dir, allow_shell=allow_shell)
    with _LOCK:
        SESSIONS[sid] = {"id": sid, "task": task, "steps": [], "status": "running",
                         "ctx": ctx, "created": time.time(), "usage": {}}
    return sid


def _emit(sid, event):
    with _LOCK:
        s = SESSIONS.get(sid)
        if s is not None:
            event["i"] = len(s["steps"])
            event["t"] = round(time.time() - s["created"], 2)
            s["steps"].append(event)


def get_steps(sid, after=0):
    with _LOCK:
        s = SESSIONS.get(sid)
        if not s:
            return None
        return {"id": sid, "status": s["status"], "task": s["task"],
                "steps": s["steps"][after:], "total": len(s["steps"]),
                "usage": s["usage"]}


def run_agent(sid: str, max_steps: int = 14):
    s = SESSIONS[sid]
    ctx = s["ctx"]
    system = SYSTEM + f"\n\nWorkspace: {ctx.workspace}"
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": s["task"]}]
    _emit(sid, {"type": "user", "content": s["task"]})
    tools = openai_tools()
    try:
        for _ in range(max_steps):
            msg, usage = engine.chat(messages, tools=tools)
            with _LOCK:
                for k, v in (usage or {}).items():
                    if isinstance(v, int):
                        s["usage"][k] = s["usage"].get(k, 0) + v
            assistant = {"role": "assistant", "content": msg.get("content") or ""}
            tcs = msg.get("tool_calls") or []
            if tcs:
                assistant["tool_calls"] = tcs
            messages.append(assistant)
            calls = [{"name": tc.get("function", {}).get("name"),
                      "args": _args(tc)} for tc in tcs]
            _emit(sid, {"type": "assistant", "content": assistant["content"], "tool_calls": calls})
            if not tcs:
                _emit(sid, {"type": "final", "content": assistant["content"]})
                with _LOCK:
                    s["status"] = "done"
                return
            for tc, call in zip(tcs, calls):
                result = execute(call["name"], call["args"], ctx)
                _emit(sid, {"type": "tool", "name": call["name"], "args": call["args"],
                            "result": result, "ok": result.get("ok", False)})
                payload = json.dumps(result, ensure_ascii=False)
                messages.append({"role": "tool", "tool_call_id": tc.get("id", call["name"]),
                                 "content": payload[:6000]})
        _emit(sid, {"type": "final", "content": "(reached the step limit without a final answer)"})
        with _LOCK:
            s["status"] = "done"
    except Exception as e:
        _emit(sid, {"type": "error", "content": repr(e)})
        with _LOCK:
            s["status"] = "error"


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