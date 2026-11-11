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

# ACP shows a plan while the turn runs; these phases are read back from the tool
# calls the loop actually made, never from a model-authored list.
_PHASE_ORDER = ("inspect", "implement", "verify", "summarise")
_PHASE_LABEL = {"inspect": "Inspect the workspace",
                "implement": "Implement the change",
                "verify": "Verify by running code or an outlet",
                "summarise": "Summarise the result"}
_PHASES = {"list_dir": ("inspect",), "read_file": ("inspect",), "grep": ("inspect",),
           "semantic_search": ("inspect",), "git_status": ("inspect",), "git_diff": ("inspect",),
           "write_file": ("implement",), "edit_file": ("implement",),
           "run_python": ("verify",), "run_shell": ("verify",),
           "web_task": ("verify",), "coworker_task": ("verify",)}


def plan_entries(used) -> list:
    """Only stages that happened, plus the one that is next.

    A skipped stage is left out rather than marked complete: a question that
    needed no edit shows "Inspect ✓ / Implement →", never a false completion.
    """
    entries, last = [], -1
    for index, phase in enumerate(_PHASE_ORDER):
        if phase in used:
            last = index
            entries.append({"content": _PHASE_LABEL[phase], "status": "completed",
                            "priority": "high" if phase in ("implement", "verify") else "medium"})
    if last + 1 < len(_PHASE_ORDER):
        phase = _PHASE_ORDER[last + 1]
        entries.append({"content": _PHASE_LABEL[phase], "status": "in_progress",
                        "priority": "high" if phase in ("implement", "verify") else "medium"})
    return entries


def _publish_plan(sink, used):
    if sink is not None:
        sink({"type": "plan", "entries": plan_entries(used)})


def new_session(task: str, workspace: str, *, allow_write: bool = False,
                allow_shell: bool = False, allow_npu: bool = False,
                allow_web: bool = False, require_approval: bool = False,
                engine_client=None, outlets=None, extra=None, system_extra: str = "") -> str:
    sid = uuid.uuid4().hex[:12]
    ctx = ToolContext(workspace=workspace, allow_write=allow_write, allow_shell=allow_shell,
                      allow_npu=allow_npu, allow_web=allow_web, require_approval=require_approval,
                      engine_client=engine_client, outlets=outlets, extra=extra)
    with _LOCK:
        _evict()
        SESSIONS[sid] = {"id": sid, "task": task, "steps": [], "status": "running",
                         "ctx": ctx, "created": time.time(), "usage": {}, "stop": None,
                         "system_extra": system_extra, "history": [],
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
                "stop": session.get("stop"), "usage": session["usage"]}


def get_history(sid):
    """The transcript a finished turn left behind, for the next turn to continue."""
    with _LOCK:
        session = SESSIONS.get(sid)
        return list(session.get("history") or []) if session else []


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


def _finish(sid, status, stop=None):
    with _LOCK:
        session = SESSIONS.get(sid)
        if session is not None and session["status"] == "running":
            session["status"] = status
            session["stop"] = stop or ("cancelled" if status == "cancelled" else "end_turn")


def _cancelled(sid):
    with _LOCK:
        session = SESSIONS.get(sid)
        return session is None or session["cancel"].is_set()


def run_agent(sid: str, max_steps: int = 14, *, sink=None, approve=None, stream=False,
              history=None):
    """Drive one agent turn.

    ``sink(event)`` receives live ``text`` / ``reasoning`` / ``plan`` events and
    needs ``stream=True`` for token deltas. ``approve(name, args)`` returns an
    ACP permission option (allow_once, allow_always, reject_once,
    reject_always) and, when present, gated tools are advertised and confirmed
    at call time instead of hidden by session flags. ``history`` is the prior
    transcript of an ACP session; the finished transcript is stored back on the
    session under ``history`` so the next turn continues this one.
    """
    with _LOCK:
        session = SESSIONS.get(sid)
    if session is None:
        return
    ctx = session["ctx"]
    client = ctx.engine_client or engine
    bridge = tokeneconomizer.Bridge(ctx.workspace, sid, session["task"])
    learned = bridge.start()
    system = SYSTEM + f"\n\nWorkspace: {ctx.workspace}"
    if session.get("system_extra"):
        system += "\n\n" + session["system_extra"]
    if learned:
        system += "\n\n" + learned
    prior = [dict(message) for message in (history or []) if isinstance(message, dict)]
    if prior:
        if prior[0].get("role") != "system":
            prior.insert(0, {"role": "system", "content": system})
        messages = prior
        messages.append({"role": "user", "content": session["task"]})
    else:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": session["task"]}]
    tools = openai_tools(ctx, advertise_all=approve is not None)
    budget = limits()
    seen: dict[str, str] = {}
    always = set()
    used_phases = set()
    _publish_plan(sink, used_phases)
    try:
        for _ in range(max(1, min(max_steps, MAX_STEPS))):
            if _cancelled(sid):
                _finish(sid, "cancelled")
                return
            payload, context_stats = compact(messages, seen=seen, **budget)
            usage = {}
            finish = None
            if stream and hasattr(client, "chat_stream"):
                msg = None
                for event in client.chat_stream(payload, tools=tools, abort=lambda: _cancelled(sid)):
                    if event["type"] == "result":
                        msg, usage = event["message"], event["usage"]
                        finish = event.get("finish_reason")
                        if event.get("cancelled") or finish == "cancelled":
                            _emit(sid, {"type": "partial", "content": (msg or {}).get("content", "")})
                            _finish(sid, "cancelled")
                            return
                    elif sink is not None and event["type"] in ("text", "reasoning"):
                        sink({"type": event["type"], "text": event["text"]})
                if msg is None:
                    raise engine.EngineError("engine stream produced no result")
            else:
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
            if sink is not None:
                sink({"type": "usage", "usage": dict(usage or {}),
                      "context_chars": context_stats["chars_after"]})
            assistant = {"role": "assistant", "content": msg.get("content") or ""}
            reasoning = msg.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                assistant["reasoning_content"] = reasoning
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                assistant["tool_calls"] = tool_calls
            messages.append(assistant)
            calls = [{"name": tc.get("function", {}).get("name"), "args": _args(tc),
                      "call_id": tc.get("id") or tc.get("function", {}).get("name")}
                     for tc in tool_calls]
            bridge.assistant(assistant["content"], [
                (tc.get("id", call["name"]), call["name"], call["args"])
                for tc, call in zip(tool_calls, calls)], usage)
            _emit(sid, {"type": "assistant", "content": assistant["content"],
                        "reasoning": reasoning or "", "tool_calls": calls})
            if sink is not None and not stream:
                if reasoning:
                    sink({"type": "reasoning", "text": reasoning})
                if assistant["content"]:
                    sink({"type": "text", "text": assistant["content"]})
            if finish == "length":
                # Truncated output: partial text stays in the transcript, a cut-off
                # tool call is never executed.
                _emit(sid, {"type": "final", "content": assistant["content"]})
                _finish(sid, "done", "max_tokens")
                return
            if not tool_calls:
                used_phases.add("summarise")
                _publish_plan(sink, used_phases)
                _emit(sid, {"type": "final", "content": assistant["content"]})
                _finish(sid, "done", "end_turn")
                return
            notes = []
            for tc, call in zip(tool_calls, calls):
                if _cancelled(sid):
                    _finish(sid, "cancelled")
                    return
                call_id = tc.get("id", call["name"])
                _emit(sid, {"type": "tool_start", "name": call["name"], "args": call["args"],
                            "call_id": call_id})
                if sink is not None:
                    sink({"type": "tool_start", "name": call["name"], "args": call["args"],
                          "call_id": call_id})
                confirm = None
                if approve is not None:
                    def confirm(name, args, _allowed=always):
                        decision = approve(name, args)
                        if decision == "allow_always":
                            _allowed.add(name)
                        return decision
                result = execute(call["name"], call["args"], ctx, confirm=confirm,
                                 allowed_always=always)
                _emit(sid, {"type": "tool", "name": call["name"], "args": call["args"],
                            "result": result, "ok": result.get("ok", False),
                            "call_id": call_id})
                if sink is not None:
                    sink({"type": "tool_end", "name": call["name"], "args": call["args"],
                          "result": result, "ok": result.get("ok", False), "call_id": call_id})
                used_phases.update(_PHASES.get(call["name"], ()))
                _publish_plan(sink, used_phases)
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
        _finish(sid, "done", "max_turn_requests")
    except Exception as exc:
        _emit(sid, {"type": "error", "content": repr(exc)})
        _finish(sid, "error")
    finally:
        bridge.end()
        with _LOCK:
            current = SESSIONS.get(sid)
            if current is not None:
                current["history"] = messages


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