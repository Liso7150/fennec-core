from __future__ import annotations

import ast
import json
import os
import re
import threading
import time
import uuid

import engine, te_bridge
from context import compact, limits
from tools import ToolContext, approval_context, execute, openai_tools

SYSTEM = """You are Fennec, a local coding agent running on the Fennec inference engine \
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

# Every step type the loop can put in `session["steps"]`, which is the body of
# `/api/agent/poll`. Shells mirror it -- `STEP_TYPES` in the PWA's
# web/app.js, `map_step` in the desktop's events.rs -- and an unknown type
# there is rendered or dropped, never a reason to stop watching the run.
# `progress` is a tool saying what it is inside while it runs, so it carries a
# `call_id` and belongs to the call it reports on rather than being a step of work.
STEP_TYPES = ("plan", "assistant", "partial", "tool_start", "progress", "tool", "queue",
              "final", "error", "cancelled", "approval_required")

# The plan these shells show is read back from the tool calls the loop actually
# made, never from a model-authored list. `_publish_plan` puts the same list on
# both channels: an ACP session update and a poll step.
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

# What a finished run hands on to the queue. The loop decides a task was complex
# by the work it actually did, not by the length of the prompt: "summarise" is
# left out of the count because every finished run reaches it, so counting it
# would call a one-line answer complex.
_SUBTASK_PHASES = ("inspect", "implement", "verify")
MAX_SUBTASKS = 3
MAX_SUBTASK_TOKENS = 320
_SUBTASK_ASK = ("From what you just did, name the follow-up tasks still unfinished in "
                "this workspace: at most three, in the order they should be done, each "
                "one an imperative instruction the agent can carry out on its own. "
                'Answer with a JSON array of strings, or with lines numbered "1. ", '
                '"2. " and nothing else. Write NONE when the work is finished.')
_LIST_MARK = re.compile(r"^(?:[-*+•]|\d+[.)])[ \t)]*(?:\[[ xX]\][ \t]+)?")

# What a work phase the run never exercised is worth handing on. The ask above is
# not enough on its own: replayed against a live run's own transcript — which had
# read the file and written the checklist but never verified it — the engine
# answered 'NONE' both to the plain ask and to the same ask with the missing phase
# named in it. A model that believes it finished cannot be made to disagree by
# being told so, so the queue is not allowed to depend on its admission: the
# phases the run really ran decide what is left, and an item names itself by the
# verb of the phase it stands for.
_UNDONE_TASK = {"inspect": "Inspect the workspace files this task names, before changing them.",
                "implement": "Make the workspace changes this task asks for.",
                "verify": ("Verify the result: re-read what this run wrote, or run the check "
                           "the task asked for.")}
_UNDONE_MARK = {"inspect": ("inspect", "read", "look", "find"),
                "implement": ("implement", "write", "create", "edit", "add", "update", "rename"),
                "verify": ("verify", "test", "check", "run", "confirm")}

# What each work phase is called by in a prompt. The tool calls a run really made
# stay the primary evidence, because they cannot be talked into lying, but they can
# under-report: measured on the live engine, session 20a8f838a133 was asked to read
# notes.txt and then write checklist4.md, emitted the read as a fenced ```read_file
# block nothing could execute, and ran only write_file. One executed phase is under
# the gate, so that run queued nothing although the work it skipped is exactly what
# a follow-up is for. What the prompt asked for does not depend on what came back,
# so it counts too.
_TASK_PHASES = {"inspect": ("read", "inspect", "look", "find", "list", "open", "show",
                            "summarise", "summarize", "grep", "search", "count", "check"),
                "implement": ("write", "create", "add", "edit", "update", "rename", "make",
                              "fix", "implement", "delete", "move", "refactor", "build"),
                "verify": ("verify", "test", "run", "confirm", "measure", "validate")}

# The subtasks this loop queued itself. A queued item comes back as an ordinary
# prompt, so without this record a complex task would decompose again every time
# it drained and the queue would never empty; depth is capped at one by that
# check alone. Bounded like SESSIONS, since the dict lives for the server's life.
_PROPOSED: dict = {}
_PROPOSED_KEEP = 256

# The reasoning levels the model dropdown offers, in the order the shells draw them.
EFFORT_LEVELS = ("auto", "low", "medium", "high")

# What a level says to the model. `reasoning_effort` on its own does not make the
# Fennec engine think: measured on the live engine, every effort value (and no
# value at all) returned an empty `reasoning_content`, while `enable_thinking`
# returned 297 characters of it. So a level is carried two ways -- the flag that
# opens the think block, and the line that says how long it should be. Saying so
# here is better than a dropdown whose middle rows do nothing.
EFFORT_LINE = {
    "low": "Reasoning effort: low. Think for one or two sentences, then answer.",
    "medium": ("Reasoning effort: medium. Think step by step in a few short "
               "paragraphs, then answer."),
    "high": ("Reasoning effort: high. Work through the problem carefully and at "
             "length, weighing the alternatives, then answer."),
}

# Tokens bought for the think block, on top of the 2048 a plain answer gets. The
# engine cannot stop a think block early, so the instruction alone would let a
# long one eat the whole completion: measured on the live engine at a 900-token
# budget, `medium` and `high` both spent all 900 on 2,060 and 2,120 characters of
# reasoning and returned an empty answer. Each level therefore gets its own room.
EFFORT_THINK_TOKENS = {"low": 0, "medium": 1024, "high": 2048}

# Words that mean the ask is work rather than a question, so `auto` spends a
# longer think block on them.
_WORK_WORDS = ("fix", "add", "create", "write", "edit", "refactor", "delete", "move",
               "test", "build", "run", "implement", "rename", "update", "check")
_PATH_MARKS = ("/", "\\", ".rs", ".py", ".md", ".js", ".json", ".toml", ".txt")


def resolve_effort(level, task: str = "") -> str:
    """The level a run actually uses: `auto` chooses, every other level stands.

    A named level is kept as given; anything unrecognised falls back to `auto`
    rather than to a level the user did not pick. `auto` reads the task: a short
    question with no file or action word gets `low`, everything else `medium`.
    """
    if level in EFFORT_LINE:
        return level
    words = (task or "").lower().split()
    if len(words) > 12 or any(w in _WORK_WORDS for w in words):
        return "medium"
    if any(mark in (task or "") for mark in _PATH_MARKS):
        return "medium"
    return "low"


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


def _publish_plan(sid, sink, used):
    """Send the one plan list to both channels a client can be watching.

    ACP gets a session update; ``/api/agent/poll`` gets a step, so a polling
    shell reads the plan from here instead of re-deriving phases from tool
    names in its own language.
    """
    entries = plan_entries(used)
    _emit(sid, {"type": "plan", "entries": entries})
    if sink is not None:
        sink({"type": "plan", "entries": entries})


def _task_phases(task: str) -> set:
    """Work phases the prompt itself asks for, matched by whole words.

    Whole words only: a substring match would call "runtime" a request to run
    something and "thread" a request to read one.
    """
    words = set(re.findall(r"[a-z]+", (task or "").lower()))
    return {phase for phase, marks in _TASK_PHASES.items() if words & set(marks)}


def is_complex(task: str, used_phases, effort: str) -> bool:
    """Whether a run did enough different kinds of work to be worth continuing.

    Two of the three work phases, at a level that thinks — either phases the run
    really ran, or phases the prompt asked for when the run got at least one of
    them done. A run that only looked something up, and one ``auto`` judged a
    short question, leave nothing to queue. A task the loop queued itself is
    never decomposed again.

    The prompt counts alongside the tool calls because the tool calls can
    under-report a task that needed several kinds of work, and the run that
    silently skipped part of its own plan is the one a follow-up is for. At least
    one phase must still have been run: a turn that touched nothing finished
    nothing either, and three subtasks handed to it would be noise.
    """
    if effort == "low" or (task or "").strip() in _PROPOSED:
        return False
    did = set(used_phases) & set(_SUBTASK_PHASES)
    return len(did) >= 2 or (bool(did) and len(_task_phases(task)) >= 2)


def parse_subtasks(text: str, goal: str = "") -> list:
    """Read the follow-up list out of a decomposition reply.

    The engine answers either as the JSON array the ask shows or as numbered
    lines, so both shapes are read. A fenced block is skipped and a lead-in
    sentence ending in a colon is not a task; an unmarked line counts only when
    the reply has more than one, because a lone "the work is finished" is an
    answer of its own. The goal is dropped: repeating it would re-run the task
    that was just finished.
    """
    body = (text or "").strip()
    if not body:
        return []
    raw: list = []
    array = re.search(r"\[[\s\S]*?\]", body)
    if array:
        try:
            data = json.loads(array.group(0))
        except (ValueError, TypeError):
            data = None
        if isinstance(data, list):
            raw = [entry if isinstance(entry, str)
                   else str(entry.get("content") or entry.get("task") or "")
                   for entry in data if isinstance(entry, (str, dict))]
    if not raw:
        marked, bare, outside_fence = [], [], True
        for line in body.splitlines():
            line = line.strip()
            if line.startswith("```"):
                outside_fence = not outside_fence
            elif not outside_fence or not line or line.endswith(":"):
                continue
            elif _LIST_MARK.match(line):
                marked.append(_LIST_MARK.sub("", line))
            else:
                bare.append(line)
        raw = marked or (bare if len(bare) > 1 else [])
    goal_lower = (goal or "").strip().lower()
    items: list = []
    for candidate in raw:
        item = re.sub(r"\s+", " ", candidate).strip("*_` ").strip()
        if len(item) < 6 or len(item) > 200 or item.lower() == goal_lower:
            continue
        items.append(item)
        if len(items) == MAX_SUBTASKS:
            break
    return items


def _undone_phases(used_phases, items) -> list:
    """Work phases the run skipped, as queue items, minus those its own answers
    already named.

    Kept off the list when an admitted item speaks the phase's verb, so a run that
    says "add a test" is not handed a second testing item.

    An inspection the run never did is worth handing on only while the change is
    still to come. Once the files are written, "inspect before you change them"
    describes a moment that has passed, and re-reading what the run wrote is what
    the verify item already asks for.
    """
    said = " ".join(items).lower()
    used = set(used_phases)
    return [phase for phase in _SUBTASK_PHASES
            if phase not in used
            and not (phase == "inspect" and "implement" in used)
            and not any(mark in said for mark in _UNDONE_MARK[phase])]


def _decompose(sid, sink, client, messages, budget, seen, task, used_phases, effort):
    """Put the run's unfinished work on the queue: what it admits, and what it
    plainly skipped.

    A polling shell owns the queue the user sees, so an ACP turn (which carries
    a sink) is asked nothing: it has no queue to fill and would only pay for the
    request. The ask runs before the ``final`` step, so the items are already in
    the shell's hands when it drains on completion. One completion, no tools,
    and nothing in here may break the run it follows — a dead engine leaves only
    the items the phase evidence supplies, which need no request at all.
    """
    if sink is not None or not is_complex(task, used_phases, effort):
        return []
    try:
        payload, _ = compact(list(messages) + [{"role": "user", "content": _SUBTASK_ASK}],
                             seen=dict(seen), **budget)
        reply, _usage = client.chat(payload, tools=None,
                                    max_tokens=MAX_SUBTASK_TOKENS, temperature=0.3)
        items = parse_subtasks((reply or {}).get("content") or "", task)
    except Exception:
        items = []
    for phase in _undone_phases(set(used_phases), items):
        if len(items) == MAX_SUBTASKS:
            break
        items.append(_UNDONE_TASK[phase])
    if not items:
        return []
    with _LOCK:
        for item in items:
            _PROPOSED[item] = None
        while len(_PROPOSED) > _PROPOSED_KEEP:
            _PROPOSED.pop(next(iter(_PROPOSED)))
    _emit(sid, {"type": "queue", "items": items})
    return items


def new_session(task: str, workspace: str, *, allow_write: bool = False,
                allow_shell: bool = False, allow_npu: bool = False,
                allow_web: bool = False, require_approval: bool = False,
                engine_client=None, outlets=None, extra=None, system_extra: str = "",
                reasoning_effort: str = "auto") -> str:
    """Open a session. ``require_approval`` makes every gated call the user's
    decision: an ACP turn asks through the approver it passed, a polling turn
    asks through `/api/agent/approve`, and nobody answering is a rejection.
    ``reasoning_effort`` is one of `EFFORT_LEVELS` and decides how much of the
    answer is spent thinking."""
    sid = uuid.uuid4().hex[:12]
    ctx = ToolContext(workspace=workspace, allow_write=allow_write, allow_shell=allow_shell,
                      allow_npu=allow_npu, allow_web=allow_web, require_approval=require_approval,
                      engine_client=engine_client, outlets=outlets, extra=extra)
    with _LOCK:
        _evict()
        SESSIONS[sid] = {"id": sid, "task": task, "steps": [], "status": "running",
                         "ctx": ctx, "created": time.time(), "usage": {}, "stop": None,
                         "system_extra": system_extra, "history": [],
                         "approvals": {}, "reasoning_effort": reasoning_effort,
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


def _pending_ids(session) -> list:
    """The approvals this session is waiting on, oldest first. Call it holding
    `_LOCK`: it reads the same records `resolve_approval` writes."""
    return [aid for aid, record in session.get("approvals", {}).items()
            if not record["waiter"].is_set()]


def list_sessions() -> list:
    """One row per session for the shells' task rails. `pending` is how many
    gated calls wait on an answer, which is what tells a rail that this run is
    stopped on a question rather than still thinking."""
    with _LOCK:
        return [{"id": sid, "task": s["task"], "status": s["status"],
                 "steps": len(s["steps"]), "pending": len(_pending_ids(s))}
                for sid, s in SESSIONS.items()]


def get_steps(sid, after=0):
    with _LOCK:
        session = SESSIONS.get(sid)
        if not session:
            return None
        return {"id": sid, "status": session["status"], "task": session["task"],
                "steps": session["steps"][max(0, after):], "total": len(session["steps"]),
                "stop": session.get("stop"), "usage": session["usage"],
                "pending": _pending_ids(session)}


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


APPROVAL_TIMEOUT = 120.0
_APPROVAL_OPTIONS = {"allow_once", "allow_always", "reject_once", "reject_always"}


def request_approval(sid, name, args):
    """Ask the client watching this session to allow or reject one tool call.

    The step it publishes carries the risk and the reason from `tools.GATES`,
    so no client has to keep its own idea of what a tool can do. Nobody
    answering — a disconnected client, a timeout, a cancel — is a rejection,
    which is what the same call already got silently; the difference is that
    now the user sees the question and the answer.
    """
    aid = uuid.uuid4().hex[:8]
    with _LOCK:
        session = SESSIONS.get(sid)
        if session is None:
            return "reject_once"
        session.setdefault("approvals", {})[aid] = {"waiter": threading.Event(),
                                                    "decision": None}
    _emit(sid, {"type": "approval_required", "approval_id": aid, "name": name,
                "args": args if isinstance(args, dict) else {},
                **approval_context(name)})
    deadline = time.monotonic() + APPROVAL_TIMEOUT
    while time.monotonic() < deadline:
        with _LOCK:
            session = SESSIONS.get(sid)
            record = (session or {}).get("approvals", {}).get(aid)
            cancelled = session is None or session["cancel"].is_set()
        if record is None:
            break
        if record["waiter"].is_set():
            break
        if cancelled:
            break
        record["waiter"].wait(0.25)
    with _LOCK:
        session = SESSIONS.get(sid)
        record = (session or {}).get("approvals", {}).get(aid) if session else None
        decision = (record or {}).get("decision")
        if session is not None and record is not None:
            session.get("approvals", {}).pop(aid, None)
    return decision if decision in _APPROVAL_OPTIONS else "reject_once"


def resolve_approval(sid, approval_id, decision):
    """Answer a pending approval. Returns False if it was never asked or is gone."""
    if decision not in _APPROVAL_OPTIONS:
        return False
    with _LOCK:
        session = SESSIONS.get(sid)
        record = (session or {}).get("approvals", {}).get(approval_id)
        if record is None:
            return False
        record["decision"] = decision
        record["waiter"].set()
        return True


def run_agent(sid: str, max_steps: int = 14, *, sink=None, approve=None, stream=False,
              history=None):
    """Drive one agent turn.

    ``sink(event)`` receives live ``text`` / ``reasoning`` / ``plan`` events and
    needs ``stream=True`` for token deltas. ``approve(name, args)`` returns an
    ACP permission option (allow_once, allow_always, reject_once,
    reject_always) and, when present, gated tools are advertised and confirmed
    at call time instead of hidden by session flags. A session opened with
    ``require_approval`` and no explicit approver gets one built from
    :func:`request_approval`, so a polling client answers over HTTP.
    ``history`` is the prior
    transcript of an ACP session; the finished transcript is stored back on the
    session under ``history`` so the next turn continues this one.
    """
    with _LOCK:
        session = SESSIONS.get(sid)
    if session is None:
        return
    ctx = session["ctx"]
    client = ctx.engine_client or engine
    bridge = te_bridge.Bridge(ctx.workspace, sid, session["task"])
    learned = bridge.start()
    effort = resolve_effort(session.get("reasoning_effort") or "auto", session["task"])
    # Both spellings go to the engine: the effort is what the user picked, the
    # flag is the one that has been measured to open the think block, and the
    # budget keeps a long think block from spending the answer's tokens.
    think_kw = {"reasoning_effort": effort, "enable_thinking": True,
                "max_tokens": 2048 + EFFORT_THINK_TOKENS[effort]}
    system = SYSTEM + f"\n\nWorkspace: {ctx.workspace}"
    system += "\n\n" + EFFORT_LINE[effort]
    if session.get("system_extra"):
        system += "\n\n" + session["system_extra"]
    if learned:
        system += "\n\n" + learned
    prior = [dict(message) for message in (history or []) if isinstance(message, dict)]
    if prior:
        if prior[0].get("role") != "system":
            prior.insert(0, {"role": "system", "content": system})
        else:
            # This turn's system text wins over the one the last turn stored: it
            # carries this run's effort level, and the old one would keep the
            # previous turn's thinking length however the dropdown is set now.
            prior[0]["content"] = system
        messages = prior
        messages.append({"role": "user", "content": session["task"]})
    else:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": session["task"]}]
    if approve is None and ctx.require_approval:
        # A polling client declared it can answer, so the question goes to it.
        def approve(name, args, _sid=sid):
            return request_approval(_sid, name, args)
    tools = openai_tools(ctx, advertise_all=approve is not None)
    budget = limits()
    seen: dict[str, str] = {}
    always = set()
    used_phases = set()
    nudged = False
    # The files this prompt promised, and the ones this run has really put on
    # disk. Both are compared resolved, so a call that spelled the path
    # differently from the prompt still counts as having written it.
    asked = [_resolve_path(ctx, path) for path in _paths_asked_to_write(session["task"])]
    written: set = set()

    def queue_subtasks():
        """Hand this run's unfinished work to the shells' queue, if it earned the ask."""
        _decompose(sid, sink, client, messages, budget, seen, session["task"],
                   used_phases, effort)

    _publish_plan(sid, sink, used_phases)
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
                for event in client.chat_stream(payload, tools=tools, abort=lambda: _cancelled(sid),
                                                **think_kw):
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
                msg, usage = client.chat(payload, tools=tools, **think_kw)
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
            if not tool_calls and assistant["content"]:
                # The engine has no function calling, so a turn that follows the
                # tool prompt arrives as a JSON object written into the content.
                call = _text_tool_call(assistant["content"], tools)
                if call is not None:
                    tool_calls = [{"id": "text-" + uuid.uuid4().hex[:8], "type": "function",
                                   "function": {"name": call[0], "arguments": json.dumps(call[1])}}]
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
            if not tool_calls and not nudged and _looks_like_how_to(assistant["content"], tools):
                # The reply handed over the call as text. One push-back asks for
                # the real thing; asking twice in one turn is nagging, and the
                # second no-call turn is allowed to be the answer.
                nudged = True
                messages.append({"role": "user", "content": HOW_TO_NUDGE})
                continue
            if (not tool_calls and not nudged and asked
                    and _pasted_an_unwritten_file(assistant["content"],
                                                  [p for p in asked if p not in written],
                                                  {t["function"]["name"] for t in tools})):
                # The reply is the file, sitting in the chat instead of the
                # workspace. Same one ask: the run can put it there itself.
                nudged = True
                messages.append({"role": "user", "content": WRITE_NUDGE})
                continue
            if not tool_calls:
                used_phases.add("summarise")
                _publish_plan(sid, sink, used_phases)
                final = assistant["content"]
                if not final and reasoning:
                    # A think block that fills the completion leaves the answer
                    # inside the reasoning. Better to show what the model got to
                    # than to hand the shells a blank final row.
                    final = reasoning.strip()
                queue_subtasks()
                _emit(sid, {"type": "final", "content": final})
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
                def report(note, _name=call["name"], _id=call_id):
                    """One live line from inside a tool that is still working."""
                    step = {"type": "progress", "name": _name, "call_id": _id,
                            "text": str(note.get("text") or ""),
                            "step": note.get("step") or 0, "of": note.get("of") or 0}
                    _emit(sid, dict(step))
                    if sink is not None:
                        sink(step)

                ctx.progress = report
                try:
                    result = execute(call["name"], call["args"], ctx, confirm=confirm,
                                     allowed_always=always)
                finally:
                    ctx.progress = None
                _emit(sid, {"type": "tool", "name": call["name"], "args": call["args"],
                            "result": result, "ok": result.get("ok", False),
                            "call_id": call_id})
                if sink is not None:
                    sink({"type": "tool_end", "name": call["name"], "args": call["args"],
                          "result": result, "ok": result.get("ok", False), "call_id": call_id})
                used_phases.update(_PHASES.get(call["name"], ()))
                _publish_plan(sid, sink, used_phases)
                rendered = json.dumps(result, ensure_ascii=False)
                if result.get("ok") and call["name"] in ("write_file", "edit_file"):
                    target = call["args"].get("path")
                    if isinstance(target, str) and target:
                        written.add(_resolve_path(ctx, target))
                bridge.tool_result(tc.get("id", call["name"]), rendered,
                                   bool(result.get("ok", False)))
                note = bridge.after_tool(call["name"], call["args"], rendered,
                                         bool(result.get("ok", False)))
                if note:
                    notes.append(note)
                messages.append({"role": "tool", "tool_call_id": tc.get("id", call["name"]),
                                 "content": rendered})
            # The model answered the push-back by making a call, so the ask is
            # spent and a later regression in this same run gets one more. The
            # budget is per stretch of prose, not per run: a long task can slip
            # twice, and a run that already proved it can call deserves to be
            # held to it.
            nudged = False
            if notes:
                messages.append({"role": "user", "content": "\n\n".join(notes)})
        queue_subtasks()
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


_PATH_ALIASES = ("source", "file_path", "filename", "file")


def _fit_path_key(args, props):
    """Move a call's file onto the parameter the tool actually declares.

    Models reach for the field they see in an editor rather than the one in the
    schema, and mean the same file.
    """
    if "path" in args or "path" not in props:
        return
    for alias in _PATH_ALIASES:
        if isinstance(args.get(alias), str):
            args["path"] = args.pop(alias)
            return


# Qwen3-Coder wraps the call it means to make in this tag. The wrapper has to
# come off before the call parsers run: with it on, the closing tag is trailing
# prose after the bracket, which is exactly what keeps a quoted example from
# executing, so it would also keep a real call from being read.
_TOOL_CODE = re.compile(r"<tool_code>\s*([\s\S]*?)\s*</tool_code>")


def _text_tool_call(text, tools):
    """Parse a tool call the model wrote as text instead of a native one.

    Only a JSON object naming an advertised tool whose remaining keys fit that
    tool's parameter list counts; quoted examples and prose JSON stay prose.
    Models spell the name key `tool`, `name` or `command`, so all three count.
    They spell the parameter bag `args` or `arguments`, so both count.
    """
    if not isinstance(text, str):
        return None
    schema = {t["function"]["name"]:
              (set(t["function"]["parameters"].get("properties", ())),
               set(t["function"]["parameters"].get("required", ())))
              for t in tools}
    text = _TOOL_CODE.sub(r"\1", text)
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        depth, in_str, esc = 0, False, False
        for end in range(start, len(text)):
            ch = text[end]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth:
                    continue
                try:
                    parsed = json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    break
                if not isinstance(parsed, dict):
                    break
                name = parsed.get("tool") or parsed.get("name") or parsed.get("command")
                if not isinstance(name, str) or name not in schema:
                    break
                args = parsed.get("args")
                if not isinstance(args, dict):
                    args = parsed.get("arguments")
                if not isinstance(args, dict):
                    args = {k: v for k, v in parsed.items()
                            if k not in ("tool", "name", "command", "args", "arguments")}
                props, required = schema[name]
                _fit_path_key(args, props)
                if not set(args) <= props or not required <= set(args):
                    break
                return name, args
    return (_xml_tool_call(text, schema) or _call_syntax_tool_call(text, schema)
            or _line_syntax_tool_call(text, schema) or _block_syntax_tool_call(text, schema))


# The call shape a Qwen-Coder checkpoint answers a tool prompt with: the tool
# name on a line of its own, then one tagged parameter after another, each value
# able to run over several lines. The chat template parks the name inside a
# fence and leaves the tags outside it, so the pieces have to be joined back.
_XML_PARAM = re.compile(r'<parameter\s+name\s*=\s*["\']?([A-Za-z_][A-Za-z0-9_]*)'
                        r'["\']?\s*>([\s\S]*?)<\s*/\s*parameter\s*>')
_XML_FUNCTION = re.compile(r'<function\s+name\s*=\s*["\']?([a-z_][a-z0-9_]*)["\']?\s*>')
_XML_NAME_LINE = re.compile(r"^[ \t`]*([a-z_][a-z0-9_]*)[ \t`]*$", re.M)


def _xml_tool_call(text, schema):
    """Read a name-plus-tagged-parameters call, only from a message that has tags.

    The name is the last one a parameter follows, so a call the model started,
    gave up on, and rewrote later in the same message does not win. A parameter
    the tool does not declare, and one the template left empty, are dropped
    rather than rejecting the call: these replies routinely carry an empty body.
    Every required parameter still has to survive, so half a call stays prose.
    """
    params = list(_XML_PARAM.finditer(text))
    if not params:
        return None
    marks = [(m.start(), m.group(1)) for m in _XML_FUNCTION.finditer(text)]
    marks += [(m.start(), m.group(1)) for m in _XML_NAME_LINE.finditer(text)]
    named = [(at, name) for at, name in sorted(marks)
             if name in schema and any(p.start() > at for p in params)]
    if not named:
        return None
    name_at, name = named[-1]
    props, required = schema[name]
    args = {}
    for p in params:
        if p.start() <= name_at:
            continue
        value = p.group(2)
        if value.startswith("\n"):
            value = value[1:]
        if value.endswith("\n"):
            value = value[:-1]
        if value.strip():
            args[p.group(1)] = value
    _fit_path_key(args, props)
    args = {key: value for key, value in args.items() if key in props}
    if required <= set(args):
        return name, args
    return None


# Some checkpoints answer a tool prompt with a call expression, not JSON.
_CALL_NAME = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _call_syntax_tool_call(text, schema):
    """Parse `write_file(path="a.md", content="hi")` as a tool call.

    The template's `tool` marker often arrives glued to the name, so
    `toolwrite_file(...)` is the same call. Only the final expression of the
    message counts: a call quoted inside a sentence has trailing prose and
    stays prose, which is what keeps an example from executing.
    """
    for m in _CALL_NAME.finditer(text):
        name = m.group(1)
        if name not in schema and name.startswith("tool") and name[4:] in schema:
            name = name[4:]
        if name not in schema:
            continue
        open_at = text.index("(", m.end() - 1)
        depth, quote, esc, close = 0, "", False, -1
        for i in range(open_at, len(text)):
            ch = text[i]
            if quote:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == quote:
                    quote = ""
            elif ch in "\"'":
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if not depth:
                    close = i
                    break
        if close < 0 or text[close + 1:].strip():
            continue
        try:
            node = ast.parse("_f(" + text[open_at + 1:close] + ")", mode="eval").body
        except SyntaxError:
            continue
        if not isinstance(node, ast.Call) or node.args:
            continue
        args = {}
        for kw in node.keywords:
            if kw.arg is None:
                args = {}
                break
            try:
                args[kw.arg] = ast.literal_eval(kw.value)
            except (ValueError, SyntaxError):
                args = {}
                break
        props, required = schema[name]
        _fit_path_key(args, props)
        if set(args) <= props and required <= set(args):
            return name, args
    return None


_LINE_CALL = re.compile(r"^(?:tool_)?([a-z_][a-z0-9_]*)\s+"
                        r"([a-z_][a-z0-9_]*)\s*[:=]\s*"
                        r"(?:\"([^\"]+)\"|'([^']+)'|(\S+))$")


def _line_syntax_tool_call(text, schema):
    """Parse `read_file path = web/app.js` as a tool call.

    Some checkpoints drop the braces and the quotes and still mean the call. The
    line has to be the last thing in the message, and the one parameter it names
    has to be that tool's only required one, so a sentence that merely mentions a
    tool, and a call that is missing half its arguments, both stay prose.
    """
    lines = [ln.strip().strip("`").strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    hit = _LINE_CALL.match(lines[-1])
    if not hit or hit.group(1) not in schema:
        return None
    name, key = hit.group(1), hit.group(2)
    value = next(group for group in hit.group(3, 4, 5) if group is not None)
    props, required = schema[name]
    if key not in props and key not in _PATH_ALIASES:
        return None
    args = {key: value}
    _fit_path_key(args, props)
    if set(args) <= props and required == set(args):
        return name, args
    return None


_BLOCK_ARG = re.compile(r"^([a-z_][a-z0-9_]*)\s*[:=]\s*(.+)$")

# The fence head the model puts the name in, measured live on 2026-11-19
# (session 068a4d2ea0c4): "tool:read_file", a prefix a code fence's language
# slot has no use for but a chat template full of tool examples reaches for.
_BLOCK_HEAD = re.compile(r"^(?:tool|call|function)[_.:/]([a-z_][a-z0-9_]*)$")


def _block_name(line, schema):
    """The tool a block header names, tolerating a "tool:" style prefix."""
    if line in schema:
        return line
    hit = _BLOCK_HEAD.match(line)
    return hit.group(1) if hit and hit.group(1) in schema else None


def _block_syntax_tool_call(text, schema):
    """Read a call whose name is the fence's language and whose parameters follow.

    Measured live three times on 2026-11-19 (sessions 298eb3e772ae, 20a8f838a133
    and 068a4d2ea0c4), the answer to a reading prompt was

        ```read_file
        path=.local-docs/queue-probe/notes.txt
        ```

    or the same block with the name qualified as ``tool:read_file``. That is a
    whole call: the chat template parks the tool name where a code fence keeps its
    language, and the parameters come under it. `_line_syntax_tool_call` reads only
    the last line, so a name and an argument on two lines parsed as nothing; the
    loop pushed back once, the model repeated the identical block, and the run
    ended with the call still unmade. Only the last such block counts, and every
    line under a name has to be a parameter the tool declares, so a block that
    trails off into prose stays prose.
    """
    lines = [ln for ln in (raw.strip().strip("`").strip()
                           for raw in (text or "").splitlines()) if ln]
    names = [_block_name(line, schema) for line in lines]
    found = []
    for index, name in enumerate(names):
        if name is None:
            continue
        props, required = schema[name]
        nxt = [j for j in range(index + 1, len(lines)) if names[j]]
        body = lines[index + 1:nxt[0] if nxt else len(lines)]
        args = {}
        for arg in body:
            hit = _BLOCK_ARG.match(arg)
            if not hit:
                args = {}
                break
            args[hit.group(1)] = hit.group(2).strip("\"'")
        if not args:
            continue
        _fit_path_key(args, props)
        if set(args) <= props and required <= set(args):
            found.append((name, args))
    return found[-1] if found else None


_FENCED = re.compile(r"```[^\n]*\n(.*?)```", re.S)
_CALL_HEAD = re.compile(r"^(?:tool_)?([a-z_][a-z0-9_]*)")
_FENCED_LANG = re.compile(r"```[ \t]*([A-Za-z0-9_+#.\-]*)[^\n]*\n(.*?)```", re.S)
_SHELL_LANGS = {"bash", "sh", "shell", "zsh", "ksh", "powershell", "pwsh", "cmd", "bat"}

# The one thing said back when a reply writes the call out instead of making it.
# Kept short because every token of it costs a round trip on a local engine.
HOW_TO_NUDGE = ("That wrote the tool call down instead of making it. Make the "
                'call now: {"name": "<tool>", "args": {<parameters>}}. Do not '
                "describe it, and do not put it in a code block.")


def _only_command_fences(text, names):
    """Whether the whole reply is command blocks the loop could have run itself.

    A shell info string is how a model hands over the command it means to run
    when it misses the call shape. The name test in `_looks_like_how_to` cannot
    see it, because `bash` is a language, not a tool. With nothing but whitespace
    outside the blocks the reply holds no answer at all, and `run_shell` being
    advertised means the loop could have made the call — so it is the call, not a
    tutorial. An answer that quotes a command keeps its prose and stays an answer.
    """
    if "run_shell" not in names:
        return False
    blocks = _FENCED_LANG.findall(text)
    if not blocks or _FENCED_LANG.sub("", text).strip():
        return False
    return any(lang.lower() in _SHELL_LANGS and body.strip() for lang, body in blocks)


def _looks_like_how_to(text, tools):
    """Whether a reply writes a tool call out instead of making it.

    A `<tool_code>` block always counts: the tag is the model's own declaration
    that it is making a call, and this only runs when nothing in the reply
    parsed into one. Beyond that, a fenced block counts when it opens with an
    advertised tool name, or when the reply is nothing but shell blocks, or when
    the reply opens as an object naming an advertised tool -- the shape a call
    that ran out of completion arrives in, which no parser can read. An answer
    that shows the user a function in their own language is an answer, and
    pushing back on that would be the loop arguing with a correct reply.
    """
    if not isinstance(text, str):
        return False
    if "<tool_code>" in text:
        return True
    names = {t["function"]["name"] for t in tools}
    if _opens_as_an_unreadable_call(text, names):
        return True
    if "```" not in text:
        return False
    if _only_command_fences(text, names):
        return True
    for body in _FENCED.findall(text):
        for line in body.strip().splitlines():
            head = _CALL_HEAD.match(line.strip().strip("`").strip())
            if head and head.group(1) in names:
                return True
    return False


# A call the model opened as JSON and then ran out of room writing, so no parser
# can read it. The name has to be one the loop advertised: `{"name": "..."}` is
# otherwise a perfectly ordinary answer in JSON.
_CALL_JSON_HEAD = re.compile(r'^\s*\{\s*"(?:name|tool|command)"\s*:\s*"([a-z_][a-z0-9_]*)"')


def _opens_as_an_unreadable_call(text, names):
    """Whether a reply this loop could not parse began as a call it could have made."""
    head = _CALL_JSON_HEAD.match(text or "")
    return bool(head and head.group(1) in names)


# How a prompt says it wants a file on disk, and how it spells the file. A path
# only counts as asked-for when one of these verbs sits in the same clause, so
# "read notes.txt, then write checklist.md" asks for one file and not two.
_WRITE_VERB = re.compile(r"\b(?:writ\w*|creat\w*|sav\w*|updat\w*|append\w*|generat\w*"
                         r"|add(?:s|ed|ing)?)\b", re.I)
_TASK_PATH = re.compile(r"[\"'`]?((?:[\w.\-]+[/\\])+[\w.\-]*|[\w.\-]+\.[A-Za-z0-9]{1,6})[\"'`]?")
_CLAUSE_MARKS = (",", ";", ":", "\n", "\r")

# Said once when a reply hands over the body of a file the run never wrote.
WRITE_NUDGE = ("The task asked for that file and this reply only pastes its "
               "contents into the chat. Put it on disk now with "
               '{"name": "write_file", "args": {"path": "<path>", '
               '"content": "<the contents you just pasted>"}}, then answer.')


def _resolve_path(ctx, path):
    """Where a path spelled in a prompt or a call lands, comparable across turns."""
    return os.path.normcase(os.path.abspath(os.path.join(str(ctx.workspace), path)))


def _paths_asked_to_write(task):
    """The files a task promises to leave behind, workspace-relative."""
    out = []
    text = str(task or "")
    for match in _TASK_PATH.finditer(text):
        window = text[max(0, match.start() - 60):match.start()]
        cut = max([0] + [window.rfind(mark) + 1 for mark in _CLAUSE_MARKS]
                  + [window.rfind("then")])
        if _WRITE_VERB.search(window[max(0, cut):]):
            path = match.group(1).strip("\"'`")
            if path not in out:
                out.append(path)
    return out


def _pasted_an_unwritten_file(text, left, names):
    """Whether a reply pastes the body of a file the run still has not written.

    The contents have to arrive in a block, because that is how the model hands
    over a file it means to leave, and `write_file` has to be advertised, because
    a run with writes switched off cannot be asked to write. Once the file is on
    disk, the same block in the closing reply is the summary the workflow asks
    for -- so this only ever fires on a deliverable that was never delivered.
    """
    return bool(left) and "write_file" in names and "```" in (text or "")


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