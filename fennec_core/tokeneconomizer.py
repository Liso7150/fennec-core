"""Shared Fennec core: optional TokenEconomizer bridge (micro-skills, loop guard, folding).

Fennec's own context economy stays where it is: this module only adds what
TokenEconomizer contributes — learned repository facts at session start, a
redirect when a turn keeps hitting the same failure, and a folded session at the
end so the next one starts from what this one learned.

Everything here fails open. Without a TokenEconomizer checkout the bridge is a
no-op, and any error inside it is swallowed so a session never breaks.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

_CANDIDATES = ("FENNEC_TOKENECONOMIZER", "TE_CORE_PATH")
_HOOK = None
_LOADED = False
_FOLD_LOCK = threading.Lock()
_ORIGINAL_SPAWN = None


def _is_core(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "tokeneconomizer", "plugin", "hook.py"))


def _requested() -> bool:
    """TokenEconomizer runs only where it was installed for Fennec.

    A pip copy that happens to be importable must not switch it on for every
    workspace, and tests that sandbox the home directory stay untouched.
    """
    if os.environ.get("FENNEC_TOKENECONOMIZER") or os.environ.get("TE_CORE_PATH"):
        return True
    return (Path.home() / ".fennec" / "tokeneconomizer.path").is_file()


def _core() -> str | None:
    for name in _CANDIDATES:
        cand = os.environ.get(name)
        if cand and _is_core(cand):
            return cand
    try:
        cand = (Path.home() / ".fennec" / "tokeneconomizer.path").read_text(encoding="utf-8").strip()
        if cand and _is_core(cand):
            return cand
    except OSError:
        pass
    for cand in (Path.home() / "TokenEconomizer", Path.home() / ".tokeneconomizer" / "core"):
        if _is_core(str(cand)):
            return str(cand)
    return None


def _load():
    global _HOOK, _LOADED
    if _LOADED:
        return _HOOK
    _LOADED = True
    if not _requested():
        return None
    core = _core()
    if core and core not in sys.path:
        sys.path.insert(0, core)
    try:
        if core:
            from tokeneconomizer.plugin import hook, hosts
        else:
            import tokeneconomizer  # noqa: F401
            from tokeneconomizer.plugin import hook, hosts
    except Exception:
        return None
    try:
        class Fennec(hosts.ClaudeCode):
            name = "fennec"
            interrupt = "Press Stop in Fennec Desktop to interrupt it."
            keeps_history = False

            def history(self, env, extra_dirs=()):
                return []

            def headless(self, env, transcript=None):
                return False

        hosts.HOSTS["fennec"] = Fennec
    except Exception:
        return None
    _HOOK = hook
    return _HOOK


def available() -> bool:
    return _load() is not None


def _install_inline_fold():
    """Fold in this process instead of a detached `te fold` child.

    A child spawned with the workspace as its cwd keeps that directory open on
    Windows, so a session started in a temporary directory could not be removed.
    """
    global _ORIGINAL_SPAWN
    from tokeneconomizer.plugin import fold as te_fold
    if _ORIGINAL_SPAWN is None:
        _ORIGINAL_SPAWN = te_fold.spawn_fold
    te_fold.spawn_fold = lambda ctx, ids: [te_fold.fold_session(ctx, sid) for sid in ids]


def _restore_spawn():
    if _ORIGINAL_SPAWN is None:
        return
    try:
        from tokeneconomizer.plugin import fold as te_fold
        te_fold.spawn_fold = _ORIGINAL_SPAWN
    except Exception:
        pass


def _usage(usage) -> dict:
    u = usage if isinstance(usage, dict) else {}
    return {"input_tokens": int(u.get("prompt_tokens") or 0),
            "output_tokens": int(u.get("completion_tokens") or 0),
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0}


def _iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + ".000Z"


class Bridge:
    """One Fennec session's connection to TokenEconomizer."""

    def __init__(self, workspace, session_id: str, task: str = ""):
        self.workspace = str(workspace)
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.task = task or ""
        self._hook = _load() if _requested() else None
        self._enabled = self._hook is not None
        self._mid = 0
        self.transcript = str(Path.home() / ".fennec" / "sessions" / f"{self.session_id}.jsonl")
        if self._enabled:
            try:
                os.makedirs(os.path.dirname(self.transcript), exist_ok=True)
            except OSError:
                self._enabled = False

    def _base(self, kind: str) -> dict:
        return {"type": kind, "sessionId": self.session_id, "cwd": self.workspace,
                "timestamp": _iso(), "uuid": uuid.uuid4().hex,
                "parentUuid": None, "isSidechain": False, "userType": "external",
                "version": "fennec", "gitBranch": "HEAD"}

    def _append(self, entry: dict) -> None:
        if not self._enabled:
            return
        try:
            os.makedirs(os.path.dirname(self.transcript), exist_ok=True)
            with open(self.transcript, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _payload(self, event: str, **extra) -> dict:
        payload = {"session_id": self.session_id, "transcript_path": self.transcript,
                   "cwd": self.workspace, "hook_event_name": event,
                   "prompt_id": self.session_id}
        payload.update(extra)
        return payload

    def _call(self, payload: dict) -> str:
        if not self._enabled:
            return ""
        try:
            out = self._hook.run(payload, os.environ, host_hint="fennec") or {}
        except Exception:
            return ""
        if not isinstance(out, dict):
            return ""
        specific = out.get("hookSpecificOutput") or {}
        return specific.get("additionalContext") or out.get("systemMessage") or ""

    def start(self) -> str:
        if not self._enabled:
            return ""
        user = self._base("user")
        user["message"] = {"role": "user", "content": [{"type": "text", "text": self.task}]}
        self._append(user)
        try:
            first = self._call(self._payload("SessionStart", source="startup"))
            then = self._call(self._payload("UserPromptSubmit", prompt=self.task))
        except Exception:
            return ""
        return "\n\n".join(part for part in (first, then) if part)

    def assistant(self, text: str, calls, usage=None) -> None:
        if not self._enabled:
            return
        self._mid += 1
        content = []
        if text:
            content.append({"type": "text", "text": text})
        for call_id, name, args in calls:
            content.append({"type": "tool_use", "id": call_id, "name": name, "input": args})
        if not content:
            content.append({"type": "text", "text": ""})
        entry = self._base("assistant")
        entry["message"] = {"role": "assistant", "id": f"fennec-{self.session_id}-{self._mid}",
                            "model": "fennec", "content": content, "usage": _usage(usage)}
        self._append(entry)

    def tool_result(self, call_id: str, text: str, ok: bool = True) -> None:
        if not self._enabled:
            return
        entry = self._base("user")
        entry["message"] = {"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": call_id,
            "content": [{"type": "text", "text": text}], "is_error": not ok}]}
        entry["toolUseResult"] = {"ok": bool(ok)}
        self._append(entry)

    def after_tool(self, name: str, args, text: str, ok: bool) -> str:
        event = "PostToolUse" if ok else "PostToolUseFailure"
        return self._call(self._payload(event, tool_name=name, tool_input=args,
                                        tool_response={"output": text,
                                                       "exit_code": 0 if ok else 1}))

    def end(self, reason: str = "exit") -> None:
        if not self._enabled:
            return
        with _FOLD_LOCK:
            try:
                _install_inline_fold()
                self._call(self._payload("Stop", stop_hook_active=False))
                self._call(self._payload("SessionEnd", reason=reason))
            except Exception:
                pass
            finally:
                _restore_spawn()
