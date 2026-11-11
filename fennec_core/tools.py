from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from itertools import islice

from .outlets import SECRET_ENV_KEYS

MAX_READ = 24000
MAX_RESULT = 7000


@dataclass
class ToolContext:
    workspace: str
    session_dir: str = ""
    allow_shell: bool = False
    exec_timeout: int = 30
    allow_write: bool = False
    allow_npu: bool = False
    allow_web: bool = False
    require_approval: bool = False
    engine_client: object = None
    cache_root: str = ""
    outlets: object = None
    extra: object = None


_DENIED = {".git", ".ssh", ".aws", ".azure", ".kube", ".gnupg", ".docker",
           ".npmrc", ".pypirc", ".netrc", "credentials", "credentials.json",
           "authorized_keys", "known_hosts", "id_rsa", "id_ed25519", "id_dsa", "id_ecdsa"}


def _credential(part):
    name = part.casefold().rstrip(" .")
    return (name in _DENIED or name == ".env" or name.startswith(".env.")
            or re.search(r"(^|[._-])(tokens?|secrets?|credentials?)([._-]|$)", name)
            or name.endswith((".pem", ".key", ".p12", ".pfx")))


def _safe(ctx: ToolContext, rel: str) -> str:
    """Resolve symlinks/junctions before checking the boundary and credential names.

    This is a filesystem guard, not an OS sandbox against concurrent hostile
    filesystem mutation or explicitly permitted subprocesses.
    """
    if not isinstance(rel, str) or "\x00" in rel:
        raise ValueError("invalid path")
    root = os.path.realpath(ctx.workspace)
    candidate = os.path.abspath(os.path.join(root, rel or "."))
    resolved = os.path.realpath(candidate)
    for path in (candidate, resolved):
        try:
            inside = os.path.commonpath([os.path.normcase(root), os.path.normcase(path)]) == os.path.normcase(root)
        except ValueError:
            inside = False
        if not inside:
            raise PermissionError("path escapes the configured workspace")
        parts = Path(os.path.relpath(path, root)).parts
        if any(_credential(part) or ":" in part or part.endswith((" ", ".")) and part != "." for part in parts):
            raise PermissionError("credential or reserved path is not accessible")
    return resolved


def _walk_files(ctx, path=".", limit=2000):
    root = _safe(ctx, path)
    if os.path.isfile(root):
        yield root
        return
    visited = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        permitted = []
        for name in dirs:
            try:
                child = _safe(ctx, os.path.join(directory, name))
                if (name not in {"__pycache__", "node_modules", ".venv"}
                        and not os.path.islink(os.path.join(directory, name))
                        and not getattr(os.path, "isjunction", lambda p: False)(os.path.join(directory, name))):
                    permitted.append(name)
            except (OSError, ValueError, PermissionError):
                continue
        dirs[:] = permitted
        for name in files:
            visited += 1
            if visited > limit:
                return
            try:
                fp = _safe(ctx, os.path.join(directory, name))
                if os.path.isfile(fp) and not fp.lower().endswith(SKIP_EXT) and os.path.getsize(fp) <= 400_000:
                    yield fp
            except (OSError, ValueError, PermissionError):
                continue


def _truncate(s: str, n: int = MAX_RESULT) -> str:
    return s if len(s) <= n else s[:n] + f"\n…[truncated {len(s)-n} chars]"


# --- handlers -------------------------------------------------------------
def list_dir(ctx, path="."):
    p = _safe(ctx, path)
    if not os.path.isdir(p):
        return {"ok": False, "error": f"not a directory: {path}"}
    entries = []
    with os.scandir(p) as children:
        for entry in islice(children, 2000):
            try:
                full = _safe(ctx, entry.path)
                entries.append({"name": entry.name + ("/" if os.path.isdir(full) else ""),
                                "size": os.path.getsize(full) if os.path.isfile(full) else None})
            except (OSError, ValueError, PermissionError):
                continue
            if len(entries) >= 500:
                break
    entries.sort(key=lambda entry: entry["name"])
    return {"ok": True, "path": os.path.relpath(p, ctx.workspace), "entries": entries}


def read_file(ctx, path, numbered=True):
    p = _safe(ctx, path)
    if not os.path.isfile(p):
        return {"ok": False, "error": f"not a file: {path}"}
    with open(p, encoding="utf-8", errors="replace") as f:
        text = f.read(MAX_READ + 1)
    lines = text.splitlines()
    if not numbered:
        return {"ok": True, "path": path, "lines": len(lines), "content": _truncate(text, MAX_READ)}
    numbered_text = "\n".join(f"{i+1}\t{ln}" for i, ln in enumerate(lines[:600]))
    return {"ok": True, "path": path, "lines": len(lines), "content": numbered_text}


def write_file(ctx, path, content):
    if not ctx.allow_write:
        raise PermissionError("writes require allow_write")
    if not isinstance(content, str) or len(content) > 1_000_000:
        raise ValueError("content must be text of at most 1000000 characters")
    p = _safe(ctx, path)
    if os.path.exists(p) and not os.path.isfile(p):
        raise ValueError("not a regular file")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    p = _safe(ctx, path)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return {"ok": True, "path": path, "bytes": len(content.encode("utf-8"))}


def edit_file(ctx, path, old, new):
    if not ctx.allow_write:
        raise PermissionError("edits require allow_write")
    if not isinstance(old, str) or not old or not isinstance(new, str) or len(new) > 1_000_000:
        raise ValueError("old must be nonempty text and new at most 1000000 characters")
    p = _safe(ctx, path)
    if not os.path.isfile(p):
        return {"ok": False, "error": f"not a file: {path}"}
    with open(p, encoding="utf-8") as f:
        text = f.read(1_000_001)
    if len(text) > 1_000_000:
        raise ValueError("file is too large to edit")
    count = text.count(old)
    if count == 0:
        return {"ok": False, "error": "old_string not found"}
    if count > 1:
        return {"ok": False, "error": f"old_string matches {count} times; make it unique"}
    with open(p, "w", encoding="utf-8") as f:
        f.write(text.replace(old, new, 1))
    return {"ok": True, "path": path, "replaced": 1}


def grep(ctx, pattern, path="."):
    root = _safe(ctx, path)
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return {"ok": False, "error": f"bad regex: {e}"}
    hits = []
    for fp in _walk_files(ctx, root):
        try:
            with open(_safe(ctx, fp), encoding="utf-8", errors="ignore") as f:
                for i, ln in enumerate(f.read(MAX_READ).splitlines(), 1):
                    if rx.search(ln):
                        hits.append({"file": os.path.relpath(fp, ctx.workspace),
                                     "line": i, "text": ln.rstrip()[:200]})
                        if len(hits) >= 200:
                            break
        except OSError:
            continue
        if len(hits) >= 200:
            break
    return {"ok": True, "pattern": pattern, "count": len(hits), "hits": hits}


def _run_process(ctx, command, *, shell=False):
    if not ctx.allow_shell:
        raise PermissionError("process execution requires allow_shell; it is NOT sandboxed")
    # Anything the model can make us spawn is third-party code on this machine.
    env = {key: value for key, value in os.environ.items() if key not in SECRET_ENV_KEYS}
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            process = subprocess.run(command, cwd=_safe(ctx, "."), shell=shell,
                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, env=env,
                timeout=ctx.exec_timeout, **options)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timeout after {ctx.exec_timeout}s; spawned descendants may still run"}
        stdout.seek(0)
        stderr.seek(0)
        return {"ok": process.returncode == 0, "exit": process.returncode,
                "stdout": _truncate(stdout.read(4001).decode("utf-8", "replace"), 4000),
                "stderr": _truncate(stderr.read(2001).decode("utf-8", "replace"), 2000)}


def run_python(ctx, code):
    return _run_process(ctx, [sys.executable, "-c", code])


def run_shell(ctx, command):
    return _run_process(ctx, command, shell=True)


def git_status(ctx):
    result = _run_process(ctx, ["git", "--no-optional-locks", "status", "--short", "--", "."])
    return {**result, "status": result.get("stdout", "")}


def git_diff(ctx):
    # Literal safe files only; never allow git's external diff/textconv drivers.
    if not ctx.allow_shell:
        raise PermissionError("git execution requires allow_shell")
    paths = [os.path.relpath(fp, ctx.workspace) for fp in _walk_files(ctx)]
    if not paths:
        return {"ok": True, "diff": "(no accessible files)"}
    result = _run_process(ctx, ["git", "--literal-pathspecs", "--no-optional-locks", "diff",
                               "--no-ext-diff", "--no-textconv", "--", *paths])
    return {**result, "diff": result.get("stdout", "")}

SKIP_EXT = (".pt", ".png", ".jpg", ".jpeg", ".gif", ".pyc", ".safetensors",
            ".exe", ".dll", ".zip", ".hnmod", ".gguf", ".bin")


def semantic_search(ctx, query, k=8):
    """Cosine search over workspace chunks embedded by the Intel NPU (bge-base)."""
    import hashlib
    import json
    import math

    if not ctx.allow_npu:
        raise PermissionError("semantic search requires explicitly enabled NPU support")
    from .engine import get_engine
    engine = ctx.engine_client or get_engine()
    root = _safe(ctx, ".")
    cache_root = Path(ctx.cache_root or engine.config.cache_root) / "fennec-embed"
    cache_root.mkdir(parents=True, exist_ok=True)
    ws_key = hashlib.sha1(os.path.abspath(root).encode()).hexdigest()[:12]
    cache_path = cache_root / (ws_key + ".json")
    try:
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    except (OSError, ValueError):
        cache = {}

    chunks = []
    for fp in islice(_walk_files(ctx), 120):
        try:
            with open(_safe(ctx, fp), encoding="utf-8", errors="ignore") as f:
                lines = f.read(24000).splitlines()
        except OSError:
            continue
        rel = os.path.relpath(fp, root)
        for i in range(0, max(1, len(lines)), 40):
            block = "\n".join(lines[i:i + 40]).strip()
            if len(block) < 20:
                continue
            key = hashlib.sha1((rel + "\n" + block).encode("utf-8", "ignore")).hexdigest()
            chunks.append({"file": rel, "line": i + 1,
                           "text": block[:1200], "key": key})
    todo = [c for c in chunks if c["key"] not in cache][:80]
    if todo:
        try:
            vecs = engine.embed([c["text"] for c in todo])
        except Exception as e:
            return {"ok": False, "error": f"NPU embedding failed: {e!r}"}
        for c, v in zip(todo, vecs):
            cache[c["key"]] = v
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f)
        except OSError:
            pass
    try:
        qv = engine.embed([query])[0]
    except Exception as e:
        return {"ok": False, "error": f"NPU embedding failed: {e!r}"}
    hits = []
    for c in chunks:
        v = cache.get(c["key"])
        if not v:
            continue
        num = sum(x * y for x, y in zip(qv, v))
        da = math.sqrt(sum(x * x for x in qv))
        db = math.sqrt(sum(y * y for y in v))
        hits.append((num / (da * db + 1e-9), c))
    hits.sort(key=lambda x: x[0], reverse=True)
    return {"ok": True, "device": engine.npu_health(timeout=1).get("device", "?"),
            "chunks_indexed": len(chunks), "embedded_now": len(todo),
            "results": [{"file": c["file"], "line": c["line"],
                         "score": round(s, 4), "preview": c["text"][:220]}
                        for s, c in hits[:max(1, min(k, 20))]]}


# --- optional web-agent outlets (Jev / OpenBot) ---------------------------
OUTLET_TOOLS = ("web_task", "coworker_task")


def web_task(ctx, goal, url="", max_steps=12):
    """One URL-scoped end state handed to the local Jev browser agent."""
    if ctx.outlets is None:
        return {"ok": False, "error": "no web outlets configured for this session"}
    return ctx.outlets.web_task(goal, url, max_steps=int(max_steps or 12))


def coworker_task(ctx, goal, agent=""):
    """One task delegated to a coworker inside the local OpenBot workspace."""
    if ctx.outlets is None:
        return {"ok": False, "error": "no web outlets configured for this session"}
    return ctx.outlets.coworker_task(goal, agent)


# --- registry (everything-is-a-plugin, harness style) ---------------------
TOOLS = {
    "list_dir": (list_dir, "List files and subdirectories in a workspace path.",
                 {"path": {"type": "string", "description": "relative dir, default '.'"}}, []),
    "read_file": (read_file, "Read a text file (returns numbered lines).",
                  {"path": {"type": "string"}}, ["path"]),
    "write_file": (write_file, "Create or overwrite a file with content.",
                   {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    "edit_file": (edit_file, "Replace a unique old_string with new_string in a file.",
                  {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}},
                  ["path", "old", "new"]),
    "grep": (grep, "Regex search across workspace files; returns file:line matches.",
             {"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"]),
    "semantic_search": (semantic_search,
                        "Natural-language code search: embeds workspace chunks on the Intel NPU "
                        "and returns the closest file:line matches for a described intent.",
                        {"query": {"type": "string"}, "k": {"type": "integer"}}, ["query"]),
    "run_python": (run_python, "Execute Python with host-user privileges (NOT sandboxed). Requires allow_shell; can write outside the workspace.",
                   {"code": {"type": "string"}}, ["code"]),
    "run_shell": (run_shell, "Run a shell command in the workspace (only if the user enabled 'allow shell').",
                  {"command": {"type": "string"}}, ["command"]),
    "git_status": (git_status, "git status --short of the workspace.", {}, []),
    "git_diff": (git_diff, "git diff of the workspace.", {}, []),
    "web_task": (web_task,
                 "Delegate one URL-scoped browsing goal to the local Jev browser agent "
                 "(browser-use/jev-ultrafast). Describe the END state, not the clicks. Only "
                 "available when that outlet is installed on this machine; it uses its own "
                 "TypeSafe policy key and never reads workspace credentials.",
                 {"goal": {"type": "string", "description": "end state to reach"},
                  "url": {"type": "string", "description": "start page, https:// or http://"},
                  "max_steps": {"type": "integer", "description": "snapshot budget, default 12"}},
                 ["goal"]),
    "coworker_task": (coworker_task,
                      "Ask one coworker in the local OpenBot workspace (CopilotKit/OpenBot) to "
                      "do a task with its own browser and files, and return what it reported. "
                      "Only available when a loopback OpenBot answers.",
                      {"goal": {"type": "string", "description": "the task, in one request"},
                       "agent": {"type": "string", "description": "OpenBot agent id or name"}},
                      ["goal"]),
}


def _extras(ctx):
    """Session-scoped tools an embedder injected, e.g. MCP servers over an ACP session."""
    return getattr(ctx, "extra", None) or {}


def _available(name, ctx):
    """Hard availability, as opposed to permission: no outlet, no tool."""
    if name in _extras(ctx):
        return True
    if name == "semantic_search":
        return ctx.allow_npu
    if name in OUTLET_TOOLS:
        outlets = getattr(ctx, "outlets", None)
        return bool(outlets is not None and outlets.available(name))
    return True


# Which session flag unlocks each tool. With an approver attached these stay
# unlocked and every gated call is confirmed instead: the user, not the flag,
# decides — that is how ACP clients expect permission to work.
_GATE = {"write_file": "allow_write", "edit_file": "allow_write",
         "run_python": "allow_shell", "run_shell": "allow_shell",
         "git_status": "allow_shell", "git_diff": "allow_shell",
         "web_task": "allow_web", "coworker_task": "allow_web"}


def _permitted(name, ctx):
    if name in _extras(ctx):
        return not ctx.require_approval and ctx.allow_shell
    gate = _GATE.get(name)
    if gate is None:
        return True
    if ctx.require_approval:
        return False
    return bool(getattr(ctx, gate))


def openai_tools(ctx=None, advertise_all=False):
    ctx = ctx or ToolContext(workspace=".")
    registry = dict(TOOLS)
    registry.update(_extras(ctx))
    out = []
    for name, (_, desc, props, required) in registry.items():
        if not _available(name, ctx):
            continue
        if not advertise_all and not _permitted(name, ctx):
            continue
        out.append({"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props,
                           "required": required, "additionalProperties": False}}})
    return out


def execute(name, args, ctx, *, confirm=None, allowed_always=None):
    extras = _extras(ctx)
    external = isinstance(name, str) and name in extras
    if not external and (not isinstance(name, str) or name not in TOOLS):
        return {"ok": False, "error": "unknown tool"}
    if not _available(name, ctx):
        return {"ok": False, "error": f"{name} is not available on this machine"}
    if not _permitted(name, ctx) and name not in (allowed_always or ()):
        if confirm is None:
            return {"ok": False, "error": f"permission denied for {name}"}
        decision = confirm(name, args if isinstance(args, dict) else {})
        if decision not in {"allow_once", "allow_always"}:
            return {"ok": False, "error": f"{name} was rejected by the user",
                    "permission": decision or "rejected"}
        if decision == "allow_always" and allowed_always is not None:
            allowed_always.add(name)
    if not isinstance(args, dict):
        return {"ok": False, "error": "tool arguments must be an object"}
    handler = extras[name][0] if external else TOOLS[name][0]
    try:
        result = handler(ctx, **args)
        import json
        if len(json.dumps(result)) > 32_000:
            return {"ok": result.get("ok", False), "truncated": True,
                    "content": _truncate(json.dumps(result), 30_000)}
        return result
    except TypeError as e:
        return {"ok": False, "error": f"bad arguments: {e}"}
    except Exception as e:
        return {"ok": False, "error": repr(e)}