from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field

MAX_READ = 24000
MAX_RESULT = 7000


@dataclass
class ToolContext:
    workspace: str
    session_dir: str = ""
    allow_shell: bool = False
    exec_timeout: int = 30


def _safe(ctx: ToolContext, rel: str) -> str:
    rel = (rel or ".").strip()
    if os.path.isabs(rel):
        p = os.path.normpath(rel)
    else:
        p = os.path.normpath(os.path.join(ctx.workspace, rel))
    root = os.path.normpath(ctx.workspace)
    if p != root and not p.startswith(root + os.sep):
        raise ValueError(f"path '{rel}' escapes the workspace jail ({root})")
    return p


def _truncate(s: str, n: int = MAX_RESULT) -> str:
    return s if len(s) <= n else s[:n] + f"\n…[truncated {len(s)-n} chars]"


# --- handlers -------------------------------------------------------------
def list_dir(ctx, path="."):
    p = _safe(ctx, path)
    if not os.path.isdir(p):
        return {"ok": False, "error": f"not a directory: {path}"}
    entries = []
    for name in sorted(os.listdir(p)):
        full = os.path.join(p, name)
        entries.append({"name": name + ("/" if os.path.isdir(full) else ""),
                        "size": os.path.getsize(full) if os.path.isfile(full) else None})
    return {"ok": True, "path": os.path.relpath(p, ctx.workspace), "entries": entries}


def read_file(ctx, path):
    p = _safe(ctx, path)
    if not os.path.isfile(p):
        return {"ok": False, "error": f"not a file: {path}"}
    with open(p, encoding="utf-8", errors="replace") as f:
        text = f.read(MAX_READ + 1)
    lines = text.splitlines()
    numbered = "\n".join(f"{i+1}\t{ln}" for i, ln in enumerate(lines[:600]))
    return {"ok": True, "path": path, "lines": len(lines), "content": numbered}


def write_file(ctx, path, content):
    p = _safe(ctx, path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return {"ok": True, "path": path, "bytes": len(content.encode("utf-8"))}


def edit_file(ctx, path, old, new):
    p = _safe(ctx, path)
    if not os.path.isfile(p):
        return {"ok": False, "error": f"not a file: {path}"}
    with open(p, encoding="utf-8") as f:
        text = f.read()
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
    files = [root] if os.path.isfile(root) else [
        os.path.join(dp, fn) for dp, dn, fns in os.walk(root)
        for fn in fns if not fn.endswith((".pt", ".png", ".pyc", ".safetensors"))
        and ".git" not in dp]
    for fp in files[:2000]:
        try:
            with open(fp, encoding="utf-8", errors="ignore") as f:
                for i, ln in enumerate(f, 1):
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


def run_python(ctx, code):
    try:
        p = subprocess.run([sys.executable, "-c", code], cwd=ctx.workspace,
                           capture_output=True, text=True, timeout=ctx.exec_timeout)
        return {"ok": p.returncode == 0, "exit": p.returncode,
                "stdout": _truncate(p.stdout, 4000), "stderr": _truncate(p.stderr, 2000)}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {ctx.exec_timeout}s"}


def run_shell(ctx, command):
    if not ctx.allow_shell:
        return {"ok": False, "error": "shell disabled; enable 'allow shell' in the UI to run commands"}
    try:
        p = subprocess.run(command, cwd=ctx.workspace, shell=True, capture_output=True,
                           text=True, timeout=ctx.exec_timeout)
        return {"ok": p.returncode == 0, "exit": p.returncode,
                "stdout": _truncate(p.stdout, 4000), "stderr": _truncate(p.stderr, 2000)}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {ctx.exec_timeout}s"}


def git_status(ctx):
    p = subprocess.run(["git", "status", "--short"], cwd=ctx.workspace,
                       capture_output=True, text=True, timeout=15)
    return {"ok": p.returncode == 0, "status": p.stdout or "(clean)", "stderr": p.stderr[:500]}


def git_diff(ctx):
    p = subprocess.run(["git", "diff"], cwd=ctx.workspace,
                       capture_output=True, text=True, timeout=15)
    return {"ok": p.returncode == 0, "diff": _truncate(p.stdout) or "(no diff)"}


EMBED_CACHE = "D:/.cache/ncm-embed"
SKIP_EXT = (".pt", ".png", ".jpg", ".jpeg", ".gif", ".pyc", ".safetensors",
            ".exe", ".dll", ".zip", ".hnmod", ".gguf", ".bin")


def semantic_search(ctx, query, k=8):
    """Cosine search over workspace chunks embedded by the Intel NPU (bge-base)."""
    import hashlib
    import json
    import math

    from . import engine
    root = _safe(ctx, ".")
    os.makedirs(EMBED_CACHE, exist_ok=True)
    ws_key = hashlib.sha1(os.path.abspath(root).encode()).hexdigest()[:12]
    cache_path = os.path.join(EMBED_CACHE, ws_key + ".json")
    try:
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    except (OSError, ValueError):
        cache = {}

    files = []
    for dp, dn, fns in os.walk(root):
        dn[:] = [d for d in dn if d not in (".git", "__pycache__", "node_modules", ".venv")]
        for fn in fns:
            if fn.endswith(SKIP_EXT):
                continue
            fp = os.path.join(dp, fn)
            try:
                if os.path.getsize(fp) > 400_000:
                    continue
            except OSError:
                continue
            files.append(fp)
    chunks = []
    for fp in files[:120]:
        try:
            with open(fp, encoding="utf-8", errors="ignore") as f:
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
    "run_python": (run_python, "Execute Python code in an isolated sandbox (30s timeout) and return stdout/stderr. Use to verify code.",
                   {"code": {"type": "string"}}, ["code"]),
    "run_shell": (run_shell, "Run a shell command in the workspace (only if the user enabled 'allow shell').",
                  {"command": {"type": "string"}}, ["command"]),
    "git_status": (git_status, "git status --short of the workspace.", {}, []),
    "git_diff": (git_diff, "git diff of the workspace.", {}, []),
}


def openai_tools():
    out = []
    for name, (_, desc, props, required) in TOOLS.items():
        out.append({"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}})
    return out


def execute(name, args, ctx):
    if name not in TOOLS:
        return {"ok": False, "error": f"unknown tool '{name}'"}
    handler = TOOLS[name][0]
    try:
        return handler(ctx, **(args or {}))
    except TypeError as e:
        return {"ok": False, "error": f"bad arguments: {e}"}
    except Exception as e:
        return {"ok": False, "error": repr(e)}