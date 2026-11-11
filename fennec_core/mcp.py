"""Minimal stdio MCP client: enough to list and call another process's tools.

ACP hands these servers to the agent in ``session/new``. Only the ``stdio``
variant is implemented, which is what the registry marks as unsupported through
``mcpCapabilities{http: false, sse: false}``.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time

PROTOCOL_VERSION = "2024-11-05"
CALL_TIMEOUT_S = 120


class McpError(RuntimeError):
    pass


class StdioServer:
    def __init__(self, name: str, command: str, args=None, env=None, cwd=None):
        self.name = _safe_name(name)
        self.command = command
        self.args = list(args or [])
        self.env = dict(env or {})
        self.cwd = cwd
        self._process = None
        self._lock = threading.Lock()
        self._next_id = 0
        self.tools: dict = {}
        self._lines: queue.Queue = queue.Queue()
        self._pumped = threading.Event()

    def start(self):
        if self._process and self._process.poll() is None:
            return self
        self._lines = queue.Queue()
        self._pumped = threading.Event()
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        merged = dict(os.environ)
        merged.update(self.env)
        try:
            self._process = subprocess.Popen([self.command, *self.args], cwd=self.cwd, env=merged,
                                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=subprocess.DEVNULL, text=True,
                                             encoding="utf-8", bufsize=1, **options)
        except OSError as exc:
            raise McpError(f"{self.name}: cannot start ({exc})") from exc
        threading.Thread(target=self._pump, args=(self._process.stdout,), daemon=True).start()
        self._call("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                                  "clientInfo": {"name": "fennec", "version": "1.9.0"}}, timeout=20)
        self._notify("notifications/initialized", {})
        return self

    def _send(self, payload):
        assert self._process and self._process.stdin
        self._process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._process.stdin.flush()

    def _pump(self, stream) -> None:
        # readline() on a pipe cannot be given a deadline, and a server that
        # starts but never answers would hold the call lock forever. Drain the
        # pipe in the background and let callers wait on the queue instead.
        try:
            for line in stream:
                self._lines.put(line)
        except (OSError, ValueError):
            pass
        self._lines.put(None)
        self._pumped.set()

    def _call(self, method, params, timeout=None):
        # Read the deadline here, not in the signature default, so a caller (or a
        # test) can lower CALL_TIMEOUT_S without patching this function.
        timeout = CALL_TIMEOUT_S if timeout is None else timeout
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                raise McpError(f"{self.name}: server is not running")
            self._next_id += 1
            request_id = self._next_id
            try:
                self._send({"jsonrpc": "2.0", "id": request_id, "method": method,
                            "params": params})
            except (OSError, ValueError) as exc:
                raise McpError(f"{self.name}: write failed ({type(exc).__name__})") from exc
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.stop()
                    raise McpError(f"{self.name}: {method} timed out after {timeout}s")
                try:
                    line = self._lines.get(timeout=min(remaining, 0.5))
                except queue.Empty:
                    continue
                if line is None:
                    raise McpError(f"{self.name}: server closed the pipe")
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(message, dict) or message.get("id") != request_id:
                    continue
                if message.get("error"):
                    raise McpError(f"{self.name}: {message['error'].get('message')}")
                return message.get("result") or {}

    def _notify(self, method, params):
        try:
            self._send({"jsonrpc": "2.0", "method": method, "params": params})
        except (OSError, ValueError):
            pass

    def list_tools(self):
        result = self._call("tools/list", {})
        specs = {}
        for item in result.get("tools") or []:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            schema = item.get("inputSchema") or {}
            props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
            required = [key for key in (schema.get("required") or []) if isinstance(key, str)]
            tool_name = f"mcp__{self.name}__{_safe_name(item['name'])}"
            specs[tool_name] = (self._handler(item["name"]),
                                (item.get("description") or f"MCP tool {item['name']} "
                                 f"from {self.name}")[:600],
                                {key: value if isinstance(value, dict) else {}
                                 for key, value in props.items()},
                                [key for key in required if key in props])
        self.tools = specs
        return specs

    def _handler(self, remote_name):
        def call(ctx, **args):
            result = self._call("tools/call", {"name": remote_name, "arguments": args})
            parts = []
            for block in result.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text") or "")
                elif isinstance(block, dict):
                    parts.append(f"[{block.get('type', 'content')} omitted]")
            return {"ok": not bool(result.get("isError")),
                    "content": "\n".join(parts)[:12_000] or "(empty result)",
                    "structured": result.get("structuredContent")
                    if isinstance(result.get("structuredContent"), dict) else None}
        return call

    def stop(self):
        process, self._process = self._process, None
        if not process:
            return
        try:
            if process.stdin:
                process.stdin.close()
        except OSError:
            pass
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        # The reader thread only finishes once the pipe hits EOF, and closing a
        # pipe under a blocked reader is not something it forgives; skipping this
        # leaves the handle to a garbage collection that may never come.
        if self._pumped.wait(1.0) and process.stdout:
            try:
                process.stdout.close()
            except (OSError, ValueError):
                pass


def _safe_name(name) -> str:
    keep = [char if (char.isalnum() or char in "._-") else "_" for char in str(name)]
    return "".join(keep).strip("_")[:48] or "server"


class McpPool:
    """Servers for one ACP session; keyed by the client-supplied name."""

    def __init__(self):
        self.servers: dict[str, StdioServer] = {}
        self.failures: dict[str, str] = {}

    @classmethod
    def from_acp(cls, servers):
        pool = cls()
        for spec in servers or []:
            if not isinstance(spec, dict):
                continue
            kind = spec.get("type", "stdio")
            name = _safe_name(spec.get("name") or "server")
            if kind != "stdio":
                pool.failures[name] = f"{kind} transport is not supported (stdio only)"
                continue
            server = StdioServer(name, spec.get("command") or "", spec.get("args") or [],
                                 {item.get("name"): item.get("value")
                                  for item in (spec.get("env") or [])
                                  if isinstance(item, dict) and item.get("name")})
            pool.servers[name] = server
        return pool

    def tools(self):
        specs = {}
        for name, server in list(self.servers.items()):
            if server.tools:
                specs.update(server.tools)
                continue
            try:
                specs.update(server.start().list_tools())
            except McpError as exc:
                server.stop()
                self.servers.pop(name, None)
                self.failures[name] = str(exc)
        return specs

    def note(self) -> str:
        return "; ".join(f"{name}: {reason}" for name, reason in sorted(self.failures.items()))

    def stop(self):
        for server in self.servers.values():
            server.stop()
        self.servers.clear()
