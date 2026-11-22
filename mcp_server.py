"""Small, stateless MCP server for connecting Fennec to Nod."""
from __future__ import annotations

import asyncio
import hmac
import json
import os

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from outlets import redact
from tools import ToolContext, grep, list_dir, read_file

MCP_MAX_BODY = 1_000_000
MCP_PROTOCOLS = ("2025-06-18", "2024-11-05")
MCP_TOOLS = (
    {
        "name": "fennec_status",
        "description": "Read Fennec engine status without starting or changing anything.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "fennec_chat",
        "description": "Ask Fennec's configured local model a text-only question.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Question for the local model."},
                "max_tokens": {"type": "integer", "minimum": 1, "maximum": 4096},
                "temperature": {"type": "number", "minimum": 0, "maximum": 2},
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
    },
    {
        "name": "fennec_list_dir",
        "description": "List a directory inside Fennec's configured workspace jail.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Relative path, default '.'"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "fennec_read_file",
        "description": "Read a text file inside Fennec's configured workspace jail.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "fennec_grep",
        "description": "Search text files inside Fennec's configured workspace jail.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "description": "Relative directory, default '.'"},
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    },
)


def _error(request_id, code: int, message: str):
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _tool_result(value, *, failed=False):
    text = redact(json.dumps(value, ensure_ascii=False))
    try:
        safe_value = json.loads(text)
    except ValueError:
        safe_value = {"ok": False, "error": "tool result could not be encoded safely"}
        text = json.dumps(safe_value)
    result = {"content": [{"type": "text", "text": text}]}
    if isinstance(safe_value, dict):
        result["structuredContent"] = safe_value
    if failed:
        result["isError"] = True
    return result


def _protocol_version(params) -> str:
    requested = params.get("protocolVersion") if isinstance(params, dict) else None
    return requested if requested in MCP_PROTOCOLS else MCP_PROTOCOLS[-1]


def _response_protocol(payload) -> str:
    message = payload[0] if isinstance(payload, list) and payload else payload
    if isinstance(message, dict) and message.get("method") == "initialize":
        return _protocol_version(message.get("params") or {})
    return MCP_PROTOCOLS[0]


def _status(client):
    try:
        running = bool(client.is_running())
    except Exception:
        running = False
    try:
        status = client.status()
    except Exception as exc:
        status = {"error": redact(str(exc))}
    return {"ok": True, "engine_running": running, "engine": status}


def _run_tool(name: str, args: dict, config, client):
    if name == "fennec_status":
        if args:
            return {"ok": False, "error": "fennec_status takes no arguments"}, True
        return _status(client), False

    ctx = ToolContext(workspace=str(config.workspace))
    try:
        if name == "fennec_list_dir":
            return list_dir(ctx, args.get("path", ".")), False
        if name == "fennec_read_file":
            return read_file(ctx, args["path"], numbered=False), False
        if name == "fennec_grep":
            return grep(ctx, args["pattern"], args.get("path", ".")), False
        if name == "fennec_chat":
            prompt = args["prompt"]
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8000:
                return {"ok": False, "error": "prompt must be 1-8000 characters"}, True
            max_tokens = args.get("max_tokens", 1024)
            temperature = args.get("temperature", 0.2)
            if type(max_tokens) is not int or not 1 <= max_tokens <= 4096:
                return {"ok": False, "error": "max_tokens must be an integer from 1 to 4096"}, True
            if type(temperature) not in (int, float) or not 0 <= temperature <= 2:
                return {"ok": False, "error": "temperature must be a number from 0 to 2"}, True
            if not client.is_running():
                return {"ok": False, "error": "Fennec engine is not running"}, True
            message, usage = client.chat([{"role": "user", "content": prompt.strip()}],
                                         max_tokens=max_tokens, temperature=float(temperature))
            content = message.get("content", "") if isinstance(message, dict) else str(message)
            return {"ok": True, "answer": redact(str(content)), "usage": usage or {}}, False
    except KeyError as exc:
        return {"ok": False, "error": f"missing argument: {exc.args[0]}"}, True
    except (OSError, PermissionError, ValueError) as exc:
        return {"ok": False, "error": redact(str(exc))}, True
    except Exception as exc:
        return {"ok": False, "error": redact(f"Fennec MCP tool failed: {exc}")}, True
    return {"ok": False, "error": f"unknown tool: {name}"}, True


def _dispatch(message, config, client):
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return _error(message.get("id") if isinstance(message, dict) else None, -32600,
                       "invalid JSON-RPC request")
    request_id = message.get("id")
    method = message.get("method")
    params = message.get("params") or {}
    if not isinstance(method, str) or not isinstance(params, dict):
        return _error(request_id, -32600, "method and params must be valid")
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": request_id, "result": {
            "protocolVersion": _protocol_version(params),
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "fennec", "version": "1.0.0"},
            "instructions": "Fennec exposes read-only workspace tools and local-model chat.",
        }}
    if method in {"notifications/initialized", "notifications/cancelled"}:
        return None
    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": list(MCP_TOOLS)}}
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if not isinstance(name, str) or not isinstance(args, dict):
            return _error(request_id, -32602, "tools/call needs a tool name and object arguments")
        value, failed = _run_tool(name, args, config, client)
        return {"jsonrpc": "2.0", "id": request_id,
                "result": _tool_result(value, failed=failed)}
    if method.startswith("notifications/"):
        return None
    return _error(request_id, -32601, f"method not found: {method}")


def _authorized(request: Request) -> bool:
    token = os.environ.get("FENNEC_MCP_TOKEN", "")
    if not token:
        return False
    supplied = request.headers.get("authorization", "")
    supplied = supplied[7:] if supplied.lower().startswith("bearer ") else ""
    return hmac.compare_digest(supplied, token)


async def handle(request: Request, config, client):
    """Handle one stateless Streamable HTTP MCP request."""
    if not os.environ.get("FENNEC_MCP_TOKEN"):
        return Response(status_code=404)
    if not _authorized(request):
        return JSONResponse({"detail": "Invalid or missing MCP token"}, status_code=401)
    raw = await request.body()
    if len(raw) > MCP_MAX_BODY:
        return JSONResponse({"detail": "MCP request is too large"}, status_code=413)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return JSONResponse(_error(None, -32700, "parse error"), status_code=400)
    messages = payload if isinstance(payload, list) else [payload]
    if not messages or len(messages) > 20:
        return JSONResponse(_error(None, -32600, "invalid JSON-RPC batch"), status_code=400)
    responses = await asyncio.gather(*(asyncio.to_thread(_dispatch, item, config, client)
                                       for item in messages))
    responses = [item for item in responses if item is not None]
    if not responses:
        return Response(status_code=202)
    body = responses[0] if not isinstance(payload, list) else responses
    return JSONResponse(body, headers={"MCP-Protocol-Version": _response_protocol(payload)})
