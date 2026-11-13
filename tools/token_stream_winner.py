#!/usr/bin/env python3
"""Compare the SSE and IPC token-stream rows and name the winner."""
import json, os, sys

sse = json.load(open(sys.argv[1], encoding="utf-8"))
ipc = None
if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
    rows = [json.loads(l) for l in open(sys.argv[2], encoding="utf-8") if l.strip()]
    got = [r for r in rows if r.get("backend") == "stream-delivered"]
    ipc = got[0] if got else None

print(f"SSE  delivered_ms={sse['elapsed_ms']} per_token={sse['per_token_ms']}")
if ipc:
    per = round(ipc["elapsed_ms"] / max(1, ipc["messages"]), 4)
    print(f"IPC  delivered_ms={ipc['elapsed_ms']} per_token={per}")
    print("WINNER=", "IPC (WebView2)" if ipc["elapsed_ms"] < sse["elapsed_ms"] else "SSE (Chromium)")
else:
    print("IPC cell did not run on this runner; SSE only")