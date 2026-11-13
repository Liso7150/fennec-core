#!/usr/bin/env python3
"""Measure SSE token delivery against a running fennec-desktop-shell.

Subscribes /api/events, triggers /api/stream_emit, times receipt of N tokens.
Browser-free, so it runs on a headless CI runner. Prints one JSON row.
"""
import json, sys, threading, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8184"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

received = []
t0 = None
done = threading.Event()

def listen():
    global t0
    r = urllib.request.urlopen(BASE + "/api/events", timeout=60)
    for line in r:
        line = line.decode().strip()
        if line.startswith("data:"):
            if t0 is None: t0 = time.time()
            received.append(1)
            if len(received) >= N:
                done.set()
                return

th = threading.Thread(target=listen, daemon=True)
th.start()
time.sleep(0.5)
urllib.request.urlopen(urllib.request.Request(
    BASE + "/api/stream_emit", data=json.dumps({"count": N}).encode(),
    headers={"Content-Type": "application/json"}), timeout=30).read()
done.wait(30)
ms = (time.time() - t0) * 1000 if t0 else 0
print(json.dumps({"shell": "chromium-sse", "backend": "stream-delivered",
                  "elapsed_ms": round(ms, 3), "messages": len(received),
                  "per_token_ms": round(ms / max(1, len(received)), 4)}))