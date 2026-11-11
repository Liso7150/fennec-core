"""Drive an installed ``jev-ultrafast`` clone without touching the clone.

Fennec owns this file; upstream owns ``jev_ultrafast.Agent``. It prints its report
as one JSON object behind a sentinel, so a chatty browser log cannot hide it from
the caller. Run it with the interpreter that can import the clone, e.g.
``uv run --quiet --project <jev-root> python <site-packages>/fennec_core/jev_driver.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

# The marker fennec_core.outlets looks for before its JSON. The driver runs inside
# the Jev checkout's own environment, so it must not import from fennec_core to get it.
SENTINEL = "@@@FENNEC-JEV-REPORT@@@"


def _emit(report: dict) -> None:
    """Print the report behind a marker, so upstream noise cannot hide it."""
    sys.stdout.write(SENTINEL)
    sys.stdout.flush()
    json.dump(report, sys.stdout)
    sys.stdout.flush()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="start page of the task")
    parser.add_argument("--goal", required=True, help="end state to reach, not the hops")
    parser.add_argument("--max-steps", type=int, default=0, help="stop after N snapshots")
    parser.add_argument("--timeout", type=float, default=240.0, help="wall-clock seconds")
    parser.add_argument("--screenshots", action="store_true")
    args = parser.parse_args(argv)

    deadline = time.monotonic() + args.timeout
    try:
        from jev_ultrafast import Agent
    except ImportError as exc:
        _emit({"ok": False, "error": f"jev_ultrafast is not importable: {exc}"})
        return 2

    steps, last = 0, None
    try:
        with Agent(args.url, args.goal, screenshots=bool(args.screenshots)) as agent:
            for state in agent.run():
                last, steps = state, steps + 1
                if args.max_steps and steps >= args.max_steps:
                    break
                if time.monotonic() > deadline:
                    break
                if isinstance(state, dict) and state.get("status") in {"done", "blocked"}:
                    break
    except Exception as exc:  # upstream owns its failures; report, never mask
        _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}", "steps": steps})
        return 1

    page = (last or {}).get("page") or {}
    _emit({"ok": True, "steps": steps, "status": (last or {}).get("status"),
           "elapsed_ms": (last or {}).get("elapsed_ms"), "url": page.get("url"),
           "title": page.get("title"), "decision": (last or {}).get("decision"),
           "history": ((last or {}).get("history") or [])[-12:],
           "text": (page.get("text") or "")[:4000]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
