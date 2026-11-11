"""Optional delegation of the ported glue to the Rust binary.

`fennec-rs serve` speaks one JSON request per line and answers in the same shape,
so the modules that used to do this arithmetic in CPython can hand their work
over. It is opt-in (`FENNEC_RUST=1`), because `bench_bridge` in
`benchmarks/bench.py` measured what the pipe actually costs on the shipped
workload: delegating every turn made a 24-turn replay several times slower than
CPython, since the parent has to re-render every byte the port has already seen
before the port can look at it. With the cold-only gate below that same replay is
a wash - 82.1 ms in CPython against 82.7 ms through the port, 0.993 - because the
warm turns stay in Python. What the port still wins is the first pass of a
transcript that is already huge: 8.08 MB / 409 messages measures 72.7 ms against
52.9 ms, 1.37x, and the crossover is 207 messages. That is why `compact` only
delegates when nothing has been seen yet and the payload is over the gate, and why
the default stays Python: both conditions together are rare.

Redaction is not delegated at all. The ported redactor is faster than CPython at
the work itself - 20 000 payloads in 18.6 ms against 63.8 - but it runs on one string
at a time, the pipe lost at every size up to 8 000 000 characters, and a live key
would have to travel down that pipe by hand: the child inherits a snapshot of the
environment taken at spawn, so a secret set afterwards would stay in the clear.

Every failure mode here is a fall-through, never an exception. A missing binary,
a child that dies mid-request or an answer that does not parse leaves the caller
with the Python result and stops asking again for the rest of the run.
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# The cold-pass break-even measured above: below this the pipe costs more than the
# Python loop even when there is nothing to dedup.
DEFAULT_MIN_CHARS = 8_000_000

_BINARY_NAMES = ("fennec-rs.exe", "fennec-rs")


def candidate_paths():
    """Where the binary may live, most specific first."""
    hint = os.environ.get("FENNEC_RS", "").strip()
    if hint and hint.lower() not in ("1", "auto", "yes", "on"):
        yield Path(hint)
        return
    for name in _BINARY_NAMES:
        yield HERE.parent / "rust" / "target" / "release" / name
    found = shutil.which("fennec-rs")
    if found:
        yield Path(found)


def binary_path():
    for path in candidate_paths():
        if path.is_file():
            return str(path)
    return None


def enabled() -> bool:
    """Off unless `FENNEC_RUST=1`: the measurements above are the reason."""
    return os.environ.get("FENNEC_RUST", "").strip().lower() in ("1", "on", "yes", "true")


def _positive_env(name: str, default: int) -> int:
    try:
        parsed = int(str(os.environ.get(name, "")).strip())
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def min_chars() -> int:
    return _positive_env("FENNEC_RUST_MIN_CHARS", DEFAULT_MIN_CHARS)


class Bridge:
    """One long-lived `fennec-rs serve` child, guarded by a lock.

    The child is stateless per request, so the only thing a restart costs is the
    next spawn; nothing has to be replayed.
    """

    def __init__(self, exe=None):
        self.exe = exe if exe is not None else binary_path()
        self._proc = None
        self._lock = threading.Lock()
        self._dead = False
        atexit.register(self.close)

    # -- pipe ownership ----------------------------------------------------
    def _start(self):
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        self._proc = None
        if not self.exe:
            self._dead = True
            return None
        flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
        try:
            self._proc = subprocess.Popen(
                [self.exe, "serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                errors="replace", creationflags=flags)
        except OSError:
            self._dead = True
        return self._proc

    def close(self):
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.stdin.write('{"op":"exit"}\n')
            proc.stdin.flush()
            proc.wait(timeout=2)
        except (OSError, ValueError, subprocess.SubprocessError):
            proc.kill()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass

    # -- one request -------------------------------------------------------
    def call(self, request: dict):
        """The child's answer, or None for 'use Python'.

        A dead child is not restarted on every call: the first failure marks the
        bridge dead so a run without the binary pays one spawn, not thousands.
        """
        if self._dead or not enabled():
            return None
        with self._lock:
            proc = self._start()
            if proc is None:
                return None
            try:
                proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
                if not line:
                    raise OSError("the rust bridge closed the pipe")
                answer = json.loads(line)
            except (OSError, ValueError):
                self._dead = True
                self.close()
                return None
            if not isinstance(answer, dict) or not answer.get("ok"):
                # a bad request is our bug, not the child's: fall back, stay alive
                return None
            return answer


_BRIDGE = None


def bridge():
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = Bridge()
    return _BRIDGE


def payload_chars(messages):
    """A cheap size gate: content only, no JSON rendering of tool calls."""
    total = 0
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            total += len(content)
    return total


def compact(messages, *, budget_chars, result_chars, floor_chars, seen):
    """`context.compact` through the bridge, or None.

    ``seen`` is updated in place, because the caller keeps one dict for the whole
    run and must not be able to tell which path produced the payload.
    """
    answer = bridge().call({"op": "compact", "messages": messages,
                           "budget_chars": budget_chars, "result_chars": result_chars,
                           "floor_chars": floor_chars, "seen": seen})
    if answer is None or not isinstance(answer.get("payload"), list):
        return None
    stats = answer.get("stats")
    if not isinstance(stats, dict):
        return None
    carried = answer.get("seen")
    if isinstance(carried, dict):
        seen.clear()
        seen.update({key: value for key, value in carried.items()
                     if isinstance(value, str)})
    return answer["payload"], stats
