"""Context economy for the agent loop.

Prefill dominates a turn on the Fennec engine, so every character the loop
re-sends each step costs real seconds. The UI keeps the full transcript; only the
payload sent to the model is compacted:

* per-result head/tail elision, so a truncated JSON blob keeps its beginning and
  its end instead of being cut mid-value,
* duplicate results become a short reference (the agent re-reads the same file or
  re-runs the same listing often),
* a total character budget shrinks the oldest tool results first, never the
  system prompt, the user task or the newest result.

Tool behaviour, permissions and the system prompt are untouched.
"""
from __future__ import annotations

import hashlib
import json
import os

DEFAULT_BUDGET_CHARS = 48_000
DEFAULT_RESULT_CHARS = 6_000
MIN_RESULT_CHARS = 600
MARKER = "\n…[{removed} chars elided; narrow the read or grep for the rest]\n"


def _positive_int(value, default, floor):
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return max(floor, parsed) if parsed > 0 else default


def limits(environ=None):
    """Character budget and per-result cap; the FENNEC_AGENT_* variables win."""
    env = os.environ if environ is None else environ
    result = _positive_int(env.get("FENNEC_AGENT_RESULT_CHARS", DEFAULT_RESULT_CHARS),
                           DEFAULT_RESULT_CHARS, MIN_RESULT_CHARS)
    budget = _positive_int(env.get("FENNEC_AGENT_CONTEXT_BUDGET", DEFAULT_BUDGET_CHARS),
                           DEFAULT_BUDGET_CHARS, 2_000)
    return {"budget_chars": max(budget, result), "result_chars": result}


def elide(text, limit):
    """Keep the head and the tail of `text`, marking what was removed."""
    if not isinstance(text, str) or len(text) <= limit:
        return text
    head = max(1, (limit * 2) // 3)
    tail = max(0, limit - head - 80)
    removed = len(text) - head - tail
    return text[:head] + MARKER.format(removed=removed) + (text[-tail:] if tail else "")


def _size(message):
    content = message.get("content")
    total = len(content) if isinstance(content, str) else 0
    calls = message.get("tool_calls")
    if calls:
        total += len(json.dumps(calls, ensure_ascii=False))
    return total


def compact(messages, *, budget_chars=DEFAULT_BUDGET_CHARS, result_chars=DEFAULT_RESULT_CHARS,
            seen=None, floor_chars=MIN_RESULT_CHARS):
    """Return ``(payload, stats)``; never mutates ``messages``.

    ``seen`` maps a result hash to the tool call that produced it, so duplicates
    are caught across steps too; pass the same dict on every call in one run.
    """
    seen = {} if seen is None else seen
    before = sum(_size(message) for message in messages)
    payload = []
    elided = deduped = 0
    for message in messages:
        copy = dict(message)
        content = copy.get("content")
        if copy.get("role") == "tool" and isinstance(content, str) and content:
            digest = hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()
            label = copy.get("tool_call_id") or "an earlier call"
            if digest in seen:
                copy["content"] = (f"[identical to the result of {seen[digest]}; "
                                   f"{len(content)} chars not repeated]")
                deduped += 1
            else:
                seen[digest] = label
                capped = elide(content, result_chars)
                if capped != content:
                    copy["content"] = capped
                    elided += 1
        payload.append(copy)

    total = sum(_size(message) for message in payload)
    for message in payload:
        if total <= budget_chars:
            break
        content = message.get("content")
        if message.get("role") != "tool" or not isinstance(content, str) or len(content) <= floor_chars:
            continue
        target = max(floor_chars, len(content) - (total - budget_chars))
        if target >= len(content):
            continue
        message["content"] = elide(content, target)
        total = sum(_size(item) for item in payload)

    stats = {"chars_before": before, "chars_after": total, "elided": elided,
             "deduped": deduped, "budget_chars": budget_chars, "result_chars": result_chars}
    return payload, stats
