"""Typed event envelope and domain models for the Fennec agent loop.

Every event the loop produces flows through one of these dataclasses before
reaching a shell (ACP, the PWA, or the desktop app). The types make the
contract explicit: a shell that pattern-matches on ``event.type`` gets a
typed object with known fields, not a loose dict.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal


@dataclass(frozen=True)
class PlanEntry:
    """One row in the plan rail the shells show."""
    content: str
    status: Literal["pending", "in_progress", "completed"]
    priority: Literal["low", "medium", "high"] = "medium"


@dataclass(frozen=True)
class PlanEvent:
    """The plan rail update: which phases happened, which is next."""
    type: Literal["plan"] = "plan"
    entries: list[PlanEntry] = field(default_factory=list)
    i: int = 0
    t: float = 0.0


@dataclass(frozen=True)
class PartialEvent:
    """A streamed text chunk before the turn completes."""
    type: Literal["partial"] = "partial"
    content: str = ""
    i: int = 0
    t: float = 0.0


@dataclass(frozen=True)
class TextEvent:
    """A complete text chunk (used by the sink channel)."""
    type: Literal["text"] = "text"
    text: str = ""
    i: int = 0
    t: float = 0.0


@dataclass(frozen=True)
class ReasoningEvent:
    """A reasoning/thinking chunk (used by the sink channel)."""
    type: Literal["reasoning"] = "reasoning"
    text: str = ""
    i: int = 0
    t: float = 0.0


@dataclass(frozen=True)
class AssistantEvent:
    """The assistant's complete response for one turn."""
    type: Literal["assistant"] = "assistant"
    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    i: int = 0
    t: float = 0.0


@dataclass(frozen=True)
class ToolStartEvent:
    """A tool call is about to execute."""
    type: Literal["tool_start"] = "tool_start"
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""
    i: int = 0
    t: float = 0.0


@dataclass(frozen=True)
class ToolEvent:
    """A tool call completed with a result."""
    type: Literal["tool"] = "tool"
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    ok: bool = False
    call_id: str = ""
    i: int = 0
    t: float = 0.0


@dataclass(frozen=True)
class ToolEndEvent:
    """Sink-channel notification that a tool call finished."""
    type: Literal["tool_end"] = "tool_end"
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    ok: bool = False
    call_id: str = ""
    i: int = 0
    t: float = 0.0


@dataclass(frozen=True)
class FinalEvent:
    """The turn produced a final answer with no more tool calls."""
    type: Literal["final"] = "final"
    content: str = ""
    i: int = 0
    t: float = 0.0


@dataclass(frozen=True)
class ErrorEvent:
    """An exception aborted the turn."""
    type: Literal["error"] = "error"
    content: str = ""
    i: int = 0
    t: float = 0.0


@dataclass(frozen=True)
class CancelledEvent:
    """The user cancelled the turn."""
    type: Literal["cancelled"] = "cancelled"
    content: str = "Cancelled by the user."
    i: int = 0
    t: float = 0.0


@dataclass(frozen=True)
class ApprovalRequiredEvent:
    """A gated tool call needs user approval before it can proceed."""
    type: Literal["approval_required"] = "approval_required"
    approval_id: str = ""
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    risk: str = "medium"
    reason: str = ""
    files: list[str] = field(default_factory=list)
    i: int = 0
    t: float = 0.0


@dataclass(frozen=True)
class UsageEvent:
    """Token usage and context stats for the turn."""
    type: Literal["usage"] = "usage"
    usage: dict[str, int] = field(default_factory=dict)
    context_chars: int = 0
    i: int = 0
    t: float = 0.0


Event = (
    PlanEvent | PartialEvent | TextEvent | ReasoningEvent | AssistantEvent |
    ToolStartEvent | ToolEvent | ToolEndEvent | FinalEvent | ErrorEvent |
    CancelledEvent | ApprovalRequiredEvent | UsageEvent
)


def to_dict(event: Event) -> dict[str, Any]:
    """Serialize an event to a plain dict for JSON transport.

    The frozen dataclasses carry their type tag, so the dict is wire-ready
    for ACP and the poll endpoint without further wrapping.
    """
    return asdict(event)


def from_dict(data: dict[str, Any]) -> Event:
    """Reconstruct a typed event from a dict (for session replay).

    The ``type`` field selects the dataclass; unknown types fall back to a
    plain dict so a future event kind does not break an old client.
    """
    kind = data.get("type")
    mapping = {
        "plan": PlanEvent,
        "partial": PartialEvent,
        "text": TextEvent,
        "reasoning": ReasoningEvent,
        "assistant": AssistantEvent,
        "tool_start": ToolStartEvent,
        "tool": ToolEvent,
        "tool_end": ToolEndEvent,
        "final": FinalEvent,
        "error": ErrorEvent,
        "cancelled": CancelledEvent,
        "approval_required": ApprovalRequiredEvent,
        "usage": UsageEvent,
    }
    cls = mapping.get(kind)
    if cls is None:
        return data  # type: ignore[return-value]
    return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
