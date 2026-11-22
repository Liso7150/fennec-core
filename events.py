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
    """A reasoning/thinking chunk.

    The sink channel sends the model's thought as one event per turn; the step
    log sends it in bounded pieces while the model is still having it, which is
    what ``part`` numbers (see ``loop._Thoughts``)."""
    type: Literal["reasoning"] = "reasoning"
    text: str = ""
    part: int = 0
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
class ProgressEvent:
    """A running tool saying what it is inside.

    It rides between ``tool_start`` and ``tool`` with the same ``call_id``, and a shell
    shows one live line per call rather than a row per event: a browser decision takes
    tens of seconds, and without this the chat looks like it stopped.
    """
    type: Literal["progress"] = "progress"
    name: str = ""
    call_id: str = ""
    text: str = ""
    step: int = 0
    of: int = 0
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
class QueueEvent:
    """Follow-up prompts the loop queued for a complex run.

    It arrives before the ``final`` step: the shells drain their queue on the
    step that ends the run, so the first follow-up has to be waiting by then.
    """
    type: Literal["queue"] = "queue"
    items: list[str] = field(default_factory=list)
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
    ToolStartEvent | ProgressEvent | ToolEvent | ToolEndEvent | QueueEvent | FinalEvent |
    ErrorEvent | CancelledEvent | ApprovalRequiredEvent | UsageEvent
)


# Every kind the loop can emit, by its ``type`` tag. loop.STEP_TYPES is the
# other copy of this list, and the desktop and PWA mirrors of both are asserted
# against it (tests/test_events.py, fennec-desktop/src/events.rs), so a new step
# kind has to be registered here as well.
EVENT_BY_KIND: dict[str, type] = {
    cls().type: cls for cls in (
        PlanEvent, PartialEvent, TextEvent, ReasoningEvent, AssistantEvent,
        ToolStartEvent, ProgressEvent, ToolEvent, ToolEndEvent, QueueEvent, FinalEvent,
        ErrorEvent, CancelledEvent, ApprovalRequiredEvent, UsageEvent,
    )
}


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
    cls = EVENT_BY_KIND.get(data.get("type"))
    if cls is None:
        return data  # type: ignore[return-value]
    return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
