"""Routing rules: what happens after each orchestrator message. Pure functions, no I/O.

Checked in order before a tool call is followed: wall-clock and token budgets, the
tool-call cap, no progress (the same tool twice in a row with no change to state), then
the gate on run_synthesis (required checks done, and in batch mode one EDA run first).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from dva.config import RunLimits
from dva.state import TURN_FIELDS

SYNTHESIS = "run_synthesis"
EDA = "run_eda"
REJECTED_SUFFIX = " (rejected)"
NOT_HASHED = {"messages", *TURN_FIELDS}

NextStep = Literal["tools", "reject_synthesis", "stop_partial", "end"]


class StopReason:
    TASK_COMPLETE = "task_complete"
    ANSWERED = "answered"
    NEEDS_USER = "needs_user"
    TOOL_CALL_CAP = "tool_call_cap"
    NO_PROGRESS = "no_progress"
    TIME_BUDGET = "time_budget"
    TOKEN_BUDGET = "token_budget"
    ENDED_WITHOUT_REPORT = "ended_without_report"


def _plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


def state_hash(state: dict[str, Any]) -> str:
    """Hash of every state field except messages and per-turn bookkeeping."""
    payload = {k: _plain(v) for k, v in state.items() if k not in NOT_HASHED}
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def checks_remaining(state: dict[str, Any]) -> list[str]:
    done = set(state.get("checks_completed") or [])
    return [c for c in state.get("required_checks") or [] if c not in done]


def synthesis_block_reason(state: dict[str, Any]) -> str | None:
    """Why run_synthesis may not run now, or None if it may."""
    remaining = checks_remaining(state)
    if remaining:
        return "required checks remain: " + ", ".join(remaining)
    if state.get("mode") == "batch" and EDA not in (state.get("tool_history") or []):
        return "in batch mode run_eda must run at least once before run_synthesis"
    return None


def finish_reason(state: dict[str, Any]) -> tuple[NextStep, str]:
    """Route for a reply without a tool call."""
    history = state.get("tool_history") or []
    if SYNTHESIS in history and state.get("report_ref"):
        return "end", StopReason.TASK_COMPLETE
    if state.get("mode") == "batch":
        return "stop_partial", StopReason.ENDED_WITHOUT_REPORT
    if not state.get("parsed_data_ref") and not state.get("raw_files"):
        return "end", StopReason.NEEDS_USER
    return "end", StopReason.ANSWERED


def hard_stop(state: dict[str, Any], tool: str, limits: RunLimits, now: float) -> str | None:
    """The first hard stop that applies before following a call to ``tool``, if any."""
    started = state.get("started_at")
    if started is not None and now - float(started) >= limits.wall_clock_seconds:
        return StopReason.TIME_BUDGET
    if int(state.get("tokens_used") or 0) >= limits.token_budget:
        return StopReason.TOKEN_BUDGET
    if int(state.get("tool_calls") or 0) >= limits.max_tool_calls:
        return StopReason.TOOL_CALL_CAP
    if tool == state.get("last_tool") and state_hash(state) == state.get("last_state_hash"):
        return StopReason.NO_PROGRESS
    return None


def decide(state: dict[str, Any], message: AIMessage, limits: RunLimits, now: float) -> dict:
    """Return the state update that records the routing decision for ``message``.

    Writes nothing; the orchestrator node returns this update to the graph.
    """
    if not message.tool_calls:
        step, reason = finish_reason(state)
        return {"next_step": step, "stop_reason": reason}
    tool = message.tool_calls[0]["name"]
    stop = hard_stop(state, tool, limits, now)
    if stop is not None:
        return {"next_step": "stop_partial", "stop_reason": stop}
    blocked = tool == SYNTHESIS and synthesis_block_reason(state) is not None
    history = [*(state.get("tool_history") or []), tool + (REJECTED_SUFFIX if blocked else "")]
    return {
        "next_step": "reject_synthesis" if blocked else "tools",
        "tool_calls": int(state.get("tool_calls") or 0) + 1,
        "last_tool": tool,
        "last_state_hash": state_hash(state),
        "tool_history": history,
    }


def route_after_orchestrator(state: dict[str, Any]) -> str:
    """Conditional edge: follow the decision recorded by the orchestrator node."""
    return state.get("next_step") or "end"
