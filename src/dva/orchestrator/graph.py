"""The orchestrator StateGraph: orchestrator -> tools | reject_synthesis | stop_partial | END.

Every tool call returns to the orchestrator. The orchestrator node calls the LLM, then
records a routing decision made in code (see routing.py); the conditional edge only
follows that decision.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from dva.agents.synthesis import fallback_wording, synthesis_inputs_from_state, write_outputs
from dva.config import RunLimits
from dva.orchestrator.prompts import (
    NOT_RUN,
    ORCHESTRATOR_SYSTEM_PROMPT,
    REJECT_SYNTHESIS,
    STATE_SUMMARY,
    STOP_MESSAGE,
)
from dva.orchestrator.routing import (
    checks_remaining,
    decide,
    route_after_orchestrator,
    synthesis_block_reason,
)
from dva.orchestrator.tools import GraphDeps, make_tools
from dva.state import PipelineState

RECURSION_LIMIT = 60  # graph steps per invocation; the tool-call cap stops runs far earlier
DEFAULT_BATCH_REQUEST = "Validate the uploaded files against the data model and write the report."
STATE_TYPES = ("Finding", "IssueType", "Severity", "FindingSource", "Draft", "DraftStatus")


def limits_for(state: dict[str, Any], deps: GraphDeps) -> RunLimits:
    limits = deps.settings.limits
    return limits.batch if state.get("mode") == "batch" else limits.chat


def state_summary(state: dict[str, Any], limits: RunLimits) -> str:
    """The short state summary the orchestrator sees: no data rows, only counts and names."""
    profile = state.get("data_profile") or {}
    findings = state.get("findings") or []
    listed = "; ".join(f"{f.field} {f.issue_type.value} ({f.count})" for f in findings[:20])
    return STATE_SUMMARY.format(
        mode=state.get("mode"), raw_files=len(state.get("raw_files") or []),
        tables=", ".join(f"{t} ({p.get('rows')} rows)" for t, p in profile.items()) or "none",
        n_findings=len(findings), findings=listed or "none",
        checks=", ".join(checks_remaining(state)) or "none",
        tool_calls=state.get("tool_calls") or 0, cap=limits.max_tool_calls,
        history=", ".join(state.get("tool_history") or []) or "none",
        report=state.get("report_ref") or "none",
    )  # fmt: skip


def keep_first_tool_call(message: AIMessage) -> AIMessage:
    """Drop all but the first tool call, including its tool_use content blocks."""
    if len(message.tool_calls) <= 1:
        return message
    kept = message.tool_calls[0]
    content = message.content
    if isinstance(content, list):
        content = [b for b in content if not (isinstance(b, dict) and b.get("type") == "tool_use"
                                              and b.get("id") != kept["id"])]  # fmt: skip
    return message.model_copy(update={"tool_calls": [kept], "content": content})


def _last_ai(state: dict[str, Any]) -> AIMessage | None:
    for message in reversed(state.get("messages") or []):
        if isinstance(message, AIMessage):
            return message
    return None


def make_orchestrator_node(deps: GraphDeps, tools: list) -> Any:
    def orchestrator(state: PipelineState) -> dict:
        """Ask the LLM for the next step, then record the routing decision made in code."""
        limits = limits_for(state, deps)
        system = ORCHESTRATOR_SYSTEM_PROMPT.format(summary=state_summary(state, limits))
        llm = deps.llm_factory("orchestrator").bind_tools(tools, tool_choice="auto",
                                                          parallel_tool_calls=False)  # fmt: skip
        reply = keep_first_tool_call(llm.invoke([SystemMessage(system), *state["messages"]]))
        usage = (reply.usage_metadata or {}).get("total_tokens", 0)
        tokens = int(state.get("tokens_used") or 0) + int(usage)
        decision = decide({**state, "tokens_used": tokens}, reply, limits, deps.clock())
        return {"messages": [reply], "tokens_used": tokens, **decision}

    return orchestrator


def reject_synthesis(state: PipelineState) -> dict:
    """Answer the blocked run_synthesis call with the reason; the orchestrator continues."""
    call = _last_ai(state).tool_calls[0]  # type: ignore[union-attr]
    reason = synthesis_block_reason(state) or "not allowed now"
    message = ToolMessage(content=REJECT_SYNTHESIS.format(reason=reason), tool_call_id=call["id"],
                          name=call["name"], status="error")  # fmt: skip
    return {"messages": [message]}


def make_stop_partial_node(deps: GraphDeps) -> Any:
    def stop_partial(state: PipelineState) -> dict:
        """Hard stop: answer any pending tool call, write a partial report without an LLM."""
        reason = state.get("stop_reason") or "stopped"
        last = state["messages"][-1] if state.get("messages") else None
        pending: list[BaseMessage] = []
        if isinstance(last, AIMessage):
            text = NOT_RUN.format(reason=reason)
            pending = [ToolMessage(content=text, tool_call_id=c["id"], name=c["name"],
                                   status="error") for c in last.tool_calls]  # fmt: skip
        inputs = synthesis_inputs_from_state(state)
        result = write_outputs(inputs, fallback_wording(inputs), deps.workspace(state),
                               stop_reason=reason, with_drafts=False)  # fmt: skip
        final = AIMessage(content=STOP_MESSAGE.format(reason=reason, report=result.report_ref))
        return {"messages": [*pending, final], "report_ref": result.report_ref,
                "report_summary": result.report_summary}  # fmt: skip

    return stop_partial


def build_graph(deps: GraphDeps, checkpointer: BaseCheckpointSaver | None = None) -> Any:
    """Build and compile the graph. Writes nothing until it is invoked."""
    tools = make_tools(deps)
    graph = StateGraph(PipelineState)
    graph.add_node("orchestrator", make_orchestrator_node(deps, tools))
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("reject_synthesis", reject_synthesis)
    graph.add_node("stop_partial", make_stop_partial_node(deps))
    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges("orchestrator", route_after_orchestrator,
                                {"tools": "tools", "reject_synthesis": "reject_synthesis",
                                 "stop_partial": "stop_partial", "end": END})  # fmt: skip
    graph.add_edge("tools", "orchestrator")
    graph.add_edge("reject_synthesis", "orchestrator")
    graph.add_edge("stop_partial", END)
    return graph.compile(checkpointer=checkpointer)


# --- entry points and persistence -------------------------------------------------------


def turn_reset(now: float) -> dict[str, Any]:
    """Per-turn counters and bookkeeping, reset at the start of every run or chat turn."""
    return {"tool_calls": 0, "tokens_used": 0, "started_at": now, "last_tool": None,
            "last_state_hash": None, "stop_reason": None, "tool_history": [],
            "next_step": None}  # fmt: skip


def session_fields(session_id: str, mode: str, deps: GraphDeps, model_name: str) -> dict:
    return {"mode": mode, "session_id": session_id, "data_model_name": model_name,
            "required_checks": deps.data_model.required_checks()}  # fmt: skip


def batch_input(deps: GraphDeps, session_id: str, files: list[str],
                request: str = DEFAULT_BATCH_REQUEST) -> dict[str, Any]:  # fmt: skip
    """Complete initial state for one batch run."""
    empty = {"parsed_data_ref": {}, "field_mapping": {}, "unmapped_columns": [],
             "missing_required_fields": [], "data_profile": {}, "findings": [],
             "checks_completed": [], "report_ref": None, "report_summary": None,
             "draft_artifacts": []}  # fmt: skip
    return {**session_fields(session_id, "batch", deps, deps.data_model.name), **empty,
            "raw_files": [str(f) for f in files], "user_request": request,
            "messages": [HumanMessage(request)], **turn_reset(deps.clock())}  # fmt: skip


def chat_input(deps: GraphDeps, session_id: str, message: str,
               uploads: list[str] | None = None) -> dict[str, Any]:  # fmt: skip
    """Input for one chat turn; session fields persist through the checkpointer.

    ``raw_files`` is only sent when files were uploaded this turn, so a checkpointed value
    is never overwritten with nothing.
    """
    update = {**session_fields(session_id, "chat", deps, deps.data_model.name),
              "user_request": message, "messages": [HumanMessage(message)],
              **turn_reset(deps.clock())}  # fmt: skip
    if uploads:
        update["raw_files"] = [str(f) for f in uploads]
    return update


def thread_config(session_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": session_id}, "recursion_limit": RECURSION_LIMIT}


def run_batch(
    deps: GraphDeps, session_id: str, files: list[str], request: str = DEFAULT_BATCH_REQUEST
) -> dict[str, Any]:
    """One batch run: a single graph invocation without a checkpointer.

    Reads: the files. Writes: everything the tools write under the session's workspace.
    Returns: the final state.
    """
    graph = build_graph(deps)
    return graph.invoke(batch_input(deps, session_id, files, request),
                        {"recursion_limit": RECURSION_LIMIT})  # fmt: skip


def message_text(message: BaseMessage) -> str:
    """The visible text of a message (thinking and tool blocks are left out)."""
    if isinstance(message.content, str):
        return message.content
    parts = [b.get("text", "") for b in message.content
             if isinstance(b, dict) and b.get("type") == "text"]  # fmt: skip
    return "".join(parts)


def final_reply(state: dict[str, Any]) -> str:
    """Text of the last assistant message in the state, or ''."""
    last = _last_ai(state)
    return message_text(last) if last is not None else ""


def checkpoint_serializer() -> JsonPlusSerializer:
    """Serializer that explicitly allows the state's own Pydantic types."""
    return JsonPlusSerializer(allowed_msgpack_modules=[("dva.state", n) for n in STATE_TYPES])


@contextmanager
def open_checkpointer(path: Path) -> Iterator[SqliteSaver]:
    """SQLite checkpointer at ``path``. Writes: the database file (and its folder)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), check_same_thread=False)
    try:
        yield SqliteSaver(connection, serde=checkpoint_serializer())
    finally:
        connection.close()
