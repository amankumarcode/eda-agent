"""The orchestrator's tools: run_parsing, run_eda, run_synthesis.

Each tool reads state through InjectedState and writes through Command(update=...),
with a short ToolMessage summarizing the result. The LLM only chooses which tool runs
(and run_eda's focus); it never passes data between tools. A failure inside a tool
becomes an error ToolMessage and leaves the rest of the state unchanged.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Annotated, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field

from dva.agents.eda import EdaError, eda_inputs_from_state, eda_state_update
from dva.agents.eda import run_eda as run_eda_agent
from dva.agents.parsing import ParseOptions, ParsingError, parse_files, parsing_state_update
from dva.agents.synthesis import SynthesisError, synthesis_inputs_from_state, synthesis_state_update
from dva.agents.synthesis import run_synthesis as run_synthesis_agent
from dva.config import Role, Settings
from dva.data_model import DataModel
from dva.orchestrator.prompts import TOOL_ERROR
from dva.state import Finding
from dva.workspace import SessionWorkspace, session_workspace

TOOL_ERRORS = (ParsingError, EdaError, SynthesisError)


def _ignore(event: str, data: dict[str, Any]) -> None:
    """Default recorder: drop tool events."""


class GraphDeps(BaseModel):
    """Everything the graph needs from outside: settings, data model, models, clocks."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    settings: Settings
    data_model: DataModel
    llm_factory: Callable[[Role], BaseChatModel]
    clock: Callable[[], float] = Field(default=time.time)
    today: Callable[[], date] = Field(default=date.today)
    # Receives (tool name, outcome details) after each tool run; evals use it to count
    # first-pass schema validity and EDA run success without adding state fields.
    recorder: Callable[[str, dict[str, Any]], None] = Field(default=_ignore)

    def workspace(self, state: dict[str, Any]) -> SessionWorkspace:
        return session_workspace(self.settings.paths.workspace_root, state["session_id"])


def _finding_counts(findings: list[Finding]) -> str:
    counts = Counter(f.issue_type.value for f in findings)
    return ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "none"


def _reply(call_id: str, name: str, text: str, update: dict | None = None) -> Command:
    message = ToolMessage(content=text, tool_call_id=call_id, name=name)
    return Command(update={**(update or {}), "messages": [message]})


def _error(call_id: str, name: str, error: Exception | str) -> Command:
    message = ToolMessage(content=TOOL_ERROR.format(error=error), tool_call_id=call_id,
                          name=name, status="error")  # fmt: skip
    return Command(update={"messages": [message]})


def _tokens(state: dict[str, Any], used: int) -> int:
    return int(state.get("tokens_used") or 0) + used


def parsing_tool(deps: GraphDeps) -> BaseTool:
    """Build run_parsing bound to ``deps``."""

    @tool
    def run_parsing(
        state: Annotated[dict, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command:
        """Load the pending raw files, map columns to the data model, and run type and
        plausibility checks. Call only when raw files are pending."""
        files = state.get("raw_files") or []
        if not files:
            return _error(tool_call_id, "run_parsing", "no raw files are pending")
        options = ParseOptions(
            retries=deps.settings.llm.structured_output_retries,
            plausibility=deps.settings.plausibility, today=deps.today(),
            existing_refs=state.get("parsed_data_ref") or {},
            existing_profile=state.get("data_profile") or {},
        )  # fmt: skip
        try:
            result = parse_files([Path(f) for f in files], deps.data_model, deps.workspace(state),
                                 deps.llm_factory("column_mapper"), options)  # fmt: skip
        except TOOL_ERRORS as exc:
            deps.recorder("run_parsing", {"ok": False, "error": str(exc)})
            return _error(tool_call_id, "run_parsing", exc)
        deps.recorder("run_parsing", {"ok": True, "first_pass": result.mapper_first_pass})
        update = parsing_state_update(state, result)
        tables = ", ".join(f"{t} ({p['rows']} rows)" for t, p in result.data_profile.items())
        text = (f"Parsed {len(files)} files into {tables or 'no tables'}. Findings: "
                f"{_finding_counts(result.findings)}. Missing required fields: "
                f"{result.missing_required_fields or 'none'}.")  # fmt: skip
        return _reply(tool_call_id, "run_parsing", text,
                      {**update, "tokens_used": _tokens(state, result.tokens_used)})  # fmt: skip

    return run_parsing


def eda_tool(deps: GraphDeps) -> BaseTool:
    """Build run_eda bound to ``deps``."""

    @tool
    def run_eda(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        focus: str | None = None,
    ) -> Command:
        """Write and run analysis code on the parsed data. Optional focus narrows the run,
        for example "check quantity units only"."""
        try:
            inputs = eda_inputs_from_state(state, focus, deps.data_model)
            result = run_eda_agent(inputs, deps.llm_factory("eda_coder"), deps.settings,
                                   deps.workspace(state))  # fmt: skip
        except TOOL_ERRORS as exc:
            deps.recorder("run_eda", {"ok": False, "error": str(exc), "first_pass": [False]})
            return _error(tool_call_id, "run_eda", exc)
        deps.recorder("run_eda", {"ok": result.succeeded, "attempts": result.attempts,
                                  "first_pass": result.first_pass_valid})  # fmt: skip
        update = eda_state_update(state, result)
        return _reply(tool_call_id, "run_eda", result.summary,
                      {**update, "tokens_used": _tokens(state, result.tokens_used)})  # fmt: skip

    return run_eda


def synthesis_tool(deps: GraphDeps) -> BaseTool:
    """Build run_synthesis bound to ``deps``."""

    @tool
    def run_synthesis(
        state: Annotated[dict, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command:
        """Write the issue report and the customer email and Jira drafts (queued for
        human review, never sent)."""
        try:
            result = run_synthesis_agent(synthesis_inputs_from_state(state),
                                   deps.llm_factory("synthesis"), deps.settings,
                                   deps.workspace(state))  # fmt: skip
        except TOOL_ERRORS as exc:
            deps.recorder("run_synthesis", {"ok": False, "error": str(exc), "first_pass": [False]})
            return _error(tool_call_id, "run_synthesis", exc)
        deps.recorder("run_synthesis", {"ok": True, "first_pass": [result.first_pass_valid]})
        update = synthesis_state_update(state, result)
        text = (f"Report written to {result.report_ref}. {len(result.drafts)} drafts queued "
                f"for review. Summary: {result.report_summary}")  # fmt: skip
        return _reply(tool_call_id, "run_synthesis", text,
                      {**update, "tokens_used": _tokens(state, result.tokens_used)})  # fmt: skip

    return run_synthesis


def make_tools(deps: GraphDeps) -> list[BaseTool]:
    """The three tools, in the order the orchestrator sees them."""
    return [parsing_tool(deps), eda_tool(deps), synthesis_tool(deps)]
