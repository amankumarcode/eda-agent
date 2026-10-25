"""Graph state and the records it carries.

The state holds references and summaries, never the customer's data rows.
Finding and Draft are Pydantic models; PipelineState is the LangGraph state.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt


class IssueType(StrEnum):
    MISSING_FIELD = "missing_field"
    UNMAPPED_COLUMN = "unmapped_column"
    TYPE_MISMATCH = "type_mismatch"
    IMPLAUSIBLE_VALUE = "implausible_value"
    UNIT_MISMATCH = "unit_mismatch"
    UNMAPPED_CATEGORY = "unmapped_category"
    JOIN_FAILURE = "join_failure"
    DUPLICATE_ROWS = "duplicate_rows"
    OTHER = "other"


class Severity(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FindingSource(StrEnum):
    PARSING = "parsing"
    PLAUSIBILITY = "plausibility"
    EDA = "eda"


class Finding(BaseModel):
    """One issue found in the customer's data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    field: str = Field(min_length=1)
    issue_type: IssueType
    severity: Severity
    detail: str = Field(min_length=1)
    count: NonNegativeInt
    evidence_ref: str | None = None
    source: FindingSource


class DraftStatus(StrEnum):
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class Draft(BaseModel):
    """A customer email or Jira ticket draft. Drafts are never sent."""

    model_config = ConfigDict(extra="forbid")

    draft_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    kind: Literal["email", "jira"]
    title: str = Field(min_length=1)
    body: str = Field(min_length=1)
    status: DraftStatus = DraftStatus.PENDING_REVIEW
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


Mode = Literal["batch", "chat"]


class PipelineState(TypedDict, total=False):
    """LangGraph state. Only ``messages`` has a reducer; every other field is replaced."""

    # Session
    mode: Mode
    session_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    data_model_name: str
    required_checks: list[str]
    raw_files: list[str] | None
    parsed_data_ref: dict[str, str]
    field_mapping: dict[str, str]
    unmapped_columns: list[str]
    missing_required_fields: list[str]
    data_profile: dict[str, Any]
    findings: list[Finding]
    checks_completed: list[str]
    report_ref: str | None
    report_summary: str | None
    draft_artifacts: list[Draft]
    # Turn
    user_request: str
    tool_calls: int
    tokens_used: int
    started_at: float
    last_tool: str | None
    last_state_hash: str | None
    stop_reason: str | None
    # Turn bookkeeping added in Phase 6: every tool call requested this turn (rejected ones
    # marked), and the routing decision taken after the orchestrator's last message.
    tool_history: list[str]
    next_step: str | None


# Fields reset at the start of every chat turn (and every batch run).
TURN_FIELDS: tuple[str, ...] = (
    "user_request",
    "tool_calls",
    "tokens_used",
    "started_at",
    "last_tool",
    "last_state_hash",
    "stop_reason",
    "tool_history",
    "next_step",
)
