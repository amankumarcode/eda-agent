"""Synthesis agent: the issue report and the customer email / Jira drafts.

The report and the issue lists in the drafts are built in code from the structured
findings, so every finding appears exactly as recorded. The LLM only writes the wording
around them: a summary, the email's opening and closing, and the Jira title and summary.
Without an LLM (a partial report after a hard stop) fixed wording is used instead.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from dva.config import Settings, session_logger
from dva.llm import StructuredOutputError, invoke_structured
from dva.review_queue import ReviewQueue
from dva.state import Draft, Finding, Severity
from dva.workspace import SessionWorkspace

REPORT_FILENAME = "report.md"
SEVERITY_ORDER = (Severity.HIGH, Severity.MEDIUM, Severity.LOW)

SYNTHESIS_SYSTEM_PROMPT = """\
You write the wording around a data validation report for a customer's data upload.
Code inserts the full list of issues between your opening and closing, so:
- Do not list every issue, and never invent issues, counts or values.
- Refer to the most important issues by field name, using the counts given.
- report_summary: 2-4 sentences for an internal reader on overall data readiness.
- email_subject, email_opening, email_closing: to the customer's data team. Polite and
  specific; ask them to correct and resend the files. No placeholders such as [Name].
- jira_title, jira_summary: an internal ticket for the onboarding team.
Plain text only."""

SYNTHESIS_USER_PROMPT = """\
Session: {session_id}
Data model: {data_model}
Tables: {tables}
Missing required fields: {missing}
Unmapped columns: {unmapped}

Findings (severity, field, issue type, rows, detail):
{findings}"""


class SynthesisWording(BaseModel):
    """What the synthesis LLM writes."""

    report_summary: str = Field(min_length=1, max_length=1500)
    email_subject: str = Field(min_length=1, max_length=200)
    email_opening: str = Field(min_length=1, max_length=2000)
    email_closing: str = Field(min_length=1, max_length=1000)
    jira_title: str = Field(min_length=1, max_length=200)
    jira_summary: str = Field(min_length=1, max_length=2000)


class SynthesisInputs(BaseModel):
    """The slice of state synthesis reads."""

    session_id: str
    data_model_name: str
    findings: list[Finding] = []
    missing_required_fields: list[str] = []
    unmapped_columns: list[str] = []
    data_profile: dict[str, Any] = {}
    required_checks: list[str] = []
    checks_completed: list[str] = []


class SynthesisResult(BaseModel):
    report_ref: str
    report_summary: str
    drafts: list[Draft]
    tokens_used: int = 0
    first_pass_valid: bool = True


class SynthesisError(RuntimeError):
    """The model gave no valid wording after the structured-output retry."""


def synthesis_inputs_from_state(state: dict[str, Any]) -> SynthesisInputs:
    """Build SynthesisInputs from graph state. Reads state only."""
    return SynthesisInputs(
        session_id=state["session_id"],
        data_model_name=state.get("data_model_name") or "",
        findings=state.get("findings") or [],
        missing_required_fields=state.get("missing_required_fields") or [],
        unmapped_columns=state.get("unmapped_columns") or [],
        data_profile=state.get("data_profile") or {},
        required_checks=state.get("required_checks") or [],
        checks_completed=state.get("checks_completed") or [],
    )


# --- deterministic content --------------------------------------------------------------


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def sorted_findings(findings: list[Finding]) -> list[Finding]:
    """High, medium, low; larger counts first within a severity."""
    rank = {s: i for i, s in enumerate(SEVERITY_ORDER)}
    return sorted(findings, key=lambda f: (rank[f.severity], -f.count, f.field))


def counts_line(findings: list[Finding]) -> str:
    """e.g. '6 issues: 3 high, 2 medium, 1 low'."""
    parts = [f"{sum(f.severity is s for f in findings)} {s.value}" for s in SEVERITY_ORDER]
    return f"{len(findings)} issues: " + ", ".join(parts)


def issue_list(findings: list[Finding]) -> str:
    """Plain-text bullets used inside the drafts."""
    if not findings:
        return "No issues were found."
    return "\n".join(f"- [{f.severity.value}] {f.field}: {f.detail} ({f.count} rows)"
                     for f in sorted_findings(findings))  # fmt: skip


def _evidence(ref: str | None, session_dir: Path) -> str:
    if not ref:
        return ""
    path = Path(ref)
    return str(path.relative_to(session_dir)) if path.is_relative_to(session_dir) else ref


def _severity_sections(findings: list[Finding], session_dir: Path) -> list[str]:
    lines: list[str] = []
    for severity in SEVERITY_ORDER:
        group = [f for f in sorted_findings(findings) if f.severity is severity]
        lines += ["", f"## {severity.value.capitalize()} severity ({len(group)})", ""]
        if not group:
            lines.append("None.")
            continue
        lines += ["| Field | Issue | Rows | Detail | Source | Evidence |",
                  "|---|---|---|---|---|---|"]  # fmt: skip
        lines += [f"| {_cell(f.field)} | {f.issue_type.value} | {f.count} | {_cell(f.detail)} "
                  f"| {f.source.value} | {_cell(_evidence(f.evidence_ref, session_dir))} |"
                  for f in group]  # fmt: skip
    return lines


def render_report(
    inputs: SynthesisInputs, summary: str, session_dir: Path, stop_reason: str | None = None
) -> str:
    """The Markdown report. Every finding is listed, grouped by severity. Writes nothing."""
    title = "Data validation report" + (" (partial)" if stop_reason else "")
    tables = ", ".join(f"{t} ({p.get('rows', '?')} rows)" for t, p in inputs.data_profile.items())
    lines = [f"# {title}: {inputs.session_id}", "",
             f"- Data model: {inputs.data_model_name}", f"- Tables: {tables or 'none parsed'}",
             f"- {counts_line(inputs.findings)}"]  # fmt: skip
    if stop_reason:
        remaining = [c for c in inputs.required_checks if c not in inputs.checks_completed]
        lines += [f"- Stopped early: {stop_reason}",
                  f"- Checks not completed: {', '.join(remaining) or 'none'}"]  # fmt: skip
    lines += ["", "## Summary", "", summary]
    lines += _severity_sections(inputs.findings, session_dir)
    missing = ", ".join(inputs.missing_required_fields) or "None."
    unmapped = ", ".join(inputs.unmapped_columns) or "None."
    lines += ["", "## Missing required fields", "", missing,
              "", "## Unmapped columns", "", unmapped, ""]  # fmt: skip
    return "\n".join(lines)


def fallback_wording(inputs: SynthesisInputs) -> SynthesisWording:
    """Fixed wording for partial reports, written without an LLM."""
    counts = counts_line(inputs.findings)
    return SynthesisWording(
        report_summary=f"Automated checks stopped before completion. {counts}.",
        email_subject=f"Data upload {inputs.session_id}: issues found",
        email_opening="Thank you for your data files. Our checks found the issues below.",
        email_closing="Could you correct these and resend the files? Thank you.",
        jira_title=f"Data validation issues for {inputs.session_id}",
        jira_summary=f"Partial validation run. {counts}.",
    )


# --- LLM wording, report, drafts --------------------------------------------------------


def build_messages(inputs: SynthesisInputs) -> list:
    """The synthesis prompt: structured findings and summaries, no data rows."""
    findings = (
        "\n".join(
            f"- {f.severity.value} | {f.field} | {f.issue_type.value} | {f.count} | {f.detail}"
            for f in sorted_findings(inputs.findings)
        )
        or "- none"
    )
    tables = ", ".join(f"{t} ({p.get('rows', '?')} rows)" for t, p in inputs.data_profile.items())
    user = SYNTHESIS_USER_PROMPT.format(
        session_id=inputs.session_id, data_model=inputs.data_model_name, tables=tables or "none",
        missing=", ".join(inputs.missing_required_fields) or "none",
        unmapped=", ".join(inputs.unmapped_columns) or "none", findings=findings,
    )  # fmt: skip
    return [SystemMessage(SYNTHESIS_SYSTEM_PROMPT), HumanMessage(user)]


def make_drafts(inputs: SynthesisInputs, wording: SynthesisWording) -> list[Draft]:
    """Email and Jira drafts: LLM wording around the code-built issue list."""
    issues = issue_list(inputs.findings)
    email = f"{wording.email_opening}\n\n{issues}\n\n{wording.email_closing}"
    jira = f"{wording.jira_summary}\n\n{counts_line(inputs.findings)}\n\n{issues}"
    return [
        Draft(draft_id=f"email-{uuid.uuid4().hex[:12]}", session_id=inputs.session_id,
              kind="email", title=wording.email_subject, body=email),
        Draft(draft_id=f"jira-{uuid.uuid4().hex[:12]}", session_id=inputs.session_id,
              kind="jira", title=wording.jira_title, body=jira),
    ]  # fmt: skip


def write_outputs(
    inputs: SynthesisInputs,
    wording: SynthesisWording,
    workspace: SessionWorkspace,
    stop_reason: str | None = None,
) -> SynthesisResult:
    """Write report.md and queue both drafts as pending_review.

    Writes: workspace/<session>/report.md and review_queue/<draft_id>.json. Sends nothing.
    """
    workspace.ensure()
    report_path = workspace.session_dir / REPORT_FILENAME
    report = render_report(inputs, wording.report_summary, workspace.session_dir, stop_reason)
    report_path.write_text(report, encoding="utf-8")
    queue = ReviewQueue(workspace)
    drafts = make_drafts(inputs, wording)
    for draft in drafts:
        queue.save(draft)
    return SynthesisResult(report_ref=str(report_path), report_summary=wording.report_summary,
                           drafts=drafts)  # fmt: skip


def run_synthesis(
    inputs: SynthesisInputs, writer: BaseChatModel, settings: Settings, workspace: SessionWorkspace
) -> SynthesisResult:
    """Ask the LLM for wording, then write the report and drafts (see write_outputs).

    Raises SynthesisError if the wording is invalid after the structured-output retry.
    """
    log = session_logger(__name__, workspace.session_id, "synthesis")
    try:
        answer = invoke_structured(writer, SynthesisWording, build_messages(inputs),
                                   retries=settings.llm.structured_output_retries,
                                   method=settings.llm.structured_output_method)  # fmt: skip
    except StructuredOutputError as exc:
        raise SynthesisError(f"the synthesis model returned no valid wording: {exc}") from exc
    result = write_outputs(inputs, answer.parsed, workspace)
    log.info("report written with %d findings and %d drafts", len(inputs.findings),
             len(result.drafts))  # fmt: skip
    return result.model_copy(update={"tokens_used": answer.tokens_used,
                                     "first_pass_valid": answer.first_pass_valid})  # fmt: skip


def synthesis_state_update(state: dict[str, Any], result: SynthesisResult) -> dict[str, Any]:
    """State fields written by synthesis; new drafts are added to earlier ones."""
    return {
        "report_ref": result.report_ref,
        "report_summary": result.report_summary,
        "draft_artifacts": [*(state.get("draft_artifacts") or []), *result.drafts],
    }
