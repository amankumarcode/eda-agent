"""EDA agent: the LLM writes analysis code, the sandbox runs it, findings are merged.

The prompt holds Parquet paths, columns and dtypes, the field mapping, the checks
remaining, a summary of known findings, the user's question and the optional focus.
It never holds data rows. A failed run gets one fix attempt with the truncated error;
a second failure is recorded as an ``other`` finding and control returns to the caller.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from dva.config import Settings, session_logger
from dva.llm import StructuredOutputError, invoke_structured
from dva.sandbox.runner import SandboxResult, run_code, truncate
from dva.state import Finding, FindingSource, IssueType, Severity
from dva.workspace import SessionWorkspace

EDA_SYSTEM_PROMPT = """\
You write Python analysis code for a data validation agent. The code runs in a sandbox.

Environment:
- Preloaded names: pd (pandas), np (numpy), DATA (dict of table name -> Parquet path),
  ARTIFACTS_DIR (folder for large outputs), emit(findings, checks_completed).
- Allowed imports: {allowed}. No OS, file-system, network or process access; eval, exec,
  getattr, open and dunder attributes are rejected before the code runs.
- Read tables with pd.read_parquet(DATA["<table>"]). Columns are data model field names.

Reporting:
- Call emit(findings, checks_completed) exactly once, at the end.
- Each finding is a dict with keys: field, issue_type (one of {issue_types}), severity
  (high, medium or low), detail (one sentence with the key numbers), count (rows
  affected, an int), evidence_ref (a path or None).
- checks_completed lists the check ids from "Checks remaining" that the code evaluated,
  whether or not it found a problem.
- Save large tables with df.to_csv(f"{{ARTIFACTS_DIR}}/<name>.csv") and use that path as
  evidence_ref. Print only short summaries; stdout is cut to {limit} characters.
- Report real problems in the customer's data only, and do not repeat known findings.

Return the code and checks_addressed (the check ids the code evaluates)."""

EDA_USER_PROMPT = """\
Tables:
{tables}

Field mapping (customer column -> model field):
{mapping}

Checks remaining: {checks}

Known findings (do not repeat):
{known}

User question: {question}
Focus: {focus}"""

EDA_FIX_PROMPT = """\
The code failed:
{error}

Fix it and return the complete corrected code."""


class EdaCode(BaseModel):
    """What the EDA LLM returns."""

    code: str = Field(min_length=1)
    checks_addressed: list[str]


class EdaInputs(BaseModel):
    """The slice of state the EDA agent reads (no data rows)."""

    parsed_data_ref: dict[str, str]
    data_profile: dict[str, Any]
    field_mapping: dict[str, str]
    required_checks: list[str]
    checks_completed: list[str]
    known_findings: list[Finding] = []
    user_request: str = ""
    focus: str | None = None

    @property
    def checks_remaining(self) -> list[str]:
        return [c for c in self.required_checks if c not in self.checks_completed]


class EdaResult(BaseModel):
    findings: list[Finding]
    checks_completed: list[str]
    artifacts: list[str]
    succeeded: bool
    attempts: int
    first_pass_valid: list[bool]
    tokens_used: int
    summary: str


class EdaError(RuntimeError):
    """EDA could not start or the model gave no valid code (after the structured retry)."""


def eda_inputs_from_state(state: dict[str, Any], focus: str | None = None) -> EdaInputs:
    """Build EdaInputs from graph state. Reads state only."""
    return EdaInputs(
        parsed_data_ref=state.get("parsed_data_ref") or {},
        data_profile=state.get("data_profile") or {},
        field_mapping=state.get("field_mapping") or {},
        required_checks=state.get("required_checks") or [],
        checks_completed=state.get("checks_completed") or [],
        known_findings=state.get("findings") or [],
        user_request=state.get("user_request") or "",
        focus=focus,
    )


def build_messages(inputs: EdaInputs, settings: Settings) -> list[BaseMessage]:
    """The EDA prompt: paths, columns, dtypes, mapping, checks, known findings, question."""
    system = EDA_SYSTEM_PROMPT.format(
        allowed=", ".join(settings.sandbox.allowed_imports),
        issue_types=", ".join(t.value for t in IssueType),
        limit=settings.sandbox.output_char_limit,
    )
    tables = "\n".join(
        f"- {name}: {path} ({inputs.data_profile.get(name, {}).get('rows', '?')} rows)\n"
        f"  columns: {json.dumps(inputs.data_profile.get(name, {}).get('dtypes', {}))}"
        for name, path in inputs.parsed_data_ref.items()
    )
    known = "\n".join(f"- {f.field}: {f.issue_type.value} ({f.count} rows)"
                      for f in inputs.known_findings) or "- none"  # fmt: skip
    user = EDA_USER_PROMPT.format(
        tables=tables, mapping=json.dumps(inputs.field_mapping),
        checks=", ".join(inputs.checks_remaining) or "none", known=known,
        question=inputs.user_request or "(none)", focus=inputs.focus or "(none)",
    )  # fmt: skip
    return [SystemMessage(system), HumanMessage(user)]


def run_eda(
    inputs: EdaInputs, coder: BaseChatModel, settings: Settings, workspace: SessionWorkspace
) -> EdaResult:
    """Get code from the EDA LLM, run it, and allow one fix attempt per settings.

    Reads: the session's Parquet files (inside the sandbox). Writes: sandbox files and
    any artifacts the code saves. Raises EdaError when there is no parsed data or the
    model returns no valid code.
    """
    if not inputs.parsed_data_ref:
        raise EdaError("no parsed data in this session; run parsing first")
    log = session_logger(__name__, workspace.session_id, "eda")
    history = build_messages(inputs, settings)
    tokens, first_pass, run = 0, [], SandboxResult(ok=False)
    attempts = settings.sandbox.fix_attempts + 1
    for attempt in range(1, attempts + 1):
        answer = _ask(coder, history, settings)
        tokens, first_pass = tokens + answer.tokens_used, first_pass + [answer.first_pass_valid]
        run = run_code(answer.parsed.code, workspace, inputs.parsed_data_ref, settings.sandbox)
        log.info("attempt %d: ok=%s in %.2fs", attempt, run.ok, run.duration_s)
        if run.ok:
            return _success(inputs, run, attempt, first_pass, tokens, settings)
        history += [AIMessage(content=answer.parsed.model_dump_json()),
                    HumanMessage(EDA_FIX_PROMPT.format(error=run.error))]  # fmt: skip
    return _failure(run, attempts, first_pass, tokens)


def _ask(coder: BaseChatModel, history: list[BaseMessage], settings: Settings) -> Any:
    try:
        return invoke_structured(
            coder, EdaCode, history, retries=settings.llm.structured_output_retries,
            method=settings.llm.structured_output_method,
        )  # fmt: skip
    except StructuredOutputError as exc:
        raise EdaError(f"the EDA model returned no valid code: {exc}") from exc


def _success(
    inputs: EdaInputs, run: SandboxResult, attempt: int, first_pass: list[bool], tokens: int,
    settings: Settings,
) -> EdaResult:  # fmt: skip
    assert run.emitted is not None
    findings = [Finding(**f.model_dump(), source=FindingSource.EDA) for f in run.emitted.findings]
    valid = set(inputs.required_checks)
    completed = [c for c in dict.fromkeys(run.emitted.checks_completed) if c in valid]
    emitted = truncate(run.emitted.model_dump_json(), settings.sandbox.output_char_limit)
    stdout = run.stdout or "(empty)"
    summary = (f"EDA succeeded on attempt {attempt}: {len(findings)} findings, checks "
               f"completed: {completed or 'none'}.\nemit: {emitted}\nstdout: {stdout}")  # fmt: skip
    return EdaResult(findings=findings, checks_completed=completed, artifacts=run.artifacts,
                     succeeded=True, attempts=attempt, first_pass_valid=first_pass,
                     tokens_used=tokens, summary=summary)  # fmt: skip


def _failure(run: SandboxResult, attempts: int, first_pass: list[bool], tokens: int) -> EdaResult:
    first_line = run.error.strip().splitlines()[-1] if run.error.strip() else "unknown error"
    finding = Finding(field="eda", issue_type=IssueType.OTHER, severity=Severity.LOW,
                      detail=f"EDA code failed after {attempts} attempts: {first_line}",
                      count=0, source=FindingSource.EDA)  # fmt: skip
    summary = f"EDA failed after {attempts} attempts. Last error:\n{run.error}"
    return EdaResult(findings=[finding], checks_completed=[], artifacts=run.artifacts,
                     succeeded=False, attempts=attempts, first_pass_valid=first_pass,
                     tokens_used=tokens, summary=summary)  # fmt: skip


def eda_state_update(state: dict[str, Any], result: EdaResult) -> dict[str, Any]:
    """Merge EDA findings and checks into state.

    A finding with the same (field, issue_type) as an existing parsing or plausibility
    finding is dropped (the deterministic one is kept); otherwise the newest wins.
    Returns: the partial state update. Writes nothing.
    """
    merged = {(f.field, f.issue_type): f for f in state.get("findings") or []}
    for finding in result.findings:
        existing = merged.get((finding.field, finding.issue_type))
        if existing is None or existing.source is FindingSource.EDA:
            merged[(finding.field, finding.issue_type)] = finding
    checks = list(dict.fromkeys([*(state.get("checks_completed") or []), *result.checks_completed]))
    return {"findings": list(merged.values()), "checks_completed": checks}
