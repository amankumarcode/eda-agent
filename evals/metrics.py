"""Eval metrics: field-mapping accuracy, issue recall and precision, EDA execution success,
first-pass schema validity, cap hit rate, and per-run tool calls, latency and tokens.

Matching findings to ground truth:
1. In code: same field and issue_type, and the count within tolerance (exact for
   ``count``, +/-10% for ``count_approx``).
2. Findings still unmatched go to an LLM judge that answers yes or no to "does this
   finding describe this ground-truth issue?" (unmatched issues first, then the rest).
3. Every decision is recorded so the owner can audit it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel

from dva.state import Finding

APPROX_TOLERANCE = 0.10

TARGETS = {
    "field_mapping_accuracy": (">=", 0.95),
    "issue_recall": (">=", 0.90),
    "issue_precision": (">=", 0.80),
    "eda_execution_success": (">=", 0.95),
    "cap_hit_rate": ("<=", 0.10),
}


class Issue(BaseModel):
    """One expected issue from ground_truth.json."""

    field: str
    issue_type: str
    count: int | None = None
    count_approx: int | None = None

    @property
    def expected(self) -> int:
        return self.count if self.count is not None else int(self.count_approx or 0)

    def count_ok(self, value: int) -> bool:
        if self.count is not None:
            return value == self.count
        return abs(value - self.expected) <= APPROX_TOLERANCE * self.expected

    def label(self) -> str:
        kind = "count" if self.count is not None else "~count"
        return f"{self.field}/{self.issue_type} ({kind} {self.expected})"


class MatchDecision(BaseModel):
    finding: str
    issue: str | None
    method: Literal["code", "llm", "none"]
    matched: bool


Judge = Callable[[Finding, Issue], bool]


def finding_label(finding: Finding) -> str:
    return f"{finding.field}/{finding.issue_type.value} ({finding.count}, {finding.source.value})"


def code_match(finding: Finding, issue: Issue) -> bool:
    """Same field and issue type, count within tolerance."""
    return (finding.field == issue.field and finding.issue_type.value == issue.issue_type
            and issue.count_ok(finding.count))  # fmt: skip


def match_findings(
    findings: list[Finding], issues: list[Issue], judge: Judge | None
) -> tuple[list[MatchDecision], set[int], int]:
    """Return (decisions, indexes of issues found, number of correct findings)."""
    found: set[int] = set()
    decisions: list[MatchDecision] = []
    leftovers: list[Finding] = []
    for finding in findings:
        hit = next((i for i, issue in enumerate(issues) if code_match(finding, issue)), None)
        if hit is None:
            leftovers.append(finding)
            continue
        found.add(hit)
        decisions.append(MatchDecision(finding=finding_label(finding), issue=issues[hit].label(),
                                       method="code", matched=True))  # fmt: skip
    for finding in leftovers:
        decisions.append(_judge_one(finding, issues, found, judge))
    correct = sum(d.matched for d in decisions)
    return decisions, found, correct


def _judge_one(finding: Finding, issues: list[Issue], found: set[int],
               judge: Judge | None) -> MatchDecision:  # fmt: skip
    order = [i for i in range(len(issues)) if i not in found] + sorted(found)
    for index in order if judge else []:
        if judge(finding, issues[index]):  # type: ignore[misc]
            found.add(index)
            return MatchDecision(finding=finding_label(finding), issue=issues[index].label(),
                                 method="llm", matched=True)  # fmt: skip
    return MatchDecision(finding=finding_label(finding), issue=None, method="none", matched=False)


def mapping_accuracy(state: dict[str, Any], expected: dict[str, str | None]) -> tuple[int, int]:
    """(correctly mapped columns, all columns) against expected_field_mapping."""
    got = {**(state.get("field_mapping") or {}),
           **{c: None for c in state.get("unmapped_columns") or []}}  # fmt: skip
    correct = sum(got.get(column, "<missing>") == field for column, field in expected.items())
    return correct, len(expected)


class RunRecord(BaseModel):
    """Everything measured for one fixture run."""

    fixture: str
    session: str
    stop_reason: str | None = None
    error: str | None = None
    mapping_correct: int = 0
    mapping_total: int = 0
    issues_found: int = 0
    issues_total: int = 0
    findings_correct: int = 0
    findings_total: int = 0
    eda_ok: int = 0
    eda_total: int = 0
    first_pass_ok: int = 0
    first_pass_total: int = 0
    tool_calls: int = 0
    tokens: int = 0
    matcher_tokens: int = 0
    latency_s: float = 0.0
    report_ref: str | None = None
    decisions: list[MatchDecision] = []


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(records: list[RunRecord]) -> dict[str, Any]:
    """Overall metrics across runs: pooled ratios, cap hit rate, mean and max per run."""

    def total(name: str) -> int:
        return sum(getattr(r, name) for r in records)

    def stats(name: str) -> dict[str, float]:
        values = [float(getattr(r, name)) for r in records] or [0.0]
        return {"mean": sum(values) / len(values), "max": max(values)}

    return {
        "field_mapping_accuracy": ratio(total("mapping_correct"), total("mapping_total")),
        "issue_recall": ratio(total("issues_found"), total("issues_total")),
        "issue_precision": ratio(total("findings_correct"), total("findings_total")),
        "eda_execution_success": ratio(total("eda_ok"), total("eda_total")),
        "first_pass_schema_rate": ratio(total("first_pass_ok"), total("first_pass_total")),
        "cap_hit_rate": ratio(sum(r.stop_reason == "tool_call_cap" for r in records), len(records)),
        "tool_calls": stats("tool_calls"),
        "latency_s": stats("latency_s"),
        "tokens": stats("tokens"),
        "runs": len(records),
        "errors": sum(r.error is not None for r in records),
    }


def meets_target(name: str, value: float | None) -> bool | None:
    """True/False against the spec's target, or None when there is no target or no data."""
    if name not in TARGETS or value is None:
        return None
    op, target = TARGETS[name]
    return value >= target if op == ">=" else value <= target
