"""Deterministic plausibility checks on parsed tables. No LLM is involved.

Checks: ranges, cross-field rules, units (cases instead of units), category codes,
duplicate rows, and joins. A check counts as complete when it was evaluated, or when it
does not apply because the field (or a join's target table) is missing; the missing
field is already reported by parsing. The units check stays open when the quantity is
present but the grouping field is not, so EDA can take it up.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from dva.agents.loaders import normalize_name
from dva.config import PlausibilitySettings
from dva.data_model import CrossFieldRule, DataModel, FieldSpec, FieldType
from dva.data_model.models import TODAY
from dva.state import Finding, FindingSource, IssueType, Severity
from dva.workspace import SessionWorkspace

COMPARE_OPS: dict[str, Callable] = {
    "<=": operator.le, ">=": operator.ge, "<": operator.lt, ">": operator.gt, "==": operator.eq,
}  # fmt: skip
ARITH_OPS: dict[type, Callable] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
}  # fmt: skip
CENTS_SHARE = 0.8  # share of above-range values that fit the range after /100


class CheckContext(BaseModel):
    """Inputs shared by every check."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tables: dict[str, pd.DataFrame]
    model: DataModel
    settings: PlausibilitySettings
    today: date
    workspace: SessionWorkspace


class CheckOutcome(BaseModel):
    findings: list[Finding] = []
    completed: bool = True


class PlausibilityResult(BaseModel):
    findings: list[Finding]
    checks_completed: list[str]


def run_plausibility(
    refs: dict[str, str],
    model: DataModel,
    settings: PlausibilitySettings,
    workspace: SessionWorkspace,
    today: date,
) -> PlausibilityResult:
    """Run every plausibility check over the parsed tables.

    Reads: the Parquet files in ``refs``. Writes: category columns back to their Parquet
    files with canonical codes, and evidence CSVs under the session's artifacts folder.
    Returns: findings and the checks completed.
    """
    workspace.ensure()
    ctx = CheckContext(tables={t: pd.read_parquet(p) for t, p in refs.items()}, model=model,
                       settings=settings, today=today, workspace=workspace)  # fmt: skip
    findings, completed = [], []
    for spec in model.iter_fields():
        for suffix in spec.check_suffixes():
            outcome = FIELD_CHECKS[suffix](spec, ctx)
            findings += outcome.findings
            check = f"{spec.name}:{suffix}"
            if outcome.completed and check not in completed:
                completed.append(check)
    findings += check_rules(ctx) + check_duplicates(ctx)
    _write_categories(ctx, refs)
    return PlausibilityResult(findings=findings, checks_completed=completed)


# --- helpers ----------------------------------------------------------------------------


def _column(spec: FieldSpec, ctx: CheckContext) -> pd.Series | None:
    frame = ctx.tables.get(spec.table)
    return None if frame is None or spec.name not in frame else frame[spec.name]


def _evidence(ctx: CheckContext, name: str, rows: pd.DataFrame) -> str:
    """Save up to max_evidence_rows offending rows; return the file path."""
    path = ctx.workspace.artifacts_dir / f"{name}.csv"
    rows.head(ctx.settings.max_evidence_rows).to_csv(path, index=False)
    return str(path)


def _finding(field: str, issue: IssueType, severity: Severity, detail: str, count: int,
             evidence: str | None = None) -> Finding:  # fmt: skip
    return Finding(field=field, issue_type=issue, severity=severity, detail=detail, count=count,
                   evidence_ref=evidence, source=FindingSource.PLAUSIBILITY)  # fmt: skip


def _fmt(value: Any) -> str:
    if isinstance(value, pd.Timestamp | date):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    return f"{value:,.2f}" if isinstance(value, float) else str(value)


# --- range ------------------------------------------------------------------------------


def check_range(spec: FieldSpec, ctx: CheckContext) -> CheckOutcome:
    """Values outside the field's range (with 'today' resolved) are implausible."""
    values = _column(spec, ctx)
    bounds = spec.resolved_range(ctx.today)
    if values is None or bounds is None:
        return CheckOutcome()
    low, high = bounds
    if spec.type is FieldType.DATE:
        low, high = pd.Timestamp(low), pd.Timestamp(high)
    present = values.dropna()
    above, below = present[present > high], present[present < low]
    count = len(above) + len(below)
    if not count:
        return CheckOutcome()
    median = _fmt(present.median())
    detail = (f"{count} values outside [{_fmt(low)}, {_fmt(high)}] "
              f"({len(below)} below, {len(above)} above); median is {median}")  # fmt: skip
    if spec.type is not FieldType.DATE and _looks_like_cents(above, low, high):
        detail += "; the high values fit the range after dividing by 100 (likely entered in cents)"
    rows = ctx.tables[spec.table].loc[above.index.union(below.index)]
    evidence = _evidence(ctx, f"{spec.name}_out_of_range", rows)
    return CheckOutcome(findings=[_finding(spec.name, IssueType.IMPLAUSIBLE_VALUE, Severity.HIGH,
                                           detail, count, evidence)])  # fmt: skip


def _looks_like_cents(above: pd.Series, low: float, high: float) -> bool:
    if above.empty:
        return False
    return bool(((above / 100).between(low, high)).mean() >= CENTS_SHARE)


# --- units ------------------------------------------------------------------------------


def check_units(spec: FieldSpec, ctx: CheckContext) -> CheckOutcome:
    """Groups (stores) whose median quantity is about pack_size times the typical level."""
    cfg = ctx.settings.unit_check
    values, frame = _column(spec, ctx), ctx.tables.get(spec.table)
    if values is None:
        return CheckOutcome()
    if frame is None or cfg.group_by not in frame:
        return CheckOutcome(completed=False)  # quantity present, grouping field missing
    positive = frame[values > 0]
    sizes = positive.groupby(cfg.group_by)[spec.name].size()
    medians = positive.groupby(cfg.group_by)[spec.name].median()[sizes >= cfg.min_rows]
    if medians.empty:
        return CheckOutcome()
    typical = float(medians.median())
    flagged = {g: m / typical for g, m in medians.items() if _is_pack_multiple(
        m / typical, _pack_size(ctx, positive[positive[cfg.group_by] == g]), cfg)}  # fmt: skip
    if not flagged:
        return CheckOutcome()
    return CheckOutcome(findings=[_units_finding(spec, ctx, frame, flagged, typical)])


def _pack_size(ctx: CheckContext, rows: pd.DataFrame) -> float | None:
    """Median pack size of the products in these rows, via the field's join, if available."""
    table, field = ctx.settings.unit_check.pack_size_field.split(".")
    master = ctx.tables.get(table)
    join_specs = [s for s in ctx.model.iter_fields() if s.join and s.join.startswith(f"{table}.")]
    for spec in join_specs:
        key = spec.join.split(".")[1]
        if master is not None and field in master and spec.name in rows and key in master:
            sizes = rows[[spec.name]].merge(master[[key, field]], left_on=spec.name,
                                            right_on=key)[field].dropna()  # fmt: skip
            if not sizes.empty:
                return float(sizes.median())
    return None


def _is_pack_multiple(ratio: float, pack: float | None, cfg: Any) -> bool:
    if ratio < cfg.min_ratio:
        return False
    packs = [pack] if pack else list(cfg.fallback_pack_sizes)
    return any(abs(ratio - p) / p <= cfg.tolerance for p in packs)


def _units_finding(spec: FieldSpec, ctx: CheckContext, frame: pd.DataFrame,
                   flagged: dict, typical: float) -> Finding:  # fmt: skip
    group_by = ctx.settings.unit_check.group_by
    rows = frame[frame[group_by].isin(list(flagged))]
    ratio = sum(flagged.values()) / len(flagged)
    groups = ", ".join(sorted(str(g) for g in flagged)[: ctx.settings.max_codes_in_detail])
    detail = (f"{len(flagged)} {group_by} values ({groups}) report {spec.name} about "
              f"{ratio:.0f}x the typical median of {typical:g}; likely cases, not units "
              f"({spec.units})")  # fmt: skip
    evidence = _evidence(ctx, f"{spec.name}_unit_mismatch", rows)
    return _finding(spec.name, IssueType.UNIT_MISMATCH, Severity.HIGH, detail, len(rows), evidence)


# --- categories -------------------------------------------------------------------------


def category_lookup(spec: FieldSpec) -> dict[str, str]:
    """Normalized code -> canonical value, for the canonical names and every variant."""
    lookup = {}
    for canonical, variants in (spec.categories or {}).items():
        for code in (canonical, *variants):
            lookup[normalize_name(code) or code.strip()] = canonical
    return lookup


def map_categories(values: pd.Series, spec: FieldSpec) -> tuple[pd.Series, pd.Series]:
    """Return (values with canonical codes where known, mask of unknown non-null codes)."""
    lookup = category_lookup(spec)
    keys = values.map(lambda v: None if pd.isna(v) else (normalize_name(v) or str(v).strip()))
    canonical = keys.map(lambda k: lookup.get(k) if k is not None else None)
    unknown = values.notna() & canonical.isna()
    return canonical.where(~unknown, values).astype("string"), unknown


def check_categories(spec: FieldSpec, ctx: CheckContext) -> CheckOutcome:
    """Codes that are neither a canonical value nor a listed variant (nulls are not counted)."""
    values = _column(spec, ctx)
    if values is None:
        return CheckOutcome()
    mapped, unknown = map_categories(values, spec)
    ctx.tables[spec.table][spec.name] = mapped
    if not unknown.any():
        return CheckOutcome()
    counts = values[unknown].value_counts()
    codes = ", ".join(
        f"{c} ({n})" for c, n in counts.head(ctx.settings.max_codes_in_detail).items()
    )
    detail = f"{int(unknown.sum())} values use codes not in the data model: {codes}"
    evidence = _evidence(ctx, f"{spec.name}_unmapped_codes", ctx.tables[spec.table][unknown])
    return CheckOutcome(findings=[_finding(spec.name, IssueType.UNMAPPED_CATEGORY,
                                           Severity.MEDIUM, detail, int(unknown.sum()),
                                           evidence)])  # fmt: skip


def _write_categories(ctx: CheckContext, refs: dict[str, str]) -> None:
    tables = {s.table for s in ctx.model.iter_fields() if s.categories} & set(ctx.tables)
    for table in tables:
        ctx.tables[table].to_parquet(Path(refs[table]), index=False)


# --- joins ------------------------------------------------------------------------------


def check_join(spec: FieldSpec, ctx: CheckContext) -> CheckOutcome:
    """Non-null values with no match in the join target. Not applicable if either side is
    missing (parsing reports the missing field)."""
    values = _column(spec, ctx)
    target_table, target_field = spec.join.split(".")  # type: ignore[union-attr]
    target = ctx.tables.get(target_table)
    if values is None or target is None or target_field not in target:
        return CheckOutcome()
    known = set(target[target_field].dropna().astype(str))
    present = values.dropna().astype(str)
    missing = present[~present.isin(known)]
    if missing.empty:
        return CheckOutcome()
    detail = (f"{len(missing)} rows ({missing.nunique()} distinct values) have a {spec.name} "
              f"not found in {spec.join}")  # fmt: skip
    evidence = _evidence(
        ctx, f"{spec.name}_join_failures", ctx.tables[spec.table].loc[missing.index]
    )
    return CheckOutcome(findings=[_finding(spec.name, IssueType.JOIN_FAILURE, Severity.HIGH,
                                           detail, len(missing), evidence)])  # fmt: skip


FIELD_CHECKS: dict[str, Callable[[FieldSpec, CheckContext], CheckOutcome]] = {
    "range": check_range,
    "units": check_units,
    "categories": check_categories,
    "join": check_join,
}


# --- duplicates and cross-field rules ---------------------------------------------------


def check_duplicates(ctx: CheckContext) -> list[Finding]:
    """Exact duplicate rows per table; count is the number of extra copies."""
    findings = []
    for table, frame in ctx.tables.items():
        extra = frame.duplicated()
        if extra.any():
            evidence = _evidence(ctx, f"{table}_duplicates", frame[frame.duplicated(keep=False)])
            findings.append(_finding("*", IssueType.DUPLICATE_ROWS, Severity.MEDIUM,
                                     f"{int(extra.sum())} rows in {table} are exact duplicates "
                                     "of another row", int(extra.sum()), evidence))  # fmt: skip
    return findings


def evaluate_expression(expression: str, frame: pd.DataFrame, today: date) -> Any:
    """Evaluate a validated rule expression (names, numbers, + - * /) over a table."""

    def walk(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Name):
            return pd.Timestamp(today) if node.id == TODAY else frame[node.id]
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -walk(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in ARITH_OPS:
            return ARITH_OPS[type(node.op)](walk(node.left), walk(node.right))
        raise ValueError(f"unsupported syntax in rule expression {expression!r}")

    return walk(ast.parse(expression, mode="eval"))


def rule_violations(rule: CrossFieldRule, frame: pd.DataFrame, today: date) -> pd.Series:
    """Mask of rows that break the rule; rows with a missing operand are skipped."""
    left = evaluate_expression(rule.left, frame, today)
    if rule.kind == "between":
        return left.notna() & ~left.between(rule.low, rule.high)
    right = evaluate_expression(rule.right, frame, today)  # type: ignore[arg-type]
    known = left.notna() & (right.notna() if isinstance(right, pd.Series) else True)
    return known & ~COMPARE_OPS[rule.op](left, right)  # type: ignore[index]


def check_rules(ctx: CheckContext) -> list[Finding]:
    """Evaluate each cross-field rule on the table that has all of its fields."""
    findings = []
    for rule in ctx.model.cross_field_rules:
        names = rule.identifiers() - {TODAY}
        table = next((t for t, f in ctx.tables.items() if names <= set(f.columns)), None)
        if table is None:
            continue
        broken = rule_violations(rule, ctx.tables[table], ctx.today)
        if broken.any():
            evidence = _evidence(ctx, f"rule_{normalize_name(rule.expression)}",
                                 ctx.tables[table][broken])  # fmt: skip
            findings.append(_finding(rule.expression, IssueType.IMPLAUSIBLE_VALUE,
                                     Severity.MEDIUM, f"{int(broken.sum())} rows in {table} "
                                     f"break the rule '{rule.expression}'",
                                     int(broken.sum()), evidence))  # fmt: skip
    return findings
