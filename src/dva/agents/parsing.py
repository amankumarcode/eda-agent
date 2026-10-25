"""Parsing agent: load files, map columns to the data model, check types, write Parquet.

Mapping runs in two steps. Exact, synonym and normalized-name matches are resolved in
code; only the columns left over go to the LLM column mapper, which must pick a field
of the assigned table or null. The mapper sees column names and a coarse kind
(numeric/text), never data values.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, create_model

from dva.agents.coercion import coerce_column, coerce_number, is_null
from dva.agents.loaders import LoadIssue, RawTable, load_files, normalize_name
from dva.agents.plausibility import run_plausibility
from dva.config import PlausibilitySettings, session_logger
from dva.data_model import DataModel, FieldSpec
from dva.llm import invoke_structured
from dva.state import Finding, FindingSource, IssueType, Severity
from dva.workspace import SessionWorkspace

MAPPER_SYSTEM_PROMPT = (
    "You map a customer's column names to fields of an internal data model table.\n"
    "For every column listed, choose exactly one target field, or null when no field "
    "means the same thing. Never map two columns to the same field. You see column "
    "names and a coarse kind only, never data values. Prefer null over a guess."
)
MAPPER_USER_PROMPT = (
    "Table: {table}\n\nTarget fields (name: type; known synonyms):\n{fields}\n\n"
    "Columns to map (name: kind):\n{columns}"
)


class TableMapping(BaseModel):
    """How one raw table maps onto one data model table."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    table: str
    raw: RawTable
    mapping: dict[str, str]  # source column -> model field
    unmapped: list[str]


class ParsingResult(BaseModel):
    """Everything run_parsing writes to state (raw_files is cleared by the state update)."""

    parsed_data_ref: dict[str, str]
    field_mapping: dict[str, str]
    unmapped_columns: list[str]
    missing_required_fields: list[str]
    data_profile: dict[str, Any]
    findings: list[Finding]
    checks_completed: list[str]
    tokens_used: int = 0
    mapper_first_pass: list[bool] = []


class ParseOptions(BaseModel):
    """Settings and session context for one parsing run."""

    retries: int
    plausibility: PlausibilitySettings
    today: date = Field(default_factory=date.today)
    existing_refs: dict[str, str] = {}  # tables parsed earlier in the session
    existing_profile: dict[str, Any] = {}


class ParsingError(RuntimeError):
    """Parsing could not complete (for example, the column mapper failed twice)."""


def parse_files(
    paths: list[Path],
    model: DataModel,
    workspace: SessionWorkspace,
    mapper: BaseChatModel | None,
    options: ParseOptions,
) -> ParsingResult:
    """Load, map, type-check and store the customer's files, then run plausibility checks.

    Reads: the files, and Parquet files of tables parsed earlier (options.existing_refs).
    Writes: workspace/<session>/data/<table>.parquet for every table found, canonical
    category codes back into those files, and evidence CSVs under artifacts/. Missing
    fields and plausibility are judged across all tables of the session.
    Raises ParsingError if the column mapper fails twice.
    """
    log = session_logger(__name__, workspace.session_id, "parsing")
    loaded = load_files(paths, _vocabulary(model))
    mappings, issues = _assign_tables(loaded.tables, model)
    issues += loaded.issues
    tokens, first_pass = 0, []
    for item in mappings:
        if item.unmapped and mapper is not None:
            used, ok = _resolve_with_llm(item, model, mapper, options.retries)
            tokens += used
            first_pass += [] if ok is None else [ok]
    log.info("loaded %d tables from %d files", len(mappings), len(paths))
    result = _build_result(mappings, issues, model, workspace, options.existing_profile)
    checks = run_plausibility({**options.existing_refs, **result.parsed_data_ref}, model,
                              options.plausibility, workspace, options.today)  # fmt: skip
    return result.model_copy(update={
        "findings": result.findings + checks.findings,
        "checks_completed": result.checks_completed + checks.checks_completed,
        "tokens_used": tokens,
        "mapper_first_pass": first_pass,
    })  # fmt: skip


# --- deterministic mapping and table assignment -----------------------------------------


def _vocabulary(model: DataModel) -> set[str]:
    words = set()
    for spec in model.iter_fields():
        words |= {normalize_name(spec.name), *(normalize_name(s) for s in spec.synonyms)}
    return words


def match_field(column: str, fields: dict[str, FieldSpec]) -> str | None:
    """Exact name, then exact synonym, then case/punctuation-insensitive match."""
    for name, spec in fields.items():
        if column == name or column in spec.synonyms:
            return name
    key = normalize_name(column)
    for name, spec in fields.items():
        if key in {normalize_name(name), *(normalize_name(s) for s in spec.synonyms)}:
            return name
    return None


def map_columns(columns: list[str], fields: dict[str, FieldSpec]) -> tuple[dict, list]:
    """Resolve columns in code. A field already taken leaves later columns unresolved."""
    mapping: dict[str, str] = {}
    unresolved: list[str] = []
    for column in columns:
        target = match_field(column, fields)
        if target is not None and target not in mapping.values():
            mapping[column] = target
        else:
            unresolved.append(column)
    return mapping, unresolved


def assign_table(raw: RawTable, model: DataModel) -> TableMapping | None:
    """Pick the model table whose fields match the most columns; the file name breaks ties."""
    columns = [str(c) for c in raw.frame.columns]
    source_key = normalize_name(raw.source)
    best: tuple[int, int, int] | None = None
    choice = None
    for order, (name, table) in enumerate(model.tables.items()):
        mapping, unresolved = map_columns(columns, table.fields)
        rank = (len(mapping), int(normalize_name(name) in source_key), -order)
        if best is None or rank > best:
            best, choice = rank, TableMapping(table=name, raw=raw, mapping=mapping,
                                              unmapped=unresolved)  # fmt: skip
    if best is None or (best[0] == 0 and best[1] == 0):
        return None
    return choice


def _assign_tables(
    raws: list[RawTable], model: DataModel
) -> tuple[list[TableMapping], list[LoadIssue]]:
    mappings, issues = [], []
    for raw in raws:
        item = assign_table(raw, model)
        if item is None:
            issues.append(LoadIssue(source=raw.source, detail="no column matches any table "
                                    "of the data model"))  # fmt: skip
        else:
            mappings.append(item)
    return mappings, issues


# --- LLM column mapper ------------------------------------------------------------------


def _column_kind(values: pd.Series) -> str:
    present = [v for v in values if not is_null(v)]
    if not present:
        return "empty"
    numbers, _ = coerce_number(pd.Series(present, dtype=object))
    return "numeric" if numbers.notna().mean() >= 0.9 else "text"


def _mapper_schema(columns: list[str], fields: list[str]) -> type[BaseModel]:
    choice = create_model(
        "ColumnChoice",
        column=(Literal[tuple(columns)], ...),
        field=(Literal[tuple(fields)] | None, ...),
    )
    return create_model("ColumnMapping", mappings=(list[choice], ...))


def _mapper_messages(item: TableMapping, model: DataModel, free: list[str]) -> list:
    fields = model.tables[item.table].fields
    field_lines = "\n".join(
        f"- {name}: {fields[name].type}; synonyms: {', '.join(fields[name].synonyms) or '-'}"
        for name in free
    )
    column_lines = "\n".join(
        f"- {col}: {_column_kind(item.raw.frame[col])}" for col in item.unmapped
    )
    prompt = MAPPER_USER_PROMPT.format(table=item.table, fields=field_lines, columns=column_lines)
    return [SystemMessage(MAPPER_SYSTEM_PROMPT), HumanMessage(prompt)]


def _resolve_with_llm(
    item: TableMapping, model: DataModel, mapper: BaseChatModel, retries: int
) -> tuple[int, bool | None]:
    """Ask the mapper about unresolved columns; update ``item`` in place.

    Returns: (tokens used, whether the first answer was valid, or None if no call was made
    because every field of the table is already taken).
    """
    free = [f for f in model.tables[item.table].fields if f not in item.mapping.values()]
    if not free:
        return 0, None
    schema = _mapper_schema(item.unmapped, free)
    expected = set(item.unmapped)

    def validate(answer: Any) -> None:
        columns = [m.column for m in answer.mappings]
        targets = [m.field for m in answer.mappings if m.field is not None]
        if set(columns) != expected or len(columns) != len(expected):
            raise ValueError(f"map each of these columns exactly once: {sorted(expected)}")
        if len(targets) != len(set(targets)):
            raise ValueError("two columns were mapped to the same field")

    try:
        result = invoke_structured(mapper, schema, _mapper_messages(item, model, free),
                                   retries=retries, validate=validate)  # fmt: skip
    except Exception as exc:
        raise ParsingError(f"column mapper failed for table {item.table!r}: {exc}") from exc
    chosen = {m.column: m.field for m in result.parsed.mappings if m.field is not None}
    item.mapping.update(chosen)
    item.unmapped = [c for c in item.unmapped if c not in chosen]
    return result.tokens_used, result.first_pass_valid


# --- typed tables, findings, Parquet ----------------------------------------------------


def _typed_frame(item: TableMapping, model: DataModel) -> tuple[pd.DataFrame, list[Finding]]:
    fields = model.tables[item.table].fields
    columns, findings = {}, []
    for source, target in item.mapping.items():
        series, report = coerce_column(item.raw.frame[source], fields[target].type)
        columns[target] = series.reset_index(drop=True)
        if report.total:
            findings.append(_type_finding(target, source, fields[target].type, report))
    return pd.DataFrame(columns), findings


def _type_finding(field: str, source: str, field_type: Any, report: Any) -> Finding:
    parts = [report.note] if report.note else []
    if report.invalid:
        parts.append(f"{report.invalid} values could not be read as {field_type} and are empty")
    return Finding(
        field=field,
        issue_type=IssueType.TYPE_MISMATCH,
        severity=Severity.HIGH if report.invalid else Severity.MEDIUM,
        detail=f"Column {source!r}: " + "; ".join(parts),
        count=report.total,
        source=FindingSource.PARSING,
    )


def _combine(items: list[TableMapping], model: DataModel) -> dict[str, tuple]:
    """Typed frame, findings and source info per model table (several files may share one)."""
    combined: dict[str, tuple[list, list, list]] = {}
    for item in items:
        frame, findings = _typed_frame(item, model)
        frames, all_findings, sources = combined.setdefault(item.table, ([], [], []))
        frames.append(frame)
        all_findings += findings
        sources.append({"source": item.raw.source, "header_row": item.raw.header_row,
                        "rows": len(frame)})  # fmt: skip
    return {
        table: (pd.concat(frames, ignore_index=True), findings, sources)
        for table, (frames, findings, sources) in combined.items()
    }


def _profile(frame: pd.DataFrame, sources: list) -> dict[str, Any]:
    return {
        "rows": len(frame),
        "columns": list(frame.columns),
        "dtypes": {c: str(t) for c, t in frame.dtypes.items()},
        "sources": sources,
    }


def _flat_mapping(items: list[TableMapping]) -> tuple[dict[str, str], list[str]]:
    mapping: dict[str, str] = {}
    unmapped: list[str] = []
    for item in items:
        for source, target in item.mapping.items():
            key = source if mapping.get(source, target) == target else f"{item.table}.{source}"
            mapping[key] = target
        unmapped += [c for c in item.unmapped if c not in unmapped]
    return mapping, unmapped


def _missing_fields(
    model: DataModel, profile: dict[str, Any]
) -> tuple[list[str], list[tuple[str, str, int]]]:
    """Required fields absent from their table: (names, [(field, table, rows)])."""
    names, details = [], []
    for table_name, table in model.tables.items():
        present = set(profile.get(table_name, {}).get("columns", []))
        rows = profile.get(table_name, {}).get("rows", 0)
        for spec in table.fields.values():
            if spec.required and spec.name not in present:
                details.append((spec.name, table_name, rows))
                if spec.name not in names:
                    names.append(spec.name)
    return names, details


def _structural_findings(
    items: list[TableMapping], missing: list[tuple[str, str, int]], issues: list[LoadIssue]
) -> list[Finding]:
    findings = []
    for item in items:
        for column in item.unmapped:
            findings.append(Finding(
                field=column, issue_type=IssueType.UNMAPPED_COLUMN, severity=Severity.LOW,
                detail=f"Column {column!r} in {item.raw.source} matches no field of "
                       f"table {item.table!r}", count=len(item.raw.frame),
                source=FindingSource.PARSING))  # fmt: skip
    for field, table, rows in missing:
        where = f"table {table!r}" if rows else f"table {table!r} (no file contained it)"
        findings.append(Finding(
            field=field, issue_type=IssueType.MISSING_FIELD, severity=Severity.HIGH,
            detail=f"Required field {field!r} is missing from {where}", count=rows,
            source=FindingSource.PARSING))  # fmt: skip
    for issue in issues:
        findings.append(Finding(
            field=issue.source, issue_type=IssueType.OTHER, severity=Severity.HIGH,
            detail=f"Could not load {issue.source}: {issue.detail}", count=0,
            source=FindingSource.PARSING))  # fmt: skip
    return findings


def _build_result(
    items: list[TableMapping],
    issues: list[LoadIssue],
    model: DataModel,
    workspace: SessionWorkspace,
    existing_profile: dict[str, Any],
) -> ParsingResult:
    workspace.ensure()
    refs, profile, type_findings = {}, {}, []
    for table, (frame, findings, sources) in _combine(items, model).items():
        path = workspace.table_path(table)
        frame.to_parquet(path, index=False)
        refs[table], profile[table] = str(path), _profile(frame, sources)
        type_findings += findings
    missing_names, missing = _missing_fields(model, {**existing_profile, **profile})
    mapping, unmapped = _flat_mapping(items)
    present_checks = [c for c in model.required_checks() if c.endswith(":present")]
    return ParsingResult(
        parsed_data_ref=refs, field_mapping=mapping, unmapped_columns=unmapped,
        missing_required_fields=missing_names, data_profile=profile,
        findings=_structural_findings(items, missing, issues) + type_findings,
        checks_completed=present_checks,
    )  # fmt: skip


# --- state update -----------------------------------------------------------------------


def parsing_state_update(state: dict[str, Any], result: ParsingResult) -> dict[str, Any]:
    """Merge a parsing result into session state and clear raw_files.

    Tables parsed now replace earlier ones with the same name. Missing-field and
    plausibility findings are recomputed over all tables on every parse, so earlier ones
    are dropped; other findings are merged by (field, issue_type), newest first.
    Returns: the partial state update. Writes nothing.
    """
    previous = [
        f
        for f in state.get("findings", [])
        if f.issue_type != IssueType.MISSING_FIELD and f.source != FindingSource.PLAUSIBILITY
    ]
    merged = {(f.field, f.issue_type): f for f in previous}
    merged.update({(f.field, f.issue_type): f for f in result.findings})
    unmapped = list(dict.fromkeys([*state.get("unmapped_columns", []), *result.unmapped_columns]))
    checks = list(dict.fromkeys([*state.get("checks_completed", []), *result.checks_completed]))
    return {
        "raw_files": None,
        "parsed_data_ref": {**state.get("parsed_data_ref", {}), **result.parsed_data_ref},
        "field_mapping": {**state.get("field_mapping", {}), **result.field_mapping},
        "unmapped_columns": unmapped,
        "missing_required_fields": result.missing_required_fields,
        "data_profile": {**state.get("data_profile", {}), **result.data_profile},
        "findings": list(merged.values()),
        "checks_completed": checks,
    }
