"""Pydantic models for a data model: tables, fields, cross-field rules, required checks."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TODAY = "today"
CHECK_SUFFIXES = ("range", "units", "categories", "join")
_COMPARE_RE = re.compile(r"^(?P<left>.+?)\s*(?P<op><=|>=|==|<|>)\s*(?P<right>.+)$")
_BETWEEN_RE = re.compile(r"^(?P<expr>.+?)\s+between\s+(?P<low>\S+)\s+and\s+(?P<high>\S+)$")
_ALLOWED_EXPR_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Constant, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub,
)  # fmt: skip

RangeBound = float | str


class FieldType(StrEnum):
    DATE = "date"
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    CATEGORY = "category"
    BOOLEAN = "boolean"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FieldSpec(_Frozen):
    """One field of one table."""

    name: str
    table: str
    type: FieldType
    required: bool = False
    synonyms: tuple[str, ...] = ()
    range: tuple[RangeBound, RangeBound] | None = None
    join: str | None = None
    units: str | None = None
    categories: dict[str, tuple[str, ...]] | None = None

    @field_validator("range", mode="before")
    @classmethod
    def _dates_to_iso(cls, value: object) -> object:
        if isinstance(value, list | tuple):
            return tuple(v.isoformat() if isinstance(v, date) else v for v in value)
        return value

    @field_validator("join")
    @classmethod
    def _join_format(cls, value: str | None) -> str | None:
        if value is not None and value.count(".") != 1:
            raise ValueError(f"join must look like 'table.field', got {value!r}")
        return value

    @model_validator(mode="after")
    def _type_rules(self) -> FieldSpec:
        if self.range is not None:
            self._check_range()
        if (self.type is FieldType.CATEGORY) != (self.categories is not None):
            raise ValueError(f"{self.qualified_name}: categories are required for, "
                             "and only allowed on, category fields")  # fmt: skip
        return self

    def _check_range(self) -> None:
        low, high = self.range  # type: ignore[misc]
        if self.type is FieldType.DATE:
            for bound in (low, high):
                if bound != TODAY:
                    date.fromisoformat(str(bound))
        elif self.type in (FieldType.NUMBER, FieldType.INTEGER):
            if not all(isinstance(b, int | float) for b in (low, high)) or low > high:
                raise ValueError(f"{self.qualified_name}: numeric range must be [low, high]")
        else:
            raise ValueError(f"{self.qualified_name}: range is not allowed on {self.type} fields")

    @property
    def qualified_name(self) -> str:
        return f"{self.table}.{self.name}"

    def resolved_range(self, today: date) -> tuple[date, date] | tuple[float, float] | None:
        """Return the range with 'today' and ISO strings resolved. Writes nothing."""
        if self.range is None:
            return None
        if self.type is FieldType.DATE:
            low, high = (today if b == TODAY else date.fromisoformat(str(b)) for b in self.range)
            return low, high
        return float(self.range[0]), float(self.range[1])

    def check_suffixes(self) -> list[str]:
        """Return the optional checks this field defines, in spec order."""
        defined = {
            "range": self.range is not None,
            "units": self.units is not None,
            "categories": self.categories is not None,
            "join": self.join is not None,
        }
        return [suffix for suffix in CHECK_SUFFIXES if defined[suffix]]


class TableSpec(_Frozen):
    name: str
    fields: dict[str, FieldSpec]


class CrossFieldRule(_Frozen):
    """A rule such as ``a <= b`` or ``expr between low and high``, validated at load time."""

    expression: str
    kind: Literal["compare", "between"]
    left: str
    op: str | None = None
    right: str | None = None
    low: float | None = None
    high: float | None = None

    @classmethod
    def parse(cls, expression: str) -> CrossFieldRule:
        """Parse one of the two supported rule forms; raise ValueError for anything else."""
        text = expression.strip()
        if match := _BETWEEN_RE.match(text):
            low, high = float(match["low"]), float(match["high"])
            if low > high:
                raise ValueError(f"rule {expression!r}: low bound is above high bound")
            rule = cls(expression=text, kind="between", left=match["expr"].strip(),
                       low=low, high=high)  # fmt: skip
        elif match := _COMPARE_RE.match(text):
            rule = cls(expression=text, kind="compare", left=match["left"].strip(),
                       op=match["op"], right=match["right"].strip())  # fmt: skip
        else:
            raise ValueError(f"unsupported cross-field rule: {expression!r}")
        rule.identifiers()  # validates the arithmetic grammar
        return rule

    def identifiers(self) -> set[str]:
        """Return the names used by the rule; raise ValueError on disallowed syntax."""
        names: set[str] = set()
        for part in (self.left, self.right):
            if part is not None:
                names |= _expression_names(part)
        return names


def _expression_names(expression: str) -> set[str]:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid rule expression {expression!r}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_EXPR_NODES):
            raise ValueError(f"disallowed syntax {type(node).__name__} in {expression!r}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, int | float):
            raise ValueError(f"only numeric constants are allowed in {expression!r}")
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


class DataModel(_Frozen):
    """A complete data model: tables, their fields, and cross-field rules."""

    name: str
    tables: dict[str, TableSpec] = Field(min_length=1)
    cross_field_rules: tuple[CrossFieldRule, ...] = ()

    @model_validator(mode="after")
    def _references_exist(self) -> DataModel:
        for spec in self.iter_fields():
            if spec.join is not None:
                table, field = spec.join.split(".")
                if field not in self.tables.get(table, TableSpec(name=table, fields={})).fields:
                    raise ValueError(f"{spec.qualified_name}: join target {spec.join} not found")
        known = self.field_names() | {TODAY}
        for rule in self.cross_field_rules:
            if unknown := rule.identifiers() - known:
                raise ValueError(f"rule {rule.expression!r} uses unknown names {sorted(unknown)}")
        return self

    def iter_fields(self) -> Iterator[FieldSpec]:
        """Yield every field in declaration order (table by table)."""
        for table in self.tables.values():
            yield from table.fields.values()

    def field_names(self) -> set[str]:
        return {spec.name for spec in self.iter_fields()}

    def required_checks(self) -> list[str]:
        """Return check ids: ``<field>:present`` per required field, then ``:range``,
        ``:units``, ``:categories``, ``:join`` where defined.

        Field names are not table-qualified (as in the spec's examples), so a field that
        appears in several tables, such as ``sku``, yields each check once.
        """
        checks: list[str] = []
        for spec in self.iter_fields():
            candidates = ["present"] if spec.required else []
            candidates += spec.check_suffixes()
            for suffix in candidates:
                check = f"{spec.name}:{suffix}"
                if check not in checks:
                    checks.append(check)
        return checks
