"""Convert raw cell values to data-model types and count what needed fixing.

Every coercion returns the typed column plus a TypeReport. Problems in the customer's
data are counted, never raised. Reports carry counts only, never cell values.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from numbers import Real

import pandas as pd
from pydantic import BaseModel

from dva.data_model import FieldType

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$")
# Tried in order for non-ISO text; month-first before day-first (US retail exports).
DATE_FORMATS = ("%m/%d/%Y", "%d-%b-%Y", "%Y%m%d", "%d/%m/%Y", "%d.%m.%Y", "%m/%d/%y", "%b %d, %Y")
PLAIN_NUMBER_RE = re.compile(r"^[+-]?\d+(\.\d+)?$")
THOUSANDS_RE = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")
DECIMAL_COMMA_RE = re.compile(r"^[+-]?\d+,\d{1,2}$")
CURRENCY_CHARS = "$€£ "
TRUE_VALUES = {"true", "t", "yes", "y", "1"}
FALSE_VALUES = {"false", "f", "no", "n", "0"}


class TypeReport(BaseModel):
    """Counts for one column: values converted from a non-standard form, and values lost."""

    coerced: int = 0
    invalid: int = 0
    note: str = ""

    @property
    def total(self) -> int:
        return self.coerced + self.invalid


def coerce_column(values: pd.Series, field_type: FieldType) -> tuple[pd.Series, TypeReport]:
    """Return the column converted to ``field_type`` and a report of what needed fixing."""
    handlers = {
        FieldType.DATE: coerce_date,
        FieldType.NUMBER: coerce_number,
        FieldType.INTEGER: coerce_integer,
        FieldType.BOOLEAN: coerce_boolean,
    }
    return handlers.get(field_type, coerce_string)(values)


def is_null(value: object) -> bool:
    """True for None, NaN/NaT, and blank strings."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def coerce_string(values: pd.Series) -> tuple[pd.Series, TypeReport]:
    """Strings stay as stripped text; whole-number floats from Excel lose their '.0'."""

    def convert(value: object) -> object:
        if is_null(value):
            return pd.NA
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value).strip()

    return values.map(convert).astype("string"), TypeReport()


def coerce_date(values: pd.Series) -> tuple[pd.Series, TypeReport]:
    """Native dates pass; ISO text parses silently; other formats parse but are counted."""
    out = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    text_idx = []
    for idx, value in values.items():
        if isinstance(value, datetime | date | pd.Timestamp):
            out[idx] = pd.Timestamp(value).normalize()
        elif not is_null(value):
            text_idx.append(idx)
    text = values[text_idx].astype(str).str.strip()
    iso = text[text.str.match(ISO_DATE_RE)]
    out[iso.index] = pd.to_datetime(iso.str[:10], format="%Y-%m-%d", errors="coerce")
    remaining = text.drop(iso.index)
    coerced = 0
    for fmt in DATE_FORMATS:
        if remaining.empty:
            break
        parsed = pd.to_datetime(remaining, format=fmt, errors="coerce").dropna()
        out[parsed.index] = parsed
        coerced += len(parsed)
        remaining = remaining.drop(parsed.index)
    note = f"{coerced} dates in non-ISO text formats were parsed" if coerced else ""
    return out, TypeReport(coerced=coerced, invalid=len(remaining), note=note)


def _parse_number_text(text: str) -> tuple[float | None, bool]:
    """Return (value, needed_cleaning). value is None when the text is not a number."""
    if PLAIN_NUMBER_RE.match(text):
        return float(text), False
    cleaned, negative = text.strip(CURRENCY_CHARS), False
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned, negative = cleaned[1:-1].strip(CURRENCY_CHARS), True
    if THOUSANDS_RE.match(cleaned):
        cleaned = cleaned.replace(",", "")
    elif DECIMAL_COMMA_RE.match(cleaned):
        cleaned = cleaned.replace(",", ".")
    if not PLAIN_NUMBER_RE.match(cleaned):
        return None, True
    return (-float(cleaned) if negative else float(cleaned)), True


def coerce_number(values: pd.Series) -> tuple[pd.Series, TypeReport]:
    """Native numbers pass; text with currency symbols, thousands separators or a decimal
    comma is converted and counted; anything else becomes NaN and is counted as invalid."""
    out, coerced, invalid = [], 0, 0
    for value in values:
        if is_null(value):
            out.append(float("nan"))
        elif isinstance(value, Real) and not isinstance(value, bool):
            out.append(float(value))
        else:
            number, cleaned = _parse_number_text(str(value).strip())
            coerced += int(cleaned and number is not None)
            invalid += int(number is None)
            out.append(float("nan") if number is None else number)
    note = f"{coerced} numbers had currency symbols or separators" if coerced else ""
    report = TypeReport(coerced=coerced, invalid=invalid, note=note)
    return pd.Series(out, index=values.index, dtype="float64"), report


def coerce_integer(values: pd.Series) -> tuple[pd.Series, TypeReport]:
    """As coerce_number, then values with a fractional part are counted as invalid."""
    numbers, report = coerce_number(values)
    fractional = numbers.notna() & (numbers % 1 != 0)
    numbers[fractional] = float("nan")
    report = report.model_copy(update={"invalid": report.invalid + int(fractional.sum())})
    return numbers.astype("Int64"), report


def coerce_boolean(values: pd.Series) -> tuple[pd.Series, TypeReport]:
    """true/false, yes/no, y/n, t/f and 1/0 (any case); anything else is invalid."""
    out, invalid = [], 0
    for value in values:
        if is_null(value):
            out.append(pd.NA)
        elif isinstance(value, bool):
            out.append(value)
        elif (key := str(value).strip().lower()) in TRUE_VALUES | FALSE_VALUES:
            out.append(key in TRUE_VALUES)
        else:
            out.append(pd.NA)
            invalid += 1
    return pd.Series(out, index=values.index, dtype="boolean"), TypeReport(invalid=invalid)
