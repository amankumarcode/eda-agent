from datetime import date, datetime

import pandas as pd

from dva.agents.coercion import (
    coerce_boolean,
    coerce_date,
    coerce_integer,
    coerce_number,
    coerce_string,
)


def _s(*values):
    return pd.Series(list(values), dtype=object)


def test_dates_iso_native_and_other_formats():
    out, report = coerce_date(
        _s(
            "2023-05-01",
            "05/02/2023",
            "03-May-2023",
            "20230504",
            datetime(2023, 5, 5, 13, 0),
            date(2023, 5, 6),
            "",
            None,
            "not a date",
        )  # fmt: skip
    )
    expected = [f"2023-05-0{d}" for d in range(1, 7)]
    assert out.dt.strftime("%Y-%m-%d").tolist()[:6] == expected
    assert out.iloc[6:].isna().all()
    assert (report.coerced, report.invalid) == (3, 1)


def test_numbers_plain_currency_separators_and_decimal_comma():
    out, report = coerce_number(_s("4.20", "$4.20", "4,20", "$1,234.50", "(3.00)", 7, 2.5,
                                   "", "abc"))  # fmt: skip
    assert out.tolist()[:7] == [4.2, 4.2, 4.2, 1234.5, -3.0, 7.0, 2.5]
    assert out.iloc[7:].isna().all()
    assert (report.coerced, report.invalid) == (4, 1)


def test_clean_numbers_report_nothing():
    _, report = coerce_number(_s("1", "-2", "3.50"))
    assert report.total == 0 and report.note == ""


def test_integers_reject_fractions():
    out, report = coerce_integer(_s("12", 24, "6.5", ""))
    assert out.tolist()[:2] == [12, 24] and out.iloc[2:].isna().all()
    assert str(out.dtype) == "Int64" and report.invalid == 1


def test_booleans():
    out, report = coerce_boolean(_s("True", "false", "Y", "0", True, "", "maybe"))
    assert out.tolist()[:5] == [True, False, True, False, True]
    assert out.iloc[5:].isna().all() and report.invalid == 1


def test_strings_strip_and_drop_excel_float_suffix():
    out, report = coerce_string(_s(" S1 ", 101.0, 7, "", None))
    assert out.tolist()[:3] == ["S1", "101", "7"] and out.iloc[3:].isna().all()
    assert report.total == 0
