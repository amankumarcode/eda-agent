import json
from datetime import date

import pandas as pd
import pytest

from dva.agents.parsing import ParseOptions, parse_files, parsing_state_update
from dva.agents.plausibility import map_categories, run_plausibility
from dva.state import Finding, FindingSource
from dva.workspace import session_workspace
from fakes import FakeChatModel

TODAY = date(2026, 10, 25)
FIXTURES = [
    "clean", "renamed_columns", "missing_field", "unit_mismatch", "implausible_values",
    "category_variants", "join_failure", "type_errors", "duplicates", "excel_messy",
    "email_table", "kitchen_sink",
]  # fmt: skip


def _gt(fixtures_dir, name):
    return json.loads((fixtures_dir / name / "ground_truth.json").read_text())


# --- every fixture, end to end through parsing ------------------------------------------


@pytest.mark.parametrize("name", FIXTURES)
def test_findings_match_ground_truth(fixtures_dir, tmp_path, retail_model, settings, name):
    """Every injected issue is found with the right count, and nothing else is reported."""
    gt = _gt(fixtures_dir, name)
    options = ParseOptions(retries=1, plausibility=settings.plausibility, today=TODAY)
    paths = [fixtures_dir / name / f for f in gt["files"].values()]
    result = parse_files(paths, retail_model, session_workspace(tmp_path, "s1"),
                         FakeChatModel(), options)  # fmt: skip
    got = {(f.field, f.issue_type.value): f.count for f in result.findings}
    assert set(got) == {(i["field"], i["issue_type"]) for i in gt["expected_issues"]}
    for item in gt["expected_issues"]:
        count = got[(item["field"], item["issue_type"])]
        if "count" in item:
            assert count == item["count"], item
        else:
            assert abs(count - item["count_approx"]) <= 0.1 * item["count_approx"], item
    assert sorted(result.checks_completed) == sorted(retail_model.required_checks())


def test_kitchen_sink_details_are_useful(fixtures_dir, tmp_path, retail_model, settings):
    gt = _gt(fixtures_dir, "kitchen_sink")
    options = ParseOptions(retries=1, plausibility=settings.plausibility, today=TODAY)
    paths = [fixtures_dir / "kitchen_sink" / f for f in gt["files"].values()]
    result = parse_files(paths, retail_model, session_workspace(tmp_path, "s1"), None, options)
    by_type = {f.issue_type.value: f for f in result.findings}
    assert "likely entered in cents" in by_type["implausible_value"].detail
    assert all(s in by_type["unit_mismatch"].detail for s in gt["case_stores"])
    assert "12x" in by_type["unit_mismatch"].detail
    assert "ZZ (25)" in by_type["unmapped_category"].detail
    for finding in result.findings:
        if finding.source is FindingSource.PLAUSIBILITY:
            assert finding.evidence_ref and pd.read_csv(finding.evidence_ref).shape[0] > 0
    channel = pd.read_parquet(result.parsed_data_ref["sales"])["channel"]
    assert set(channel.dropna()) == {"store", "online", "marketplace", "ZZ"}  # rewritten


# --- single checks on small tables ------------------------------------------------------


def _sales(n=30, **overrides):
    data = {
        "transaction_date": pd.to_datetime(["2024-01-01"] * n),
        "store_id": pd.array(["S1"] * n, dtype="string"),
        "sku": pd.array(["A"] * n, dtype="string"),
        "quantity": [4.0] * n,
        "unit_price": [3.0] * n,
        "channel": pd.array(["store"] * n, dtype="string"),
    }
    data.update(overrides)
    frame = pd.DataFrame(data)
    frame["row"] = range(n)  # keeps rows distinct so duplicates are opt-in
    return frame


def _run(tmp_path, retail_model, settings, **tables):
    ws = session_workspace(tmp_path, "s1")
    ws.ensure()
    refs = {}
    for name, frame in tables.items():
        frame.to_parquet(ws.table_path(name), index=False)
        refs[name] = str(ws.table_path(name))
    result = run_plausibility(refs, retail_model, settings.plausibility, ws, TODAY)
    return result, {(f.field, f.issue_type.value): f for f in result.findings}, refs


def test_range_reports_both_sides_and_dates(tmp_path, retail_model, settings):
    dates = pd.to_datetime(["2024-01-01"] * 28 + ["2014-12-31", "2027-01-01"])
    sales = _sales(unit_price=[3.0] * 27 + [0.0, 700.0, 3.5], transaction_date=dates)
    _, found, _ = _run(tmp_path, retail_model, settings, sales=sales)
    price = found[("unit_price", "implausible_value")]
    assert price.count == 2 and "(1 below, 1 above)" in price.detail
    assert "cents" in price.detail  # 700 / 100 = 7.00 fits the range
    date_finding = found[("transaction_date", "implausible_value")]
    assert date_finding.count == 2 and "2026-10-25" in date_finding.detail  # 'today' resolved


def test_no_cents_hint_when_values_do_not_fit(tmp_path, retail_model, settings):
    _, found, _ = _run(tmp_path, retail_model, settings,
                       sales=_sales(unit_price=[3.0] * 29 + [90000.0]))  # fmt: skip
    assert "cents" not in found[("unit_price", "implausible_value")].detail


def test_cross_field_rules(tmp_path, retail_model, settings):
    sales = _sales(quantity=[4.0] * 28 + [4000.0, None], unit_price=[3.0] * 28 + [20.0, 9.0])
    _, found, _ = _run(tmp_path, retail_model, settings, sales=sales)
    rule = found[("quantity * unit_price between -10000 and 50000", "implausible_value")]
    assert rule.count == 1  # 80,000; the row with a missing quantity is skipped


def test_categories_normalize_and_skip_nulls(tmp_path, retail_model, settings):
    codes = ["Store", " web ", "B&M", "1", "MKT", "3", "zz", None] + ["store"] * 22
    result, found, refs = _run(tmp_path, retail_model, settings,
                               sales=_sales(channel=pd.array(codes, dtype="string")))  # fmt: skip
    assert found[("channel", "unmapped_category")].count == 1
    stored = pd.read_parquet(refs["sales"])["channel"].tolist()[:8]
    assert stored[:6] == ["store", "online", "store", "store", "marketplace", "marketplace"]
    assert stored[6] == "zz" and pd.isna(stored[7])
    assert "channel:categories" in result.checks_completed


def test_map_categories_is_idempotent(retail_model):
    spec = retail_model.tables["sales"].fields["channel"]
    once, unknown = map_categories(pd.Series(["B&M", "Web", "ZZ"], dtype="string"), spec)
    twice, unknown_again = map_categories(once, spec)
    assert once.tolist() == twice.tolist() == ["store", "online", "ZZ"]
    assert unknown.tolist() == unknown_again.tolist() == [False, False, True]


def _store_rows(store, qty, n=25):
    return _sales(n=n, store_id=pd.array([store] * n, dtype="string"), quantity=[qty] * n)


def test_units_use_pack_size_from_the_product_master(tmp_path, retail_model, settings):
    sales = pd.concat([_store_rows(f"S{i}", 4.0) for i in range(5)] + [_store_rows("S9", 48.0)])
    products = pd.DataFrame({"sku": pd.array(["A"], dtype="string"),
                             "category": pd.array(["x"], dtype="string"),
                             "pack_size": pd.array([12], dtype="Int64")})  # fmt: skip
    _, found, _ = _run(tmp_path, retail_model, settings, sales=sales.reset_index(drop=True),
                       products=products)  # fmt: skip
    assert found[("quantity", "unit_mismatch")].count == 25


def test_units_need_ratio_near_pack_and_enough_rows(tmp_path, retail_model, settings):
    sales = pd.concat([_store_rows(f"S{i}", 4.0) for i in range(5)]
                      + [_store_rows("S8", 34.0), _store_rows("S9", 48.0, n=5)])  # fmt: skip
    # S8 is 8.5x (not within 20% of 6, 12 or 24); S9 is 12x but has only 5 rows.
    _, found, _ = _run(tmp_path, retail_model, settings, sales=sales.reset_index(drop=True))
    assert ("quantity", "unit_mismatch") not in found


def test_units_check_stays_open_without_grouping_field(tmp_path, retail_model, settings):
    result, _, _ = _run(tmp_path, retail_model, settings, sales=_sales().drop(columns="store_id"))
    assert "quantity:units" not in result.checks_completed
    assert "quantity:range" in result.checks_completed


def test_join_not_applicable_without_target_table(tmp_path, retail_model, settings):
    result, found, _ = _run(tmp_path, retail_model, settings, sales=_sales())
    assert not any(issue == "join_failure" for _, issue in found)
    assert {"sku:join", "store_id:join"} <= set(result.checks_completed)


def test_duplicates_and_evidence_cap(tmp_path, retail_model, settings):
    sales = _sales().drop(columns="row")  # 30 identical rows -> 29 extra copies
    capped = settings.plausibility.model_copy(update={"max_evidence_rows": 5})
    ws = session_workspace(tmp_path, "s1").ensure()
    sales.to_parquet(ws.table_path("sales"), index=False)
    result = run_plausibility({"sales": str(ws.table_path("sales"))}, retail_model, capped, ws,
                              TODAY)  # fmt: skip
    (dup,) = [f for f in result.findings if f.issue_type.value == "duplicate_rows"]
    assert dup.field == "*" and dup.count == 29
    assert len(pd.read_csv(dup.evidence_ref)) == 5


def test_state_update_replaces_old_plausibility_findings(tmp_path, retail_model, settings,
                                                         fixtures_dir):  # fmt: skip
    old = Finding(field="unit_price", issue_type="implausible_value", severity="high",
                  detail="stale", count=99, source="plausibility")  # fmt: skip
    keep = Finding(field="quantity", issue_type="other", severity="low", detail="eda",
                   count=1, source="eda")  # fmt: skip
    options = ParseOptions(retries=1, plausibility=settings.plausibility, today=TODAY)
    paths = [fixtures_dir / "clean" / f for f in ("sales.csv", "products.csv", "stores.csv")]
    result = parse_files(paths, retail_model, session_workspace(tmp_path, "s1"), None, options)
    update = parsing_state_update({"findings": [old, keep]}, result)
    assert [f.detail for f in update["findings"]] == ["eda"]
