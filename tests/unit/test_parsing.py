import json

import pandas as pd
import pytest
from langchain_core.messages import HumanMessage

from dva.agents.loaders import RawTable
from dva.agents.parsing import ParsingError, assign_table, parse_files, parsing_state_update
from dva.state import Finding, IssueType
from dva.workspace import session_workspace
from fakes import FakeChatModel

FIXTURES = [
    "clean", "renamed_columns", "missing_field", "unit_mismatch", "implausible_values",
    "category_variants", "join_failure", "type_errors", "duplicates", "excel_messy",
    "email_table", "kitchen_sink",
]  # fmt: skip
SALES_HEADER = "Txn_Date,Store_No,Item_No,Units,Sell_Price,Sales_Channel"


def _gt(fixtures_dir, name):
    return json.loads((fixtures_dir / name / "ground_truth.json").read_text())


def _files(fixtures_dir, name):
    return [fixtures_dir / name / f for f in _gt(fixtures_dir, name)["files"].values()]


def _parse(paths, model, tmp_path, mapper=None):
    ws = session_workspace(tmp_path, "s1")
    return parse_files(paths, model, ws, mapper or FakeChatModel(), retries=1)


def _counts(findings):
    return {(f.field, f.issue_type.value): f.count for f in findings}


def _expected_counts(gt):
    return {(i["field"], i["issue_type"]): i["count"] for i in gt["expected_issues"]}


def _mapping_accuracy(result, expected):
    got = {**result.field_mapping, **{c: None for c in result.unmapped_columns}}
    assert set(got) == set(expected)
    return sum(got[column] == field for column, field in expected.items()) / len(expected)


@pytest.mark.parametrize("name", FIXTURES)
def test_field_mapping_accuracy_on_every_fixture(fixtures_dir, tmp_path, retail_model, name):
    """Every fixture maps fully in code; the unmatched columns have no free field left,
    so the mapper is never asked (an empty fake script would fail on any call)."""
    mapper = FakeChatModel()
    result = _parse(_files(fixtures_dir, name), retail_model, tmp_path, mapper)
    assert _mapping_accuracy(result, _gt(fixtures_dir, name)["expected_field_mapping"]) == 1.0
    assert mapper.calls == []


@pytest.fixture
def renamed_without_promo(fixtures_dir, tmp_path):
    """renamed_columns with On_Promo dropped: promo_flag is free, so the mapper is asked
    about Cashier_Notes and Legacy_Code."""
    out = tmp_path / "input"
    out.mkdir()
    sales = pd.read_csv(fixtures_dir / "renamed_columns" / "sales.csv", dtype=str)
    sales.drop(columns=["On_Promo"]).to_csv(out / "sales.csv", index=False)
    for name in ("products.csv", "stores.csv"):
        (out / name).write_bytes((fixtures_dir / "renamed_columns" / name).read_bytes())
    expected = dict(_gt(fixtures_dir, "renamed_columns")["expected_field_mapping"])
    del expected["On_Promo"]
    return [out / f for f in ("sales.csv", "products.csv", "stores.csv")], expected


NULL_ANSWER = {"mappings": [{"column": "Cashier_Notes", "field": None},
                            {"column": "Legacy_Code", "field": None}]}  # fmt: skip


def test_fake_mapper_resolves_leftover_columns(renamed_without_promo, tmp_path, retail_model):
    paths, expected = renamed_without_promo
    mapper = FakeChatModel([NULL_ANSWER])
    result = _parse(paths, retail_model, tmp_path, mapper)
    assert _mapping_accuracy(result, expected) == 1.0
    assert mapper.remaining == 0 and result.mapper_first_pass == [True]
    assert "promo_flag" not in result.missing_required_fields  # promo_flag is optional


def test_mapper_prompt_has_names_but_no_data_values(renamed_without_promo, tmp_path, retail_model):
    paths, _ = renamed_without_promo
    mapper = FakeChatModel([NULL_ANSWER])
    _parse(paths, retail_model, tmp_path, mapper)
    prompt = "\n".join(str(m.content) for m in mapper.calls[0])
    assert "Cashier_Notes: text" in prompt and "Legacy_Code: text" in prompt
    assert "- promo_flag: boolean" in prompt and "- quantity:" not in prompt  # free fields only
    for value in ("price override", "loyalty card", "L0", "SKU00", "S0", "2023-"):
        assert value not in prompt


def test_mapper_failing_twice_fails_parsing_clearly(renamed_without_promo, tmp_path, retail_model):
    paths, _ = renamed_without_promo
    mapper = FakeChatModel([{"mappings": []}, {"mappings": [{"column": "nope", "field": None}]}])
    with pytest.raises(ParsingError, match="column mapper failed for table 'sales'"):
        _parse(paths, retail_model, tmp_path, mapper)


def _units_csv(tmp_path):
    path = tmp_path / "sales.csv"
    path.write_text(f"{SALES_HEADER}\n2024-01-02,S001,SKU00001,3,4.20,store\n")
    return path


def test_llm_maps_a_column_code_cannot_resolve(tmp_path, retail_model):
    mapper = FakeChatModel([{"mappings": [{"column": "Units", "field": "quantity"}]}])
    result = _parse([_units_csv(tmp_path)], retail_model, tmp_path, mapper)
    assert result.field_mapping["Units"] == "quantity"
    assert result.unmapped_columns == [] and result.mapper_first_pass == [True]


def test_invalid_mapper_answer_is_retried(tmp_path, retail_model):
    mapper = FakeChatModel([
        {"mappings": [{"column": "Units", "field": "sku"}]},  # sku is already taken
        {"mappings": [{"column": "Units", "field": "quantity"}]},
    ])  # fmt: skip
    result = _parse([_units_csv(tmp_path)], retail_model, tmp_path, mapper)
    assert result.field_mapping["Units"] == "quantity" and result.mapper_first_pass == [False]


def test_clean_fixture_has_no_findings_and_writes_parquet(fixtures_dir, tmp_path, retail_model):
    result = _parse(_files(fixtures_dir, "clean"), retail_model, tmp_path)
    assert result.findings == [] and result.missing_required_fields == []
    sales = pd.read_parquet(result.parsed_data_ref["sales"])
    assert list(sales.columns) == list(retail_model.tables["sales"].fields)
    assert str(sales["transaction_date"].dtype).startswith("datetime64")
    assert sales["quantity"].dtype == "float64" and str(sales["promo_flag"].dtype) == "boolean"
    assert str(pd.read_parquet(result.parsed_data_ref["products"])["pack_size"].dtype) == "Int64"
    assert result.data_profile["sales"]["rows"] == 2000
    assert result.data_profile["sales"]["sources"][0]["source"] == "sales.csv"
    assert all(c.endswith(":present") for c in result.checks_completed)


def test_type_errors_become_findings_with_ground_truth_counts(fixtures_dir, tmp_path, retail_model):
    gt = _gt(fixtures_dir, "type_errors")
    result = _parse(_files(fixtures_dir, "type_errors"), retail_model, tmp_path)
    assert _counts(result.findings) == _expected_counts(gt)
    sales = pd.read_parquet(result.parsed_data_ref["sales"])
    assert sales["transaction_date"].notna().all() and sales["unit_price"].notna().all()
    assert sales["transaction_date"].between("2023-01-01", "2024-12-31").all()


@pytest.mark.parametrize("name", ["missing_field", "renamed_columns"])
def test_missing_and_unmapped_match_ground_truth(fixtures_dir, tmp_path, retail_model, name):
    gt = _gt(fixtures_dir, name)
    result = _parse(_files(fixtures_dir, name), retail_model, tmp_path)
    assert _counts(result.findings) == _expected_counts(gt)
    assert result.missing_required_fields == gt["expected_missing_required_fields"]


@pytest.mark.parametrize("name", ["excel_messy", "email_table"])
def test_excel_and_email_fixtures_parse_cleanly(fixtures_dir, tmp_path, retail_model, name):
    result = _parse(_files(fixtures_dir, name), retail_model, tmp_path)
    assert result.findings == []
    assert {t: p["rows"] for t, p in result.data_profile.items()} == _gt(fixtures_dir, name)["rows"]
    first_source = {t: p["sources"][0] for t, p in result.data_profile.items()}
    if name == "excel_messy":
        assert first_source["sales"] == {"source": "sales.xlsx#Sales", "header_row": 3,
                                         "rows": 3000}  # fmt: skip
    else:
        assert first_source["sales"]["source"] == "sales.eml#sales_export.csv"
        assert first_source["stores"]["source"] == "stores.eml#html-table-1"


def test_missing_table_and_unknown_file(tmp_path, retail_model):
    (tmp_path / "notes.csv").write_text("foo,bar\n1,2\n")
    (tmp_path / "stores.csv").write_text("store_id,region\nS1,North\n")
    result = _parse([tmp_path / "notes.csv", tmp_path / "stores.csv"], retail_model, tmp_path)
    assert set(result.parsed_data_ref) == {"stores"}
    assert {"sku", "category"} <= set(result.missing_required_fields)
    other = [f for f in result.findings if f.issue_type is IssueType.OTHER]
    assert other[0].field == "notes.csv" and "no column matches" in other[0].detail
    no_file = [f for f in result.findings if f.field == "category"]
    assert no_file[0].count == 0 and "no file contained it" in no_file[0].detail


def test_file_name_breaks_ties_between_tables(retail_model):
    raw = RawTable(source="stores_master.csv", frame=pd.DataFrame(columns=["store_id"]))
    assert assign_table(raw, retail_model).table == "stores"
    raw = RawTable(source="export.csv", frame=pd.DataFrame(columns=["store_id"]))
    assert assign_table(raw, retail_model).table == "sales"  # declaration order


def test_state_update_clears_raw_files_and_merges(fixtures_dir, tmp_path, retail_model):
    result = _parse(_files(fixtures_dir, "missing_field"), retail_model, tmp_path)
    stale = Finding(field="channel", issue_type="missing_field", severity="high",
                    detail="old", count=1, source="parsing")  # fmt: skip
    eda = Finding(field="quantity", issue_type="other", severity="low", detail="eda note",
                  count=1, source="eda")  # fmt: skip
    state = {
        "raw_files": ["x.csv"],
        "findings": [stale, eda],
        "checks_completed": ["sku:join"],
        "parsed_data_ref": {"old": "old.parquet"},
        "messages": [HumanMessage("hi")],
    }
    update = parsing_state_update(state, result)
    assert update["raw_files"] is None
    assert update["parsed_data_ref"]["old"] == "old.parquet"
    assert "sales" in update["parsed_data_ref"]
    assert {"sku:join", "channel:present"} <= set(update["checks_completed"])
    details = [f.detail for f in update["findings"]]
    assert "old" not in details and "eda note" in details
    assert update["missing_required_fields"] == ["channel"]
    assert "messages" not in update
