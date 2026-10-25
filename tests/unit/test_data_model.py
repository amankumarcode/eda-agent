from datetime import date

import pytest
import yaml
from pydantic import ValidationError

from dva.data_model import CrossFieldRule, DataModelNotFoundError, FieldType
from dva.data_model.yaml_provider import YamlDataModelProvider, data_model_from_dict

EXPECTED_REQUIRED_CHECKS = [
    "transaction_date:present", "transaction_date:range",
    "store_id:present", "store_id:join",
    "sku:present", "sku:join",
    "quantity:present", "quantity:range", "quantity:units",
    "unit_price:present", "unit_price:range",
    "channel:present", "channel:categories",
    "category:present",
    "pack_size:present", "pack_size:range",
    "region:present",
]  # fmt: skip


def test_retail_model_loads(retail_model):
    assert retail_model.name == "retail_sales"
    assert list(retail_model.tables) == ["sales", "products", "stores"]
    sales = retail_model.tables["sales"].fields
    assert sales["channel"].type is FieldType.CATEGORY
    assert sales["channel"].categories["store"] == ("Store", "B&M", "1")
    assert sales["sku"].join == "products.sku"
    assert sales["quantity"].units == "each"
    assert sales["promo_flag"].required is False
    assert "Units_Sold" in sales["quantity"].synonyms


def test_required_checks_match_spec_derivation(retail_model):
    assert retail_model.required_checks() == EXPECTED_REQUIRED_CHECKS


def test_ranges_resolve_today_and_numbers(retail_model):
    sales = retail_model.tables["sales"].fields
    today = date(2026, 10, 24)
    assert sales["transaction_date"].resolved_range(today) == (date(2015, 1, 1), today)
    assert sales["quantity"].resolved_range(today) == (-500.0, 5000.0)
    assert sales["sku"].resolved_range(today) is None


def test_cross_field_rules_parse(retail_model):
    first, second = retail_model.cross_field_rules
    assert (first.kind, first.left, first.op, first.right) == ("compare", "transaction_date",
                                                               "<=", "today")  # fmt: skip
    assert (second.kind, second.left, second.low, second.high) == (
        "between", "quantity * unit_price", -10000.0, 50000.0)  # fmt: skip
    assert second.identifiers() == {"quantity", "unit_price"}


@pytest.mark.parametrize(
    "rule",
    [
        "quantity",
        "quantity != 3",
        "__import__('os') <= 1",
        "quantity.real <= 1",
        "quantity <= 'a'",
        "quantity * unit_price between 10 and 1",
    ],  # fmt: skip
)
def test_unsupported_rules_fail_at_load(rule):
    with pytest.raises(ValueError):
        CrossFieldRule.parse(rule)


def _raw(project_root):
    return yaml.safe_load((project_root / "config/data_model/retail_sales.yaml").read_text())


def test_rule_with_unknown_field_is_rejected(project_root):
    raw = _raw(project_root)
    raw["cross_field_rules"].append("discount <= 1")
    with pytest.raises(ValidationError, match="unknown names"):
        data_model_from_dict(raw)


def test_join_to_missing_field_is_rejected(project_root):
    raw = _raw(project_root)
    raw["tables"]["sales"]["fields"]["sku"]["join"] = "products.product_id"
    with pytest.raises(ValidationError, match="join target"):
        data_model_from_dict(raw)


@pytest.mark.parametrize(
    ("field", "patch"),
    [
        ("channel", {"categories": None}),
        ("sku", {"categories": {"a": ["b"]}}),
        ("sku", {"range": [1, 2]}),
        ("quantity", {"range": [10, 1]}),
        ("transaction_date", {"range": ["not-a-date", "today"]}),
    ],  # fmt: skip
)
def test_invalid_field_specs_are_rejected(project_root, field, patch):
    raw = _raw(project_root)
    raw["tables"]["sales"]["fields"][field].update(patch)
    with pytest.raises(ValidationError):
        data_model_from_dict(raw)


def test_provider_errors(project_root, tmp_path):
    provider = YamlDataModelProvider(project_root / "config" / "data_model")
    with pytest.raises(DataModelNotFoundError):
        provider.get_model("no_such_model")
    with pytest.raises(DataModelNotFoundError):
        provider.get_model("../settings")
    (tmp_path / "other.yaml").write_text(yaml.safe_dump({**_raw(project_root)}))
    with pytest.raises(ValueError, match="declares name"):
        YamlDataModelProvider(tmp_path).get_model("other")
