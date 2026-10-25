import json
from datetime import date

import pytest

from dva.agents.eda import (
    EdaCode,
    EdaError,
    EdaInputs,
    build_messages,
    eda_inputs_from_state,
    eda_state_update,
    run_eda,
)
from dva.agents.parsing import ParseOptions, parse_files, parsing_state_update
from dva.state import Finding, FindingSource, IssueType
from dva.workspace import session_workspace
from fakes import FakeChatModel

UNITS_CODE = """\
sales = pd.read_parquet(DATA["sales"])
positive = sales[sales["quantity"] > 0]
medians = positive.groupby("store_id")["quantity"].median()
ratio = medians / medians.median()
flagged = sorted(ratio[(ratio > 8) & (ratio < 16)].index)
rows = sales[sales["store_id"].isin(flagged)]
path = f"{ARTIFACTS_DIR}/case_stores.csv"
rows.to_csv(path, index=False)
print("flagged:", flagged)
emit([{"field": "quantity", "issue_type": "unit_mismatch", "severity": "high",
       "detail": f"{len(flagged)} stores report about 12x the typical quantity",
       "count": int(len(rows)), "evidence_ref": path}], ["quantity:units"])
"""
PRICE_CODE = """\
sales = pd.read_parquet(DATA["sales"])
high = sales[sales["unit_price"] > 500]
emit([{"field": "unit_price", "issue_type": "implausible_value", "severity": "high",
       "detail": f"{len(high)} prices above 500", "count": int(len(high))}], ["unit_price:range"])
"""
JOIN_CODE = """\
sales = pd.read_parquet(DATA["sales"])
products = pd.read_parquet(DATA["products"])
orphans = sales[~sales["sku"].isin(products["sku"])]
emit([{"field": "sku", "issue_type": "join_failure", "severity": "high",
       "detail": f"{len(orphans)} rows with unknown SKUs", "count": int(len(orphans))}],
     ["sku:join"])
"""


def _code(code, checks):
    return EdaCode(code=code, checks_addressed=checks)


def _parsed_inputs(fixtures_dir, tmp_path, retail_model, settings, name, reopen):
    """Parse a fixture, then reopen some checks so EDA has something to address."""
    gt = json.loads((fixtures_dir / name / "ground_truth.json").read_text())
    options = ParseOptions(retries=1, plausibility=settings.plausibility, today=date(2026, 10, 25))
    ws = session_workspace(tmp_path, "s1")
    paths = [fixtures_dir / name / f for f in gt["files"].values()]
    state = parsing_state_update({}, parse_files(paths, retail_model, ws, None, options))
    state["required_checks"] = retail_model.required_checks()
    state["checks_completed"] = [c for c in state["checks_completed"] if c not in reopen]
    state["user_request"] = "Check the data"
    return gt, ws, state


@pytest.mark.parametrize(
    ("name", "code", "check", "key"),
    [
        ("unit_mismatch", UNITS_CODE, "quantity:units", ("quantity", "unit_mismatch")),
        ("implausible_values", PRICE_CODE, "unit_price:range", ("unit_price", "implausible_value")),
        ("join_failure", JOIN_CODE, "sku:join", ("sku", "join_failure")),
    ],  # fmt: skip
)
def test_canned_code_matches_ground_truth(fixtures_dir, tmp_path, retail_model, settings,
                                          name, code, check, key):  # fmt: skip
    gt, ws, state = _parsed_inputs(fixtures_dir, tmp_path, retail_model, settings, name, {check})
    coder = FakeChatModel([_code(code, [check])])
    result = run_eda(eda_inputs_from_state(state), coder, settings, ws)
    assert result.succeeded and result.attempts == 1 and result.checks_completed == [check]
    (finding,) = result.findings
    assert (finding.field, finding.issue_type.value) == key and finding.source is FindingSource.EDA
    expected = next(i for i in gt["expected_issues"] if (i["field"], i["issue_type"]) == key)
    target = expected.get("count", expected.get("count_approx"))
    tolerance = 0 if "count" in expected else 0.1 * target
    assert abs(finding.count - target) <= tolerance


def test_failing_first_attempt_recovers_on_retry(fixtures_dir, tmp_path, retail_model, settings):
    _, ws, state = _parsed_inputs(fixtures_dir, tmp_path, retail_model, settings,
                                  "unit_mismatch", {"quantity:units"})  # fmt: skip
    broken = UNITS_CODE.replace('["quantity"] > 0', '["qty"] > 0')
    coder = FakeChatModel(
        [_code(broken, ["quantity:units"]), _code(UNITS_CODE, ["quantity:units"])]
    )
    result = run_eda(eda_inputs_from_state(state), coder, settings, ws)
    assert result.succeeded and result.attempts == 2
    fix_prompt = coder.calls[1][-1].content
    assert (
        "The code failed" in fix_prompt and "KeyError" in fix_prompt and "analysis.py" in fix_prompt
    )


def test_second_failure_records_other_finding(fixtures_dir, tmp_path, retail_model, settings):
    _, ws, state = _parsed_inputs(fixtures_dir, tmp_path, retail_model, settings, "clean", set())
    coder = FakeChatModel([_code("1 / 0", []), _code("x = undefined_name", [])])
    result = run_eda(eda_inputs_from_state(state), coder, settings, ws)
    assert not result.succeeded and result.attempts == 2 and result.checks_completed == []
    (finding,) = result.findings
    assert finding.issue_type is IssueType.OTHER and finding.field == "eda"
    assert "NameError" in finding.detail and coder.remaining == 0


def test_policy_violation_is_sent_back_for_a_fix(fixtures_dir, tmp_path, retail_model, settings):
    _, ws, state = _parsed_inputs(fixtures_dir, tmp_path, retail_model, settings,
                                  "implausible_values", {"unit_price:range"})  # fmt: skip
    coder = FakeChatModel([_code("import os\n" + PRICE_CODE, ["unit_price:range"]),
                           _code(PRICE_CODE, ["unit_price:range"])])  # fmt: skip
    result = run_eda(eda_inputs_from_state(state), coder, settings, ws)
    assert result.succeeded and result.attempts == 2
    assert "import of 'os' is not allowed" in coder.calls[1][-1].content


def test_unknown_check_ids_are_not_marked_complete(fixtures_dir, tmp_path, retail_model, settings):
    _, ws, state = _parsed_inputs(fixtures_dir, tmp_path, retail_model, settings, "clean", set())
    coder = FakeChatModel([_code("emit([], ['made:up', 'sku:join'])", [])])
    result = run_eda(eda_inputs_from_state(state), coder, settings, ws)
    assert result.checks_completed == ["sku:join"]


def test_prompt_has_structure_but_no_data_rows(fixtures_dir, tmp_path, retail_model, settings):
    _, ws, state = _parsed_inputs(fixtures_dir, tmp_path, retail_model, settings,
                                  "kitchen_sink", {"quantity:units"})  # fmt: skip
    messages = build_messages(eda_inputs_from_state(state, focus="check quantity units only"),
                              settings)  # fmt: skip
    prompt = "\n".join(m.content for m in messages)
    for expected in (
        state["parsed_data_ref"]["sales"],
        '"unit_price": "float64"',
        '"Units_Sold": "quantity"',
        "Checks remaining: quantity:units",
        "User question: Check the data",
        "Focus: check quantity units only",
        "- unit_price: implausible_value (37 rows)",
        "Allowed imports: pandas",
    ):
        assert expected in prompt, expected
    for value in ("SKU00", "S004", "manager ok", "2023-", "3.83"):
        assert value not in prompt, value


def test_errors_without_data_or_valid_code(fixtures_dir, tmp_path, retail_model, settings):
    ws = session_workspace(tmp_path, "s1")
    empty = EdaInputs(parsed_data_ref={}, data_profile={}, field_mapping={},
                      required_checks=[], checks_completed=[])  # fmt: skip
    with pytest.raises(EdaError, match="no parsed data"):
        run_eda(empty, FakeChatModel(), settings, ws)
    _, ws, state = _parsed_inputs(fixtures_dir, tmp_path, retail_model, settings, "clean", set())
    with pytest.raises(EdaError, match="no valid code"):
        run_eda(eda_inputs_from_state(state), FakeChatModel([{"code": ""}, {"nope": 1}]),
                settings, ws)  # fmt: skip


def _finding(field, issue, source, detail):
    return Finding(field=field, issue_type=issue, severity="high", detail=detail, count=1,
                   source=source)  # fmt: skip


def test_state_update_keeps_deterministic_findings(fixtures_dir, tmp_path, retail_model, settings):
    plaus = _finding("quantity", "unit_mismatch", "plausibility", "from plausibility")
    old_eda = _finding("region", "other", "eda", "old eda")
    _, ws, state = _parsed_inputs(fixtures_dir, tmp_path, retail_model, settings,
                                  "unit_mismatch", {"quantity:units"})  # fmt: skip
    coder = FakeChatModel([_code(UNITS_CODE, ["quantity:units"])])
    result = run_eda(eda_inputs_from_state(state), coder, settings, ws)
    new_eda = result.model_copy(update={"findings": result.findings + [
        _finding("region", "other", "eda", "new eda")]})  # fmt: skip
    update = eda_state_update({"findings": [plaus, old_eda], "checks_completed": ["a:b"]}, new_eda)
    details = sorted(f.detail for f in update["findings"])
    assert details == ["from plausibility", "new eda"]
    assert update["checks_completed"] == ["a:b", "quantity:units"]
