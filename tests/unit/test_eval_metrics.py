import pytest

from dva.state import Finding
from metrics import (
    Issue,
    RunRecord,
    code_match,
    mapping_accuracy,
    match_findings,
    meets_target,
    summarize,
)
from run_evals import score


def _f(field, issue_type, count, source="plausibility"):
    return Finding(field=field, issue_type=issue_type, severity="high", detail="d", count=count,
                   source=source)  # fmt: skip


ISSUES = [
    Issue(field="unit_price", issue_type="implausible_value", count=37),
    Issue(field="quantity", issue_type="unit_mismatch", count_approx=400),
    Issue(field="sku", issue_type="join_failure", count=412),
]


@pytest.mark.parametrize(
    ("finding", "issue", "expected"),
    [
        (_f("unit_price", "implausible_value", 37), ISSUES[0], True),
        (_f("unit_price", "implausible_value", 36), ISSUES[0], False),
        (_f("quantity", "unit_mismatch", 440), ISSUES[1], True),
        (_f("quantity", "unit_mismatch", 441), ISSUES[1], False),
        (_f("quantity", "implausible_value", 400), ISSUES[1], False),
    ],  # fmt: skip
)
def test_code_match_tolerance(finding, issue, expected):
    assert code_match(finding, issue) is expected


def test_match_findings_uses_judge_only_for_leftovers():
    asked = []

    def judge(finding, issue):
        asked.append((finding.field, issue.field))
        return finding.field == "Item_No" and issue.field == "sku"

    findings = [
        _f("unit_price", "implausible_value", 37),
        _f("Item_No", "other", 410, "eda"),  # described differently
        _f("region", "other", 1, "eda"),
    ]  # fmt: skip  (not a real issue)
    decisions, found, correct = match_findings(findings, ISSUES, judge)
    assert found == {0, 2} and correct == 2
    assert [(d.method, d.matched) for d in decisions] == [("code", True), ("llm", True),
                                                          ("none", False)]  # fmt: skip
    assert ("unit_price", "unit_price") not in asked  # code matches never reach the judge
    assert asked[0] == ("Item_No", "quantity")  # unmatched issues are asked first


def test_match_without_judge():
    decisions, found, correct = match_findings([_f("x", "other", 1, "eda")], ISSUES, None)
    assert found == set() and correct == 0 and decisions[0].method == "none"


def test_mapping_accuracy():
    state = {"field_mapping": {"Qty": "quantity", "Item": "sku"}, "unmapped_columns": ["Notes"]}
    expected = {"Qty": "quantity", "Item": "store_id", "Notes": None, "Gone": "region"}
    assert mapping_accuracy(state, expected) == (2, 4)


def test_summarize_and_targets():
    records = [
        RunRecord(fixture="a", session="a", stop_reason="task_complete", mapping_correct=9,
                  mapping_total=10, issues_found=2, issues_total=2, findings_correct=2,
                  findings_total=3, eda_ok=1, eda_total=1, tool_calls=3, tokens=100,
                  latency_s=10.0),
        RunRecord(fixture="b", session="b", stop_reason="tool_call_cap", mapping_correct=10,
                  mapping_total=10, issues_found=0, issues_total=1, findings_correct=1,
                  findings_total=1, eda_ok=0, eda_total=1, tool_calls=6, tokens=300,
                  latency_s=30.0),
    ]  # fmt: skip
    overall = summarize(records)
    assert overall["field_mapping_accuracy"] == 0.95 and overall["issue_recall"] == 2 / 3
    assert overall["issue_precision"] == 0.75 and overall["cap_hit_rate"] == 0.5
    assert overall["tokens"] == {"mean": 200.0, "max": 300.0}
    assert meets_target("field_mapping_accuracy", 0.95) is True
    assert meets_target("cap_hit_rate", 0.5) is False
    assert meets_target("first_pass_schema_rate", 0.5) is None
    assert summarize([])["issue_recall"] is None


def test_score_reads_state_and_tool_events():
    state = {"stop_reason": "task_complete", "tool_calls": 3, "tokens_used": 1234,
             "report_ref": "r.md", "field_mapping": {"sku": "sku"}, "unmapped_columns": [],
             "findings": [_f("sku", "join_failure", 412)]}  # fmt: skip
    gt = {"expected_issues": [{"field": "sku", "issue_type": "join_failure", "count": 412}],
          "expected_field_mapping": {"sku": "sku"}}  # fmt: skip
    events = [("run_parsing", {"ok": True, "first_pass": []}),
              ("run_eda", {"ok": False, "attempts": 2, "first_pass": [True, False]}),
              ("run_eda", {"ok": True, "attempts": 1, "first_pass": [True]}),
              ("run_synthesis", {"ok": True, "first_pass": [True]})]  # fmt: skip
    record = score(RunRecord(fixture="j", session="j"), state, gt, events, None)
    assert (record.issues_found, record.findings_correct, record.mapping_correct) == (1, 1, 1)
    assert (record.eda_ok, record.eda_total) == (1, 2)
    assert (record.first_pass_ok, record.first_pass_total) == (3, 4)
    assert (record.tool_calls, record.tokens, record.report_ref) == (3, 1234, "r.md")
