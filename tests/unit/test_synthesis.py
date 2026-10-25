import json
from datetime import date
from pathlib import Path

import pytest

from dva.agents.parsing import ParseOptions, parse_files, parsing_state_update
from dva.agents.synthesis import (
    SynthesisError,
    SynthesisInputs,
    SynthesisWording,
    build_messages,
    fallback_wording,
    issue_list,
    run_synthesis,
    synthesis_inputs_from_state,
    synthesis_state_update,
    write_outputs,
)
from dva.review_queue import ReviewQueue
from dva.state import DraftStatus, Finding
from dva.workspace import session_workspace
from fakes import FakeChatModel

WORDING = SynthesisWording(
    report_summary="The upload is not ready: prices and quantities need fixing.",
    email_subject="Your sales upload: a few fixes needed",
    email_opening="Thanks for the files. A few issues need fixing before we can forecast.",
    email_closing="Please correct these and resend. Thank you!",
    jira_title="cust-418: data issues before first forecast",
    jira_summary="Customer upload has blocking issues in unit_price and quantity.",
)


@pytest.fixture
def kitchen(fixtures_dir, tmp_path, retail_model, settings):
    """Session state after parsing kitchen_sink: 6 findings of all three severities."""
    gt = json.loads((fixtures_dir / "kitchen_sink" / "ground_truth.json").read_text())
    ws = session_workspace(tmp_path, "cust-418")
    options = ParseOptions(retries=1, plausibility=settings.plausibility, today=date(2026, 10, 25))
    paths = [fixtures_dir / "kitchen_sink" / f for f in gt["files"].values()]
    state = parsing_state_update({}, parse_files(paths, retail_model, ws, None, options))
    state.update(session_id="cust-418", data_model_name="retail_sales",
                 required_checks=retail_model.required_checks())  # fmt: skip
    return state, ws


def test_every_finding_appears_in_report_grouped_by_severity(kitchen, settings):
    state, ws = kitchen
    result = run_synthesis(synthesis_inputs_from_state(state), FakeChatModel([WORDING]),
                           settings, ws)  # fmt: skip
    report = Path(result.report_ref).read_text()
    assert result.report_ref == str(ws.session_dir / "report.md")
    for finding in state["findings"]:
        row = [line for line in report.splitlines() if f"| {finding.count} |" in line
               and finding.issue_type.value in line]  # fmt: skip
        assert len(row) == 1, finding
        assert finding.detail.replace("|", "\\|") in row[0]
    high, medium, low = (report.index(f"## {s} severity") for s in ("High", "Medium", "Low"))
    assert high < medium < low
    assert report.index("unit_mismatch") < medium < report.index("duplicate_rows") < low
    assert "artifacts/unit_price_out_of_range.csv" in report
    assert WORDING.report_summary in report and "6 issues: 3 high, 2 medium, 1 low" in report
    assert "## Unmapped columns\n\nCashier_Notes" in report


def test_drafts_are_queued_pending_with_every_finding(kitchen, settings):
    state, ws = kitchen
    result = run_synthesis(synthesis_inputs_from_state(state), FakeChatModel([WORDING]),
                           settings, ws)  # fmt: skip
    queued = ReviewQueue(ws).drafts()
    assert sorted(d.kind for d in queued) == ["email", "jira"]
    assert all(d.status is DraftStatus.PENDING_REVIEW for d in queued)
    assert {d.draft_id for d in queued} == {d.draft_id for d in result.drafts}
    email = next(d for d in queued if d.kind == "email")
    assert email.title == WORDING.email_subject
    assert email.body.startswith(WORDING.email_opening)
    assert email.body.endswith(WORDING.email_closing)
    for finding in state["findings"]:
        assert finding.detail in email.body
    jira = next(d for d in queued if d.kind == "jira")
    assert jira.title == WORDING.jira_title and "6 issues" in jira.body


def test_prompt_has_findings_but_no_data_rows(kitchen):
    state, _ = kitchen
    prompt = "\n".join(m.content for m in build_messages(synthesis_inputs_from_state(state)))
    assert "high | unit_price | implausible_value | 37 |" in prompt
    assert "Unmapped columns: Cashier_Notes" in prompt
    for value in ("SKU00", "manager ok", "2023-", "Txn_Date"):
        assert value not in prompt


def test_invalid_wording_is_retried_then_fails_clearly(kitchen, settings):
    state, ws = kitchen
    inputs = synthesis_inputs_from_state(state)
    bad = WORDING.model_dump() | {"email_subject": ""}
    retried = run_synthesis(inputs, FakeChatModel([bad, WORDING]), settings, ws)
    assert not retried.first_pass_valid
    with pytest.raises(SynthesisError, match="no valid wording"):
        run_synthesis(inputs, FakeChatModel([bad, {"report_summary": "x"}]), settings, ws)


def test_partial_report_without_llm(tmp_path):
    finding = Finding(field="unit_price", issue_type="implausible_value", severity="high",
                      detail="37 prices above 500", count=37, source="plausibility")  # fmt: skip
    inputs = SynthesisInputs(session_id="s1", data_model_name="retail_sales",
                             findings=[finding], required_checks=["a:present", "b:join"],
                             checks_completed=["a:present"])  # fmt: skip
    ws = session_workspace(tmp_path, "s1")
    result = write_outputs(inputs, fallback_wording(inputs), ws, stop_reason="tool_call_cap")
    report = Path(result.report_ref).read_text()
    assert "(partial)" in report and "Stopped early: tool_call_cap" in report
    assert "Checks not completed: b:join" in report and "37 prices above 500" in report
    assert len(ReviewQueue(ws).drafts()) == 2


def test_empty_findings_and_pipes_are_handled(tmp_path):
    finding = Finding(field="a|b", issue_type="other", severity="low", detail="x | y",
                      count=1, source="eda")  # fmt: skip
    inputs = SynthesisInputs(session_id="s1", data_model_name="m", findings=[finding])
    result = write_outputs(inputs, WORDING, session_workspace(tmp_path, "s1"))
    assert "| a\\|b | other | 1 | x \\| y |" in Path(result.report_ref).read_text()
    assert issue_list([]) == "No issues were found."


def test_state_update_appends_drafts(kitchen, settings):
    state, ws = kitchen
    result = run_synthesis(synthesis_inputs_from_state(state), FakeChatModel([WORDING]),
                           settings, ws)  # fmt: skip
    update = synthesis_state_update({"draft_artifacts": ["earlier"]}, result)
    assert update["report_ref"] == result.report_ref
    assert update["report_summary"] == WORDING.report_summary
    assert update["draft_artifacts"][0] == "earlier" and len(update["draft_artifacts"]) == 3
