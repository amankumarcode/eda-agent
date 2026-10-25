import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from dva.state import (
    TURN_FIELDS,
    Draft,
    DraftStatus,
    Finding,
    IssueType,
    PipelineState,
    Severity,
)


def _finding(**overrides):
    data = {
        "field": "unit_price",
        "issue_type": "implausible_value",
        "severity": "high",
        "detail": "37 prices above $500",
        "count": 37,
        "source": "plausibility",
    }
    return Finding(**{**data, **overrides})


def test_finding_parses_spec_example():
    finding = _finding(evidence_ref="workspace/cust-418/artifacts/unit_price_outliers.csv")
    assert finding.issue_type is IssueType.IMPLAUSIBLE_VALUE
    assert finding.severity is Severity.HIGH
    assert finding.model_dump(mode="json")["source"] == "plausibility"


def test_issue_type_enum_matches_spec():
    assert {t.value for t in IssueType} == {
        "missing_field", "unmapped_column", "type_mismatch", "implausible_value",
        "unit_mismatch", "unmapped_category", "join_failure", "duplicate_rows", "other",
    }  # fmt: skip


@pytest.mark.parametrize(
    "overrides",
    [
        {"count": -1},
        {"issue_type": "bogus"},
        {"severity": "critical"},
        {"source": "llm"},
        {"field": ""},
        {"unexpected": 1},
    ],  # fmt: skip
)
def test_finding_rejects_invalid_values(overrides):
    with pytest.raises(ValidationError):
        _finding(**overrides)


def test_draft_defaults_to_pending_review():
    draft = Draft(draft_id="d1", session_id="cust-418", kind="email", title="t", body="b")
    assert draft.status is DraftStatus.PENDING_REVIEW
    assert draft.created_at.tzinfo is not None
    with pytest.raises(ValidationError):
        Draft(draft_id="d1", session_id="s", kind="slack", title="t", body="b")


def test_turn_fields_are_state_keys():
    assert set(TURN_FIELDS) <= set(PipelineState.__annotations__)


def test_state_works_in_a_graph_with_message_reducer():
    def node(state: PipelineState) -> dict:
        return {"messages": [AIMessage(content="done")], "findings": [_finding()], "tool_calls": 1}

    graph = StateGraph(PipelineState)
    graph.add_node("n", node)
    graph.add_edge(START, "n")
    graph.add_edge("n", END)
    out = graph.compile().invoke({"messages": [HumanMessage(content="hi")], "findings": []})
    assert [m.content for m in out["messages"]] == ["hi", "done"]
    assert out["findings"] == [_finding()]
    assert out["tool_calls"] == 1
