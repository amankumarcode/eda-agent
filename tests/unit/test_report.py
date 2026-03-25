from pathlib import Path
from unittest.mock import patch

import plotly.graph_objects as go
import pytest

from core.schema import (
    AgentResult,
    AnalysisPlan,
    EvaluationResult,
    OutputType,
    Verdict,
)
from core.state import AgentState
from outputs.report import (
    _get_viz_findings,
    _reconstruct_charts,
    _verdict_badge,
    generate_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def viz_chart_dict() -> dict:
    """A simple Plotly Scatter figure serialised as a dict."""
    fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name="test"))
    return fig.to_dict()


@pytest.fixture
def state_with_report_data(base_state: AgentState, viz_chart_dict: dict) -> AgentState:
    return {
        **base_state,
        "narrative": "Income is strongly driven by age in this dataset.",
        "key_insights": ["Age explains 85% of income variance (r=0.85)"],
        "caveats": ["Small sample size of 10 rows"],
        "agent_results": [
            AgentResult(
                agent_name="profiler",
                success=True,
                findings={"notable_columns": ["income"]},
                confidence=0.9,
            ),
            AgentResult(
                agent_name="viz_agent",
                success=True,
                findings={
                    "charts": [viz_chart_dict],
                    "chart_descriptions": ["Scatter of age vs income"],
                    "recommended_primary_chart": "age_income_scatter",
                },
                confidence=0.8,
            ),
        ],
        "evaluation": EvaluationResult(
            goal_coverage=0.9,
            insight_quality=0.85,
            evidence_quality=0.9,
            overall_score=0.89,
            strengths=["Directly addresses goal"],
            gaps=[],
            verdict=Verdict.STRONG,
            retry_instructions=None,
        ),
    }


@pytest.fixture
def state_no_caveats(state_with_report_data: AgentState) -> AgentState:
    return {**state_with_report_data, "caveats": []}


# ---------------------------------------------------------------------------
# _get_viz_findings
# ---------------------------------------------------------------------------

def test_get_viz_findings_returns_empty_when_no_viz_agent(
    base_state: AgentState,
) -> None:
    assert _get_viz_findings(base_state) == {}


def test_get_viz_findings_returns_findings_when_present(
    state_with_report_data: AgentState,
) -> None:
    result = _get_viz_findings(state_with_report_data)
    assert isinstance(result, dict)
    assert "charts" in result


def test_get_viz_findings_ignores_failed_viz_agent(base_state: AgentState) -> None:
    state = {
        **base_state,
        "agent_results": [
            AgentResult(
                agent_name="viz_agent",
                success=False,
                findings={"charts": [{"data": [], "layout": {}}]},
                confidence=0.0,
            )
        ],
    }
    assert _get_viz_findings(state) == {}


# ---------------------------------------------------------------------------
# _reconstruct_charts
# ---------------------------------------------------------------------------

def test_reconstruct_charts_empty_input() -> None:
    assert _reconstruct_charts({}) == []


def test_reconstruct_charts_empty_charts_list() -> None:
    assert _reconstruct_charts({"charts": []}) == []


def test_reconstruct_charts_returns_list_of_strings(viz_chart_dict: dict) -> None:
    result = _reconstruct_charts({"charts": [viz_chart_dict]})
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], str)


def test_reconstruct_charts_contains_div(viz_chart_dict: dict) -> None:
    result = _reconstruct_charts({"charts": [viz_chart_dict]})
    assert "<div" in result[0]


def test_reconstruct_charts_skips_invalid_entries(viz_chart_dict: dict) -> None:
    """Bad entry skipped, valid one still returned."""
    result = _reconstruct_charts({"charts": ["not_a_dict", viz_chart_dict]})
    assert len(result) == 1


# ---------------------------------------------------------------------------
# _verdict_badge
# ---------------------------------------------------------------------------

def test_verdict_badge_strong_contains_strong() -> None:
    badge = _verdict_badge(Verdict.STRONG)
    assert "strong" in badge


def test_verdict_badge_adequate_contains_adequate() -> None:
    badge = _verdict_badge(Verdict.ADEQUATE)
    assert "adequate" in badge


def test_verdict_badge_weak_contains_weak() -> None:
    badge = _verdict_badge(Verdict.WEAK)
    assert "weak" in badge


def test_verdict_badge_none_contains_not_evaluated() -> None:
    badge = _verdict_badge(None)
    assert "Not evaluated" in badge


def test_verdict_badge_returns_span() -> None:
    assert _verdict_badge(Verdict.STRONG).startswith("<span")


# ---------------------------------------------------------------------------
# generate_report — file output
# ---------------------------------------------------------------------------

def test_generate_report_returns_string_path(
    state_with_report_data: AgentState, tmp_path: Path
) -> None:
    with patch("outputs.report.os.getenv", return_value=str(tmp_path)):
        result = generate_report(state_with_report_data)

    assert isinstance(result, str)
    assert result.endswith(".html")


def test_generate_report_file_exists(
    state_with_report_data: AgentState, tmp_path: Path
) -> None:
    with patch("outputs.report.os.getenv", return_value=str(tmp_path)):
        path = generate_report(state_with_report_data)

    assert Path(path).exists()


def test_generate_report_contains_doctype(
    state_with_report_data: AgentState, tmp_path: Path
) -> None:
    with patch("outputs.report.os.getenv", return_value=str(tmp_path)):
        path = generate_report(state_with_report_data)

    content = Path(path).read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content


def test_generate_report_contains_goal(
    state_with_report_data: AgentState, tmp_path: Path
) -> None:
    with patch("outputs.report.os.getenv", return_value=str(tmp_path)):
        path = generate_report(state_with_report_data)

    content = Path(path).read_text(encoding="utf-8")
    assert state_with_report_data["request"].goal in content


def test_generate_report_contains_narrative(
    state_with_report_data: AgentState, tmp_path: Path
) -> None:
    with patch("outputs.report.os.getenv", return_value=str(tmp_path)):
        path = generate_report(state_with_report_data)

    content = Path(path).read_text(encoding="utf-8")
    assert "Income is strongly driven by age" in content


def test_generate_report_contains_key_insight(
    state_with_report_data: AgentState, tmp_path: Path
) -> None:
    with patch("outputs.report.os.getenv", return_value=str(tmp_path)):
        path = generate_report(state_with_report_data)

    content = Path(path).read_text(encoding="utf-8")
    assert "Age explains 85% of income variance" in content


def test_generate_report_contains_plotly_when_charts_present(
    state_with_report_data: AgentState, tmp_path: Path
) -> None:
    with patch("outputs.report.os.getenv", return_value=str(tmp_path)):
        path = generate_report(state_with_report_data)

    content = Path(path).read_text(encoding="utf-8")
    assert "plotly" in content.lower()


def test_generate_report_contains_evaluation_scorecard(
    state_with_report_data: AgentState, tmp_path: Path
) -> None:
    with patch("outputs.report.os.getenv", return_value=str(tmp_path)):
        path = generate_report(state_with_report_data)

    content = Path(path).read_text(encoding="utf-8")
    assert "0.89" in content  # overall_score


def test_generate_report_omits_empty_caveats_section(
    state_no_caveats: AgentState, tmp_path: Path
) -> None:
    with patch("outputs.report.os.getenv", return_value=str(tmp_path)):
        path = generate_report(state_no_caveats)

    content = Path(path).read_text(encoding="utf-8")
    assert "Caveats" not in content


def test_generate_report_creates_output_directory(
    state_with_report_data: AgentState, tmp_path: Path
) -> None:
    nested = tmp_path / "new_nested_dir"
    with patch("outputs.report.os.getenv", return_value=str(nested)):
        path = generate_report(state_with_report_data)

    assert Path(path).parent.exists()


def test_generate_report_filename_contains_session_id(
    state_with_report_data: AgentState, tmp_path: Path
) -> None:
    with patch("outputs.report.os.getenv", return_value=str(tmp_path)):
        path = generate_report(state_with_report_data)

    assert "test-session-001" in Path(path).name


def test_generate_report_raises_on_write_failure(
    state_with_report_data: AgentState, tmp_path: Path
) -> None:
    with (
        patch("outputs.report.os.getenv", return_value=str(tmp_path)),
        patch("builtins.open", side_effect=PermissionError("no write access")),
    ):
        with pytest.raises(RuntimeError, match="generate_report failed"):
            generate_report(state_with_report_data)
