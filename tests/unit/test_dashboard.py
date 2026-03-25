"""Tests for outputs/dashboard.py."""
import os
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pandas as pd
import pytest

from core.schema import AgentResult, EDARequest, EvaluationResult, OutputType, Verdict
from core.state import AgentState
from outputs.dashboard import (
    _build_dashboard_html,
    _get_chart_descriptions,
    _get_viz_findings,
    _reconstruct_charts,
    generate_dashboard,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age":    [25, 32, 47, 51, 23],
        "income": [40000, 58000, 92000, 105000, 36000],
        "region": ["N", "S", "E", "W", "N"],
    })


def _make_chart_spec(title: str = "Test Chart") -> Dict[str, Any]:
    return {
        "data": [{"type": "bar", "x": [1, 2, 3], "y": [4, 5, 6]}],
        "layout": {"title": title},
    }


def _make_profiler_result() -> AgentResult:
    return AgentResult(
        agent_name="profiler",
        success=True,
        findings={"schema": {}, "notable_columns": []},
        confidence=0.9,
    )


def _make_viz_result(charts=None, descriptions=None, recommended="") -> AgentResult:
    return AgentResult(
        agent_name="viz_agent",
        success=True,
        findings={
            "charts": charts or [_make_chart_spec()],
            "chart_descriptions": descriptions or ["A test chart."],
            "recommended_primary_chart": recommended,
        },
        confidence=0.9,
    )


def _base_state(sample_df: pd.DataFrame, **overrides) -> AgentState:
    request = EDARequest(
        goal="Find income drivers",
        session_id="dash-test-001",
        output_formats=[OutputType.DASHBOARD],
        metadata={"shape": list(sample_df.shape), "columns": list(sample_df.columns)},
    )
    state: AgentState = {
        "request": request,
        "plan": None,
        "plan_approved": False,
        "agent_results": [],
        "rerun_agent": None,
        "rerun_count": 0,
        "narrative": "This dataset shows income correlates with age.",
        "key_insights": ["Age is positively correlated with income."],
        "caveats": ["Small sample size."],
        "output_paths": {},
        "messages": [],
        "evaluation": None,
        "evaluation_count": 0,
        "dispatched_agents": [],
        "completed_agents": [],
        "next_action": None,
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# _get_viz_findings
# ---------------------------------------------------------------------------

def test_get_viz_findings_returns_dict_when_present(sample_df):
    viz = _make_viz_result()
    state = _base_state(sample_df, agent_results=[viz])
    findings = _get_viz_findings(state)
    assert isinstance(findings, dict)
    assert "charts" in findings


def test_get_viz_findings_returns_empty_when_absent(sample_df):
    state = _base_state(sample_df)
    assert _get_viz_findings(state) == {}


def test_get_viz_findings_skips_failed_result(sample_df):
    failed = AgentResult(
        agent_name="viz_agent",
        success=False,
        findings={"charts": [_make_chart_spec()]},
        confidence=0.0,
    )
    state = _base_state(sample_df, agent_results=[failed])
    assert _get_viz_findings(state) == {}


# ---------------------------------------------------------------------------
# _get_chart_descriptions
# ---------------------------------------------------------------------------

def test_get_chart_descriptions_returns_list():
    findings = {"chart_descriptions": ["Chart A", "Chart B"]}
    descs = _get_chart_descriptions(findings)
    assert descs == ["Chart A", "Chart B"]


def test_get_chart_descriptions_empty_when_absent():
    assert _get_chart_descriptions({}) == []


# ---------------------------------------------------------------------------
# _reconstruct_charts
# ---------------------------------------------------------------------------

def test_reconstruct_charts_returns_html_strings():
    findings = {"charts": [_make_chart_spec()]}
    divs = _reconstruct_charts(findings)
    assert len(divs) == 1
    assert isinstance(divs[0], str)
    assert "<div" in divs[0]


def test_reconstruct_charts_returns_empty_for_no_charts():
    divs = _reconstruct_charts({})
    assert divs == []


def test_reconstruct_charts_skips_invalid_entry():
    findings = {"charts": [{"bad": "data"}, _make_chart_spec()]}
    divs = _reconstruct_charts(findings)
    assert len(divs) >= 1


def test_reconstruct_charts_does_not_include_full_plotly_js():
    findings = {"charts": [_make_chart_spec()]}
    divs = _reconstruct_charts(findings)
    combined = " ".join(divs)
    assert "cdn.plot.ly" not in combined


# ---------------------------------------------------------------------------
# _build_dashboard_html
# ---------------------------------------------------------------------------

def test_build_dashboard_html_contains_plotly_cdn(sample_df):
    state = _base_state(sample_df)
    html = _build_dashboard_html(state, [])
    assert "plotly" in html.lower()


def test_build_dashboard_html_contains_session_id(sample_df):
    state = _base_state(sample_df)
    html = _build_dashboard_html(state, [])
    assert "dash-test-001" in html


def test_build_dashboard_html_contains_goal(sample_df):
    state = _base_state(sample_df)
    html = _build_dashboard_html(state, [])
    assert "Find income drivers" in html


def test_build_dashboard_html_embeds_chart_divs(sample_df):
    state = _base_state(sample_df)
    fake_div = "<div id='fake-chart'>chart here</div>"
    html = _build_dashboard_html(state, [fake_div])
    assert "fake-chart" in html


def test_build_dashboard_html_includes_narrative(sample_df):
    state = _base_state(sample_df, narrative="Income correlates with age.")
    html = _build_dashboard_html(state, [])
    assert "Income correlates with age." in html


def test_build_dashboard_html_includes_key_insights(sample_df):
    state = _base_state(sample_df, key_insights=["Insight one", "Insight two"])
    html = _build_dashboard_html(state, [])
    assert "Insight one" in html
    assert "Insight two" in html


def test_build_dashboard_html_includes_agent_confidence(sample_df):
    profiler = _make_profiler_result()
    state = _base_state(sample_df, agent_results=[profiler])
    html = _build_dashboard_html(state, [])
    assert "profiler" in html
    assert "90%" in html


def test_build_dashboard_html_omits_narrative_when_empty(sample_df):
    state = _base_state(sample_df, narrative="")
    html = _build_dashboard_html(state, [])
    assert "Summary" not in html


def test_build_dashboard_html_omits_insights_when_empty(sample_df):
    state = _base_state(sample_df, key_insights=[])
    html = _build_dashboard_html(state, [])
    assert "Key Insights" not in html


def test_build_dashboard_html_marks_primary_chart(sample_df):
    viz = _make_viz_result(
        descriptions=["Distribution of age"],
        recommended="Distribution of age",
    )
    state = _base_state(sample_df, agent_results=[viz])
    divs = _reconstruct_charts(viz.findings)
    html = _build_dashboard_html(state, divs)
    assert "Primary" in html


# ---------------------------------------------------------------------------
# generate_dashboard
# ---------------------------------------------------------------------------

def test_generate_dashboard_returns_path_string(sample_df, tmp_path):
    state = _base_state(sample_df)
    with patch.dict(os.environ, {"OUTPUT_DIR": str(tmp_path)}):
        result = generate_dashboard(state)
    assert isinstance(result, str)


def test_generate_dashboard_file_ends_with_html(sample_df, tmp_path):
    state = _base_state(sample_df)
    with patch.dict(os.environ, {"OUTPUT_DIR": str(tmp_path)}):
        result = generate_dashboard(state)
    assert result.endswith(".html")


def test_generate_dashboard_file_exists(sample_df, tmp_path):
    state = _base_state(sample_df)
    with patch.dict(os.environ, {"OUTPUT_DIR": str(tmp_path)}):
        result = generate_dashboard(state)
    assert Path(result).exists()


def test_generate_dashboard_filename_contains_session_id(sample_df, tmp_path):
    state = _base_state(sample_df)
    with patch.dict(os.environ, {"OUTPUT_DIR": str(tmp_path)}):
        result = generate_dashboard(state)
    assert "dash-test-001" in Path(result).name


def test_generate_dashboard_html_contains_plotly(sample_df, tmp_path):
    state = _base_state(sample_df)
    with patch.dict(os.environ, {"OUTPUT_DIR": str(tmp_path)}):
        result = generate_dashboard(state)
    content = Path(result).read_text()
    assert "plotly" in content.lower()


def test_generate_dashboard_html_contains_session_id(sample_df, tmp_path):
    state = _base_state(sample_df)
    with patch.dict(os.environ, {"OUTPUT_DIR": str(tmp_path)}):
        result = generate_dashboard(state)
    content = Path(result).read_text()
    assert "dash-test-001" in content


def test_generate_dashboard_creates_output_dir(sample_df, tmp_path):
    state = _base_state(sample_df)
    target = tmp_path / "new_subdir"
    with patch.dict(os.environ, {"OUTPUT_DIR": str(target)}):
        generate_dashboard(state)
    assert target.exists()


def test_generate_dashboard_raises_runtime_error_on_failure(sample_df):
    state = _base_state(sample_df)
    with patch("outputs.dashboard.open", side_effect=OSError("disk full")):
        with pytest.raises(RuntimeError, match="generate_dashboard failed"):
            generate_dashboard(state)


def test_generate_dashboard_with_charts(sample_df, tmp_path):
    viz = _make_viz_result()
    state = _base_state(sample_df, agent_results=[viz])
    with patch.dict(os.environ, {"OUTPUT_DIR": str(tmp_path)}):
        result = generate_dashboard(state)
    content = Path(result).read_text()
    assert "Chart 1" in content
