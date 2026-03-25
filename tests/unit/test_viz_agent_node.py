from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from core.config import LLMConfig
from core.schema import AgentResult, ProfilerFindings, StatFindings, VizFindings
from core.state import AgentState
from graph.nodes.viz_agent import (
    _build_viz_input,
    _extract_chart_specs,
    _get_upstream_findings,
    viz_agent_node,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_viz_findings_one_chart() -> VizFindings:
    """Single chart — triggers confidence=0.6."""
    return VizFindings(
        charts=[{"data": [{"type": "histogram"}], "layout": {"title": "Age Distribution"}}],
        chart_descriptions=["Distribution of age values"],
        recommended_primary_chart="Age Distribution",
    )


@pytest.fixture
def mock_viz_findings_three_charts() -> VizFindings:
    """Three charts — triggers confidence=0.9."""
    return VizFindings(
        charts=[
            {"data": [{"type": "histogram"}], "layout": {"title": "Age Distribution"}},
            {"data": [{"type": "scatter"}], "layout": {"title": "Age vs Income"}},
            {"data": [{"type": "heatmap"}], "layout": {"title": "Correlation Heatmap"}},
        ],
        chart_descriptions=[
            "Distribution of age values",
            "Scatter plot of age vs income",
            "Correlation heatmap for all numeric columns",
        ],
        recommended_primary_chart="Correlation Heatmap",
    )


@pytest.fixture
def state_with_upstream(base_state: AgentState, sample_df: pd.DataFrame) -> AgentState:
    """State that already has profiler and stat_analyst AgentResults."""
    profiler_result = AgentResult(
        agent_name="profiler",
        success=True,
        findings=ProfilerFindings(
            schema={
                "numeric_cols": ["age", "income", "score"],
                "categorical_cols": ["region"],
                "columns": ["age", "income", "score", "region"],
                "shape": [10, 4],
                "dtypes": {},
            },
            null_report={},
            distributions={},
            notable_columns=["income", "age"],
            categorical_summary={},
            data_quality={},
        ).model_dump(),
        confidence=0.9,
    )
    stat_result = AgentResult(
        agent_name="stat_analyst",
        success=True,
        findings=StatFindings(
            correlations={"age": {"income": 0.85}},
            outliers={},
            normality={},
            skewness={},
            feature_ranking=[{"feature": "age", "importance": 0.85}],
            notable_findings=["Income is strongly correlated with age"],
        ).model_dump(),
        confidence=0.9,
    )
    return {**base_state, "agent_results": [profiler_result, stat_result]}


def _make_mock_llm(findings: VizFindings) -> MagicMock:
    """Return a mock ChatAnthropic whose structured output call returns findings."""
    structured = MagicMock()
    structured.invoke.return_value = findings
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


def _make_mock_agent(content: str = "Generated charts.") -> MagicMock:
    """Return a mock ReAct agent whose invoke returns a messages dict."""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {
        "messages": [AIMessage(content=content)]
    }
    return mock_agent


# ---------------------------------------------------------------------------
# _get_upstream_findings
# ---------------------------------------------------------------------------

def test_get_upstream_findings_both_none_when_empty(base_state: AgentState) -> None:
    profiler, stat = _get_upstream_findings(base_state)
    assert profiler is None
    assert stat is None


def test_get_upstream_findings_returns_profiler_when_present(
    state_with_upstream: AgentState,
) -> None:
    profiler, stat = _get_upstream_findings(state_with_upstream)
    assert profiler is not None
    assert isinstance(profiler, ProfilerFindings)
    assert profiler.notable_columns == ["income", "age"]


def test_get_upstream_findings_returns_stat_when_present(
    state_with_upstream: AgentState,
) -> None:
    profiler, stat = _get_upstream_findings(state_with_upstream)
    assert stat is not None
    assert isinstance(stat, StatFindings)
    assert stat.notable_findings == ["Income is strongly correlated with age"]


def test_get_upstream_findings_ignores_failed_results(
    base_state: AgentState,
) -> None:
    failed_profiler = AgentResult(
        agent_name="profiler",
        success=False,
        findings={},
        confidence=0.0,
    )
    state = {**base_state, "agent_results": [failed_profiler]}
    profiler, stat = _get_upstream_findings(state)
    assert profiler is None
    assert stat is None


# ---------------------------------------------------------------------------
# _build_viz_input
# ---------------------------------------------------------------------------

def test_build_viz_input_contains_goal(base_state: AgentState) -> None:
    request = base_state["request"]
    content = _build_viz_input(request, None, None)
    assert request.goal in content


def test_build_viz_input_contains_notable_columns_when_profiler_present(
    base_state: AgentState,
) -> None:
    request = base_state["request"]
    profiler = ProfilerFindings(
        schema={"numeric_cols": ["age"], "categorical_cols": [], "columns": ["age"],
                "shape": [10, 1], "dtypes": {}},
        null_report={},
        distributions={},
        notable_columns=["income"],
        categorical_summary={},
        data_quality={},
    )
    content = _build_viz_input(request, profiler, None)
    assert "income" in content


def test_build_viz_input_contains_stat_findings_when_present(
    base_state: AgentState,
) -> None:
    request = base_state["request"]
    stat = StatFindings(
        correlations={},
        outliers={},
        normality={},
        notable_findings=["High correlation found"],
    )
    content = _build_viz_input(request, None, stat)
    assert "High correlation found" in content


# ---------------------------------------------------------------------------
# _extract_chart_specs
# ---------------------------------------------------------------------------

def test_extract_chart_specs_empty_on_no_messages() -> None:
    charts = _extract_chart_specs([])
    assert charts == []


def test_extract_chart_specs_finds_valid_spec_in_json_text() -> None:
    import json
    spec = {"data": [{"type": "histogram"}], "layout": {"title": "Test"}}
    msg = MagicMock()
    msg.content = f"Here is the chart: {json.dumps(spec)}"
    charts = _extract_chart_specs([msg])
    assert len(charts) == 1
    assert charts[0]["layout"]["title"] == "Test"


def test_extract_chart_specs_skips_dicts_without_data_key() -> None:
    import json
    spec = {"layout": {"title": "Missing data key"}}
    msg = MagicMock()
    msg.content = json.dumps(spec)
    charts = _extract_chart_specs([msg])
    assert charts == []


def test_extract_chart_specs_skips_dicts_without_layout_key() -> None:
    import json
    spec = {"data": [{"type": "bar"}]}
    msg = MagicMock()
    msg.content = json.dumps(spec)
    charts = _extract_chart_specs([msg])
    assert charts == []


def test_extract_chart_specs_does_not_raise_on_malformed_content() -> None:
    msg = MagicMock()
    msg.content = "{this is not valid json}"
    # Should not raise
    charts = _extract_chart_specs([msg])
    assert isinstance(charts, list)


def test_extract_chart_specs_handles_none_content() -> None:
    msg = MagicMock()
    msg.content = None
    charts = _extract_chart_specs([msg])
    assert charts == []


# ---------------------------------------------------------------------------
# Happy-path node tests
# ---------------------------------------------------------------------------

def test_viz_agent_node_returns_required_keys(
    base_state: AgentState,
    mock_viz_findings_one_chart: VizFindings,
) -> None:
    with (
        patch("graph.nodes.viz_agent.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.viz_agent.ChatAnthropic",
              return_value=_make_mock_llm(mock_viz_findings_one_chart)),
        patch("graph.nodes.viz_agent.create_react_agent",
              return_value=_make_mock_agent()),
        patch("graph.nodes.viz_agent.inject_dataframe"),
    ):
        result = viz_agent_node(base_state)

    assert "agent_results" in result
    assert "messages" in result


def test_viz_agent_node_agent_results_has_one_item(
    base_state: AgentState,
    mock_viz_findings_one_chart: VizFindings,
) -> None:
    with (
        patch("graph.nodes.viz_agent.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.viz_agent.ChatAnthropic",
              return_value=_make_mock_llm(mock_viz_findings_one_chart)),
        patch("graph.nodes.viz_agent.create_react_agent",
              return_value=_make_mock_agent()),
        patch("graph.nodes.viz_agent.inject_dataframe"),
    ):
        result = viz_agent_node(base_state)

    assert isinstance(result["agent_results"], list)
    assert len(result["agent_results"]) == 1


def test_viz_agent_node_agent_name(
    base_state: AgentState,
    mock_viz_findings_one_chart: VizFindings,
) -> None:
    with (
        patch("graph.nodes.viz_agent.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.viz_agent.ChatAnthropic",
              return_value=_make_mock_llm(mock_viz_findings_one_chart)),
        patch("graph.nodes.viz_agent.create_react_agent",
              return_value=_make_mock_agent()),
        patch("graph.nodes.viz_agent.inject_dataframe"),
    ):
        result = viz_agent_node(base_state)

    assert result["agent_results"][0].agent_name == "viz_agent"


def test_viz_agent_node_success_true_on_happy_path(
    base_state: AgentState,
    mock_viz_findings_one_chart: VizFindings,
) -> None:
    with (
        patch("graph.nodes.viz_agent.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.viz_agent.ChatAnthropic",
              return_value=_make_mock_llm(mock_viz_findings_one_chart)),
        patch("graph.nodes.viz_agent.create_react_agent",
              return_value=_make_mock_agent()),
        patch("graph.nodes.viz_agent.inject_dataframe"),
    ):
        result = viz_agent_node(base_state)

    assert result["agent_results"][0].success is True


def test_viz_agent_node_findings_non_empty(
    base_state: AgentState,
    mock_viz_findings_one_chart: VizFindings,
) -> None:
    with (
        patch("graph.nodes.viz_agent.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.viz_agent.ChatAnthropic",
              return_value=_make_mock_llm(mock_viz_findings_one_chart)),
        patch("graph.nodes.viz_agent.create_react_agent",
              return_value=_make_mock_agent()),
        patch("graph.nodes.viz_agent.inject_dataframe"),
    ):
        result = viz_agent_node(base_state)

    findings = result["agent_results"][0].findings
    assert isinstance(findings, dict)
    assert findings != {}


def test_viz_agent_node_findings_has_charts_key(
    base_state: AgentState,
    mock_viz_findings_one_chart: VizFindings,
) -> None:
    with (
        patch("graph.nodes.viz_agent.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.viz_agent.ChatAnthropic",
              return_value=_make_mock_llm(mock_viz_findings_one_chart)),
        patch("graph.nodes.viz_agent.create_react_agent",
              return_value=_make_mock_agent()),
        patch("graph.nodes.viz_agent.inject_dataframe"),
    ):
        result = viz_agent_node(base_state)

    assert "charts" in result["agent_results"][0].findings


def test_viz_agent_node_charts_have_data_and_layout(
    base_state: AgentState,
    mock_viz_findings_one_chart: VizFindings,
) -> None:
    with (
        patch("graph.nodes.viz_agent.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.viz_agent.ChatAnthropic",
              return_value=_make_mock_llm(mock_viz_findings_one_chart)),
        patch("graph.nodes.viz_agent.create_react_agent",
              return_value=_make_mock_agent()),
        patch("graph.nodes.viz_agent.inject_dataframe"),
    ):
        result = viz_agent_node(base_state)

    charts = result["agent_results"][0].findings["charts"]
    for chart in charts:
        assert "data" in chart
        assert "layout" in chart


# ---------------------------------------------------------------------------
# Confidence tests
# ---------------------------------------------------------------------------

def test_viz_agent_node_confidence_06_when_fewer_than_3_charts(
    base_state: AgentState,
    mock_viz_findings_one_chart: VizFindings,
) -> None:
    """1 chart → confidence=0.6."""
    with (
        patch("graph.nodes.viz_agent.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.viz_agent.ChatAnthropic",
              return_value=_make_mock_llm(mock_viz_findings_one_chart)),
        patch("graph.nodes.viz_agent.create_react_agent",
              return_value=_make_mock_agent()),
        patch("graph.nodes.viz_agent.inject_dataframe"),
    ):
        result = viz_agent_node(base_state)

    assert result["agent_results"][0].confidence == pytest.approx(0.6)


def test_viz_agent_node_confidence_09_when_3_or_more_charts(
    base_state: AgentState,
    mock_viz_findings_three_charts: VizFindings,
) -> None:
    """3 charts → confidence=0.9."""
    with (
        patch("graph.nodes.viz_agent.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.viz_agent.ChatAnthropic",
              return_value=_make_mock_llm(mock_viz_findings_three_charts)),
        patch("graph.nodes.viz_agent.create_react_agent",
              return_value=_make_mock_agent("Generated 3 charts.")),
        patch("graph.nodes.viz_agent.inject_dataframe"),
    ):
        result = viz_agent_node(base_state)

    assert result["agent_results"][0].confidence == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Message content
# ---------------------------------------------------------------------------

def test_viz_agent_node_message_contains_react(
    base_state: AgentState,
    mock_viz_findings_one_chart: VizFindings,
) -> None:
    with (
        patch("graph.nodes.viz_agent.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.viz_agent.ChatAnthropic",
              return_value=_make_mock_llm(mock_viz_findings_one_chart)),
        patch("graph.nodes.viz_agent.create_react_agent",
              return_value=_make_mock_agent()),
        patch("graph.nodes.viz_agent.inject_dataframe"),
    ):
        result = viz_agent_node(base_state)

    assert "react" in result["messages"][0].content


# ---------------------------------------------------------------------------
# create_react_agent always called
# ---------------------------------------------------------------------------

def test_viz_agent_node_always_calls_create_react_agent(
    base_state: AgentState,
    mock_viz_findings_one_chart: VizFindings,
) -> None:
    with (
        patch("graph.nodes.viz_agent.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.viz_agent.ChatAnthropic",
              return_value=_make_mock_llm(mock_viz_findings_one_chart)),
        patch("graph.nodes.viz_agent.create_react_agent",
              return_value=_make_mock_agent()) as mock_create,
        patch("graph.nodes.viz_agent.inject_dataframe"),
    ):
        viz_agent_node(base_state)

    mock_create.assert_called_once()


def test_viz_agent_node_inject_dataframe_called_once(
    base_state: AgentState,
    mock_viz_findings_one_chart: VizFindings,
) -> None:
    with (
        patch("graph.nodes.viz_agent.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.viz_agent.ChatAnthropic",
              return_value=_make_mock_llm(mock_viz_findings_one_chart)),
        patch("graph.nodes.viz_agent.create_react_agent",
              return_value=_make_mock_agent()),
        patch("graph.nodes.viz_agent.inject_dataframe") as mock_inject,
    ):
        viz_agent_node(base_state)

    mock_inject.assert_called_once()


# ---------------------------------------------------------------------------
# Upstream findings integration
# ---------------------------------------------------------------------------

def test_viz_agent_node_uses_upstream_findings(
    state_with_upstream: AgentState,
    mock_viz_findings_three_charts: VizFindings,
) -> None:
    """Node should run successfully when upstream profiler + stat results are present."""
    with (
        patch("graph.nodes.viz_agent.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.viz_agent.ChatAnthropic",
              return_value=_make_mock_llm(mock_viz_findings_three_charts)),
        patch("graph.nodes.viz_agent.create_react_agent",
              return_value=_make_mock_agent("Generated 3 charts.")),
        patch("graph.nodes.viz_agent.inject_dataframe"),
    ):
        result = viz_agent_node(state_with_upstream)

    assert result["agent_results"][0].success is True
    assert result["agent_results"][0].confidence == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Exception path
# ---------------------------------------------------------------------------

def test_viz_agent_node_failure_on_exception(base_state: AgentState) -> None:
    with (
        patch("graph.nodes.viz_agent.get_llm_config",
              side_effect=RuntimeError("config failed")),
        patch("graph.nodes.viz_agent.inject_dataframe"),
    ):
        result = viz_agent_node(base_state)

    agent_result: AgentResult = result["agent_results"][0]
    assert agent_result.success is False
    assert agent_result.confidence == 0.0
    assert agent_result.findings == {}
    assert len(agent_result.warnings) > 0


def test_viz_agent_node_get_llm_config_called_once(
    base_state: AgentState,
    mock_viz_findings_one_chart: VizFindings,
) -> None:
    with (
        patch("graph.nodes.viz_agent.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)) as mock_cfg,
        patch("graph.nodes.viz_agent.ChatAnthropic",
              return_value=_make_mock_llm(mock_viz_findings_one_chart)),
        patch("graph.nodes.viz_agent.create_react_agent",
              return_value=_make_mock_agent()),
        patch("graph.nodes.viz_agent.inject_dataframe"),
    ):
        viz_agent_node(base_state)

    mock_cfg.assert_called_once()
