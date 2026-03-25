from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from core.config import AgentConfig, LLMConfig
from core.schema import AgentResult, ProfilerFindings, StatFindings
from core.state import AgentState
from graph.nodes.stat_analyst import (
    _get_profiler_findings,
    stat_analyst_node,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_stat_findings() -> StatFindings:
    return StatFindings(
        correlations={"age": {"income": 0.85}},
        outliers={"income": {"outlier_count": 1}},
        normality={"age": {"is_normal": True}},
        skewness={"age": {"skewness": 0.3, "highly_skewed": False}},
        feature_ranking=[{"feature": "age", "importance": 0.85}],
        notable_findings=["Income is strongly correlated with age"],
    )


@pytest.fixture
def state_with_profiler_result(base_state: AgentState, sample_df: pd.DataFrame) -> AgentState:
    """State that already has a profiler AgentResult in agent_results."""
    profiler_result = AgentResult(
        agent_name="profiler",
        success=True,
        findings=ProfilerFindings(
            schema={
                "numeric_cols": ["age", "income"],
                "categorical_cols": ["region"],
                "columns": ["age", "income", "region"],
                "shape": [10, 3],
                "dtypes": {},
            },
            null_report={},
            distributions={},
            notable_columns=["income"],
            categorical_summary={},
            data_quality={},
        ).model_dump(),
        confidence=0.9,
    )
    return {**base_state, "agent_results": [profiler_result]}


def _make_mock_llm(findings: StatFindings) -> MagicMock:
    """Return a mock ChatAnthropic whose structured output call returns findings."""
    structured = MagicMock()
    structured.invoke.return_value = findings
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


def _direct_patches(findings: StatFindings):
    """Context manager stack for a standard direct-mode run."""
    return (
        patch("graph.nodes.stat_analyst.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.stat_analyst.get_agent_config",
              return_value=AgentConfig(stat_analyst_use_react=False)),
        patch("graph.nodes.stat_analyst.ChatAnthropic",
              return_value=_make_mock_llm(findings)),
    )


# ---------------------------------------------------------------------------
# _get_profiler_findings
# ---------------------------------------------------------------------------

def test_get_profiler_findings_returns_none_when_empty(base_state: AgentState) -> None:
    assert _get_profiler_findings(base_state) is None


def test_get_profiler_findings_returns_findings_when_present(
    state_with_profiler_result: AgentState,
) -> None:
    result = _get_profiler_findings(state_with_profiler_result)
    assert result is not None
    assert isinstance(result, ProfilerFindings)
    assert result.notable_columns == ["income"]


# ---------------------------------------------------------------------------
# Happy-path (direct mode)
# ---------------------------------------------------------------------------

def test_stat_analyst_node_returns_required_keys(
    base_state: AgentState,
    mock_stat_findings: StatFindings,
) -> None:
    with (
        patch("graph.nodes.stat_analyst.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.stat_analyst.get_agent_config",
              return_value=AgentConfig(stat_analyst_use_react=False)),
        patch("graph.nodes.stat_analyst.ChatAnthropic",
              return_value=_make_mock_llm(mock_stat_findings)),
    ):
        result = stat_analyst_node(base_state)

    assert "agent_results" in result
    assert "messages" in result


def test_stat_analyst_node_agent_results_has_one_item(
    base_state: AgentState,
    mock_stat_findings: StatFindings,
) -> None:
    with (
        patch("graph.nodes.stat_analyst.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.stat_analyst.get_agent_config",
              return_value=AgentConfig(stat_analyst_use_react=False)),
        patch("graph.nodes.stat_analyst.ChatAnthropic",
              return_value=_make_mock_llm(mock_stat_findings)),
    ):
        result = stat_analyst_node(base_state)

    assert isinstance(result["agent_results"], list)
    assert len(result["agent_results"]) == 1


def test_stat_analyst_node_agent_name(
    base_state: AgentState,
    mock_stat_findings: StatFindings,
) -> None:
    with (
        patch("graph.nodes.stat_analyst.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.stat_analyst.get_agent_config",
              return_value=AgentConfig(stat_analyst_use_react=False)),
        patch("graph.nodes.stat_analyst.ChatAnthropic",
              return_value=_make_mock_llm(mock_stat_findings)),
    ):
        result = stat_analyst_node(base_state)

    assert result["agent_results"][0].agent_name == "stat_analyst"


def test_stat_analyst_node_success_true_on_happy_path(
    base_state: AgentState,
    mock_stat_findings: StatFindings,
) -> None:
    with (
        patch("graph.nodes.stat_analyst.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.stat_analyst.get_agent_config",
              return_value=AgentConfig(stat_analyst_use_react=False)),
        patch("graph.nodes.stat_analyst.ChatAnthropic",
              return_value=_make_mock_llm(mock_stat_findings)),
    ):
        result = stat_analyst_node(base_state)

    assert result["agent_results"][0].success is True


def test_stat_analyst_node_findings_non_empty(
    base_state: AgentState,
    mock_stat_findings: StatFindings,
) -> None:
    with (
        patch("graph.nodes.stat_analyst.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.stat_analyst.get_agent_config",
              return_value=AgentConfig(stat_analyst_use_react=False)),
        patch("graph.nodes.stat_analyst.ChatAnthropic",
              return_value=_make_mock_llm(mock_stat_findings)),
    ):
        result = stat_analyst_node(base_state)

    findings = result["agent_results"][0].findings
    assert isinstance(findings, dict)
    assert findings != {}


def test_stat_analyst_node_confidence_1_when_no_tool_errors(
    base_state: AgentState,
    mock_stat_findings: StatFindings,
) -> None:
    with (
        patch("graph.nodes.stat_analyst.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.stat_analyst.get_agent_config",
              return_value=AgentConfig(stat_analyst_use_react=False)),
        patch("graph.nodes.stat_analyst.ChatAnthropic",
              return_value=_make_mock_llm(mock_stat_findings)),
    ):
        result = stat_analyst_node(base_state)

    assert result["agent_results"][0].confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Mode routing
# ---------------------------------------------------------------------------

def test_direct_mode_message_contains_direct(
    base_state: AgentState,
    mock_stat_findings: StatFindings,
) -> None:
    with (
        patch("graph.nodes.stat_analyst.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.stat_analyst.get_agent_config",
              return_value=AgentConfig(stat_analyst_use_react=False)),
        patch("graph.nodes.stat_analyst.ChatAnthropic",
              return_value=_make_mock_llm(mock_stat_findings)),
    ):
        result = stat_analyst_node(base_state)

    assert "direct" in result["messages"][0].content


def test_react_mode_message_contains_react(
    base_state: AgentState,
    mock_stat_findings: StatFindings,
) -> None:
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {
        "messages": [AIMessage(content="Statistical analysis complete.")]
    }

    with (
        patch("graph.nodes.stat_analyst.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.stat_analyst.get_agent_config",
              return_value=AgentConfig(stat_analyst_use_react=True)),
        patch("graph.nodes.stat_analyst.ChatAnthropic",
              return_value=_make_mock_llm(mock_stat_findings)),
        patch("graph.nodes.stat_analyst.create_react_agent",
              return_value=mock_agent),
    ):
        result = stat_analyst_node(base_state)

    assert "react" in result["messages"][0].content


def test_direct_mode_does_not_call_create_react_agent(
    base_state: AgentState,
    mock_stat_findings: StatFindings,
) -> None:
    with (
        patch("graph.nodes.stat_analyst.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.stat_analyst.get_agent_config",
              return_value=AgentConfig(stat_analyst_use_react=False)),
        patch("graph.nodes.stat_analyst.ChatAnthropic",
              return_value=_make_mock_llm(mock_stat_findings)),
        patch("graph.nodes.stat_analyst.create_react_agent") as mock_create,
    ):
        stat_analyst_node(base_state)

    mock_create.assert_not_called()


def test_react_mode_calls_create_react_agent(
    base_state: AgentState,
    mock_stat_findings: StatFindings,
) -> None:
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {
        "messages": [AIMessage(content="done")]
    }

    with (
        patch("graph.nodes.stat_analyst.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.stat_analyst.get_agent_config",
              return_value=AgentConfig(stat_analyst_use_react=True)),
        patch("graph.nodes.stat_analyst.ChatAnthropic",
              return_value=_make_mock_llm(mock_stat_findings)),
        patch("graph.nodes.stat_analyst.create_react_agent",
              return_value=mock_agent) as mock_create,
    ):
        stat_analyst_node(base_state)

    mock_create.assert_called_once()


# ---------------------------------------------------------------------------
# Full node exception path
# ---------------------------------------------------------------------------

def test_stat_analyst_node_failure_on_exception(base_state: AgentState) -> None:
    with (
        patch("graph.nodes.stat_analyst.get_llm_config",
              side_effect=RuntimeError("config failed")),
        patch("graph.nodes.stat_analyst.get_agent_config",
              return_value=AgentConfig(stat_analyst_use_react=False)),
    ):
        result = stat_analyst_node(base_state)

    agent_result: AgentResult = result["agent_results"][0]
    assert agent_result.success is False
    assert agent_result.confidence == 0.0
    assert agent_result.findings == {}
    assert len(agent_result.warnings) > 0


# ---------------------------------------------------------------------------
# Config call counts
# ---------------------------------------------------------------------------

def test_get_llm_config_called_once(
    base_state: AgentState,
    mock_stat_findings: StatFindings,
) -> None:
    with (
        patch("graph.nodes.stat_analyst.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)) as mock_cfg,
        patch("graph.nodes.stat_analyst.get_agent_config",
              return_value=AgentConfig(stat_analyst_use_react=False)),
        patch("graph.nodes.stat_analyst.ChatAnthropic",
              return_value=_make_mock_llm(mock_stat_findings)),
    ):
        stat_analyst_node(base_state)

    mock_cfg.assert_called_once()


def test_get_agent_config_called_once(
    base_state: AgentState,
    mock_stat_findings: StatFindings,
) -> None:
    with (
        patch("graph.nodes.stat_analyst.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.stat_analyst.get_agent_config",
              return_value=AgentConfig(stat_analyst_use_react=False)) as mock_acfg,
        patch("graph.nodes.stat_analyst.ChatAnthropic",
              return_value=_make_mock_llm(mock_stat_findings)),
    ):
        stat_analyst_node(base_state)

    mock_acfg.assert_called_once()
