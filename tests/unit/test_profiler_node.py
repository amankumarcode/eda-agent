from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from core.config import LLMConfig
from core.schema import AgentResult, ProfilerFindings
from core.state import AgentState
from graph.nodes.profiler import _run_all_profiling_tools, profiler_node

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_profiler_findings() -> ProfilerFindings:
    return ProfilerFindings(
        schema={
            "columns": ["age", "income"],
            "shape": [10, 2],
            "numeric_cols": ["age", "income"],
            "categorical_cols": [],
            "dtypes": {},
        },
        null_report={},
        distributions={"age": {"mean": 35.0}},
        notable_columns=["income"],
        categorical_summary={},
        data_quality={},
    )


def _make_mock_llm(findings: ProfilerFindings) -> MagicMock:
    """Return a mock ChatAnthropic whose structured output call returns findings."""
    structured = MagicMock()
    structured.invoke.return_value = findings
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

def test_profiler_node_returns_required_keys(
    base_state: AgentState,
    mock_profiler_findings: ProfilerFindings,
) -> None:
    with (
        patch("graph.nodes.profiler.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.profiler.ChatAnthropic",
              return_value=_make_mock_llm(mock_profiler_findings)),
        patch("graph.nodes.profiler.inject_dataframe") as mock_inject,
    ):
        result = profiler_node(base_state)

    assert "agent_results" in result
    assert "messages" in result


def test_profiler_node_agent_results_has_one_item(
    base_state: AgentState,
    mock_profiler_findings: ProfilerFindings,
) -> None:
    with (
        patch("graph.nodes.profiler.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.profiler.ChatAnthropic",
              return_value=_make_mock_llm(mock_profiler_findings)),
        patch("graph.nodes.profiler.inject_dataframe"),
    ):
        result = profiler_node(base_state)

    assert isinstance(result["agent_results"], list)
    assert len(result["agent_results"]) == 1


def test_profiler_node_agent_name(
    base_state: AgentState,
    mock_profiler_findings: ProfilerFindings,
) -> None:
    with (
        patch("graph.nodes.profiler.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.profiler.ChatAnthropic",
              return_value=_make_mock_llm(mock_profiler_findings)),
        patch("graph.nodes.profiler.inject_dataframe"),
    ):
        result = profiler_node(base_state)

    agent_result: AgentResult = result["agent_results"][0]
    assert agent_result.agent_name == "profiler"


def test_profiler_node_success_true_on_happy_path(
    base_state: AgentState,
    mock_profiler_findings: ProfilerFindings,
) -> None:
    with (
        patch("graph.nodes.profiler.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.profiler.ChatAnthropic",
              return_value=_make_mock_llm(mock_profiler_findings)),
        patch("graph.nodes.profiler.inject_dataframe"),
    ):
        result = profiler_node(base_state)

    assert result["agent_results"][0].success is True


def test_profiler_node_findings_non_empty_on_happy_path(
    base_state: AgentState,
    mock_profiler_findings: ProfilerFindings,
) -> None:
    with (
        patch("graph.nodes.profiler.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.profiler.ChatAnthropic",
              return_value=_make_mock_llm(mock_profiler_findings)),
        patch("graph.nodes.profiler.inject_dataframe"),
    ):
        result = profiler_node(base_state)

    assert isinstance(result["agent_results"][0].findings, dict)
    assert result["agent_results"][0].findings != {}


def test_profiler_node_confidence_1_when_no_tool_errors(
    base_state: AgentState,
    mock_profiler_findings: ProfilerFindings,
) -> None:
    with (
        patch("graph.nodes.profiler.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.profiler.ChatAnthropic",
              return_value=_make_mock_llm(mock_profiler_findings)),
        patch("graph.nodes.profiler.inject_dataframe"),
    ):
        result = profiler_node(base_state)

    assert result["agent_results"][0].confidence == pytest.approx(1.0)


def test_profiler_node_warnings_empty_on_clean_run(
    base_state: AgentState,
    mock_profiler_findings: ProfilerFindings,
) -> None:
    with (
        patch("graph.nodes.profiler.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.profiler.ChatAnthropic",
              return_value=_make_mock_llm(mock_profiler_findings)),
        patch("graph.nodes.profiler.inject_dataframe"),
    ):
        result = profiler_node(base_state)

    assert result["agent_results"][0].warnings == []


# ---------------------------------------------------------------------------
# Tool-error path
# ---------------------------------------------------------------------------

def test_profiler_node_confidence_reduced_when_tool_fails(
    base_state: AgentState,
    mock_profiler_findings: ProfilerFindings,
) -> None:
    """One tool failure → confidence < 1.0 and warnings non-empty."""
    with (
        patch("graph.nodes.profiler.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.profiler.ChatAnthropic",
              return_value=_make_mock_llm(mock_profiler_findings)),
        patch("graph.nodes.profiler.inject_dataframe"),
        patch("graph.nodes.profiler.get_null_report",
              side_effect=RuntimeError("null report failed")),
    ):
        result = profiler_node(base_state)

    agent_result: AgentResult = result["agent_results"][0]
    assert agent_result.confidence < 1.0
    assert len(agent_result.warnings) > 0


def test_profiler_node_warnings_non_empty_when_tool_fails(
    base_state: AgentState,
    mock_profiler_findings: ProfilerFindings,
) -> None:
    with (
        patch("graph.nodes.profiler.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.profiler.ChatAnthropic",
              return_value=_make_mock_llm(mock_profiler_findings)),
        patch("graph.nodes.profiler.inject_dataframe"),
        patch("graph.nodes.profiler.get_duplicate_report",
              side_effect=ValueError("dup report failed")),
    ):
        result = profiler_node(base_state)

    warnings = result["agent_results"][0].warnings
    assert any("dup report failed" in w for w in warnings)


# ---------------------------------------------------------------------------
# Full node exception path
# ---------------------------------------------------------------------------

def test_profiler_node_failure_on_exception(base_state: AgentState) -> None:
    """If the entire node body raises, return success=False with empty findings."""
    with (
        patch("graph.nodes.profiler.inject_dataframe",
              side_effect=RuntimeError("inject failed")),
    ):
        result = profiler_node(base_state)

    agent_result: AgentResult = result["agent_results"][0]
    assert agent_result.success is False
    assert agent_result.confidence == 0.0
    assert agent_result.findings == {}
    assert len(agent_result.warnings) > 0


# ---------------------------------------------------------------------------
# Call-count / argument assertions
# ---------------------------------------------------------------------------

def test_profiler_node_inject_dataframe_called_once(
    base_state: AgentState,
    mock_profiler_findings: ProfilerFindings,
) -> None:
    with (
        patch("graph.nodes.profiler.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.profiler.ChatAnthropic",
              return_value=_make_mock_llm(mock_profiler_findings)),
        patch("graph.nodes.profiler.inject_dataframe") as mock_inject,
    ):
        profiler_node(base_state)

    mock_inject.assert_called_once()


def test_profiler_node_get_llm_config_called_once(
    base_state: AgentState,
    mock_profiler_findings: ProfilerFindings,
) -> None:
    with (
        patch("graph.nodes.profiler.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)) as mock_cfg,
        patch("graph.nodes.profiler.ChatAnthropic",
              return_value=_make_mock_llm(mock_profiler_findings)),
        patch("graph.nodes.profiler.inject_dataframe"),
    ):
        profiler_node(base_state)

    mock_cfg.assert_called_once()


def test_profiler_node_run_all_tools_called_with_correct_df(
    base_state: AgentState,
    mock_profiler_findings: ProfilerFindings,
) -> None:
    from core.dataframe_registry import get as get_df
    expected_df = get_df(base_state["request"].session_id)

    with (
        patch("graph.nodes.profiler.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.profiler.ChatAnthropic",
              return_value=_make_mock_llm(mock_profiler_findings)),
        patch("graph.nodes.profiler.inject_dataframe"),
        patch("graph.nodes.profiler._run_all_profiling_tools",
              wraps=_run_all_profiling_tools) as mock_run,
    ):
        profiler_node(base_state)

    mock_run.assert_called_once()
    called_df = mock_run.call_args[0][0]
    assert called_df is expected_df
