from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from core.config import LLMConfig
from core.schema import AgentResult, NarratorResult
from core.state import AgentState
from graph.nodes.narrator import (
    _build_narrator_prompt,
    _get_all_findings,
    narrator_node,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_narrator_result() -> NarratorResult:
    return NarratorResult(
        narrative=(
            "Income is strongly driven by age in this dataset. "
            "Median income is $74,000 with a correlation of 0.85 with age."
        ),
        key_insights=[
            "Age explains 85% of income variance (r=0.85)",
            "3 high-income outliers above $110,000 detected",
            "Northern region has highest median income at $92,000",
        ],
        caveats=["Sample size of 10 rows limits statistical confidence"],
        recommended_next_steps=[
            "Run regression analysis with age as predictor",
            "Investigate regional income differences with larger sample",
        ],
    )


@pytest.fixture
def state_with_all_results(base_state: AgentState) -> AgentState:
    """State with profiler, stat_analyst, viz_agent results."""
    return {
        **base_state,
        "agent_results": [
            AgentResult(
                agent_name="profiler",
                success=True,
                findings={
                    "schema": {
                        "numeric_cols": ["age", "income"],
                        "categorical_cols": ["region"],
                        "shape": [10, 3],
                    },
                    "notable_columns": ["income", "age"],
                    "null_report": {},
                    "distributions": {"income": {"mean": 74000, "std": 27000}},
                    "categorical_summary": {},
                    "data_quality": {},
                },
                confidence=0.9,
                warnings=[],
            ),
            AgentResult(
                agent_name="stat_analyst",
                success=True,
                findings={
                    "correlations": {"age": {"income": 0.85}},
                    "outliers": {"income": {"outlier_count": 3}},
                    "normality": {"age": {"is_normal": True}},
                    "skewness": {},
                    "feature_ranking": [{"feature": "age", "importance": 0.85}],
                    "notable_findings": ["Income strongly correlated with age (r=0.85)"],
                },
                confidence=0.85,
                warnings=[],
            ),
            AgentResult(
                agent_name="viz_agent",
                success=True,
                findings={
                    "charts": [{"data": [], "layout": {}}],
                    "chart_descriptions": ["Income distribution"],
                    "recommended_primary_chart": "income_histogram",
                },
                confidence=0.8,
                warnings=[],
            ),
        ],
        "completed_agents": [
            "profiler", "stat_analyst", "viz_agent", "insight_critic"
        ],
    }


def _make_mock_llm(narrator_result: NarratorResult) -> MagicMock:
    structured = MagicMock()
    structured.invoke.return_value = narrator_result
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


# ---------------------------------------------------------------------------
# _get_all_findings
# ---------------------------------------------------------------------------

def test_get_all_findings_overall_confidence_mean(
    state_with_all_results: AgentState,
) -> None:
    findings = _get_all_findings(state_with_all_results)
    expected = (0.9 + 0.85 + 0.8) / 3
    assert findings["overall_confidence"] == pytest.approx(expected, abs=1e-6)


def test_get_all_findings_overall_confidence_zero_when_empty(
    base_state: AgentState,
) -> None:
    findings = _get_all_findings(base_state)
    assert findings["overall_confidence"] == 0.0


def test_get_all_findings_flattens_warnings(base_state: AgentState) -> None:
    state = {
        **base_state,
        "agent_results": [
            AgentResult(
                agent_name="profiler",
                success=True,
                findings={},
                confidence=0.9,
                warnings=["null report failed"],
            ),
            AgentResult(
                agent_name="stat_analyst",
                success=True,
                findings={},
                confidence=0.8,
                warnings=["normality test skipped"],
            ),
        ],
    }
    findings = _get_all_findings(state)
    assert "null report failed" in findings["any_warnings"]
    assert "normality test skipped" in findings["any_warnings"]
    assert len(findings["any_warnings"]) == 2


def test_get_all_findings_chart_count(state_with_all_results: AgentState) -> None:
    findings = _get_all_findings(state_with_all_results)
    assert findings["chart_count"] == 1


def test_get_all_findings_returns_profiler_dict(
    state_with_all_results: AgentState,
) -> None:
    findings = _get_all_findings(state_with_all_results)
    assert isinstance(findings["profiler"], dict)
    assert "notable_columns" in findings["profiler"]


def test_get_all_findings_returns_stat_dict(
    state_with_all_results: AgentState,
) -> None:
    findings = _get_all_findings(state_with_all_results)
    assert isinstance(findings["stat_analyst"], dict)
    assert "correlations" in findings["stat_analyst"]


def test_get_all_findings_ignores_failed_results(base_state: AgentState) -> None:
    state = {
        **base_state,
        "agent_results": [
            AgentResult(
                agent_name="profiler",
                success=False,
                findings={"schema": {"shape": [10, 3]}},
                confidence=0.0,
                warnings=["inject failed"],
            ),
        ],
    }
    findings = _get_all_findings(state)
    # Failed result should not contribute to profiler findings
    assert findings["profiler"] == {}
    # But its confidence should not count in the mean
    assert findings["overall_confidence"] == 0.0


def test_get_all_findings_any_warnings_empty_when_no_warnings(
    state_with_all_results: AgentState,
) -> None:
    findings = _get_all_findings(state_with_all_results)
    assert findings["any_warnings"] == []


# ---------------------------------------------------------------------------
# _build_narrator_prompt
# ---------------------------------------------------------------------------

def test_build_narrator_prompt_contains_goal(
    base_state: AgentState,
    state_with_all_results: AgentState,
) -> None:
    request = base_state["request"]
    all_findings = _get_all_findings(state_with_all_results)
    prompt = _build_narrator_prompt(request, all_findings)
    assert request.goal in prompt


def test_build_narrator_prompt_contains_confidence(
    base_state: AgentState,
    state_with_all_results: AgentState,
) -> None:
    request = base_state["request"]
    all_findings = _get_all_findings(state_with_all_results)
    prompt = _build_narrator_prompt(request, all_findings)
    assert "0.85" in prompt or "confidence" in prompt.lower()


def test_build_narrator_prompt_contains_notable_columns(
    base_state: AgentState,
    state_with_all_results: AgentState,
) -> None:
    request = base_state["request"]
    all_findings = _get_all_findings(state_with_all_results)
    prompt = _build_narrator_prompt(request, all_findings)
    assert "income" in prompt


# ---------------------------------------------------------------------------
# Happy path node tests
# ---------------------------------------------------------------------------

def test_narrator_node_returns_expected_keys(
    state_with_all_results: AgentState,
    mock_narrator_result: NarratorResult,
) -> None:
    with (
        patch("graph.nodes.narrator.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.narrator.ChatAnthropic",
              return_value=_make_mock_llm(mock_narrator_result)),
    ):
        result = narrator_node(state_with_all_results)

    for key in ("narrative", "key_insights", "caveats",
                "next_action", "completed_agents", "messages"):
        assert key in result


def test_narrator_node_next_action_is_evaluate(
    state_with_all_results: AgentState,
    mock_narrator_result: NarratorResult,
) -> None:
    with (
        patch("graph.nodes.narrator.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.narrator.ChatAnthropic",
              return_value=_make_mock_llm(mock_narrator_result)),
    ):
        result = narrator_node(state_with_all_results)

    assert result["next_action"] == "evaluate"


def test_narrator_node_completed_agents_contains_narrator(
    state_with_all_results: AgentState,
    mock_narrator_result: NarratorResult,
) -> None:
    with (
        patch("graph.nodes.narrator.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.narrator.ChatAnthropic",
              return_value=_make_mock_llm(mock_narrator_result)),
    ):
        result = narrator_node(state_with_all_results)

    assert "narrator" in result["completed_agents"]


def test_narrator_node_narrative_non_empty(
    state_with_all_results: AgentState,
    mock_narrator_result: NarratorResult,
) -> None:
    with (
        patch("graph.nodes.narrator.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.narrator.ChatAnthropic",
              return_value=_make_mock_llm(mock_narrator_result)),
    ):
        result = narrator_node(state_with_all_results)

    assert isinstance(result["narrative"], str)
    assert len(result["narrative"]) > 0


def test_narrator_node_key_insights_non_empty(
    state_with_all_results: AgentState,
    mock_narrator_result: NarratorResult,
) -> None:
    with (
        patch("graph.nodes.narrator.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.narrator.ChatAnthropic",
              return_value=_make_mock_llm(mock_narrator_result)),
    ):
        result = narrator_node(state_with_all_results)

    assert isinstance(result["key_insights"], list)
    assert len(result["key_insights"]) > 0


def test_narrator_node_caveats_is_list(
    state_with_all_results: AgentState,
    mock_narrator_result: NarratorResult,
) -> None:
    with (
        patch("graph.nodes.narrator.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.narrator.ChatAnthropic",
              return_value=_make_mock_llm(mock_narrator_result)),
    ):
        result = narrator_node(state_with_all_results)

    assert isinstance(result["caveats"], list)


def test_narrator_node_key_insights_at_top_level(
    state_with_all_results: AgentState,
    mock_narrator_result: NarratorResult,
) -> None:
    """key_insights must be a top-level key, not nested inside a findings dict."""
    with (
        patch("graph.nodes.narrator.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.narrator.ChatAnthropic",
              return_value=_make_mock_llm(mock_narrator_result)),
    ):
        result = narrator_node(state_with_all_results)

    assert "key_insights" in result
    assert isinstance(result["key_insights"], list)


def test_narrator_node_messages_is_list(
    state_with_all_results: AgentState,
    mock_narrator_result: NarratorResult,
) -> None:
    with (
        patch("graph.nodes.narrator.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.narrator.ChatAnthropic",
              return_value=_make_mock_llm(mock_narrator_result)),
    ):
        result = narrator_node(state_with_all_results)

    assert isinstance(result["messages"], list)
    assert len(result["messages"]) == 1


def test_narrator_node_narrative_matches_mock(
    state_with_all_results: AgentState,
    mock_narrator_result: NarratorResult,
) -> None:
    with (
        patch("graph.nodes.narrator.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.narrator.ChatAnthropic",
              return_value=_make_mock_llm(mock_narrator_result)),
    ):
        result = narrator_node(state_with_all_results)

    assert result["narrative"] == mock_narrator_result.narrative


def test_narrator_node_key_insights_match_mock(
    state_with_all_results: AgentState,
    mock_narrator_result: NarratorResult,
) -> None:
    with (
        patch("graph.nodes.narrator.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.narrator.ChatAnthropic",
              return_value=_make_mock_llm(mock_narrator_result)),
    ):
        result = narrator_node(state_with_all_results)

    assert result["key_insights"] == mock_narrator_result.key_insights


# ---------------------------------------------------------------------------
# Exception path
# ---------------------------------------------------------------------------

def test_exception_path_next_action_is_evaluate(base_state: AgentState) -> None:
    with patch("graph.nodes.narrator.get_llm_config",
               side_effect=RuntimeError("config failed")):
        result = narrator_node(base_state)

    assert result["next_action"] == "evaluate"


def test_exception_path_narrative_non_empty(base_state: AgentState) -> None:
    with patch("graph.nodes.narrator.get_llm_config",
               side_effect=RuntimeError("config failed")):
        result = narrator_node(base_state)

    assert isinstance(result["narrative"], str)
    assert len(result["narrative"]) > 0


def test_exception_path_key_insights_empty(base_state: AgentState) -> None:
    with patch("graph.nodes.narrator.get_llm_config",
               side_effect=RuntimeError("config failed")):
        result = narrator_node(base_state)

    assert result["key_insights"] == []


def test_exception_path_completed_agents_contains_narrator(
    base_state: AgentState,
) -> None:
    with patch("graph.nodes.narrator.get_llm_config",
               side_effect=RuntimeError("config failed")):
        result = narrator_node(base_state)

    assert "narrator" in result["completed_agents"]


def test_exception_does_not_propagate(base_state: AgentState) -> None:
    with patch("graph.nodes.narrator.get_llm_config",
               side_effect=RuntimeError("config failed")):
        result = narrator_node(base_state)

    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Config / LLM call assertions
# ---------------------------------------------------------------------------

def test_get_llm_config_called_once(
    state_with_all_results: AgentState,
    mock_narrator_result: NarratorResult,
) -> None:
    with (
        patch("graph.nodes.narrator.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)) as mock_cfg,
        patch("graph.nodes.narrator.ChatAnthropic",
              return_value=_make_mock_llm(mock_narrator_result)),
    ):
        narrator_node(state_with_all_results)

    mock_cfg.assert_called_once()


def test_with_structured_output_called_with_narrator_result(
    state_with_all_results: AgentState,
    mock_narrator_result: NarratorResult,
) -> None:
    mock_llm = _make_mock_llm(mock_narrator_result)
    with (
        patch("graph.nodes.narrator.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.narrator.ChatAnthropic",
              return_value=mock_llm),
    ):
        narrator_node(state_with_all_results)

    mock_llm.with_structured_output.assert_called_once_with(NarratorResult)
