from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from core.config import LLMConfig
from core.schema import AgentResult, CriticOutput
from core.state import AgentState
from graph.nodes.insight_critic import (
    _build_critic_prompt,
    _format_agent_results,
    insight_critic_node,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def three_specialist_results():
    """Three AgentResults simulating completed specialist nodes."""
    return [
        AgentResult(
            agent_name="profiler",
            success=True,
            findings={
                "schema": {}, "null_report": {},
                "distributions": {}, "notable_columns": ["income"],
                "categorical_summary": {}, "data_quality": {},
            },
            confidence=0.9,
        ),
        AgentResult(
            agent_name="stat_analyst",
            success=True,
            findings={
                "correlations": {}, "outliers": {},
                "normality": {}, "skewness": {},
                "feature_ranking": [],
                "notable_findings": ["income correlated with age"],
            },
            confidence=0.85,
        ),
        AgentResult(
            agent_name="viz_agent",
            success=True,
            findings={
                "charts": [{"data": [], "layout": {}}],
                "chart_descriptions": ["Income histogram"],
                "recommended_primary_chart": "income_histogram",
            },
            confidence=0.8,
        ),
    ]


@pytest.fixture
def mock_critic_output_no_rerun(three_specialist_results):
    return CriticOutput(
        scored_results=three_specialist_results,
        rerun_agent=None,
        rerun_reason=None,
        overall_quality=0.85,
    )


@pytest.fixture
def mock_critic_output_with_rerun(three_specialist_results):
    low_conf = three_specialist_results.copy()
    low_conf[2] = AgentResult(
        agent_name="viz_agent",
        success=True,
        findings=three_specialist_results[2].findings,
        confidence=0.45,
    )
    return CriticOutput(
        scored_results=low_conf,
        rerun_agent="viz_agent",
        rerun_reason=(
            "Only 1 chart generated. Need at least 3 charts "
            "covering distributions, correlations, and categorical breakdown."
        ),
        overall_quality=0.65,
    )


@pytest.fixture
def state_with_specialists(base_state: AgentState, three_specialist_results):
    return {
        **base_state,
        "agent_results": three_specialist_results,
        "completed_agents": ["profiler", "stat_analyst", "viz_agent"],
    }


def _make_mock_llm(critic_output: CriticOutput) -> MagicMock:
    """Return a mock ChatAnthropic whose structured output call returns critic_output."""
    structured = MagicMock()
    structured.invoke.return_value = critic_output
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


# ---------------------------------------------------------------------------
# _format_agent_results
# ---------------------------------------------------------------------------

def test_format_agent_results_returns_non_empty_string(three_specialist_results) -> None:
    result = _format_agent_results(three_specialist_results)
    assert isinstance(result, str)
    assert len(result) > 0


def test_format_agent_results_contains_all_agent_names(three_specialist_results) -> None:
    result = _format_agent_results(three_specialist_results)
    assert "profiler" in result
    assert "stat_analyst" in result
    assert "viz_agent" in result


def test_format_agent_results_shows_findings_summary_not_raw_data(
    three_specialist_results,
) -> None:
    """Should show key types/lengths, not dump full findings."""
    result = _format_agent_results(three_specialist_results)
    # Should have summary strings like "dict(" or "list("
    assert "dict(" in result or "list(" in result


def test_format_agent_results_empty_list_returns_empty_string() -> None:
    result = _format_agent_results([])
    assert result == ""


# ---------------------------------------------------------------------------
# _build_critic_prompt
# ---------------------------------------------------------------------------

def test_build_critic_prompt_contains_goal(
    base_state: AgentState,
    three_specialist_results,
) -> None:
    request = base_state["request"]
    formatted = _format_agent_results(three_specialist_results)
    prompt = _build_critic_prompt(request, three_specialist_results, formatted)
    assert request.goal in prompt


def test_build_critic_prompt_contains_agent_count(
    base_state: AgentState,
    three_specialist_results,
) -> None:
    request = base_state["request"]
    formatted = _format_agent_results(three_specialist_results)
    prompt = _build_critic_prompt(request, three_specialist_results, formatted)
    assert "3" in prompt


def test_build_critic_prompt_contains_formatted_results(
    base_state: AgentState,
    three_specialist_results,
) -> None:
    request = base_state["request"]
    formatted = _format_agent_results(three_specialist_results)
    prompt = _build_critic_prompt(request, three_specialist_results, formatted)
    assert formatted in prompt


# ---------------------------------------------------------------------------
# Happy path — no rerun
# ---------------------------------------------------------------------------

def test_insight_critic_node_returns_expected_keys(
    state_with_specialists: AgentState,
    mock_critic_output_no_rerun: CriticOutput,
) -> None:
    with (
        patch("graph.nodes.insight_critic.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.insight_critic.ChatAnthropic",
              return_value=_make_mock_llm(mock_critic_output_no_rerun)),
    ):
        result = insight_critic_node(state_with_specialists)

    for key in ("scored_results", "rerun_agent", "rerun_count",
                "next_action", "completed_agents", "messages"):
        assert key in result


def test_next_action_is_narrate_when_no_rerun(
    state_with_specialists: AgentState,
    mock_critic_output_no_rerun: CriticOutput,
) -> None:
    with (
        patch("graph.nodes.insight_critic.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.insight_critic.ChatAnthropic",
              return_value=_make_mock_llm(mock_critic_output_no_rerun)),
    ):
        result = insight_critic_node(state_with_specialists)

    assert result["next_action"] == "narrate"


def test_rerun_agent_is_none_when_no_rerun(
    state_with_specialists: AgentState,
    mock_critic_output_no_rerun: CriticOutput,
) -> None:
    with (
        patch("graph.nodes.insight_critic.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.insight_critic.ChatAnthropic",
              return_value=_make_mock_llm(mock_critic_output_no_rerun)),
    ):
        result = insight_critic_node(state_with_specialists)

    assert result["rerun_agent"] is None


def test_completed_agents_contains_insight_critic(
    state_with_specialists: AgentState,
    mock_critic_output_no_rerun: CriticOutput,
) -> None:
    with (
        patch("graph.nodes.insight_critic.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.insight_critic.ChatAnthropic",
              return_value=_make_mock_llm(mock_critic_output_no_rerun)),
    ):
        result = insight_critic_node(state_with_specialists)

    assert "insight_critic" in result["completed_agents"]


def test_rerun_count_incremented_by_1(
    state_with_specialists: AgentState,
    mock_critic_output_no_rerun: CriticOutput,
) -> None:
    initial_count = state_with_specialists.get("rerun_count", 0)
    with (
        patch("graph.nodes.insight_critic.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.insight_critic.ChatAnthropic",
              return_value=_make_mock_llm(mock_critic_output_no_rerun)),
    ):
        result = insight_critic_node(state_with_specialists)

    assert result["rerun_count"] == initial_count + 1


def test_rerun_hint_is_none_for_all_agents_when_no_rerun(
    state_with_specialists: AgentState,
    mock_critic_output_no_rerun: CriticOutput,
) -> None:
    with (
        patch("graph.nodes.insight_critic.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.insight_critic.ChatAnthropic",
              return_value=_make_mock_llm(mock_critic_output_no_rerun)),
    ):
        result = insight_critic_node(state_with_specialists)

    for ar in result["agent_results"]:
        assert ar.rerun_hint is None


def test_messages_is_list(
    state_with_specialists: AgentState,
    mock_critic_output_no_rerun: CriticOutput,
) -> None:
    with (
        patch("graph.nodes.insight_critic.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.insight_critic.ChatAnthropic",
              return_value=_make_mock_llm(mock_critic_output_no_rerun)),
    ):
        result = insight_critic_node(state_with_specialists)

    assert isinstance(result["messages"], list)
    assert len(result["messages"]) == 1


def test_agent_results_is_list(
    state_with_specialists: AgentState,
    mock_critic_output_no_rerun: CriticOutput,
) -> None:
    with (
        patch("graph.nodes.insight_critic.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.insight_critic.ChatAnthropic",
              return_value=_make_mock_llm(mock_critic_output_no_rerun)),
    ):
        result = insight_critic_node(state_with_specialists)

    assert isinstance(result["agent_results"], list)


# ---------------------------------------------------------------------------
# Rerun path
# ---------------------------------------------------------------------------

def test_next_action_is_dispatch_when_rerun_set(
    state_with_specialists: AgentState,
    mock_critic_output_with_rerun: CriticOutput,
) -> None:
    with (
        patch("graph.nodes.insight_critic.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.insight_critic.ChatAnthropic",
              return_value=_make_mock_llm(mock_critic_output_with_rerun)),
    ):
        result = insight_critic_node(state_with_specialists)

    assert result["next_action"] == "dispatch"


def test_rerun_agent_set_correctly(
    state_with_specialists: AgentState,
    mock_critic_output_with_rerun: CriticOutput,
) -> None:
    with (
        patch("graph.nodes.insight_critic.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.insight_critic.ChatAnthropic",
              return_value=_make_mock_llm(mock_critic_output_with_rerun)),
    ):
        result = insight_critic_node(state_with_specialists)

    assert result["rerun_agent"] == "viz_agent"


def test_rerun_hint_set_on_rerun_agent_only(
    state_with_specialists: AgentState,
    mock_critic_output_with_rerun: CriticOutput,
) -> None:
    with (
        patch("graph.nodes.insight_critic.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.insight_critic.ChatAnthropic",
              return_value=_make_mock_llm(mock_critic_output_with_rerun)),
    ):
        result = insight_critic_node(state_with_specialists)

    rerun_agent_name = mock_critic_output_with_rerun.rerun_agent
    for ar in result["agent_results"]:
        if ar.agent_name == rerun_agent_name:
            assert ar.rerun_hint is not None
            assert ar.rerun_hint == mock_critic_output_with_rerun.rerun_reason
        else:
            assert ar.rerun_hint is None


def test_rerun_count_incremented_on_rerun_path(
    state_with_specialists: AgentState,
    mock_critic_output_with_rerun: CriticOutput,
) -> None:
    initial_count = state_with_specialists.get("rerun_count", 0)
    with (
        patch("graph.nodes.insight_critic.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.insight_critic.ChatAnthropic",
              return_value=_make_mock_llm(mock_critic_output_with_rerun)),
    ):
        result = insight_critic_node(state_with_specialists)

    assert result["rerun_count"] == initial_count + 1


# ---------------------------------------------------------------------------
# Early return — empty agent_results
# ---------------------------------------------------------------------------

def test_early_return_when_no_specialist_results(base_state: AgentState) -> None:
    result = insight_critic_node(base_state)
    assert result["next_action"] == "narrate"


def test_early_return_completed_agents_contains_insight_critic(
    base_state: AgentState,
) -> None:
    result = insight_critic_node(base_state)
    assert "insight_critic" in result["completed_agents"]


def test_early_return_filters_non_specialist_results(base_state: AgentState) -> None:
    """A result from 'supervisor' should not count as a specialist result."""
    supervisor_result = AgentResult(
        agent_name="supervisor",
        success=True,
        findings={"plan": "some plan"},
        confidence=1.0,
    )
    state = {**base_state, "agent_results": [supervisor_result]}
    result = insight_critic_node(state)
    assert result["next_action"] == "narrate"


# ---------------------------------------------------------------------------
# Exception path
# ---------------------------------------------------------------------------

def test_exception_path_next_action_is_narrate(base_state: AgentState) -> None:
    with (
        patch("graph.nodes.insight_critic.get_llm_config",
              side_effect=RuntimeError("config failed")),
    ):
        state = {**base_state, "agent_results": [
            AgentResult(
                agent_name="profiler", success=True,
                findings={"schema": {}}, confidence=0.9,
            )
        ]}
        result = insight_critic_node(state)

    assert result["next_action"] == "narrate"


def test_exception_path_completed_agents_contains_insight_critic(
    base_state: AgentState,
) -> None:
    with (
        patch("graph.nodes.insight_critic.get_llm_config",
              side_effect=RuntimeError("config failed")),
    ):
        state = {**base_state, "agent_results": [
            AgentResult(
                agent_name="profiler", success=True,
                findings={"schema": {}}, confidence=0.9,
            )
        ]}
        result = insight_critic_node(state)

    assert "insight_critic" in result["completed_agents"]


def test_exception_does_not_propagate(base_state: AgentState) -> None:
    with (
        patch("graph.nodes.insight_critic.get_llm_config",
              side_effect=RuntimeError("config failed")),
    ):
        state = {**base_state, "agent_results": [
            AgentResult(
                agent_name="profiler", success=True,
                findings={"schema": {}}, confidence=0.9,
            )
        ]}
        # Should not raise
        result = insight_critic_node(state)

    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Config / LLM call assertions
# ---------------------------------------------------------------------------

def test_get_llm_config_called_once(
    state_with_specialists: AgentState,
    mock_critic_output_no_rerun: CriticOutput,
) -> None:
    with (
        patch("graph.nodes.insight_critic.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)) as mock_cfg,
        patch("graph.nodes.insight_critic.ChatAnthropic",
              return_value=_make_mock_llm(mock_critic_output_no_rerun)),
    ):
        insight_critic_node(state_with_specialists)

    mock_cfg.assert_called_once()


def test_with_structured_output_called_with_critic_output(
    state_with_specialists: AgentState,
    mock_critic_output_no_rerun: CriticOutput,
) -> None:
    mock_llm = _make_mock_llm(mock_critic_output_no_rerun)
    with (
        patch("graph.nodes.insight_critic.get_llm_config",
              return_value=LLMConfig(model="claude-sonnet-3-7", temperature=0)),
        patch("graph.nodes.insight_critic.ChatAnthropic",
              return_value=mock_llm),
    ):
        insight_critic_node(state_with_specialists)

    mock_llm.with_structured_output.assert_called_once_with(CriticOutput)
