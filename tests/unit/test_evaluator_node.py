from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from core.config import AgentConfig, LLMConfig
from core.schema import AgentResult, EvaluationResult, Verdict
from core.state import AgentState
from graph.nodes.evaluator import (
    _build_evaluator_prompt,
    evaluator_node,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def state_with_narrative(base_state: AgentState) -> AgentState:
    return {
        **base_state,
        "narrative": "Income is strongly driven by age (r=0.85).",
        "key_insights": [
            "Age explains 85% of income variance",
            "3 outliers above $110,000 detected",
        ],
        "caveats": ["Small sample size of 10 rows"],
        "agent_results": [
            AgentResult(
                agent_name="profiler",
                success=True,
                findings={"notable_columns": ["income"]},
                confidence=0.9,
            ),
            AgentResult(
                agent_name="stat_analyst",
                success=True,
                findings={"notable_findings": ["income correlated with age"]},
                confidence=0.85,
            ),
        ],
        "evaluation_count": 0,
        "completed_agents": ["profiler", "stat_analyst", "narrator"],
    }


@pytest.fixture
def mock_evaluation_strong() -> EvaluationResult:
    return EvaluationResult(
        goal_coverage=0.9,
        insight_quality=0.85,
        evidence_quality=0.9,
        overall_score=0.89,
        strengths=["Directly addresses revenue driver goal"],
        gaps=[],
        verdict=Verdict.STRONG,
        retry_instructions=None,
    )


@pytest.fixture
def mock_evaluation_adequate() -> EvaluationResult:
    return EvaluationResult(
        goal_coverage=0.6,
        insight_quality=0.55,
        evidence_quality=0.6,
        overall_score=0.58,
        strengths=["Some goal alignment"],
        gaps=["Missing feature importance analysis"],
        verdict=Verdict.ADEQUATE,
        retry_instructions=None,
    )


@pytest.fixture
def mock_evaluation_weak() -> EvaluationResult:
    return EvaluationResult(
        goal_coverage=0.3,
        insight_quality=0.4,
        evidence_quality=0.5,
        overall_score=0.37,
        strengths=[],
        gaps=["Narrative does not address revenue drivers directly"],
        verdict=Verdict.WEAK,
        retry_instructions=(
            "Re-focus analysis on revenue drivers. Run feature "
            "importance with revenue as target. Ensure key insights "
            "mention revenue explicitly."
        ),
    )


def _make_mock_llm(evaluation: EvaluationResult) -> MagicMock:
    structured = MagicMock()
    structured.invoke.return_value = evaluation
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


_ENABLED_CONFIG = AgentConfig(stat_analyst_use_react=False, evaluator_enabled=True, agents_required=["profiler", "stat_analyst"])
_DISABLED_CONFIG = AgentConfig(stat_analyst_use_react=False, evaluator_enabled=False, agents_required=["profiler", "stat_analyst"])
_LLM_CONFIG = LLMConfig(model="claude-sonnet-3-7", temperature=0)


# ---------------------------------------------------------------------------
# _build_evaluator_prompt
# ---------------------------------------------------------------------------

def test_build_evaluator_prompt_contains_goal(
    state_with_narrative: AgentState,
) -> None:
    request = state_with_narrative["request"]
    prompt = _build_evaluator_prompt(request, state_with_narrative)
    assert request.goal in prompt


def test_build_evaluator_prompt_contains_narrative(
    state_with_narrative: AgentState,
) -> None:
    request = state_with_narrative["request"]
    prompt = _build_evaluator_prompt(request, state_with_narrative)
    assert state_with_narrative["narrative"] in prompt


def test_build_evaluator_prompt_contains_key_insights(
    state_with_narrative: AgentState,
) -> None:
    request = state_with_narrative["request"]
    prompt = _build_evaluator_prompt(request, state_with_narrative)
    assert "Age explains 85% of income variance" in prompt


# ---------------------------------------------------------------------------
# Happy path — strong verdict
# ---------------------------------------------------------------------------

def test_evaluator_node_returns_expected_keys(
    state_with_narrative: AgentState,
    mock_evaluation_strong: EvaluationResult,
) -> None:
    with (
        patch("graph.nodes.evaluator.get_agent_config", return_value=_ENABLED_CONFIG),
        patch("graph.nodes.evaluator.get_llm_config", return_value=_LLM_CONFIG),
        patch("graph.nodes.evaluator.ChatAnthropic",
              return_value=_make_mock_llm(mock_evaluation_strong)),
    ):
        result = evaluator_node(state_with_narrative)

    for key in ("evaluation", "evaluation_count", "next_action",
                "completed_agents", "messages"):
        assert key in result


def test_next_action_output_on_strong_verdict(
    state_with_narrative: AgentState,
    mock_evaluation_strong: EvaluationResult,
) -> None:
    with (
        patch("graph.nodes.evaluator.get_agent_config", return_value=_ENABLED_CONFIG),
        patch("graph.nodes.evaluator.get_llm_config", return_value=_LLM_CONFIG),
        patch("graph.nodes.evaluator.ChatAnthropic",
              return_value=_make_mock_llm(mock_evaluation_strong)),
    ):
        result = evaluator_node(state_with_narrative)

    assert result["next_action"] == "output"


def test_next_action_output_on_adequate_verdict(
    state_with_narrative: AgentState,
    mock_evaluation_adequate: EvaluationResult,
) -> None:
    with (
        patch("graph.nodes.evaluator.get_agent_config", return_value=_ENABLED_CONFIG),
        patch("graph.nodes.evaluator.get_llm_config", return_value=_LLM_CONFIG),
        patch("graph.nodes.evaluator.ChatAnthropic",
              return_value=_make_mock_llm(mock_evaluation_adequate)),
    ):
        result = evaluator_node(state_with_narrative)

    assert result["next_action"] == "output"


def test_evaluation_is_evaluation_result_instance(
    state_with_narrative: AgentState,
    mock_evaluation_strong: EvaluationResult,
) -> None:
    with (
        patch("graph.nodes.evaluator.get_agent_config", return_value=_ENABLED_CONFIG),
        patch("graph.nodes.evaluator.get_llm_config", return_value=_LLM_CONFIG),
        patch("graph.nodes.evaluator.ChatAnthropic",
              return_value=_make_mock_llm(mock_evaluation_strong)),
    ):
        result = evaluator_node(state_with_narrative)

    assert isinstance(result["evaluation"], EvaluationResult)


def test_evaluation_count_incremented(
    state_with_narrative: AgentState,
    mock_evaluation_strong: EvaluationResult,
) -> None:
    with (
        patch("graph.nodes.evaluator.get_agent_config", return_value=_ENABLED_CONFIG),
        patch("graph.nodes.evaluator.get_llm_config", return_value=_LLM_CONFIG),
        patch("graph.nodes.evaluator.ChatAnthropic",
              return_value=_make_mock_llm(mock_evaluation_strong)),
    ):
        result = evaluator_node(state_with_narrative)

    assert result["evaluation_count"] == state_with_narrative["evaluation_count"] + 1


def test_completed_agents_contains_evaluator(
    state_with_narrative: AgentState,
    mock_evaluation_strong: EvaluationResult,
) -> None:
    with (
        patch("graph.nodes.evaluator.get_agent_config", return_value=_ENABLED_CONFIG),
        patch("graph.nodes.evaluator.get_llm_config", return_value=_LLM_CONFIG),
        patch("graph.nodes.evaluator.ChatAnthropic",
              return_value=_make_mock_llm(mock_evaluation_strong)),
    ):
        result = evaluator_node(state_with_narrative)

    assert "evaluator" in result["completed_agents"]


def test_retry_instructions_none_on_strong_verdict(
    state_with_narrative: AgentState,
    mock_evaluation_strong: EvaluationResult,
) -> None:
    with (
        patch("graph.nodes.evaluator.get_agent_config", return_value=_ENABLED_CONFIG),
        patch("graph.nodes.evaluator.get_llm_config", return_value=_LLM_CONFIG),
        patch("graph.nodes.evaluator.ChatAnthropic",
              return_value=_make_mock_llm(mock_evaluation_strong)),
    ):
        result = evaluator_node(state_with_narrative)

    assert result["evaluation"].retry_instructions is None


# ---------------------------------------------------------------------------
# Weak verdict — replan path
# ---------------------------------------------------------------------------

def test_next_action_replan_on_weak_verdict_first_run(
    state_with_narrative: AgentState,
    mock_evaluation_weak: EvaluationResult,
) -> None:
    """evaluation_count=0 → new_count=1 < 2 → replan."""
    with (
        patch("graph.nodes.evaluator.get_agent_config", return_value=_ENABLED_CONFIG),
        patch("graph.nodes.evaluator.get_llm_config", return_value=_LLM_CONFIG),
        patch("graph.nodes.evaluator.ChatAnthropic",
              return_value=_make_mock_llm(mock_evaluation_weak)),
    ):
        result = evaluator_node(state_with_narrative)

    assert result["next_action"] == "replan"


def test_retry_instructions_non_none_on_weak_verdict(
    state_with_narrative: AgentState,
    mock_evaluation_weak: EvaluationResult,
) -> None:
    with (
        patch("graph.nodes.evaluator.get_agent_config", return_value=_ENABLED_CONFIG),
        patch("graph.nodes.evaluator.get_llm_config", return_value=_LLM_CONFIG),
        patch("graph.nodes.evaluator.ChatAnthropic",
              return_value=_make_mock_llm(mock_evaluation_weak)),
    ):
        result = evaluator_node(state_with_narrative)

    assert result["evaluation"].retry_instructions is not None


def test_loop_guard_next_action_output_when_evaluation_count_ge_2(
    state_with_narrative: AgentState,
    mock_evaluation_weak: EvaluationResult,
) -> None:
    """evaluation_count=2 → new_count=3, guard triggers → output."""
    guarded_state = {**state_with_narrative, "evaluation_count": 2}
    with (
        patch("graph.nodes.evaluator.get_agent_config", return_value=_ENABLED_CONFIG),
        patch("graph.nodes.evaluator.get_llm_config", return_value=_LLM_CONFIG),
        patch("graph.nodes.evaluator.ChatAnthropic",
              return_value=_make_mock_llm(mock_evaluation_weak)),
    ):
        result = evaluator_node(guarded_state)

    assert result["next_action"] == "output"


def test_loop_guard_evaluation_count_still_incremented(
    state_with_narrative: AgentState,
    mock_evaluation_weak: EvaluationResult,
) -> None:
    guarded_state = {**state_with_narrative, "evaluation_count": 2}
    with (
        patch("graph.nodes.evaluator.get_agent_config", return_value=_ENABLED_CONFIG),
        patch("graph.nodes.evaluator.get_llm_config", return_value=_LLM_CONFIG),
        patch("graph.nodes.evaluator.ChatAnthropic",
              return_value=_make_mock_llm(mock_evaluation_weak)),
    ):
        result = evaluator_node(guarded_state)

    assert result["evaluation_count"] == 3


# ---------------------------------------------------------------------------
# Disabled path
# ---------------------------------------------------------------------------

def test_evaluator_disabled_next_action_output(
    state_with_narrative: AgentState,
) -> None:
    with patch("graph.nodes.evaluator.get_agent_config",
               return_value=_DISABLED_CONFIG) as mock_acfg:
        result = evaluator_node(state_with_narrative)

    assert result["next_action"] == "output"


def test_evaluator_disabled_evaluation_is_none(
    state_with_narrative: AgentState,
) -> None:
    with patch("graph.nodes.evaluator.get_agent_config",
               return_value=_DISABLED_CONFIG):
        result = evaluator_node(state_with_narrative)

    assert result["evaluation"] is None


def test_evaluator_disabled_evaluation_count_zero(
    state_with_narrative: AgentState,
) -> None:
    with patch("graph.nodes.evaluator.get_agent_config",
               return_value=_DISABLED_CONFIG):
        result = evaluator_node(state_with_narrative)

    assert result["evaluation_count"] == 0


def test_evaluator_disabled_completed_agents_contains_evaluator(
    state_with_narrative: AgentState,
) -> None:
    with patch("graph.nodes.evaluator.get_agent_config",
               return_value=_DISABLED_CONFIG):
        result = evaluator_node(state_with_narrative)

    assert "evaluator" in result["completed_agents"]


def test_evaluator_disabled_no_llm_calls(
    state_with_narrative: AgentState,
) -> None:
    with (
        patch("graph.nodes.evaluator.get_agent_config",
              return_value=_DISABLED_CONFIG),
        patch("graph.nodes.evaluator.get_llm_config") as mock_llm_cfg,
    ):
        evaluator_node(state_with_narrative)

    mock_llm_cfg.assert_not_called()


# ---------------------------------------------------------------------------
# Exception path
# ---------------------------------------------------------------------------

def test_exception_path_next_action_output(
    state_with_narrative: AgentState,
) -> None:
    with (
        patch("graph.nodes.evaluator.get_agent_config", return_value=_ENABLED_CONFIG),
        patch("graph.nodes.evaluator.get_llm_config",
              side_effect=RuntimeError("config failed")),
    ):
        result = evaluator_node(state_with_narrative)

    assert result["next_action"] == "output"


def test_exception_path_verdict_weak(
    state_with_narrative: AgentState,
) -> None:
    with (
        patch("graph.nodes.evaluator.get_agent_config", return_value=_ENABLED_CONFIG),
        patch("graph.nodes.evaluator.get_llm_config",
              side_effect=RuntimeError("config failed")),
    ):
        result = evaluator_node(state_with_narrative)

    assert result["evaluation"].verdict == Verdict.WEAK


def test_exception_path_evaluation_count_incremented(
    state_with_narrative: AgentState,
) -> None:
    with (
        patch("graph.nodes.evaluator.get_agent_config", return_value=_ENABLED_CONFIG),
        patch("graph.nodes.evaluator.get_llm_config",
              side_effect=RuntimeError("config failed")),
    ):
        result = evaluator_node(state_with_narrative)

    assert result["evaluation_count"] == state_with_narrative["evaluation_count"] + 1


def test_exception_does_not_propagate(
    state_with_narrative: AgentState,
) -> None:
    with (
        patch("graph.nodes.evaluator.get_agent_config", return_value=_ENABLED_CONFIG),
        patch("graph.nodes.evaluator.get_llm_config",
              side_effect=RuntimeError("config failed")),
    ):
        result = evaluator_node(state_with_narrative)

    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Config call assertions
# ---------------------------------------------------------------------------

def test_get_agent_config_called_once_when_enabled(
    state_with_narrative: AgentState,
    mock_evaluation_strong: EvaluationResult,
) -> None:
    with (
        patch("graph.nodes.evaluator.get_agent_config",
              return_value=_ENABLED_CONFIG) as mock_acfg,
        patch("graph.nodes.evaluator.get_llm_config", return_value=_LLM_CONFIG),
        patch("graph.nodes.evaluator.ChatAnthropic",
              return_value=_make_mock_llm(mock_evaluation_strong)),
    ):
        evaluator_node(state_with_narrative)

    mock_acfg.assert_called_once()


def test_get_agent_config_called_once_when_disabled(
    state_with_narrative: AgentState,
) -> None:
    with patch("graph.nodes.evaluator.get_agent_config",
               return_value=_DISABLED_CONFIG) as mock_acfg:
        evaluator_node(state_with_narrative)

    mock_acfg.assert_called_once()


def test_get_llm_config_called_once_when_enabled(
    state_with_narrative: AgentState,
    mock_evaluation_strong: EvaluationResult,
) -> None:
    with (
        patch("graph.nodes.evaluator.get_agent_config", return_value=_ENABLED_CONFIG),
        patch("graph.nodes.evaluator.get_llm_config",
              return_value=_LLM_CONFIG) as mock_cfg,
        patch("graph.nodes.evaluator.ChatAnthropic",
              return_value=_make_mock_llm(mock_evaluation_strong)),
    ):
        evaluator_node(state_with_narrative)

    mock_cfg.assert_called_once()


def test_with_structured_output_called_with_evaluation_result(
    state_with_narrative: AgentState,
    mock_evaluation_strong: EvaluationResult,
) -> None:
    mock_llm = _make_mock_llm(mock_evaluation_strong)
    with (
        patch("graph.nodes.evaluator.get_agent_config", return_value=_ENABLED_CONFIG),
        patch("graph.nodes.evaluator.get_llm_config", return_value=_LLM_CONFIG),
        patch("graph.nodes.evaluator.ChatAnthropic", return_value=mock_llm),
    ):
        evaluator_node(state_with_narrative)

    mock_llm.with_structured_output.assert_called_once_with(EvaluationResult)
