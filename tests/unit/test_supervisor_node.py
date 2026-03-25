from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from core.config import LLMConfig
from core.schema import (
    AnalysisPlan,
    EvaluationResult,
    OutputType,
    Verdict,
)
from core.state import AgentState
from graph.nodes.supervisor import (
    _all_dispatched_complete,
    _determine_mode,
    _format_plan_for_display,
    _generate_plan,
    _handle_dispatch_mode,
    _handle_plan_mode,
    _handle_replan_mode,
    _handle_route_mode,
    _inspect_dataset,
    supervisor_node,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_LLM_CONFIG = LLMConfig(model="claude-sonnet-3-7", temperature=0)


@pytest.fixture
def mock_plan(sample_request) -> AnalysisPlan:
    return AnalysisPlan(
        session_id="test-session-001",
        goal="Find key drivers of income",
        steps=["Profile dataset", "Run correlations", "Visualise"],
        agents=["profiler", "stat_analyst", "viz_agent"],
        parallel=[["profiler", "stat_analyst", "viz_agent"]],
        output_formats=[OutputType.REPORT, OutputType.JSON],
    )


@pytest.fixture
def state_plan_approved(base_state: AgentState, mock_plan: AnalysisPlan) -> AgentState:
    return {
        **base_state,
        "plan": mock_plan,
        "plan_approved": True,
        "next_action": "dispatch",
        "dispatched_agents": [],
        "completed_agents": [],
    }


@pytest.fixture
def state_specialists_complete(
    base_state: AgentState, mock_plan: AnalysisPlan
) -> AgentState:
    return {
        **base_state,
        "plan": mock_plan,
        "plan_approved": True,
        "next_action": None,
        "dispatched_agents": ["profiler", "stat_analyst", "viz_agent"],
        "completed_agents": ["profiler", "stat_analyst", "viz_agent"],
    }


@pytest.fixture
def state_post_critic_narrate(
    base_state: AgentState, mock_plan: AnalysisPlan
) -> AgentState:
    return {
        **base_state,
        "plan": mock_plan,
        "plan_approved": True,
        "next_action": "narrate",
        "dispatched_agents": ["profiler", "stat_analyst", "viz_agent"],
        "completed_agents": [
            "profiler", "stat_analyst", "viz_agent", "insight_critic"
        ],
    }


@pytest.fixture
def state_replan(base_state: AgentState, mock_plan: AnalysisPlan) -> AgentState:
    return {
        **base_state,
        "plan": mock_plan,
        "plan_approved": True,
        "next_action": "replan",
        "evaluation": EvaluationResult(
            goal_coverage=0.3,
            insight_quality=0.4,
            evidence_quality=0.5,
            overall_score=0.37,
            strengths=[],
            gaps=["Missing revenue focus"],
            verdict=Verdict.WEAK,
            retry_instructions="Focus on revenue drivers explicitly.",
        ),
        "evaluation_count": 1,
        "dispatched_agents": ["profiler", "stat_analyst", "viz_agent"],
        "completed_agents": [
            "profiler", "stat_analyst", "viz_agent",
            "insight_critic", "narrator", "evaluator",
        ],
    }


def _make_mock_llm(plan: AnalysisPlan) -> MagicMock:
    structured = MagicMock()
    structured.invoke.return_value = plan
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


_FAKE_DATASET_INFO = {"schema": {"numeric_cols": ["age", "income"]}, "sample": []}


# ---------------------------------------------------------------------------
# _determine_mode
# ---------------------------------------------------------------------------

def test_determine_mode_plan_when_no_plan(base_state: AgentState) -> None:
    assert _determine_mode(base_state) == "plan"


def test_determine_mode_plan_when_plan_not_approved(
    base_state: AgentState, mock_plan: AnalysisPlan
) -> None:
    state = {**base_state, "plan": mock_plan, "plan_approved": False}
    assert _determine_mode(state) == "plan"


def test_determine_mode_replan_when_next_action_replan(
    state_replan: AgentState,
) -> None:
    assert _determine_mode(state_replan) == "replan"


def test_determine_mode_dispatch_when_approved_and_none_dispatched(
    state_plan_approved: AgentState,
) -> None:
    assert _determine_mode(state_plan_approved) == "dispatch"


def test_determine_mode_route_when_all_dispatched_complete(
    state_specialists_complete: AgentState,
) -> None:
    assert _determine_mode(state_specialists_complete) == "route"


def test_determine_mode_route_when_next_action_narrate(
    state_post_critic_narrate: AgentState,
) -> None:
    assert _determine_mode(state_post_critic_narrate) == "route"


def test_determine_mode_route_when_next_action_output(
    base_state: AgentState, mock_plan: AnalysisPlan
) -> None:
    state = {
        **base_state,
        "plan": mock_plan,
        "plan_approved": True,
        "next_action": "output",
        "dispatched_agents": ["profiler"],
        "completed_agents": ["profiler"],
    }
    assert _determine_mode(state) == "route"


# ---------------------------------------------------------------------------
# _all_dispatched_complete
# ---------------------------------------------------------------------------

def test_all_dispatched_complete_true(state_specialists_complete: AgentState) -> None:
    assert _all_dispatched_complete(state_specialists_complete) is True


def test_all_dispatched_complete_false_when_partial(
    base_state: AgentState,
) -> None:
    state = {
        **base_state,
        "dispatched_agents": ["profiler", "stat_analyst"],
        "completed_agents": ["profiler"],
    }
    assert _all_dispatched_complete(state) is False


def test_all_dispatched_complete_false_when_empty(base_state: AgentState) -> None:
    assert _all_dispatched_complete(base_state) is False


# ---------------------------------------------------------------------------
# _format_plan_for_display
# ---------------------------------------------------------------------------

def test_format_plan_for_display_contains_goal(mock_plan: AnalysisPlan) -> None:
    text = _format_plan_for_display(mock_plan)
    assert mock_plan.goal in text


def test_format_plan_for_display_contains_agents(mock_plan: AnalysisPlan) -> None:
    text = _format_plan_for_display(mock_plan)
    for agent in mock_plan.agents:
        assert agent in text


def test_format_plan_for_display_contains_steps(mock_plan: AnalysisPlan) -> None:
    text = _format_plan_for_display(mock_plan)
    for step in mock_plan.steps:
        assert step in text


# ---------------------------------------------------------------------------
# _handle_plan_mode — approval
# ---------------------------------------------------------------------------

def test_handle_plan_mode_calls_interrupt_once(
    base_state: AgentState, mock_plan: AnalysisPlan
) -> None:
    with (
        patch("graph.nodes.supervisor.get_llm_config", return_value=_LLM_CONFIG),
        patch("graph.nodes.supervisor.ChatAnthropic",
              return_value=_make_mock_llm(mock_plan)),
        patch("graph.nodes.supervisor._inspect_dataset",
              return_value=_FAKE_DATASET_INFO),
        patch("graph.nodes.supervisor.interrupt", return_value="approve") as mock_intr,
    ):
        _handle_plan_mode(base_state)

    mock_intr.assert_called_once()


def test_handle_plan_mode_approved_sets_plan_approved_true(
    base_state: AgentState, mock_plan: AnalysisPlan
) -> None:
    with (
        patch("graph.nodes.supervisor.get_llm_config", return_value=_LLM_CONFIG),
        patch("graph.nodes.supervisor.ChatAnthropic",
              return_value=_make_mock_llm(mock_plan)),
        patch("graph.nodes.supervisor._inspect_dataset",
              return_value=_FAKE_DATASET_INFO),
        patch("graph.nodes.supervisor.interrupt", return_value="approve"),
    ):
        result = _handle_plan_mode(base_state)

    assert result["plan_approved"] is True


def test_handle_plan_mode_approved_sets_plan(
    base_state: AgentState, mock_plan: AnalysisPlan
) -> None:
    with (
        patch("graph.nodes.supervisor.get_llm_config", return_value=_LLM_CONFIG),
        patch("graph.nodes.supervisor.ChatAnthropic",
              return_value=_make_mock_llm(mock_plan)),
        patch("graph.nodes.supervisor._inspect_dataset",
              return_value=_FAKE_DATASET_INFO),
        patch("graph.nodes.supervisor.interrupt", return_value="approve"),
    ):
        result = _handle_plan_mode(base_state)

    assert result["plan"] is not None
    assert isinstance(result["plan"], AnalysisPlan)


def test_handle_plan_mode_approved_next_action_dispatch(
    base_state: AgentState, mock_plan: AnalysisPlan
) -> None:
    with (
        patch("graph.nodes.supervisor.get_llm_config", return_value=_LLM_CONFIG),
        patch("graph.nodes.supervisor.ChatAnthropic",
              return_value=_make_mock_llm(mock_plan)),
        patch("graph.nodes.supervisor._inspect_dataset",
              return_value=_FAKE_DATASET_INFO),
        patch("graph.nodes.supervisor.interrupt", return_value="approve"),
    ):
        result = _handle_plan_mode(base_state)

    assert result["next_action"] == "dispatch"


def test_handle_plan_mode_rejected_sets_plan_none(
    base_state: AgentState, mock_plan: AnalysisPlan
) -> None:
    with (
        patch("graph.nodes.supervisor.get_llm_config", return_value=_LLM_CONFIG),
        patch("graph.nodes.supervisor.ChatAnthropic",
              return_value=_make_mock_llm(mock_plan)),
        patch("graph.nodes.supervisor._inspect_dataset",
              return_value=_FAKE_DATASET_INFO),
        patch("graph.nodes.supervisor.interrupt",
              return_value="needs more focus on revenue"),
    ):
        result = _handle_plan_mode(base_state)

    assert result["plan"] is None


def test_handle_plan_mode_rejected_plan_approved_false(
    base_state: AgentState, mock_plan: AnalysisPlan
) -> None:
    with (
        patch("graph.nodes.supervisor.get_llm_config", return_value=_LLM_CONFIG),
        patch("graph.nodes.supervisor.ChatAnthropic",
              return_value=_make_mock_llm(mock_plan)),
        patch("graph.nodes.supervisor._inspect_dataset",
              return_value=_FAKE_DATASET_INFO),
        patch("graph.nodes.supervisor.interrupt",
              return_value="needs more focus on revenue"),
    ):
        result = _handle_plan_mode(base_state)

    assert result["plan_approved"] is False


def test_handle_plan_mode_calls_inspect_dataset_once(
    base_state: AgentState, mock_plan: AnalysisPlan
) -> None:
    with (
        patch("graph.nodes.supervisor.get_llm_config", return_value=_LLM_CONFIG),
        patch("graph.nodes.supervisor.ChatAnthropic",
              return_value=_make_mock_llm(mock_plan)),
        patch("graph.nodes.supervisor._inspect_dataset",
              return_value=_FAKE_DATASET_INFO) as mock_insp,
        patch("graph.nodes.supervisor.interrupt", return_value="approve"),
    ):
        _handle_plan_mode(base_state)

    mock_insp.assert_called_once()


def test_handle_plan_mode_calls_generate_plan_once(
    base_state: AgentState, mock_plan: AnalysisPlan
) -> None:
    with (
        patch("graph.nodes.supervisor.get_llm_config", return_value=_LLM_CONFIG),
        patch("graph.nodes.supervisor.ChatAnthropic",
              return_value=_make_mock_llm(mock_plan)),
        patch("graph.nodes.supervisor._inspect_dataset",
              return_value=_FAKE_DATASET_INFO),
        patch("graph.nodes.supervisor._generate_plan",
              return_value=mock_plan) as mock_gen,
        patch("graph.nodes.supervisor.interrupt", return_value="approve"),
    ):
        _handle_plan_mode(base_state)

    mock_gen.assert_called_once()


# ---------------------------------------------------------------------------
# _handle_dispatch_mode
# ---------------------------------------------------------------------------

def test_handle_dispatch_mode_dispatches_plan_agents(
    state_plan_approved: AgentState, mock_plan: AnalysisPlan
) -> None:
    result = _handle_dispatch_mode(state_plan_approved)
    assert set(result["dispatched_agents"]) == set(mock_plan.agents)


def test_handle_dispatch_mode_dispatches_only_rerun_agent(
    state_plan_approved: AgentState,
) -> None:
    state = {
        **state_plan_approved,
        "rerun_agent": "viz_agent",
        "rerun_count": 1,
        "completed_agents": ["profiler", "stat_analyst"],
    }
    result = _handle_dispatch_mode(state)
    assert result["dispatched_agents"] == ["viz_agent"]


def test_handle_dispatch_mode_next_action_wait(
    state_plan_approved: AgentState,
) -> None:
    result = _handle_dispatch_mode(state_plan_approved)
    assert result["next_action"] == "wait"


def test_handle_dispatch_mode_messages_is_list(
    state_plan_approved: AgentState,
) -> None:
    result = _handle_dispatch_mode(state_plan_approved)
    assert isinstance(result["messages"], list)
    assert len(result["messages"]) > 0


# ---------------------------------------------------------------------------
# _handle_replan_mode
# ---------------------------------------------------------------------------

def test_handle_replan_mode_plan_approved_true(
    state_replan: AgentState, mock_plan: AnalysisPlan
) -> None:
    with (
        patch("graph.nodes.supervisor.get_llm_config", return_value=_LLM_CONFIG),
        patch("graph.nodes.supervisor.ChatAnthropic",
              return_value=_make_mock_llm(mock_plan)),
        patch("graph.nodes.supervisor._inspect_dataset",
              return_value=_FAKE_DATASET_INFO),
    ):
        result = _handle_replan_mode(state_replan)

    assert result["plan_approved"] is True


def test_handle_replan_mode_next_action_dispatch(
    state_replan: AgentState, mock_plan: AnalysisPlan
) -> None:
    with (
        patch("graph.nodes.supervisor.get_llm_config", return_value=_LLM_CONFIG),
        patch("graph.nodes.supervisor.ChatAnthropic",
              return_value=_make_mock_llm(mock_plan)),
        patch("graph.nodes.supervisor._inspect_dataset",
              return_value=_FAKE_DATASET_INFO),
    ):
        result = _handle_replan_mode(state_replan)

    assert result["next_action"] == "dispatch"


def test_handle_replan_mode_rerun_agent_cleared(
    state_replan: AgentState, mock_plan: AnalysisPlan
) -> None:
    with (
        patch("graph.nodes.supervisor.get_llm_config", return_value=_LLM_CONFIG),
        patch("graph.nodes.supervisor.ChatAnthropic",
              return_value=_make_mock_llm(mock_plan)),
        patch("graph.nodes.supervisor._inspect_dataset",
              return_value=_FAKE_DATASET_INFO),
    ):
        result = _handle_replan_mode(state_replan)

    assert result["rerun_agent"] is None


def test_handle_replan_mode_passes_retry_instructions(
    state_replan: AgentState, mock_plan: AnalysisPlan
) -> None:
    with (
        patch("graph.nodes.supervisor.get_llm_config", return_value=_LLM_CONFIG),
        patch("graph.nodes.supervisor.ChatAnthropic",
              return_value=_make_mock_llm(mock_plan)),
        patch("graph.nodes.supervisor._inspect_dataset",
              return_value=_FAKE_DATASET_INFO),
        patch("graph.nodes.supervisor._generate_plan",
              return_value=mock_plan) as mock_gen,
    ):
        _handle_replan_mode(state_replan)

    call_kwargs = mock_gen.call_args
    # retry_instructions should be passed as 4th positional arg or keyword
    args, kwargs = call_kwargs
    retry = args[3] if len(args) > 3 else kwargs.get("retry_instructions")
    assert retry == "Focus on revenue drivers explicitly."


# ---------------------------------------------------------------------------
# _handle_route_mode
# ---------------------------------------------------------------------------

def test_handle_route_mode_critique_when_specialists_done_and_critic_not_run(
    state_specialists_complete: AgentState,
) -> None:
    result = _handle_route_mode(state_specialists_complete)
    assert result["next_action"] == "critique"


def test_handle_route_mode_narrate_when_next_action_narrate(
    state_post_critic_narrate: AgentState,
) -> None:
    result = _handle_route_mode(state_post_critic_narrate)
    assert result["next_action"] == "narrate"


def test_handle_route_mode_output_when_next_action_output(
    base_state: AgentState, mock_plan: AnalysisPlan
) -> None:
    state = {
        **base_state,
        "plan": mock_plan,
        "plan_approved": True,
        "next_action": "output",
        "completed_agents": [
            "profiler", "stat_analyst", "viz_agent",
            "insight_critic", "narrator", "evaluator",
        ],
    }
    result = _handle_route_mode(state)
    assert result["next_action"] == "output"


def test_handle_route_mode_complete_for_unknown_next_action(
    base_state: AgentState, mock_plan: AnalysisPlan
) -> None:
    state = {
        **base_state,
        "plan": mock_plan,
        "plan_approved": True,
        "next_action": "unknown_signal",
        "completed_agents": [
            "profiler", "stat_analyst", "viz_agent", "insight_critic",
        ],
    }
    result = _handle_route_mode(state)
    assert result["next_action"] == "complete"


# ---------------------------------------------------------------------------
# supervisor_node dispatch tests
# ---------------------------------------------------------------------------

def test_supervisor_node_calls_handle_plan_mode_when_no_plan(
    base_state: AgentState, mock_plan: AnalysisPlan
) -> None:
    with (
        patch("graph.nodes.supervisor._handle_plan_mode",
              return_value={"plan": mock_plan, "plan_approved": True,
                            "next_action": "dispatch", "messages": []}) as mock_handler,
        patch("graph.nodes.supervisor._determine_mode", return_value="plan"),
    ):
        supervisor_node(base_state)

    mock_handler.assert_called_once()


def test_supervisor_node_calls_handle_dispatch_mode(
    state_plan_approved: AgentState,
) -> None:
    with (
        patch("graph.nodes.supervisor._handle_dispatch_mode",
              return_value={"dispatched_agents": [], "next_action": "wait",
                            "messages": []}) as mock_handler,
        patch("graph.nodes.supervisor._determine_mode", return_value="dispatch"),
    ):
        supervisor_node(state_plan_approved)

    mock_handler.assert_called_once()


def test_supervisor_node_calls_handle_replan_mode(
    state_replan: AgentState, mock_plan: AnalysisPlan
) -> None:
    with (
        patch("graph.nodes.supervisor._handle_replan_mode",
              return_value={"plan": mock_plan, "plan_approved": True,
                            "next_action": "dispatch", "rerun_agent": None,
                            "messages": []}) as mock_handler,
        patch("graph.nodes.supervisor._determine_mode", return_value="replan"),
    ):
        supervisor_node(state_replan)

    mock_handler.assert_called_once()


def test_supervisor_node_calls_handle_route_mode(
    state_specialists_complete: AgentState,
) -> None:
    with (
        patch("graph.nodes.supervisor._handle_route_mode",
              return_value={"next_action": "critique",
                            "messages": []}) as mock_handler,
        patch("graph.nodes.supervisor._determine_mode", return_value="route"),
    ):
        supervisor_node(state_specialists_complete)

    mock_handler.assert_called_once()


def test_supervisor_node_wait_mode_returns_messages(
    base_state: AgentState,
) -> None:
    with patch("graph.nodes.supervisor._determine_mode", return_value="wait"):
        result = supervisor_node(base_state)

    assert "messages" in result
    assert isinstance(result["messages"], list)
