import pytest
from langgraph.graph import END
from langgraph.types import Send

from core.schema import AnalysisPlan, OutputType
from core.state import AgentState
from graph.builder import (
    build_graph,
    get_graph,
    route_after_evaluator,
    route_after_supervisor,
)
from memory.checkpointer import get_memory_checkpointer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def checkpointer():
    return get_memory_checkpointer()


@pytest.fixture
def compiled_graph(checkpointer):
    return build_graph(checkpointer)


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
def dispatch_state(base_state: AgentState, mock_plan: AnalysisPlan) -> AgentState:
    return {
        **base_state,
        "plan": mock_plan,
        "plan_approved": True,
        "next_action": "dispatch",
        "dispatched_agents": [],
        "completed_agents": [],
        "rerun_agent": None,
        "rerun_count": 0,
    }


@pytest.fixture
def dispatch_state_with_completed(
    base_state: AgentState, mock_plan: AnalysisPlan
) -> AgentState:
    """Profiler already done — should not appear in Send list."""
    return {
        **base_state,
        "plan": mock_plan,
        "plan_approved": True,
        "next_action": "dispatch",
        "dispatched_agents": ["profiler"],
        "completed_agents": ["profiler"],
        "rerun_agent": None,
        "rerun_count": 0,
    }


@pytest.fixture
def rerun_state(base_state: AgentState, mock_plan: AnalysisPlan) -> AgentState:
    return {
        **base_state,
        "plan": mock_plan,
        "plan_approved": True,
        "next_action": "dispatch",
        "dispatched_agents": ["profiler", "stat_analyst", "viz_agent"],
        "completed_agents": [
            "profiler", "stat_analyst", "viz_agent", "insight_critic"
        ],
        "rerun_agent": "viz_agent",
        "rerun_count": 1,
    }


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

def test_build_graph_compiles_without_error(checkpointer) -> None:
    graph = build_graph(checkpointer)
    assert graph is not None


def test_compiled_graph_has_all_node_names(compiled_graph) -> None:
    expected = {
        "supervisor", "profiler", "stat_analyst", "viz_agent",
        "insight_critic", "narrator", "evaluator", "output_router",
    }
    actual = set(compiled_graph.get_graph().nodes.keys())
    assert expected.issubset(actual)


def test_get_graph_returns_compiled_graph(checkpointer) -> None:
    g1 = build_graph(checkpointer)
    g2 = get_graph(checkpointer)
    # Both should have the same node structure
    assert (
        set(g1.get_graph().nodes.keys()) == set(g2.get_graph().nodes.keys())
    )


# ---------------------------------------------------------------------------
# route_after_supervisor — dispatch fan-out
# ---------------------------------------------------------------------------

def test_route_after_supervisor_dispatch_returns_send_list(
    dispatch_state: AgentState,
) -> None:
    result = route_after_supervisor(dispatch_state)
    assert isinstance(result, list)
    assert all(isinstance(s, Send) for s in result)


def test_route_after_supervisor_dispatch_sends_all_plan_agents(
    dispatch_state: AgentState,
) -> None:
    result = route_after_supervisor(dispatch_state)
    sent_nodes = [s.node for s in result]
    assert set(sent_nodes) == {"profiler", "stat_analyst", "viz_agent"}


def test_route_after_supervisor_dispatch_skips_completed_agents(
    dispatch_state_with_completed: AgentState,
) -> None:
    result = route_after_supervisor(dispatch_state_with_completed)
    assert isinstance(result, list)
    sent_nodes = [s.node for s in result]
    assert "profiler" not in sent_nodes
    assert "stat_analyst" in sent_nodes
    assert "viz_agent" in sent_nodes


def test_route_after_supervisor_rerun_returns_single_send(
    rerun_state: AgentState,
) -> None:
    result = route_after_supervisor(rerun_state)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].node == "viz_agent"


# ---------------------------------------------------------------------------
# route_after_supervisor — serial routing
# ---------------------------------------------------------------------------

def test_route_after_supervisor_critique(
    base_state: AgentState, mock_plan: AnalysisPlan
) -> None:
    state = {**base_state, "plan": mock_plan, "next_action": "critique",
             "dispatched_agents": [], "completed_agents": [],
             "rerun_agent": None, "rerun_count": 0}
    result = route_after_supervisor(state)
    assert result == "insight_critic"


def test_route_after_supervisor_narrate(
    base_state: AgentState, mock_plan: AnalysisPlan
) -> None:
    state = {**base_state, "plan": mock_plan, "next_action": "narrate",
             "dispatched_agents": [], "completed_agents": [],
             "rerun_agent": None, "rerun_count": 0}
    result = route_after_supervisor(state)
    assert result == "narrator"


def test_route_after_supervisor_complete_returns_end(
    base_state: AgentState, mock_plan: AnalysisPlan
) -> None:
    state = {**base_state, "plan": mock_plan, "next_action": "complete",
             "dispatched_agents": [], "completed_agents": [],
             "rerun_agent": None, "rerun_count": 0}
    result = route_after_supervisor(state)
    assert result == END


def test_route_after_supervisor_unknown_defaults_to_end(
    base_state: AgentState,
) -> None:
    state = {**base_state, "next_action": "bogus_value",
             "dispatched_agents": [], "completed_agents": [],
             "rerun_agent": None, "rerun_count": 0}
    result = route_after_supervisor(state)
    assert result == END


# ---------------------------------------------------------------------------
# route_after_evaluator
# ---------------------------------------------------------------------------

def test_route_after_evaluator_replan_returns_supervisor(
    base_state: AgentState,
) -> None:
    state = {**base_state, "next_action": "replan"}
    result = route_after_evaluator(state)
    assert result == "supervisor"


def test_route_after_evaluator_output_returns_output_router(
    base_state: AgentState,
) -> None:
    state = {**base_state, "next_action": "output"}
    result = route_after_evaluator(state)
    assert result == "output_router"


def test_route_after_evaluator_unknown_defaults_to_output_router(
    base_state: AgentState,
) -> None:
    state = {**base_state, "next_action": "unexpected"}
    result = route_after_evaluator(state)
    assert result == "output_router"


def test_route_after_evaluator_missing_next_action_defaults_to_output_router(
    base_state: AgentState,
) -> None:
    # next_action not set
    result = route_after_evaluator(base_state)
    assert result == "output_router"
