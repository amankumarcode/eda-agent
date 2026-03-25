import operator

import pandas as pd
import pytest

from core.schema import AgentResult, AnalysisPlan, EDARequest, EvaluationResult, OutputType, Verdict
from core.state import AgentState


@pytest.fixture
def small_df() -> pd.DataFrame:
    return pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})


@pytest.fixture
def sample_request(small_df: pd.DataFrame) -> EDARequest:
    return EDARequest(
        goal="Find patterns",
        session_id="test-session-001",
        output_formats=[OutputType.REPORT, OutputType.JSON],
    )


def make_result(name: str) -> AgentResult:
    return AgentResult(
        agent_name=name,
        success=True,
        findings={"key": "value"},
        confidence=0.9,
    )


def make_evaluation(verdict: Verdict = Verdict.STRONG) -> EvaluationResult:
    return EvaluationResult(
        goal_coverage=0.9,
        insight_quality=0.85,
        evidence_quality=0.8,
        overall_score=0.875,
        verdict=verdict,
    )


def _base_state(request: EDARequest, **overrides) -> AgentState:
    """Helper: construct a complete AgentState with sensible defaults."""
    defaults: AgentState = AgentState(
        request=request,
        plan=None,
        plan_approved=False,
        agent_results=[],
        rerun_agent=None,
        rerun_count=0,
        narrative=None,
        key_insights=[],
        caveats=[],
        output_paths={},
        messages=[],
        evaluation=None,
        evaluation_count=0,
        dispatched_agents=[],
        completed_agents=[],
        next_action=None,
    )
    defaults.update(overrides)  # type: ignore[attr-defined]
    return defaults


def test_agent_state_construction(sample_request: EDARequest) -> None:
    state = _base_state(sample_request)
    assert state["request"] is sample_request
    assert state["plan"] is None
    assert state["plan_approved"] is False
    assert state["agent_results"] == []
    assert state["rerun_agent"] is None
    assert state["rerun_count"] == 0
    assert state["narrative"] is None
    assert state["key_insights"] == []
    assert state["caveats"] == []
    assert state["output_paths"] == {}
    assert state["messages"] == []
    assert state["evaluation"] is None
    assert state["evaluation_count"] == 0


def test_agent_results_appends_via_operator_add() -> None:
    r1 = make_result("profiler")
    r2 = make_result("stat_analyst")
    merged = operator.add([r1], [r2])
    assert len(merged) == 2
    assert merged[0].agent_name == "profiler"
    assert merged[1].agent_name == "stat_analyst"


def test_messages_appends_via_operator_add() -> None:
    batch1 = ["plan generated"]
    batch2 = ["plan approved"]
    merged = operator.add(batch1, batch2)
    assert merged == ["plan generated", "plan approved"]


def test_three_parallel_results_merge() -> None:
    r_profiler = make_result("profiler")
    r_stat = make_result("stat_analyst")
    r_viz = make_result("viz_agent")
    combined = operator.add(operator.add([r_profiler], [r_stat]), [r_viz])
    assert len(combined) == 3
    names = [r.agent_name for r in combined]
    assert names == ["profiler", "stat_analyst", "viz_agent"]


def test_plan_defaults_to_none(sample_request: EDARequest) -> None:
    state = _base_state(sample_request)
    assert state["plan"] is None


def test_rerun_count_starts_at_zero(sample_request: EDARequest) -> None:
    state = _base_state(sample_request)
    assert state["rerun_count"] == 0


def test_schema_types_usable_in_state(small_df: pd.DataFrame) -> None:
    request = EDARequest(
        goal="test",
        session_id="s1",
        output_formats=[OutputType.DASHBOARD],
    )
    plan = AnalysisPlan(
        session_id="s1",
        goal="test",
        steps=["step1"],
        agents=["profiler"],
        parallel=[["profiler"]],
        output_formats=[OutputType.DASHBOARD],
    )
    result = AgentResult(
        agent_name="profiler",
        success=True,
        findings={"shape": (3, 2)},
        confidence=0.75,
    )
    state = _base_state(
        request,
        plan=plan,
        plan_approved=True,
        agent_results=[result],
        narrative="Some narrative",
        key_insights=["insight1"],
        caveats=["caveat1"],
        output_paths={"report": "./output/s1/report.html"},
        messages=["approved"],
    )
    assert state["plan"].session_id == "s1"
    assert state["agent_results"][0].agent_name == "profiler"
    assert state["output_paths"]["report"] == "./output/s1/report.html"


# ---------------------------------------------------------------------------
# evaluation + evaluation_count
# ---------------------------------------------------------------------------

def test_evaluation_defaults_to_none(sample_request: EDARequest) -> None:
    state = _base_state(sample_request)
    assert state["evaluation"] is None


def test_evaluation_count_starts_at_zero(sample_request: EDARequest) -> None:
    state = _base_state(sample_request)
    assert state["evaluation_count"] == 0


def test_state_with_evaluation_result(sample_request: EDARequest) -> None:
    ev = make_evaluation(Verdict.STRONG)
    state = _base_state(sample_request, evaluation=ev, evaluation_count=1)
    assert state["evaluation"] is ev
    assert state["evaluation"].verdict == Verdict.STRONG
    assert state["evaluation_count"] == 1


def test_evaluator_update_increments_count(sample_request: EDARequest) -> None:
    """Simulate a LangGraph partial-dict merge from the evaluator node."""
    state = _base_state(sample_request)
    ev = make_evaluation(Verdict.ADEQUATE)
    # LangGraph merges by updating the state dict with the node's return value
    update = {"evaluation": ev, "evaluation_count": state["evaluation_count"] + 1}
    state.update(update)  # type: ignore[attr-defined]
    assert state["evaluation_count"] == 1
    assert state["evaluation"].verdict == Verdict.ADEQUATE


def test_second_evaluator_run_increments_to_two(sample_request: EDARequest) -> None:
    """Guard condition: evaluation_count >= 2 forces exit regardless of verdict."""
    state = _base_state(sample_request, evaluation=make_evaluation(Verdict.WEAK), evaluation_count=1)
    update = {"evaluation": make_evaluation(Verdict.WEAK), "evaluation_count": state["evaluation_count"] + 1}
    state.update(update)  # type: ignore[attr-defined]
    assert state["evaluation_count"] == 2
    assert state["evaluation"].verdict == Verdict.WEAK


# ---------------------------------------------------------------------------
# dispatched_agents + completed_agents + next_action
# ---------------------------------------------------------------------------

def test_dispatched_agents_defaults_to_empty_list(sample_request: EDARequest) -> None:
    state = _base_state(sample_request)
    assert state["dispatched_agents"] == []


def test_completed_agents_defaults_to_empty_list(sample_request: EDARequest) -> None:
    state = _base_state(sample_request)
    assert state["completed_agents"] == []


def test_next_action_defaults_to_none(sample_request: EDARequest) -> None:
    state = _base_state(sample_request)
    assert state["next_action"] is None


def test_dispatched_agents_appends_across_two_updates(sample_request: EDARequest) -> None:
    """Simulate supervisor dispatching profiler then stat_analyst on separate turns."""
    state = _base_state(sample_request)
    state.update({"dispatched_agents": operator.add(state["dispatched_agents"], ["profiler"])})  # type: ignore[attr-defined]
    state.update({"dispatched_agents": operator.add(state["dispatched_agents"], ["stat_analyst"])})  # type: ignore[attr-defined]
    assert state["dispatched_agents"] == ["profiler", "stat_analyst"]


def test_completed_agents_appends_across_two_updates(sample_request: EDARequest) -> None:
    """Simulate profiler then stat_analyst reporting back."""
    state = _base_state(sample_request)
    state.update({"completed_agents": operator.add(state["completed_agents"], ["profiler"])})  # type: ignore[attr-defined]
    state.update({"completed_agents": operator.add(state["completed_agents"], ["stat_analyst"])})  # type: ignore[attr-defined]
    assert state["completed_agents"] == ["profiler", "stat_analyst"]


def test_next_action_accepts_all_valid_values(sample_request: EDARequest) -> None:
    valid_actions = ["dispatch", "critique", "narrate", "evaluate", "output", "complete", "replan"]
    for action in valid_actions:
        state = _base_state(sample_request, next_action=action)
        assert state["next_action"] == action


def test_next_action_can_be_none(sample_request: EDARequest) -> None:
    state = _base_state(sample_request, next_action=None)
    assert state["next_action"] is None
