import pandas as pd
import pytest
from pydantic import ValidationError

from core.schema import (
    AgentResult,
    AnalysisPlan,
    CriticOutput,
    EDARequest,
    EvaluationResult,
    NarratorResult,
    OutputType,
    ProfilerFindings,
    REPLOutput,
    REPLRequest,
    StatFindings,
    Verdict,
    VizFindings,
)


@pytest.fixture
def small_df() -> pd.DataFrame:
    return pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})


# ---------------------------------------------------------------------------
# OutputType
# ---------------------------------------------------------------------------

def test_output_type_has_exactly_four_values() -> None:
    assert set(OutputType) == {
        OutputType.REPORT,
        OutputType.EMAIL,
        OutputType.JSON,
        OutputType.DASHBOARD,
    }


def test_output_type_values() -> None:
    assert OutputType.REPORT    == "report"
    assert OutputType.EMAIL     == "email"
    assert OutputType.JSON      == "json"
    assert OutputType.DASHBOARD == "dashboard"


# ---------------------------------------------------------------------------
# EDARequest
# ---------------------------------------------------------------------------

def test_eda_request_valid(small_df: pd.DataFrame) -> None:
    req = EDARequest(
        goal="Find revenue drivers",
        session_id="sess-001",
    )
    assert req.goal == "Find revenue drivers"
    assert req.session_id == "sess-001"
    assert req.output_formats == [OutputType.REPORT, OutputType.JSON]
    assert req.metadata == {}


def test_eda_request_custom_output_formats(small_df: pd.DataFrame) -> None:
    req = EDARequest(
        goal="Summarise data",
        session_id="sess-002",
        output_formats=[OutputType.EMAIL, OutputType.DASHBOARD],
    )
    assert req.output_formats == [OutputType.EMAIL, OutputType.DASHBOARD]


def test_eda_request_metadata(small_df: pd.DataFrame) -> None:
    req = EDARequest(
        goal="Analyse trends",
        session_id="sess-003",
        metadata={"filename": "data.csv", "rows": 3},
    )
    assert req.metadata["filename"] == "data.csv"


def test_eda_request_missing_goal(small_df: pd.DataFrame) -> None:
    with pytest.raises(ValidationError):
        EDARequest(session_id="sess-004")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# AnalysisPlan
# ---------------------------------------------------------------------------

def test_analysis_plan_valid() -> None:
    plan = AnalysisPlan(
        session_id="sess-001",
        goal="Find revenue drivers",
        steps=["Profile schema", "Run correlations", "Generate charts"],
        agents=["profiler", "stat_analyst", "viz_agent"],
        parallel=[["profiler", "stat_analyst", "viz_agent"]],
        output_formats=[OutputType.REPORT, OutputType.JSON],
    )
    assert plan.session_id == "sess-001"
    assert plan.steps == ["Profile schema", "Run correlations", "Generate charts"]
    assert plan.agents == ["profiler", "stat_analyst", "viz_agent"]


def test_analysis_plan_parallel_nested_lists() -> None:
    plan = AnalysisPlan(
        session_id="sess-005",
        goal="Multi-stage analysis",
        steps=["Step 1", "Step 2"],
        agents=["profiler", "stat_analyst", "viz_agent"],
        parallel=[["profiler", "stat_analyst"], ["viz_agent"]],
        output_formats=[OutputType.REPORT],
    )
    assert plan.parallel == [["profiler", "stat_analyst"], ["viz_agent"]]
    assert isinstance(plan.parallel[0], list)
    assert isinstance(plan.parallel[1], list)


# ---------------------------------------------------------------------------
# AgentResult
# ---------------------------------------------------------------------------

def test_agent_result_valid() -> None:
    result = AgentResult(
        agent_name="profiler",
        success=True,
        findings={"null_count": 0, "row_count": 3},
        confidence=0.9,
    )
    assert result.agent_name == "profiler"
    assert result.success is True
    assert result.confidence == 0.9


def test_agent_result_defaults() -> None:
    result = AgentResult(
        agent_name="stat_analyst",
        success=True,
        findings={},
        confidence=0.5,
    )
    assert result.warnings == []
    assert result.rerun_hint is None


def test_agent_result_confidence_too_high() -> None:
    with pytest.raises(ValidationError):
        AgentResult(
            agent_name="profiler",
            success=True,
            findings={},
            confidence=1.1,
        )


def test_agent_result_confidence_too_low() -> None:
    with pytest.raises(ValidationError):
        AgentResult(
            agent_name="profiler",
            success=True,
            findings={},
            confidence=-0.1,
        )


def test_agent_result_confidence_boundary_values() -> None:
    low = AgentResult(agent_name="a", success=True, findings={}, confidence=0.0)
    high = AgentResult(agent_name="b", success=True, findings={}, confidence=1.0)
    assert low.confidence == 0.0
    assert high.confidence == 1.0


def test_agent_result_rerun_hint_set() -> None:
    result = AgentResult(
        agent_name="viz_agent",
        success=False,
        findings={},
        confidence=0.3,
        rerun_hint="Use scatter plot instead of histogram",
    )
    assert result.rerun_hint == "Use scatter plot instead of histogram"


def test_agent_result_warnings() -> None:
    result = AgentResult(
        agent_name="stat_analyst",
        success=True,
        findings={},
        confidence=0.7,
        warnings=["Low sample size", "Skewed distribution"],
    )
    assert len(result.warnings) == 2
    assert "Low sample size" in result.warnings


# ---------------------------------------------------------------------------
# ProfilerFindings
# ---------------------------------------------------------------------------

def test_profiler_findings_valid() -> None:
    pf = ProfilerFindings(
        schema={"columns": ["x", "y"], "shape": [3, 2]},
        null_report={"x": 0, "y": 0},
        distributions={"x": {"mean": 2.0, "std": 1.0}},
    )
    assert pf.schema["shape"] == [3, 2]
    assert pf.notable_columns == []
    assert pf.categorical_summary == {}
    assert pf.data_quality == {}


def test_profiler_findings_missing_schema() -> None:
    with pytest.raises(ValidationError):
        ProfilerFindings(  # type: ignore[call-arg]
            null_report={},
            distributions={},
        )


def test_profiler_findings_with_notable_columns() -> None:
    pf = ProfilerFindings(
        schema={"columns": ["age", "income"]},
        null_report={"income": 2},
        distributions={"age": {"mean": 35.0}},
        notable_columns=["income"],
        categorical_summary={"region": {"top": "N"}},
        data_quality={"duplicates": 0, "constant_columns": []},
    )
    assert pf.notable_columns == ["income"]
    assert pf.categorical_summary["region"]["top"] == "N"


# ---------------------------------------------------------------------------
# StatFindings
# ---------------------------------------------------------------------------

def test_stat_findings_valid() -> None:
    sf = StatFindings(
        correlations={"age": {"income": 0.8}},
        outliers={"income": {"count": 1}},
        normality={"age": {"p_value": 0.12}},
    )
    assert sf.correlations["age"]["income"] == 0.8
    assert sf.notable_findings == []
    assert sf.feature_ranking == []


def test_stat_findings_notable_findings_defaults_empty() -> None:
    sf = StatFindings(
        correlations={},
        outliers={},
        normality={},
    )
    assert sf.notable_findings == []


def test_stat_findings_with_all_fields() -> None:
    sf = StatFindings(
        correlations={"a": {"b": 0.5}},
        outliers={"a": {"iqr_bounds": [1, 10]}},
        normality={"a": {"normal": True}},
        skewness={"a": 0.3},
        feature_ranking=[{"feature": "a", "score": 0.9}],
        notable_findings=["Strong positive correlation between a and b"],
    )
    assert sf.skewness["a"] == 0.3
    assert sf.feature_ranking[0]["feature"] == "a"
    assert len(sf.notable_findings) == 1


# ---------------------------------------------------------------------------
# VizFindings
# ---------------------------------------------------------------------------

def test_viz_findings_valid() -> None:
    vf = VizFindings(
        charts=[{"data": [], "layout": {"title": "histogram"}}],
        chart_descriptions=["Distribution of age"],
        recommended_primary_chart="histogram",
    )
    assert len(vf.charts) == 1
    assert vf.recommended_primary_chart == "histogram"


def test_viz_findings_charts_accepts_list_of_dicts() -> None:
    charts = [
        {"data": [{"type": "histogram"}], "layout": {}},
        {"data": [{"type": "scatter"}],   "layout": {}},
    ]
    vf = VizFindings(charts=charts, recommended_primary_chart="scatter")
    assert len(vf.charts) == 2
    assert vf.chart_descriptions == []


# ---------------------------------------------------------------------------
# CriticOutput
# ---------------------------------------------------------------------------

def _make_result(name: str, confidence: float = 0.8) -> AgentResult:
    return AgentResult(agent_name=name, success=True, findings={}, confidence=confidence)


def test_critic_output_valid() -> None:
    co = CriticOutput(
        scored_results=[_make_result("profiler"), _make_result("stat_analyst")],
        overall_quality=0.85,
    )
    assert len(co.scored_results) == 2
    assert co.rerun_agent is None
    assert co.rerun_reason is None
    assert co.overall_quality == 0.85


def test_critic_output_rerun_agent_defaults_to_none() -> None:
    co = CriticOutput(scored_results=[], overall_quality=0.5)
    assert co.rerun_agent is None


def test_critic_output_overall_quality_too_high() -> None:
    with pytest.raises(ValidationError):
        CriticOutput(scored_results=[], overall_quality=1.1)


def test_critic_output_overall_quality_too_low() -> None:
    with pytest.raises(ValidationError):
        CriticOutput(scored_results=[], overall_quality=-0.1)


def test_critic_output_with_rerun() -> None:
    co = CriticOutput(
        scored_results=[_make_result("viz_agent", confidence=0.3)],
        rerun_agent="viz_agent",
        rerun_reason="Charts lacked labels",
        overall_quality=0.3,
    )
    assert co.rerun_agent == "viz_agent"
    assert co.rerun_reason == "Charts lacked labels"


# ---------------------------------------------------------------------------
# NarratorResult
# ---------------------------------------------------------------------------

def test_narrator_result_valid() -> None:
    nr = NarratorResult(
        narrative="Income is strongly driven by age and region.",
        key_insights=["Age correlates with income (r=0.8)", "Northern region earns most"],
        caveats=["Small sample size of 10"],
        recommended_next_steps=["Collect more data"],
    )
    assert nr.narrative.startswith("Income")
    assert len(nr.key_insights) == 2
    assert len(nr.caveats) == 1
    assert len(nr.recommended_next_steps) == 1


def test_narrator_result_recommended_next_steps_defaults_empty() -> None:
    nr = NarratorResult(narrative="Some analysis.")
    assert nr.recommended_next_steps == []
    assert nr.key_insights == []
    assert nr.caveats == []


def test_narrator_result_key_insights_non_empty_when_populated() -> None:
    nr = NarratorResult(
        narrative="Analysis complete.",
        key_insights=["Insight 1", "Insight 2", "Insight 3"],
    )
    assert len(nr.key_insights) == 3
    assert all(isinstance(i, str) for i in nr.key_insights)


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def test_verdict_has_exactly_three_values() -> None:
    assert set(Verdict) == {Verdict.STRONG, Verdict.ADEQUATE, Verdict.WEAK}


def test_verdict_values() -> None:
    assert Verdict.STRONG   == "strong"
    assert Verdict.ADEQUATE == "adequate"
    assert Verdict.WEAK     == "weak"


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------

def test_evaluation_result_valid() -> None:
    er = EvaluationResult(
        goal_coverage=0.9,
        insight_quality=0.8,
        evidence_quality=0.85,
        overall_score=0.865,
        strengths=["Clear correlation analysis"],
        gaps=["Missing time-series breakdown"],
        verdict=Verdict.STRONG,
        retry_instructions=None,
    )
    assert er.goal_coverage == 0.9
    assert er.verdict == Verdict.STRONG
    assert er.retry_instructions is None


def test_evaluation_result_retry_instructions_defaults_none() -> None:
    er = EvaluationResult(
        goal_coverage=0.5,
        insight_quality=0.5,
        evidence_quality=0.5,
        overall_score=0.5,
        verdict=Verdict.ADEQUATE,
    )
    assert er.retry_instructions is None
    assert er.strengths == []
    assert er.gaps == []


def test_evaluation_result_strengths_and_gaps_default_empty() -> None:
    er = EvaluationResult(
        goal_coverage=0.3,
        insight_quality=0.2,
        evidence_quality=0.4,
        overall_score=0.29,
        verdict=Verdict.WEAK,
    )
    assert er.strengths == []
    assert er.gaps == []


def test_evaluation_result_goal_coverage_too_high() -> None:
    with pytest.raises(ValidationError):
        EvaluationResult(
            goal_coverage=1.1,
            insight_quality=0.5,
            evidence_quality=0.5,
            overall_score=0.5,
            verdict=Verdict.ADEQUATE,
        )


def test_evaluation_result_goal_coverage_too_low() -> None:
    with pytest.raises(ValidationError):
        EvaluationResult(
            goal_coverage=-0.1,
            insight_quality=0.5,
            evidence_quality=0.5,
            overall_score=0.5,
            verdict=Verdict.ADEQUATE,
        )


def test_evaluation_result_overall_score_too_high() -> None:
    with pytest.raises(ValidationError):
        EvaluationResult(
            goal_coverage=0.5,
            insight_quality=0.5,
            evidence_quality=0.5,
            overall_score=1.5,
            verdict=Verdict.ADEQUATE,
        )


def test_evaluation_result_overall_score_too_low() -> None:
    with pytest.raises(ValidationError):
        EvaluationResult(
            goal_coverage=0.5,
            insight_quality=0.5,
            evidence_quality=0.5,
            overall_score=-0.1,
            verdict=Verdict.ADEQUATE,
        )


def test_evaluation_result_weak_with_retry_instructions() -> None:
    er = EvaluationResult(
        goal_coverage=0.2,
        insight_quality=0.3,
        evidence_quality=0.1,
        overall_score=0.22,
        verdict=Verdict.WEAK,
        retry_instructions="Focus on the revenue drivers and include segment breakdowns.",
    )
    assert er.verdict == Verdict.WEAK
    assert er.retry_instructions is not None
    assert "revenue" in er.retry_instructions


# ---------------------------------------------------------------------------
# REPLRequest
# ---------------------------------------------------------------------------

def test_repl_request_valid() -> None:
    req = REPLRequest(
        code="print(df.shape)",
        description="Print the shape of the dataframe.",
    )
    assert req.code == "print(df.shape)"
    assert req.description == "Print the shape of the dataframe."


# ---------------------------------------------------------------------------
# REPLOutput
# ---------------------------------------------------------------------------

def test_repl_output_success() -> None:
    out = REPLOutput(
        success=True,
        result={"rows": 100, "cols": 5},
        stdout_raw='{"rows": 100, "cols": 5}',
    )
    assert out.success is True
    assert out.result == {"rows": 100, "cols": 5}
    assert out.error is None


def test_repl_output_failure_has_error() -> None:
    out = REPLOutput(
        success=False,
        result={"raw": ""},
        stdout_raw="",
        error="NameError: name 'df' is not defined",
    )
    assert out.success is False
    assert out.error is not None
    assert "NameError" in out.error


def test_repl_output_result_is_always_dict() -> None:
    out_json = REPLOutput(
        success=True,
        result={"mean": 42.0},
        stdout_raw='{"mean": 42.0}',
    )
    out_raw = REPLOutput(
        success=True,
        result={"raw": "some plain text"},
        stdout_raw="some plain text",
    )
    assert isinstance(out_json.result, dict)
    assert isinstance(out_raw.result, dict)
