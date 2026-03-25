import json
from pathlib import Path
from unittest.mock import patch

import pytest

from core.schema import (
    AgentResult,
    AnalysisPlan,
    EvaluationResult,
    OutputType,
    Verdict,
)
from core.state import AgentState
from outputs.json_summary import (
    _compute_mean_confidence,
    _serialize_evaluation,
    generate_json,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def full_state(base_state: AgentState) -> AgentState:
    return {
        **base_state,
        "plan": AnalysisPlan(
            session_id="test-session-001",
            goal="Find key drivers of income",
            steps=["Profile", "Analyse", "Visualise"],
            agents=["profiler", "stat_analyst", "viz_agent"],
            parallel=[["profiler", "stat_analyst", "viz_agent"]],
            output_formats=[OutputType.REPORT, OutputType.JSON],
        ),
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
                findings={"notable_findings": ["income correlated"]},
                confidence=0.85,
            ),
        ],
        "narrative": "Income is driven by age.",
        "key_insights": ["Age explains 85% of variance"],
        "caveats": ["Small sample size"],
        "evaluation": EvaluationResult(
            goal_coverage=0.9,
            insight_quality=0.85,
            evidence_quality=0.9,
            overall_score=0.89,
            strengths=["Strong goal alignment"],
            gaps=[],
            verdict=Verdict.STRONG,
            retry_instructions=None,
        ),
        "output_paths": {},
    }


# ---------------------------------------------------------------------------
# _compute_mean_confidence
# ---------------------------------------------------------------------------

def test_compute_mean_confidence_empty_list() -> None:
    assert _compute_mean_confidence([]) == 0.0


def test_compute_mean_confidence_skips_failed_agents() -> None:
    results = [
        AgentResult(agent_name="profiler", success=True,
                    findings={}, confidence=0.9),
        AgentResult(agent_name="stat_analyst", success=False,
                    findings={}, confidence=0.0),
    ]
    assert _compute_mean_confidence(results) == pytest.approx(0.9)


def test_compute_mean_confidence_all_failed_returns_zero() -> None:
    results = [
        AgentResult(agent_name="profiler", success=False,
                    findings={}, confidence=0.0),
    ]
    assert _compute_mean_confidence(results) == 0.0


def test_compute_mean_confidence_two_agents() -> None:
    results = [
        AgentResult(agent_name="profiler", success=True,
                    findings={}, confidence=0.9),
        AgentResult(agent_name="stat_analyst", success=True,
                    findings={}, confidence=0.85),
    ]
    assert _compute_mean_confidence(results) == pytest.approx(0.875)


# ---------------------------------------------------------------------------
# _serialize_evaluation
# ---------------------------------------------------------------------------

def test_serialize_evaluation_returns_none_when_none() -> None:
    assert _serialize_evaluation(None) is None


def test_serialize_evaluation_returns_dict() -> None:
    ev = EvaluationResult(
        goal_coverage=0.9,
        insight_quality=0.85,
        evidence_quality=0.9,
        overall_score=0.89,
        strengths=[],
        gaps=[],
        verdict=Verdict.STRONG,
        retry_instructions=None,
    )
    result = _serialize_evaluation(ev)
    assert isinstance(result, dict)


def test_serialize_evaluation_verdict_is_string() -> None:
    ev = EvaluationResult(
        goal_coverage=0.9,
        insight_quality=0.85,
        evidence_quality=0.9,
        overall_score=0.89,
        strengths=[],
        gaps=[],
        verdict=Verdict.STRONG,
        retry_instructions=None,
    )
    result = _serialize_evaluation(ev)
    assert isinstance(result["verdict"], str)
    assert result["verdict"] == "strong"


def test_serialize_evaluation_contains_overall_score() -> None:
    ev = EvaluationResult(
        goal_coverage=0.9,
        insight_quality=0.85,
        evidence_quality=0.9,
        overall_score=0.89,
        strengths=[],
        gaps=[],
        verdict=Verdict.STRONG,
        retry_instructions=None,
    )
    result = _serialize_evaluation(ev)
    assert result["overall_score"] == pytest.approx(0.89)


# ---------------------------------------------------------------------------
# generate_json — file output
# ---------------------------------------------------------------------------

def test_generate_json_returns_string_path(
    full_state: AgentState, tmp_path: Path
) -> None:
    with patch("outputs.json_summary.os.getenv", return_value=str(tmp_path)):
        result = generate_json(full_state)

    assert isinstance(result, str)
    assert result.endswith(".json")


def test_generate_json_file_exists(
    full_state: AgentState, tmp_path: Path
) -> None:
    with patch("outputs.json_summary.os.getenv", return_value=str(tmp_path)):
        path = generate_json(full_state)

    assert Path(path).exists()


def test_generate_json_valid_json(
    full_state: AgentState, tmp_path: Path
) -> None:
    with patch("outputs.json_summary.os.getenv", return_value=str(tmp_path)):
        path = generate_json(full_state)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)


def test_generate_json_contains_metadata(
    full_state: AgentState, tmp_path: Path
) -> None:
    with patch("outputs.json_summary.os.getenv", return_value=str(tmp_path)):
        path = generate_json(full_state)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    assert "metadata" in data
    assert data["metadata"]["session_id"] == "test-session-001"
    assert data["metadata"]["goal"] == "Find key drivers of income"


def test_generate_json_contains_narrative(
    full_state: AgentState, tmp_path: Path
) -> None:
    with patch("outputs.json_summary.os.getenv", return_value=str(tmp_path)):
        path = generate_json(full_state)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    assert "narrative" in data
    assert isinstance(data["narrative"]["key_insights"], list)
    assert len(data["narrative"]["key_insights"]) > 0


def test_generate_json_contains_analysis(
    full_state: AgentState, tmp_path: Path
) -> None:
    with patch("outputs.json_summary.os.getenv", return_value=str(tmp_path)):
        path = generate_json(full_state)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    assert "analysis" in data
    assert isinstance(data["analysis"]["agent_results"], list)


def test_generate_json_contains_evaluation(
    full_state: AgentState, tmp_path: Path
) -> None:
    with patch("outputs.json_summary.os.getenv", return_value=str(tmp_path)):
        path = generate_json(full_state)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    assert "evaluation" in data
    assert data["evaluation"]["overall_score"] == pytest.approx(0.89)


def test_generate_json_metadata_dataset_shape(
    full_state: AgentState, tmp_path: Path, sample_df
) -> None:
    with patch("outputs.json_summary.os.getenv", return_value=str(tmp_path)):
        path = generate_json(full_state)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["metadata"]["dataset_shape"] == [10, 4]


def test_generate_json_overall_confidence(
    full_state: AgentState, tmp_path: Path
) -> None:
    with patch("outputs.json_summary.os.getenv", return_value=str(tmp_path)):
        path = generate_json(full_state)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["analysis"]["overall_confidence"] == pytest.approx(0.875)


def test_generate_json_creates_output_directory(
    full_state: AgentState, tmp_path: Path
) -> None:
    nested = tmp_path / "new_dir"
    with patch("outputs.json_summary.os.getenv", return_value=str(nested)):
        path = generate_json(full_state)

    assert Path(path).parent.exists()


def test_generate_json_filename_contains_session_id(
    full_state: AgentState, tmp_path: Path
) -> None:
    with patch("outputs.json_summary.os.getenv", return_value=str(tmp_path)):
        path = generate_json(full_state)

    assert "test-session-001" in Path(path).name


def test_generate_json_raises_on_write_failure(
    full_state: AgentState, tmp_path: Path
) -> None:
    with (
        patch("outputs.json_summary.os.getenv", return_value=str(tmp_path)),
        patch("builtins.open", side_effect=PermissionError("no write access")),
    ):
        with pytest.raises(RuntimeError, match="generate_json failed"):
            generate_json(full_state)
