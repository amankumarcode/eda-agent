"""Tests for outputs/email_drafter.py."""
import os
import smtplib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from core.schema import AgentResult, EDARequest, EvaluationResult, OutputType, Verdict
from core.state import AgentState
from outputs.email_drafter import (
    _build_body,
    _build_subject,
    _compute_mean_confidence,
    _smtp_configured,
    draft_email,
    send_email,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age":    [25, 32, 47, 51, 23],
        "income": [40000, 58000, 92000, 105000, 36000],
    })


def _make_agent_result(name: str = "profiler", success: bool = True, confidence: float = 0.8) -> AgentResult:
    return AgentResult(
        agent_name=name,
        success=success,
        findings={},
        confidence=confidence,
    )


def _make_evaluation(verdict: Verdict = Verdict.STRONG, score: float = 0.9) -> EvaluationResult:
    return EvaluationResult(
        goal_coverage=0.9,
        insight_quality=0.85,
        evidence_quality=0.8,
        overall_score=score,
        verdict=verdict,
    )


def _base_state(sample_df: pd.DataFrame, **overrides) -> AgentState:
    request = EDARequest(
        goal="Find key drivers of income",
        session_id="email-test-001",
        output_formats=[OutputType.EMAIL],
    )
    state: AgentState = {
        "request": request,
        "plan": None,
        "plan_approved": False,
        "agent_results": [_make_agent_result()],
        "rerun_agent": None,
        "rerun_count": 0,
        "narrative": "Age and income are positively correlated in this dataset.",
        "key_insights": ["Age correlates with income.", "Region affects earnings."],
        "caveats": ["Small sample size."],
        "output_paths": {},
        "messages": [],
        "evaluation": None,
        "evaluation_count": 0,
        "dispatched_agents": [],
        "completed_agents": [],
        "next_action": None,
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# _compute_mean_confidence
# ---------------------------------------------------------------------------

def test_compute_mean_confidence_returns_zero_for_empty():
    assert _compute_mean_confidence([]) == 0.0


def test_compute_mean_confidence_skips_failed_agents():
    results = [
        _make_agent_result("a", success=True, confidence=0.8),
        _make_agent_result("b", success=False, confidence=0.2),
    ]
    assert _compute_mean_confidence(results) == pytest.approx(0.8)


def test_compute_mean_confidence_averages_successful():
    results = [
        _make_agent_result("a", success=True, confidence=0.6),
        _make_agent_result("b", success=True, confidence=1.0),
    ]
    assert _compute_mean_confidence(results) == pytest.approx(0.8)


def test_compute_mean_confidence_all_failed_returns_zero():
    results = [_make_agent_result("a", success=False, confidence=0.5)]
    assert _compute_mean_confidence(results) == 0.0


# ---------------------------------------------------------------------------
# _smtp_configured
# ---------------------------------------------------------------------------

def test_smtp_configured_returns_false_when_no_env_vars():
    env_without_smtp = {
        k: v for k, v in os.environ.items()
        if k not in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM")
    }
    with patch.dict(os.environ, env_without_smtp, clear=True):
        assert _smtp_configured() is False


def test_smtp_configured_returns_true_when_all_vars_set():
    smtp_env = {
        "SMTP_HOST": "smtp.example.com",
        "SMTP_USER": "user@example.com",
        "SMTP_PASSWORD": "secret123",
        "SMTP_FROM": "noreply@example.com",
    }
    with patch.dict(os.environ, smtp_env):
        assert _smtp_configured() is True


def test_smtp_configured_returns_false_when_one_var_missing():
    env_without = {k: v for k, v in os.environ.items() if k != "SMTP_FROM"}
    smtp_partial = {
        "SMTP_HOST": "smtp.example.com",
        "SMTP_USER": "user@example.com",
        "SMTP_PASSWORD": "secret123",
    }
    with patch.dict(env_without, smtp_partial, clear=True):
        assert _smtp_configured() is False


def test_smtp_configured_returns_false_when_var_is_empty():
    smtp_env = {
        "SMTP_HOST": "smtp.example.com",
        "SMTP_USER": "user@example.com",
        "SMTP_PASSWORD": "",
        "SMTP_FROM": "noreply@example.com",
    }
    with patch.dict(os.environ, smtp_env):
        assert _smtp_configured() is False


# ---------------------------------------------------------------------------
# _build_subject
# ---------------------------------------------------------------------------

def test_build_subject_includes_goal(sample_df):
    state = _base_state(sample_df)
    subject = _build_subject(state)
    assert "Find key drivers of income" in subject


def test_build_subject_includes_verdict_when_evaluation_present(sample_df):
    ev = _make_evaluation(Verdict.STRONG)
    state = _base_state(sample_df, evaluation=ev)
    subject = _build_subject(state)
    assert "strong" in subject.lower()


def test_build_subject_includes_confidence_when_no_evaluation(sample_df):
    state = _base_state(sample_df, evaluation=None)
    subject = _build_subject(state)
    assert "confidence" in subject.lower()


def test_build_subject_truncates_long_goal(sample_df):
    long_goal = "A" * 80
    request = EDARequest(
        goal=long_goal,
        session_id="s1",
        output_formats=[OutputType.EMAIL],
    )
    state = _base_state(sample_df)
    state["request"] = request
    subject = _build_subject(state)
    assert "..." in subject
    assert len(subject) < len(f"EDA Report: {long_goal} [x]")


def test_build_subject_starts_with_eda_report(sample_df):
    state = _base_state(sample_df)
    subject = _build_subject(state)
    assert subject.startswith("EDA Report:")


def test_build_subject_adequate_verdict(sample_df):
    ev = _make_evaluation(Verdict.ADEQUATE, score=0.7)
    state = _base_state(sample_df, evaluation=ev)
    subject = _build_subject(state)
    assert "adequate" in subject.lower()


# ---------------------------------------------------------------------------
# _build_body
# ---------------------------------------------------------------------------

def test_build_body_contains_goal(sample_df):
    state = _base_state(sample_df)
    body = _build_body(state)
    assert "Find key drivers of income" in body


def test_build_body_contains_narrative(sample_df):
    state = _base_state(sample_df)
    body = _build_body(state)
    assert "Age and income are positively correlated" in body


def test_build_body_truncates_long_narrative(sample_df):
    long_narrative = "X" * 600
    state = _base_state(sample_df, narrative=long_narrative)
    body = _build_body(state)
    assert long_narrative not in body
    assert "..." in body


def test_build_body_contains_key_insights(sample_df):
    state = _base_state(sample_df)
    body = _build_body(state)
    assert "Age correlates with income." in body


def test_build_body_contains_none_identified_when_caveats_empty(sample_df):
    state = _base_state(sample_df, caveats=[])
    body = _build_body(state)
    assert "None identified." in body


def test_build_body_contains_caveats_when_present(sample_df):
    state = _base_state(sample_df, caveats=["Small sample.", "Possible bias."])
    body = _build_body(state)
    assert "Small sample." in body


def test_build_body_contains_verdict_when_evaluation_present(sample_df):
    ev = _make_evaluation(Verdict.STRONG, score=0.9)
    state = _base_state(sample_df, evaluation=ev)
    body = _build_body(state)
    assert "strong" in body.lower()


def test_build_body_contains_confidence_when_no_evaluation(sample_df):
    state = _base_state(sample_df, evaluation=None)
    body = _build_body(state)
    assert "Confidence:" in body


def test_build_body_under_2000_chars(sample_df):
    state = _base_state(sample_df)
    body = _build_body(state)
    assert len(body) <= 2000


def test_build_body_contains_session_id(sample_df):
    state = _base_state(sample_df)
    body = _build_body(state)
    assert "email-test-001" in body


def test_build_body_report_path_not_generated_when_no_path(sample_df):
    state = _base_state(sample_df, output_paths={})
    body = _build_body(state)
    assert "not generated" in body


def test_build_body_report_path_shown_when_available(sample_df):
    state = _base_state(sample_df, output_paths={"report": "/output/s1/report.html"})
    body = _build_body(state)
    assert "/output/s1/report.html" in body


# ---------------------------------------------------------------------------
# draft_email
# ---------------------------------------------------------------------------

def test_draft_email_returns_path_string(sample_df, tmp_path):
    state = _base_state(sample_df)
    with patch.dict(os.environ, {"OUTPUT_DIR": str(tmp_path)}):
        result = draft_email(state)
    assert isinstance(result, str)


def test_draft_email_file_ends_with_txt(sample_df, tmp_path):
    state = _base_state(sample_df)
    with patch.dict(os.environ, {"OUTPUT_DIR": str(tmp_path)}):
        result = draft_email(state)
    assert result.endswith(".txt")


def test_draft_email_file_exists(sample_df, tmp_path):
    state = _base_state(sample_df)
    with patch.dict(os.environ, {"OUTPUT_DIR": str(tmp_path)}):
        result = draft_email(state)
    assert Path(result).exists()


def test_draft_email_content_starts_with_subject(sample_df, tmp_path):
    state = _base_state(sample_df)
    with patch.dict(os.environ, {"OUTPUT_DIR": str(tmp_path)}):
        result = draft_email(state)
    content = Path(result).read_text()
    assert content.startswith("Subject:")


def test_draft_email_content_contains_goal(sample_df, tmp_path):
    state = _base_state(sample_df)
    with patch.dict(os.environ, {"OUTPUT_DIR": str(tmp_path)}):
        result = draft_email(state)
    content = Path(result).read_text()
    assert "Find key drivers of income" in content


def test_draft_email_content_contains_key_insight(sample_df, tmp_path):
    state = _base_state(sample_df)
    with patch.dict(os.environ, {"OUTPUT_DIR": str(tmp_path)}):
        result = draft_email(state)
    content = Path(result).read_text()
    assert "Age correlates with income." in content


def test_draft_email_content_contains_narrative(sample_df, tmp_path):
    state = _base_state(sample_df)
    with patch.dict(os.environ, {"OUTPUT_DIR": str(tmp_path)}):
        result = draft_email(state)
    content = Path(result).read_text()
    assert "Age and income are positively correlated" in content


def test_draft_email_reraises_on_write_failure(sample_df):
    state = _base_state(sample_df)
    with patch("outputs.email_drafter.Path.write_text", side_effect=OSError("disk full")):
        with pytest.raises(RuntimeError, match="draft_email failed"):
            draft_email(state)


# ---------------------------------------------------------------------------
# send_email
# ---------------------------------------------------------------------------

def test_send_email_returns_false_when_not_configured(sample_df, tmp_path):
    state = _base_state(sample_df)
    with patch.dict(os.environ, {"OUTPUT_DIR": str(tmp_path)}):
        draft_path = draft_email(state)

    env_without_smtp = {
        k: v for k, v in os.environ.items()
        if k not in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM")
    }
    with patch.dict(os.environ, env_without_smtp, clear=True):
        with patch("smtplib.SMTP") as mock_smtp_cls:
            result = send_email(draft_path, "test@example.com")
    assert result is False
    mock_smtp_cls.assert_not_called()


def test_send_email_returns_true_on_successful_send(sample_df, tmp_path):
    state = _base_state(sample_df)
    with patch.dict(os.environ, {"OUTPUT_DIR": str(tmp_path)}):
        draft_path = draft_email(state)

    mock_server = MagicMock()
    mock_smtp_cls = MagicMock()
    mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
    mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

    with patch("smtplib.SMTP", mock_smtp_cls):
        with patch("outputs.email_drafter._smtp_configured", return_value=True):
            result = send_email(draft_path, "to@example.com")

    assert result is True
    mock_server.starttls.assert_called_once()
    mock_server.login.assert_called_once()
    mock_server.sendmail.assert_called_once()


def test_send_email_returns_false_on_smtp_exception(sample_df, tmp_path):
    state = _base_state(sample_df)
    with patch.dict(os.environ, {"OUTPUT_DIR": str(tmp_path)}):
        draft_path = draft_email(state)

    with patch("outputs.email_drafter._smtp_configured", return_value=True):
        with patch("smtplib.SMTP", side_effect=smtplib.SMTPException("connection refused")):
            result = send_email(draft_path, "to@example.com")

    assert result is False
