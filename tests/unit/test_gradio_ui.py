"""Tests for adapters/gradio_ui.py."""
from unittest.mock import MagicMock, patch

import gradio as gr
import pytest

from adapters.gradio_ui import (
    _answer_question,
    _build_qa_context,
    _extract_progress_message,
    _format_outputs_as_markdown,
    _format_plan_as_markdown,
    _is_pipeline_complete,
    build_gradio_app,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plan(goal="Find revenue drivers", steps=None, agents=None):
    return {
        "goal": goal,
        "steps": steps or ["Profile dataset", "Run stat analysis"],
        "agents": agents or ["profiler", "stat_analyst"],
        "parallel": [["profiler", "stat_analyst"]],
        "output_formats": ["report", "json"],
    }


# ---------------------------------------------------------------------------
# _format_plan_as_markdown
# ---------------------------------------------------------------------------

def test_format_plan_contains_analysis_plan_heading():
    md = _format_plan_as_markdown(_make_plan())
    assert "Analysis Plan" in md


def test_format_plan_contains_goal():
    md = _format_plan_as_markdown(_make_plan(goal="Find revenue drivers"))
    assert "Find revenue drivers" in md


def test_format_plan_contains_numbered_steps():
    md = _format_plan_as_markdown(_make_plan(steps=["Step A", "Step B"]))
    assert "1. Step A" in md
    assert "2. Step B" in md


def test_format_plan_contains_approve_instruction():
    md = _format_plan_as_markdown(_make_plan())
    assert "approve" in md.lower()


def test_format_plan_contains_agents():
    md = _format_plan_as_markdown(_make_plan(agents=["profiler", "viz_agent"]))
    assert "profiler" in md
    assert "viz_agent" in md


def test_format_plan_includes_message_when_provided():
    md = _format_plan_as_markdown(_make_plan(), message="Please review carefully.")
    assert "Please review carefully." in md


def test_format_plan_returns_string():
    assert isinstance(_format_plan_as_markdown(_make_plan()), str)


# ---------------------------------------------------------------------------
# _format_outputs_as_markdown
# ---------------------------------------------------------------------------

def test_format_outputs_contains_analysis_complete():
    md = _format_outputs_as_markdown({"report": "/out/report.html"})
    assert "Analysis Complete" in md


def test_format_outputs_contains_each_key():
    md = _format_outputs_as_markdown({
        "report": "/out/report.html",
        "json": "/out/summary.json",
    })
    assert "report" in md
    assert "json" in md


def test_format_outputs_contains_path_values():
    md = _format_outputs_as_markdown({"report": "/out/s1/report.html"})
    assert "/out/s1/report.html" in md


def test_format_outputs_contains_download_instruction():
    md = _format_outputs_as_markdown({"json": "/out/summary.json"})
    assert "Download" in md


def test_format_outputs_returns_string():
    assert isinstance(_format_outputs_as_markdown({}), str)


# ---------------------------------------------------------------------------
# _extract_progress_message
# ---------------------------------------------------------------------------

def test_extract_progress_returns_none_when_no_messages_key():
    result = _extract_progress_message("profiler", {"agent_results": []})
    assert result is None


def test_extract_progress_returns_none_when_messages_empty():
    result = _extract_progress_message("profiler", {"messages": []})
    assert result is None


def test_extract_progress_returns_formatted_string():
    msg = MagicMock()
    msg.content = "Profiler complete."
    result = _extract_progress_message("profiler", {"messages": [msg]})
    assert result is not None
    assert "profiler" in result
    assert "Profiler complete." in result


def test_extract_progress_truncates_long_content():
    msg = MagicMock()
    msg.content = "X" * 300
    result = _extract_progress_message("profiler", {"messages": [msg]})
    assert result is not None
    assert "..." in result
    assert "X" * 300 not in result


def test_extract_progress_uses_last_message():
    msg1 = MagicMock()
    msg1.content = "First message."
    msg2 = MagicMock()
    msg2.content = "Last message."
    result = _extract_progress_message("node", {"messages": [msg1, msg2]})
    assert "Last message." in result
    assert "First message." not in result


def test_extract_progress_returns_none_for_non_dict_output():
    result = _extract_progress_message("node", "not a dict")
    assert result is None


# ---------------------------------------------------------------------------
# build_gradio_app
# ---------------------------------------------------------------------------

def test_build_gradio_app_returns_blocks_instance():
    mock_graph = MagicMock()
    mock_checkpointer = MagicMock()
    app = build_gradio_app(mock_graph, mock_checkpointer)
    assert isinstance(app, gr.Blocks)


# ---------------------------------------------------------------------------
# _is_pipeline_complete
# ---------------------------------------------------------------------------

def test_is_pipeline_complete_false_for_empty_state():
    assert _is_pipeline_complete({}) is False


def test_is_pipeline_complete_false_when_awaiting_hitl():
    state = {
        "awaiting_hitl": True,
        "completed_agents": ["output_router"],
        "session_id": "abc",
    }
    assert _is_pipeline_complete(state) is False


def test_is_pipeline_complete_false_when_output_router_not_in_completed():
    state = {
        "awaiting_hitl": False,
        "completed_agents": ["profiler", "stat_analyst", "narrator"],
        "session_id": "abc",
    }
    assert _is_pipeline_complete(state) is False


def test_is_pipeline_complete_true_when_pipeline_done():
    state = {
        "awaiting_hitl": False,
        "completed_agents": ["profiler", "stat_analyst", "narrator", "output_router"],
        "session_id": "abc",
    }
    assert _is_pipeline_complete(state) is True


# ---------------------------------------------------------------------------
# _build_qa_context
# ---------------------------------------------------------------------------

def _make_full_state(goal="Find income drivers", insights=None, caveats=None):
    request = MagicMock()
    request.goal = goal
    request.metadata = {"shape": [10, 4], "columns": ["age", "income"]}

    result = MagicMock()
    result.agent_name = "stat_analyst"
    result.confidence = 0.85
    result.findings = {"notable_findings": ["Income correlates strongly with age"]}

    return {
        "request": request,
        "narrative": "The dataset shows strong income patterns.",
        "key_insights": insights or ["Reveals income-age correlation of 0.99"],
        "caveats": caveats or ["Small sample size of 10 rows"],
        "agent_results": [result],
        "scored_results": [],
    }


def test_build_qa_context_contains_goal():
    ctx = _build_qa_context(_make_full_state(goal="Find income drivers"))
    assert "Find income drivers" in ctx


def test_build_qa_context_contains_key_insight():
    ctx = _build_qa_context(
        _make_full_state(insights=["Reveals income-age correlation of 0.99"])
    )
    assert "income-age correlation" in ctx


# ---------------------------------------------------------------------------
# _answer_question
# ---------------------------------------------------------------------------

def test_answer_question_returns_history_with_assistant_message():
    mock_graph = MagicMock()
    mock_graph.get_state.return_value.values = _make_full_state()

    mock_response = MagicMock()
    mock_response.content = "The strongest correlation is age-income at 0.996"

    with patch("adapters.gradio_ui.ChatAnthropic") as MockLLM:
        MockLLM.return_value.invoke.return_value = mock_response
        history = _answer_question(
            "What is the strongest relationship?",
            {"session_id": "test-123"},
            mock_graph,
            [],
        )

    assert len(history) == 1
    assert history[0]["role"] == "assistant"
    assert "0.996" in history[0]["content"]


def test_answer_question_returns_error_message_on_exception():
    mock_graph = MagicMock()
    mock_graph.get_state.side_effect = Exception("connection failed")

    history = _answer_question(
        "What is the strongest relationship?",
        {"session_id": "test-123"},
        mock_graph,
        [],
    )

    assert len(history) == 1
    assert history[0]["role"] == "assistant"
    assert "couldn't answer" in history[0]["content"].lower()
