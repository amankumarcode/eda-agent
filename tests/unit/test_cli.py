"""Tests for adapters/cli.py."""
import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from adapters.cli import (
    _handle_hitl,
    _load_dataframe,
    _parse_output_formats,
    _print_outputs,
    _print_plan,
    _print_progress,
    build_arg_parser,
)
from core.schema import OutputType


# ---------------------------------------------------------------------------
# _parse_output_formats
# ---------------------------------------------------------------------------

def test_parse_report_and_json():
    result = _parse_output_formats("report,json")
    assert OutputType.REPORT in result
    assert OutputType.JSON in result
    assert len(result) == 2


def test_parse_all_four_formats():
    result = _parse_output_formats("report,email,dashboard,json")
    assert set(result) == {
        OutputType.REPORT,
        OutputType.EMAIL,
        OutputType.DASHBOARD,
        OutputType.JSON,
    }


def test_parse_unknown_value_skipped_with_warning(capsys):
    result = _parse_output_formats("report,nope")
    assert OutputType.REPORT in result
    assert len(result) == 1
    captured = capsys.readouterr()
    assert "nope" in captured.out
    assert "Warning" in captured.out


def test_parse_empty_string_returns_default():
    result = _parse_output_formats("")
    assert result == [OutputType.REPORT, OutputType.JSON]


def test_parse_all_invalid_returns_default(capsys):
    result = _parse_output_formats("foo,bar")
    assert result == [OutputType.REPORT, OutputType.JSON]


def test_parse_single_dashboard():
    result = _parse_output_formats("dashboard")
    assert result == [OutputType.DASHBOARD]


def test_parse_whitespace_around_values():
    result = _parse_output_formats(" report , json ")
    assert OutputType.REPORT in result
    assert OutputType.JSON in result


# ---------------------------------------------------------------------------
# _load_dataframe
# ---------------------------------------------------------------------------

def test_load_csv(tmp_path, capsys):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n1,2\n3,4\n")
    df = _load_dataframe(str(csv_file))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    captured = capsys.readouterr()
    assert "2 rows" in captured.out
    assert "data.csv" in captured.out


def test_load_excel(tmp_path, capsys):
    xlsx_file = tmp_path / "data.xlsx"
    pd.DataFrame({"x": [1, 2], "y": [3, 4]}).to_excel(xlsx_file, index=False)
    df = _load_dataframe(str(xlsx_file))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    captured = capsys.readouterr()
    assert "data.xlsx" in captured.out


def test_load_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        _load_dataframe("/nonexistent/path/file.csv")


def test_load_raises_value_error_for_unsupported_extension(tmp_path):
    bad_file = tmp_path / "data.parquet"
    bad_file.write_text("placeholder")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        _load_dataframe(str(bad_file))


# ---------------------------------------------------------------------------
# _print_plan
# ---------------------------------------------------------------------------

def test_print_plan_contains_analysis_plan_heading(capsys):
    plan = {
        "goal": "Find revenue drivers",
        "steps": ["Step 1", "Step 2"],
        "agents": ["profiler", "stat_analyst"],
        "parallel": [["profiler", "stat_analyst"]],
        "output_formats": ["report", "json"],
    }
    _print_plan(plan)
    captured = capsys.readouterr()
    assert "ANALYSIS PLAN" in captured.out


def test_print_plan_contains_goal(capsys):
    plan = {
        "goal": "Find revenue drivers",
        "steps": [],
        "agents": [],
        "parallel": [],
        "output_formats": [],
    }
    _print_plan(plan)
    captured = capsys.readouterr()
    assert "Find revenue drivers" in captured.out


def test_print_plan_contains_agents(capsys):
    plan = {
        "goal": "Test",
        "steps": [],
        "agents": ["profiler", "viz_agent"],
        "parallel": [],
        "output_formats": [],
    }
    _print_plan(plan)
    captured = capsys.readouterr()
    assert "profiler" in captured.out
    assert "viz_agent" in captured.out


def test_print_plan_lists_steps(capsys):
    plan = {
        "goal": "Test",
        "steps": ["Run profiler", "Run stat analyst"],
        "agents": [],
        "parallel": [],
        "output_formats": [],
    }
    _print_plan(plan)
    captured = capsys.readouterr()
    assert "Run profiler" in captured.out
    assert "Run stat analyst" in captured.out


# ---------------------------------------------------------------------------
# _handle_hitl
# ---------------------------------------------------------------------------

def test_handle_hitl_returns_user_input(capsys):
    interrupt_data = {
        "plan": {
            "goal": "Test goal",
            "steps": [],
            "agents": [],
            "parallel": [],
            "output_formats": [],
        },
        "message": "Approve this analysis plan?",
    }
    with patch("builtins.input", return_value="approve"):
        result = _handle_hitl(interrupt_data)
    assert result == "approve"


def test_handle_hitl_calls_print_plan(capsys):
    interrupt_data = {
        "plan": {
            "goal": "Find patterns",
            "steps": ["step1"],
            "agents": ["profiler"],
            "parallel": [],
            "output_formats": [],
        },
        "message": "Approve?",
    }
    with patch("builtins.input", return_value="yes"):
        _handle_hitl(interrupt_data)
    captured = capsys.readouterr()
    assert "ANALYSIS PLAN" in captured.out
    assert "Find patterns" in captured.out


def test_handle_hitl_prints_instructions(capsys):
    interrupt_data = {
        "plan": {"goal": "g", "steps": [], "agents": [], "parallel": [], "output_formats": []},
        "message": "Approve?",
    }
    with patch("builtins.input", return_value="approve"):
        _handle_hitl(interrupt_data)
    captured = capsys.readouterr()
    assert "approve" in captured.out.lower()


# ---------------------------------------------------------------------------
# _print_progress
# ---------------------------------------------------------------------------

def test_print_progress_prints_node_and_content(capsys):
    msg = MagicMock()
    msg.content = "Profiler complete."
    _print_progress({"profiler": {"messages": [msg]}})
    captured = capsys.readouterr()
    assert "[profiler]" in captured.out
    assert "Profiler complete." in captured.out


def test_print_progress_skips_events_with_no_messages(capsys):
    _print_progress({"profiler": {"agent_results": []}})
    captured = capsys.readouterr()
    assert captured.out == ""


def test_print_progress_truncates_long_content(capsys):
    msg = MagicMock()
    msg.content = "A" * 200
    _print_progress({"profiler": {"messages": [msg]}})
    captured = capsys.readouterr()
    assert "..." in captured.out
    # Line should not contain the full 200-char string
    assert "A" * 200 not in captured.out


def test_print_progress_skips_interrupt_key(capsys):
    _print_progress({"__interrupt__": {"value": "something"}})
    captured = capsys.readouterr()
    assert captured.out == ""


# ---------------------------------------------------------------------------
# _print_outputs
# ---------------------------------------------------------------------------

def test_print_outputs_contains_output_files_heading(capsys):
    _print_outputs({"report": "/output/s1/report.html"})
    captured = capsys.readouterr()
    assert "OUTPUT FILES" in captured.out


def test_print_outputs_contains_path_strings(capsys):
    _print_outputs({
        "report": "/output/s1/report.html",
        "json": "/output/s1/summary.json",
    })
    captured = capsys.readouterr()
    assert "/output/s1/report.html" in captured.out
    assert "/output/s1/summary.json" in captured.out


def test_print_outputs_contains_format_keys(capsys):
    _print_outputs({"report": "/output/s1/report.html"})
    captured = capsys.readouterr()
    assert "report" in captured.out


# ---------------------------------------------------------------------------
# build_arg_parser
# ---------------------------------------------------------------------------

def test_build_arg_parser_returns_parser():
    parser = build_arg_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def test_build_arg_parser_accepts_file_goal_output_mode():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--mode", "cli",
        "--file", "data.csv",
        "--goal", "Find patterns",
        "--output", "report,json",
    ])
    assert args.mode == "cli"
    assert args.file == "data.csv"
    assert args.goal == "Find patterns"
    assert args.output == "report,json"


def test_build_arg_parser_default_mode_is_gradio():
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.mode == "gradio"


def test_build_arg_parser_default_output_is_report_json():
    parser = build_arg_parser()
    args = parser.parse_args(["--mode", "cli", "--file", "f.csv", "--goal", "g"])
    assert args.output == "report,json"
