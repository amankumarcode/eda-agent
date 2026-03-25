import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import List

import pandas as pd
from langgraph.types import Command

from core.dataframe_registry import register as register_df
from core.schema import EDARequest, OutputType
from core.state import AgentState
from memory.checkpointer import make_config

_BORDER = "╔══════════════════════════════════════════╗"
_BORDER_BOTTOM = "╚══════════════════════════════════════════╝"


def _parse_output_formats(output_str: str) -> List[OutputType]:
    """Parse comma-separated format string into OutputType list.

    Unknown values are skipped with a printed warning.
    Falls back to [REPORT, JSON] if empty or all invalid.
    """
    _DEFAULT = [OutputType.REPORT, OutputType.JSON]
    if not output_str or not output_str.strip():
        return _DEFAULT

    valid_map = {ot.value: ot for ot in OutputType}
    result: List[OutputType] = []
    for token in output_str.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token in valid_map:
            result.append(valid_map[token])
        else:
            print(f"Warning: unknown output format '{token}' — skipping.")

    return result if result else _DEFAULT


def _load_dataframe(file_path: str) -> pd.DataFrame:
    """Load CSV or Excel file into a DataFrame.

    Prints loaded shape on success.
    Raises FileNotFoundError or ValueError on failure.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. Use .csv, .xlsx, or .xls."
        )

    rows, cols = df.shape
    print(f"Loaded {rows} rows × {cols} columns from {path.name}")
    return df


def _print_plan(plan_dict: dict) -> None:
    """Print a formatted analysis plan to stdout."""
    print(_BORDER)
    print("║           ANALYSIS PLAN                  ║")
    print(_BORDER_BOTTOM)
    print(f"Goal: {plan_dict.get('goal', '')}")
    print("Steps:")
    for step in plan_dict.get("steps", []):
        print(f"  {step}")
    agents = ", ".join(plan_dict.get("agents", []))
    parallel = plan_dict.get("parallel", [])
    formats = ", ".join(
        f.value if hasattr(f, "value") else str(f)
        for f in plan_dict.get("output_formats", [])
    )
    print(f"\nAgents: {agents}")
    print(f"Parallel groups: {parallel}")
    print(f"Output formats: {formats}")


def _handle_hitl(plan_interrupt: dict) -> str:
    """Handle the HITL interrupt from the supervisor node.

    Prints the plan, then reads user input from stdin.
    Returns the raw input string.
    """
    _print_plan(plan_interrupt.get("plan", {}))
    print(plan_interrupt.get("message", ""))
    print("Type 'approve' to proceed, or enter feedback to revise the plan.")
    return input("> ")


def _print_progress(event: dict) -> None:
    """Print a progress update for a single streamed graph event."""
    for node_name, node_output in event.items():
        if node_name == "__interrupt__":
            continue
        if not isinstance(node_output, dict):
            continue
        messages = node_output.get("messages")
        if not messages:
            continue
        last = messages[-1]
        content = getattr(last, "content", str(last))
        if len(content) > 120:
            content = content[:120] + "..."
        print(f"[{node_name}] {content}")


def _print_outputs(output_paths: dict) -> None:
    """Print the final output file paths."""
    print(_BORDER)
    print("║           OUTPUT FILES                   ║")
    print(_BORDER_BOTTOM)
    for fmt, path in output_paths.items():
        print(f"  {fmt:<12}→ {path}")


def run_cli(graph, args) -> None:
    """Main CLI entry point. Called by run.py with a compiled graph and parsed args."""
    # 1. Load dataframe
    try:
        df = _load_dataframe(args.file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    # 2. Session ID
    session_id = str(uuid.uuid4())[:8]

    # Register DataFrame in registry before building request
    register_df(session_id, df)

    # 3. Output formats
    output_formats = _parse_output_formats(args.output)

    # 4. EDARequest
    rows, cols = df.shape
    request = EDARequest(
        goal=args.goal,
        session_id=session_id,
        output_formats=output_formats,
        metadata={
            "filename": Path(args.file).name,
            "source": "cli",
            "shape": list(df.shape),
            "columns": list(df.columns),
        },
    )

    # 5. LangGraph config
    config = make_config(session_id)

    # 6. Startup banner
    print(f"Starting EDA Agent | Session: {session_id}")
    print(f"Goal: {args.goal}")
    print(f"File: {args.file} ({rows}×{cols})")

    # 7. Initial state
    initial_state: AgentState = {
        "request": request,
        "plan": None,
        "plan_approved": False,
        "agent_results": [],
        "rerun_agent": None,
        "rerun_count": 0,
        "narrative": None,
        "key_insights": [],
        "caveats": [],
        "output_paths": {},
        "messages": [],
        "evaluation": None,
        "evaluation_count": 0,
        "dispatched_agents": [],
        "completed_agents": [],
        "next_action": None,
    }

    # 8. Stream with interrupt handling
    while True:
        interrupted = False
        for event in graph.stream(
            initial_state,
            config={**config, "recursion_limit": 100},
            stream_mode="updates",
        ):
            for node_name, node_output in event.items():
                _print_progress({node_name: node_output})

            if "__interrupt__" in event:
                interrupted = True
                interrupt_data = event["__interrupt__"][0].value
                human_input = _handle_hitl(interrupt_data)
                initial_state = Command(resume=human_input)
                break

        if interrupted:
            continue
        break

    # 9. Final output paths
    final_state = graph.get_state(config)
    output_paths = final_state.values.get("output_paths", {})
    if output_paths:
        _print_outputs(output_paths)
    else:
        print("No output files were generated.")

    # 10. Completion banner
    print(f"EDA Agent complete. Session: {session_id}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="EDA Agent — multi-agent exploratory data analysis",
    )
    parser.add_argument(
        "--mode",
        choices=["gradio", "cli"],
        default="gradio",
        help="Interface mode (default: gradio)",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to CSV or Excel file",
    )
    parser.add_argument(
        "--goal",
        type=str,
        help="Analysis goal in plain English",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="report,json",
        help="Comma-separated output formats: report,json,email,dashboard",
    )
    return parser
