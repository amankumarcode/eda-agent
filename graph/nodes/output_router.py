from typing import List

from langchain_core.messages import AIMessage

from core.schema import OutputType
from core.state import AgentState
from outputs.dashboard import generate_dashboard
from outputs.email_drafter import draft_email
from outputs.json_summary import generate_json
from outputs.report import generate_report
from outputs.slack_notifier import notify_slack

# Maps each OutputType to its handler function.
# This is the single source of truth for output format dispatch.
OUTPUT_HANDLERS = {
    OutputType.REPORT:    generate_report,
    OutputType.JSON:      generate_json,
    OutputType.DASHBOARD: generate_dashboard,
    OutputType.EMAIL:     draft_email,
}

# Maps OutputType to the handler's module-level name, used to look up
# the current binding at call time so unittest.mock patches take effect.
_HANDLER_NAMES = {fmt: fn.__name__ for fmt, fn in OUTPUT_HANDLERS.items()}


def output_router_node(state: AgentState) -> dict:
    """LangGraph node: routes completed analysis to requested output formats.

    Each format is generated independently — a failure in one never
    prevents others from completing. Always returns next_action = "complete".
    """
    try:
        request = state["request"]
        output_formats = request.output_formats

        # Resolve handlers through the module's current globals so that
        # unittest.mock patches applied to the individual function names
        # are picked up at call time rather than at import time.
        _globals = globals()

        output_paths: dict = {}
        failed: List[str] = []

        for fmt in output_formats:
            fn_name = _HANDLER_NAMES.get(fmt)
            if fn_name is None:
                failed.append(f"{fmt}: no handler registered")
                continue
            handler = _globals.get(fn_name)
            if handler is None:
                failed.append(f"{fmt.value}: handler not found in module")
                continue
            try:
                path = handler(state)
                output_paths[fmt.value] = path
            except Exception as e:
                failed.append(f"{fmt.value}: {str(e)}")

        notify_slack(state)

        success_count = len(output_paths)
        fail_count = len(failed)
        msg = (
            f"Output router complete. "
            f"{success_count} outputs generated: "
            f"{list(output_paths.keys())}. "
            f"{fail_count} failed: {failed if failed else 'none'}."
        )

        return {
            "output_paths": output_paths,
            "next_action": "complete",
            "completed_agents": ["output_router"],
            "messages": [AIMessage(content=msg)],
        }

    except Exception as e:
        return {
            "output_paths": {},
            "next_action": "complete",
            "completed_agents": ["output_router"],
            "messages": [AIMessage(content=f"Output router failed entirely: {str(e)}")],
        }
