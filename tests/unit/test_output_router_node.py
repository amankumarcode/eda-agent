from contextlib import ExitStack
from unittest.mock import patch

import pytest

from core.schema import EDARequest, OutputType
from core.state import AgentState
from graph.nodes.output_router import OUTPUT_HANDLERS, output_router_node


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def state_with_narrative(base_state: AgentState) -> AgentState:
    return {
        **base_state,
        "narrative": "Income is driven by age.",
        "key_insights": ["Age explains 85% of income variance"],
        "caveats": ["Small sample"],
        "evaluation": None,
        "agent_results": [],
    }


@pytest.fixture
def state_all_formats(state_with_narrative: AgentState) -> AgentState:
    """Request with all four output formats."""
    request = EDARequest(
        goal="test",
        session_id="test-session-001",
        output_formats=[
            OutputType.REPORT,
            OutputType.JSON,
            OutputType.DASHBOARD,
            OutputType.EMAIL,
        ],
    )
    return {**state_with_narrative, "request": request}


_REPORT_PATH    = "./output/test-session-001/report.html"
_JSON_PATH      = "./output/test-session-001/summary.json"
_DASHBOARD_PATH = "./output/test-session-001/dashboard.html"
_EMAIL_PATH     = "./output/test-session-001/email.txt"

_ALL_PATCHES = {
    "graph.nodes.output_router.generate_report":    _REPORT_PATH,
    "graph.nodes.output_router.generate_json":      _JSON_PATH,
    "graph.nodes.output_router.generate_dashboard": _DASHBOARD_PATH,
    "graph.nodes.output_router.draft_email":        _EMAIL_PATH,
}


def _patch_all_handlers(overrides: dict = None):
    """ExitStack context manager patching all four output handlers."""
    targets = dict(_ALL_PATCHES)
    if overrides:
        targets.update(overrides)
    stack = ExitStack()
    mocks = {}
    for target, return_value in targets.items():
        if isinstance(return_value, Exception):
            mocks[target] = stack.enter_context(
                patch(target, side_effect=return_value)
            )
        else:
            mocks[target] = stack.enter_context(
                patch(target, return_value=return_value)
            )
    return stack, mocks


# ---------------------------------------------------------------------------
# OUTPUT_HANDLERS
# ---------------------------------------------------------------------------

def test_output_handlers_maps_all_four_types() -> None:
    assert OutputType.REPORT    in OUTPUT_HANDLERS
    assert OutputType.JSON      in OUTPUT_HANDLERS
    assert OutputType.DASHBOARD in OUTPUT_HANDLERS
    assert OutputType.EMAIL     in OUTPUT_HANDLERS


def test_output_handlers_has_callable_values() -> None:
    for handler in OUTPUT_HANDLERS.values():
        assert callable(handler)


# ---------------------------------------------------------------------------
# Return shape
# ---------------------------------------------------------------------------

def test_output_router_returns_expected_keys(
    state_with_narrative: AgentState,
) -> None:
    with (
        patch("graph.nodes.output_router.generate_report", return_value=_REPORT_PATH),
        patch("graph.nodes.output_router.generate_json",   return_value=_JSON_PATH),
    ):
        result = output_router_node(state_with_narrative)

    for key in ("output_paths", "next_action", "completed_agents", "messages"):
        assert key in result


def test_next_action_always_complete(state_with_narrative: AgentState) -> None:
    with (
        patch("graph.nodes.output_router.generate_report", return_value=_REPORT_PATH),
        patch("graph.nodes.output_router.generate_json",   return_value=_JSON_PATH),
    ):
        result = output_router_node(state_with_narrative)

    assert result["next_action"] == "complete"


def test_completed_agents_contains_output_router(
    state_with_narrative: AgentState,
) -> None:
    with (
        patch("graph.nodes.output_router.generate_report", return_value=_REPORT_PATH),
        patch("graph.nodes.output_router.generate_json",   return_value=_JSON_PATH),
    ):
        result = output_router_node(state_with_narrative)

    assert "output_router" in result["completed_agents"]


def test_messages_is_list(state_with_narrative: AgentState) -> None:
    with (
        patch("graph.nodes.output_router.generate_report", return_value=_REPORT_PATH),
        patch("graph.nodes.output_router.generate_json",   return_value=_JSON_PATH),
    ):
        result = output_router_node(state_with_narrative)

    assert isinstance(result["messages"], list)
    assert len(result["messages"]) == 1


# ---------------------------------------------------------------------------
# All-success path
# ---------------------------------------------------------------------------

def test_output_paths_contains_all_formats_on_success(
    state_all_formats: AgentState,
) -> None:
    with _patch_all_handlers()[0]:
        result = output_router_node(state_all_formats)

    paths = result["output_paths"]
    assert "report"    in paths
    assert "json"      in paths
    assert "dashboard" in paths
    assert "email"     in paths


def test_output_paths_values_are_strings(state_all_formats: AgentState) -> None:
    with _patch_all_handlers()[0]:
        result = output_router_node(state_all_formats)

    for path in result["output_paths"].values():
        assert isinstance(path, str)


def test_output_paths_keys_are_value_strings_not_enum_members(
    state_all_formats: AgentState,
) -> None:
    """Keys must be "report", "json" etc — not OutputType.REPORT."""
    with _patch_all_handlers()[0]:
        result = output_router_node(state_all_formats)

    for key in result["output_paths"]:
        assert isinstance(key, str)
        assert not hasattr(key, "value")  # not an enum


def test_message_contains_success_count(state_all_formats: AgentState) -> None:
    with _patch_all_handlers()[0]:
        result = output_router_node(state_all_formats)

    content = result["messages"][0].content
    assert "4" in content  # 4 outputs generated


# ---------------------------------------------------------------------------
# Partial-failure path
# ---------------------------------------------------------------------------

def test_partial_failure_does_not_prevent_other_handlers(
    state_all_formats: AgentState,
) -> None:
    stack, mocks = _patch_all_handlers(overrides={
        "graph.nodes.output_router.generate_report": Exception("disk full"),
    })
    with stack:
        result = output_router_node(state_all_formats)

    mocks["graph.nodes.output_router.generate_json"].assert_called_once()
    mocks["graph.nodes.output_router.generate_dashboard"].assert_called_once()
    mocks["graph.nodes.output_router.draft_email"].assert_called_once()


def test_partial_failure_output_paths_contains_only_successful(
    state_all_formats: AgentState,
) -> None:
    stack, _ = _patch_all_handlers(overrides={
        "graph.nodes.output_router.generate_report": Exception("disk full"),
    })
    with stack:
        result = output_router_node(state_all_formats)

    paths = result["output_paths"]
    assert "report" not in paths
    assert "json" in paths
    assert "dashboard" in paths
    assert "email" in paths


def test_partial_failure_next_action_still_complete(
    state_all_formats: AgentState,
) -> None:
    stack, _ = _patch_all_handlers(overrides={
        "graph.nodes.output_router.generate_report": Exception("disk full"),
    })
    with stack:
        result = output_router_node(state_all_formats)

    assert result["next_action"] == "complete"


def test_message_contains_fail_count_on_partial_failure(
    state_all_formats: AgentState,
) -> None:
    stack, _ = _patch_all_handlers(overrides={
        "graph.nodes.output_router.generate_report": Exception("disk full"),
    })
    with stack:
        result = output_router_node(state_all_formats)

    content = result["messages"][0].content
    assert "1" in content  # 1 failed


# ---------------------------------------------------------------------------
# Total failure path
# ---------------------------------------------------------------------------

def test_total_failure_output_paths_empty(base_state: AgentState) -> None:
    # Force outer try/except by passing a dict with no "request" key
    result = output_router_node({})
    assert result["output_paths"] == {}


def test_total_failure_next_action_complete(base_state: AgentState) -> None:
    result = output_router_node({})
    assert result["next_action"] == "complete"


def test_total_failure_completed_agents_contains_output_router(
    base_state: AgentState,
) -> None:
    result = output_router_node({})
    assert "output_router" in result["completed_agents"]


def test_total_failure_does_not_propagate(base_state: AgentState) -> None:
    result = output_router_node({})
    assert isinstance(result, dict)
