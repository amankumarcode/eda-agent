from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from memory.checkpointer import (
    get_checkpointer,
    get_memory_checkpointer,
    get_thread_state,
    make_config,
)


# ---------------------------------------------------------------------------
# Minimal graph fixture — used only to test get_thread_state
# ---------------------------------------------------------------------------

class _SimpleState(TypedDict):
    value: int


def _build_minimal_graph(checkpointer: MemorySaver):
    """Compile a trivial single-node graph with the given checkpointer."""
    g = StateGraph(_SimpleState)
    g.add_node("noop", lambda state: state)
    g.add_edge(START, "noop")
    g.add_edge("noop", END)
    return g.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# get_checkpointer
# ---------------------------------------------------------------------------

def test_get_checkpointer_returns_memory_saver() -> None:
    cp = get_checkpointer()
    assert isinstance(cp, MemorySaver)


# ---------------------------------------------------------------------------
# get_memory_checkpointer
# ---------------------------------------------------------------------------

def test_get_memory_checkpointer_returns_memory_saver() -> None:
    cp = get_memory_checkpointer()
    assert isinstance(cp, MemorySaver)


def test_get_memory_checkpointer_is_independent() -> None:
    # Two in-memory checkpointers must not share state
    cp1 = get_memory_checkpointer()
    cp2 = get_memory_checkpointer()
    assert cp1 is not cp2


# ---------------------------------------------------------------------------
# make_config
# ---------------------------------------------------------------------------

def test_make_config_structure() -> None:
    config = make_config("test-123")
    assert "configurable" in config
    assert "thread_id" in config["configurable"]
    assert config["configurable"]["thread_id"] == "test-123"


def test_make_config_thread_id_matches_session_id() -> None:
    session_id = "session-abc-999"
    config = make_config(session_id)
    assert config["configurable"]["thread_id"] == session_id


def test_make_config_different_sessions_produce_different_configs() -> None:
    cfg_a = make_config("session-a")
    cfg_b = make_config("session-b")
    assert cfg_a["configurable"]["thread_id"] != cfg_b["configurable"]["thread_id"]
    assert cfg_a != cfg_b


# ---------------------------------------------------------------------------
# get_thread_state
# ---------------------------------------------------------------------------

def test_get_thread_state_returns_none_for_unknown_session() -> None:
    cp = get_memory_checkpointer()
    graph = _build_minimal_graph(cp)
    result = get_thread_state(graph, "never-ran-session")
    assert result is None
