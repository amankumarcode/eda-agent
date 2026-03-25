import sqlite3
from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver


def get_checkpointer() -> MemorySaver:
    """Return an in-memory checkpointer for the current process.

    SqliteSaver cannot serialize pd.DataFrame (used in EDARequest), so
    MemorySaver is the correct choice here.  HITL pause/resume works
    because the graph process stays alive between the interrupt and the
    resume call — no cross-process persistence is required.
    """
    return MemorySaver()


def get_memory_checkpointer() -> MemorySaver:
    """Return an in-memory checkpointer.

    Identical to get_checkpointer(); kept as a separate name so test
    fixtures and integration tests can import it by its historical name
    without breaking.
    """
    return MemorySaver()


def make_config(session_id: str) -> dict:
    """Return the LangGraph config dict for a given session.

    Pass this as the ``config`` argument to graph.invoke() / graph.stream().
    LangGraph uses thread_id to isolate checkpoints per session.

    Example::

        graph.invoke({"request": req}, config=make_config(req.session_id))
    """
    return {"configurable": {"thread_id": session_id}}


def get_thread_state(graph: Any, session_id: str) -> Optional[dict]:
    """Return the current checkpointed state dict for a session, or None.

    Args:
        graph:      A compiled LangGraph StateGraph (with checkpointer attached).
        session_id: The session whose checkpoint to retrieve.

    Returns:
        The state values dict if a checkpoint exists, otherwise None.
    """
    snapshot = graph.get_state(make_config(session_id))
    return snapshot.values if snapshot.values else None
