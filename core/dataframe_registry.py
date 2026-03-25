"""In-memory registry mapping session_id → pd.DataFrame.

Nodes call get(session_id) instead of reading df from AgentState,
keeping the checkpointer state fully JSON-serialisable.
"""
from typing import Optional

import pandas as pd

_registry: dict[str, pd.DataFrame] = {}


def register(session_id: str, df: pd.DataFrame) -> None:
    """Store a DataFrame keyed by session_id."""
    _registry[session_id] = df


def get(session_id: str) -> Optional[pd.DataFrame]:
    """Retrieve a DataFrame by session_id. Returns None if not found."""
    return _registry.get(session_id)


def remove(session_id: str) -> None:
    """Remove a DataFrame from the registry when a session ends."""
    _registry.pop(session_id, None)


def clear() -> None:
    """Clear all registered DataFrames. Used in tests."""
    _registry.clear()
