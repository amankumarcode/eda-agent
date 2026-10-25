"""Session folders and paths under the workspace root.

Layout per session::

    <workspace_root>/<session_id>/
        data/            cleaned tables as Parquet, columns renamed to model fields
        artifacts/       large EDA outputs referenced by path from findings
        review_queue/    drafts awaiting human review
        sandbox/         emit output from sandbox runs
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
TABLE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
EMIT_FILENAME = "emit.json"


class SessionWorkspace(BaseModel):
    """Paths for one session. Constructing it writes nothing; call ensure() to create folders."""

    model_config = ConfigDict(frozen=True)

    root: Path
    session_id: str

    @field_validator("session_id")
    @classmethod
    def _safe_session_id(cls, value: str) -> str:
        if not SESSION_ID_PATTERN.match(value):
            raise ValueError(
                "session_id must be 1-64 characters of letters, digits, '_' or '-', "
                "starting with a letter or digit"
            )
        return value

    @property
    def session_dir(self) -> Path:
        return self.root / self.session_id

    @property
    def data_dir(self) -> Path:
        return self.session_dir / "data"

    @property
    def artifacts_dir(self) -> Path:
        return self.session_dir / "artifacts"

    @property
    def review_queue_dir(self) -> Path:
        return self.session_dir / "review_queue"

    @property
    def sandbox_dir(self) -> Path:
        return self.session_dir / "sandbox"

    @property
    def emit_path(self) -> Path:
        return self.sandbox_dir / EMIT_FILENAME

    def table_path(self, table: str) -> Path:
        """Return the Parquet path for a cleaned table. Writes nothing."""
        if not TABLE_NAME_PATTERN.match(table):
            raise ValueError(f"invalid table name: {table!r}")
        return self.data_dir / f"{table}.parquet"

    def ensure(self) -> SessionWorkspace:
        """Create the session folders if missing. Writes: the four session sub-folders."""
        for folder in (self.data_dir, self.artifacts_dir, self.review_queue_dir, self.sandbox_dir):
            folder.mkdir(parents=True, exist_ok=True)
        return self


def session_workspace(workspace_root: Path, session_id: str) -> SessionWorkspace:
    """Return the workspace for a session without creating any folders."""
    return SessionWorkspace(root=workspace_root, session_id=session_id)
