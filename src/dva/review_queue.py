"""Review queue: drafts on disk waiting for a human. Nothing here ever sends anything.

Each draft is workspace/<session_id>/review_queue/<draft_id>.json. Marking a draft only
changes its status.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from dva.state import Draft, DraftStatus
from dva.workspace import SESSION_ID_PATTERN, SessionWorkspace, session_workspace

DRAFT_ID_PATTERN = re.compile(r"^[a-z]+-[0-9a-f]{8,32}$")
REVIEW_STATUSES = (DraftStatus.APPROVED, DraftStatus.REJECTED)


class DraftNotFoundError(LookupError):
    """No draft with that id in this session's queue."""


class ReviewQueue:
    """The review queue of one session."""

    def __init__(self, workspace: SessionWorkspace) -> None:
        self._workspace = workspace

    def _path(self, draft_id: str) -> Path:
        if not DRAFT_ID_PATTERN.match(draft_id):
            raise DraftNotFoundError(f"invalid draft id: {draft_id!r}")
        return self._workspace.review_queue_dir / f"{draft_id}.json"

    def save(self, draft: Draft) -> Path:
        """Write the draft as JSON. Writes: review_queue/<draft_id>.json. Returns its path."""
        self._workspace.ensure()
        path = self._path(draft.draft_id)
        path.write_text(draft.model_dump_json(indent=2), encoding="utf-8")
        return path

    def get(self, draft_id: str) -> Draft:
        """Read one draft. Raises DraftNotFoundError."""
        path = self._path(draft_id)
        if not path.is_file():
            raise DraftNotFoundError(
                f"no draft {draft_id!r} in session {self._workspace.session_id}"
            )
        return Draft.model_validate_json(path.read_text(encoding="utf-8"))

    @property
    def session_id(self) -> str:
        return self._workspace.session_id

    def drafts(self) -> list[Draft]:
        """All drafts in this session, oldest first. Reads: review_queue/*.json."""
        folder = self._workspace.review_queue_dir
        if not folder.is_dir():
            return []
        drafts = [Draft.model_validate_json(p.read_text(encoding="utf-8"))
                  for p in folder.glob("*.json")]  # fmt: skip
        return sorted(drafts, key=lambda d: (d.created_at, d.draft_id))

    def mark(self, draft_id: str, status: DraftStatus | str) -> Draft:
        """Set a draft to approved or rejected. Changes the status field only; sends nothing.

        Writes: review_queue/<draft_id>.json.
        """
        new_status = DraftStatus(status)
        if new_status not in REVIEW_STATUSES:
            raise ValueError("a draft can only be marked approved or rejected")
        draft = self.get(draft_id).model_copy(update={"status": new_status})
        self.save(draft)
        return draft


def iter_queues(workspace_root: Path) -> Iterator[ReviewQueue]:
    """Yield the review queue of every session under the workspace root that has one."""
    if not workspace_root.is_dir():
        return
    for folder in sorted(workspace_root.iterdir()):
        if SESSION_ID_PATTERN.match(folder.name) and (folder / "review_queue").is_dir():
            yield ReviewQueue(session_workspace(workspace_root, folder.name))
