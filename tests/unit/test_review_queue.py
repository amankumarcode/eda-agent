import ast
from pathlib import Path

import pytest

from dva.review_queue import DraftNotFoundError, ReviewQueue, iter_queues
from dva.state import Draft, DraftStatus
from dva.workspace import session_workspace

SEND_MODULES = {"smtplib", "requests", "httpx", "urllib", "http", "socket", "email.mime"}


def _draft(draft_id="email-0123456789ab", session_id="s1", **overrides):
    return Draft(draft_id=draft_id, session_id=session_id, kind="email", title="Subject",
                 body="Body", **overrides)  # fmt: skip


@pytest.fixture
def queue(tmp_path):
    return ReviewQueue(session_workspace(tmp_path, "s1"))


def test_save_writes_pending_json(queue, tmp_path):
    path = queue.save(_draft())
    assert path == tmp_path / "s1" / "review_queue" / "email-0123456789ab.json"
    assert queue.get("email-0123456789ab").status is DraftStatus.PENDING_REVIEW


def test_drafts_are_listed_oldest_first(queue):
    first = _draft("jira-aaaaaaaa0000")
    second = _draft("email-bbbbbbbb0000")
    queue.save(second.model_copy(update={"created_at": second.created_at.replace(year=2030)}))
    queue.save(first)
    assert [d.draft_id for d in queue.drafts()] == ["jira-aaaaaaaa0000", "email-bbbbbbbb0000"]


@pytest.mark.parametrize("status", ["approved", "rejected"])
def test_mark_changes_status_only(queue, status):
    original = _draft()
    queue.save(original)
    marked = queue.mark(original.draft_id, status)
    assert marked.status.value == status
    stored = queue.get(original.draft_id)
    assert stored.model_dump(exclude={"status"}) == original.model_dump(exclude={"status"})


def test_mark_rejects_other_statuses_and_unknown_ids(queue):
    queue.save(_draft())
    with pytest.raises(ValueError):
        queue.mark("email-0123456789ab", "pending_review")
    with pytest.raises(ValueError):
        queue.mark("email-0123456789ab", "sent")
    with pytest.raises(DraftNotFoundError):
        queue.mark("email-ffffffffffff", "approved")
    with pytest.raises(DraftNotFoundError, match="invalid draft id"):
        queue.get("../../etc/passwd")


def test_empty_queue_and_queues_across_sessions(tmp_path):
    assert ReviewQueue(session_workspace(tmp_path, "none")).drafts() == []
    ReviewQueue(session_workspace(tmp_path, "a1")).save(_draft(session_id="a1"))
    ReviewQueue(session_workspace(tmp_path, "b2")).save(_draft(session_id="b2"))
    (tmp_path / "not a session").mkdir()
    assert [q.session_id for q in iter_queues(tmp_path)] == ["a1", "b2"]
    assert list(iter_queues(tmp_path / "missing")) == []


@pytest.mark.parametrize("module", ["review_queue.py", "agents/synthesis.py"])
def test_nothing_can_be_sent(module):
    """Neither the queue nor synthesis imports anything that could send a message."""
    source = Path(__file__).parents[2] / "src" / "dva" / module
    imported = set()
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not {m for m in imported if m.split(".")[0] in SEND_MODULES or m in SEND_MODULES}
