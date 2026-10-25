import pytest
from pydantic import ValidationError

from dva.workspace import session_workspace


def test_paths_follow_layout(tmp_path):
    ws = session_workspace(tmp_path, "cust-418")
    assert ws.session_dir == tmp_path / "cust-418"
    assert ws.table_path("sales") == tmp_path / "cust-418" / "data" / "sales.parquet"
    assert ws.artifacts_dir == tmp_path / "cust-418" / "artifacts"
    assert ws.review_queue_dir == tmp_path / "cust-418" / "review_queue"
    assert ws.emit_path == tmp_path / "cust-418" / "sandbox" / "emit.json"


def test_construction_writes_nothing_and_ensure_creates_folders(tmp_path):
    ws = session_workspace(tmp_path, "s1")
    assert not ws.session_dir.exists()
    ws.ensure()
    for folder in (ws.data_dir, ws.artifacts_dir, ws.review_queue_dir, ws.sandbox_dir):
        assert folder.is_dir()
    ws.ensure()  # idempotent


@pytest.mark.parametrize("bad", ["", "../escape", "a/b", ".hidden", "x" * 65, "white space"])
def test_unsafe_session_ids_are_rejected(tmp_path, bad):
    with pytest.raises(ValidationError):
        session_workspace(tmp_path, bad)


@pytest.mark.parametrize("bad", ["../sales", "Sales", "sales.parquet", ""])
def test_unsafe_table_names_are_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        session_workspace(tmp_path, "s1").table_path(bad)
