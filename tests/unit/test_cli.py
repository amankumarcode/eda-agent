import json
from datetime import date

import pytest
from typer.testing import CliRunner

import dva.cli as cli
from dva.agents.eda import EdaCode
from dva.agents.synthesis import SynthesisWording
from dva.orchestrator.graph import build_graph, open_checkpointer
from dva.orchestrator.tools import GraphDeps
from dva.review_queue import ReviewQueue
from dva.state import Draft
from dva.workspace import session_workspace
from fakes import FakeChatModel, tool_call

WORDING = SynthesisWording(report_summary="Not ready.", email_subject="Fixes needed",
                           email_opening="Hello.", email_closing="Thanks.",
                           jira_title="Issues", jira_summary="Issues found.")  # fmt: skip
NOTE = EdaCode(code="emit([{'field': 'region', 'issue_type': 'other', 'severity': 'low', "
               "'detail': 'note', 'count': 1}], [])", checks_addressed=[])  # fmt: skip
runner = CliRunner()


@pytest.fixture
def local_settings(settings, tmp_path, monkeypatch):
    paths = settings.paths.model_copy(update={"workspace_root": tmp_path / "workspace",
                                              "checkpoint_db": tmp_path / "cp.sqlite"})  # fmt: skip
    local = settings.model_copy(update={"paths": paths})
    monkeypatch.setattr(cli, "load_settings", lambda: local)
    monkeypatch.setattr(cli, "configure_logging", lambda s: None)
    return local


def _fake_deps(settings, retail_model, orchestrator, eda=()):
    fakes = {
        "orchestrator": FakeChatModel(orchestrator),
        "column_mapper": FakeChatModel(),
        "eda_coder": FakeChatModel(list(eda)),
        "synthesis": FakeChatModel([WORDING]),
    }
    return GraphDeps(settings=settings, data_model=retail_model, llm_factory=fakes.__getitem__,
                     today=lambda: date(2026, 10, 25))  # fmt: skip


@pytest.fixture
def kitchen_files(fixtures_dir):
    gt = json.loads((fixtures_dir / "kitchen_sink" / "ground_truth.json").read_text())
    return [str(fixtures_dir / "kitchen_sink" / f) for f in gt["files"].values()]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            ["run", "--files", "a", "b", "c", "--session", "s"],
            ["run", "--files", "a", "--files", "b", "--files", "c", "--session", "s"],
        ),
        (["run", "--session", "s", "--files", "a"], ["run", "--session", "s", "--files", "a"]),
        (["run", "--files"], ["run", "--files"]),
        (["review", "list"], ["review", "list"]),
    ],  # fmt: skip
)
def test_files_option_takes_space_separated_values(argv, expected):
    assert cli.expand_multi_value(argv) == expected


def test_run_prints_report_and_stop_reason(local_settings, retail_model, kitchen_files,
                                           monkeypatch):  # fmt: skip
    script = [tool_call("run_parsing", call_id="c1"), tool_call("run_eda", call_id="c2"),
              tool_call("run_synthesis", call_id="c3"), "Done."]  # fmt: skip
    deps = _fake_deps(local_settings, retail_model, script, [NOTE])
    monkeypatch.setattr(cli, "build_deps", lambda settings, name: deps)
    args = cli.expand_multi_value(["run", "--files", *kitchen_files, "--session", "cust-418",
                                   "--data-model", "retail_sales"])  # fmt: skip
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    report = local_settings.paths.workspace_root / "cust-418" / "report.md"
    assert f"Report: {report}" in result.output
    assert "Summary: Not ready." in result.output and "Stop reason: task_complete" in result.output
    assert "tools: run_parsing, run_eda, run_synthesis" in result.output


def test_run_exit_code_1_on_partial_and_2_on_bad_session(local_settings, retail_model,
                                                         kitchen_files, monkeypatch):  # fmt: skip
    deps = _fake_deps(local_settings, retail_model, ["Nothing to do."])
    monkeypatch.setattr(cli, "build_deps", lambda settings, name: deps)
    partial = runner.invoke(cli.app, ["run", "--files", kitchen_files[0], "--session", "s1"])
    assert partial.exit_code == 1 and "ended_without_report" in partial.output
    bad = runner.invoke(cli.app, ["run", "--files", kitchen_files[0], "--session", "../x"])
    assert bad.exit_code == 2 and "invalid --session" in bad.output


def test_chat_loop_uploads_and_keeps_state(local_settings, retail_model, kitchen_files, tmp_path):
    script = [tool_call("run_parsing", call_id="c1"), "Parsed: 6 issues.", "37 prices in cents."]
    deps = _fake_deps(local_settings, retail_model, script)
    lines = iter(["/upload " + " ".join(kitchen_files), "/upload nowhere.csv",
                  "check my files", "", "how many prices are wrong?", "/quit"])  # fmt: skip
    out: list[str] = []
    with open_checkpointer(tmp_path / "chat.sqlite") as saver:
        cli.chat_loop(build_graph(deps, saver), deps, "chat-1", lambda: next(lines), out.append)
    assert out[0] == cli.CHAT_HELP
    assert out[1] == "3 file(s) ready; they are sent with your next message."
    assert out[2] == "Not found: nowhere.csv"
    assert out[3] == "agent> Parsed: 6 issues." and "tools: run_parsing" in out[4]
    assert out[5] == "agent> 37 prices in cents." and "tools: none" in out[6]
    assert len(out) == 7


def test_chat_loop_ends_on_end_of_input(local_settings, retail_model, tmp_path):
    deps = _fake_deps(local_settings, retail_model, [])

    def read():
        raise EOFError

    out: list[str] = []
    cli.chat_loop(build_graph(deps), deps, "chat-2", read, out.append)
    assert out == [cli.CHAT_HELP]


def _queue_draft(settings, session, draft_id):
    queue = ReviewQueue(session_workspace(settings.paths.workspace_root, session))
    queue.save(Draft(draft_id=draft_id, session_id=session, kind="email", title="Fix prices",
                     body="Please fix the prices."))  # fmt: skip
    return queue


def test_review_list_show_and_mark(local_settings):
    queue = _queue_draft(local_settings, "cust-418", "email-0123456789ab")
    _queue_draft(local_settings, "cust-500", "jira-0123456789ab")
    listed = runner.invoke(cli.app, ["review", "list"])
    assert listed.exit_code == 0
    assert "cust-418" in listed.output and "jira-0123456789ab" in listed.output
    only = runner.invoke(cli.app, ["review", "list", "--session", "cust-500"])
    assert "email-0123456789ab" not in only.output
    shown = runner.invoke(cli.app, ["review", "show", "email-0123456789ab"])
    assert "Please fix the prices." in shown.output and "[pending_review]" in shown.output
    marked = runner.invoke(cli.app, ["review", "mark", "email-0123456789ab", "approved"])
    assert marked.exit_code == 0 and "approved (nothing was sent)" in marked.output
    assert queue.get("email-0123456789ab").status.value == "approved"


def test_review_errors(local_settings):
    _queue_draft(local_settings, "cust-418", "email-0123456789ab")
    assert runner.invoke(cli.app, ["review", "mark", "email-0123456789ab", "sent"]).exit_code == 2
    unknown = runner.invoke(cli.app, ["review", "mark", "email-ffffffffffff", "approved"])
    assert unknown.exit_code == 2 and "no draft" in unknown.output
    empty = runner.invoke(cli.app, ["review", "list", "--session", "nobody"])
    assert "No drafts." in empty.output
