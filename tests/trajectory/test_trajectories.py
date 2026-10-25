"""Whole-graph runs with scripted fake models. Each test asserts the exact sequence of tool
calls (tool_history), the final state fields, and stop_reason."""

import itertools
import json
from datetime import date
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from dva.agents.eda import EdaCode
from dva.agents.synthesis import SynthesisWording
from dva.orchestrator.graph import (
    batch_input,
    build_graph,
    chat_input,
    open_checkpointer,
    thread_config,
)
from dva.orchestrator.tools import GraphDeps
from dva.review_queue import ReviewQueue
from dva.state import Finding
from dva.workspace import session_workspace
from fakes import FakeChatModel, tool_call

WORDING = SynthesisWording(
    report_summary="Not ready: prices in cents and quantities in cases.",
    email_subject="Fixes needed in your sales upload",
    email_opening="Thanks for the files. Please fix the issues below.",
    email_closing="Please resend the corrected files.",
    jira_title="kitchen_sink: blocking data issues",
    jira_summary="Upload has blocking issues.",
)


def _note_code(field: str) -> EdaCode:
    """EDA code that reports one new, distinct finding (so state changes every run)."""
    code = (f"emit([{{'field': '{field}', 'issue_type': 'other', 'severity': 'low', "
            f"'detail': 'note about {field}', 'count': 1}}], [])")  # fmt: skip
    return EdaCode(code=code, checks_addressed=[])


EMPTY_CODE = EdaCode(code="emit([], [])", checks_addressed=[])


def call(name, n, **args):
    return tool_call(name, args or None, call_id=f"call_{n}")


class Harness:
    """Fake models per role, deps, and a graph for one test."""

    def __init__(self, settings, retail_model, tmp_path, orchestrator, eda=(), clock=None,
                 checkpointer=None, tokens_per_call=0):  # fmt: skip
        paths = settings.paths.model_copy(update={"workspace_root": tmp_path / "workspace"})
        self.settings = settings.model_copy(update={"paths": paths})
        self.fakes = {
            "orchestrator": FakeChatModel(orchestrator, tokens_per_call=tokens_per_call),
            "column_mapper": FakeChatModel(),
            "eda_coder": FakeChatModel(list(eda)),
            "synthesis": FakeChatModel([WORDING]),
        }
        extra = {"clock": clock} if clock else {}
        self.deps = GraphDeps(settings=self.settings, data_model=retail_model,
                              llm_factory=self.fakes.__getitem__,
                              today=lambda: date(2026, 10, 25), **extra)  # fmt: skip
        self.graph = build_graph(self.deps, checkpointer)

    def batch(self, files, session="cust-418"):
        return self.graph.invoke(batch_input(self.deps, session, files))

    def chat(self, message, uploads=None, session="chat-1"):
        return self.graph.invoke(chat_input(self.deps, session, message, uploads),
                                 thread_config(session))  # fmt: skip

    def workspace(self, session):
        return session_workspace(self.settings.paths.workspace_root, session)


@pytest.fixture
def kitchen_files(fixtures_dir):
    gt = json.loads((fixtures_dir / "kitchen_sink" / "ground_truth.json").read_text())
    return [str(fixtures_dir / "kitchen_sink" / f) for f in gt["files"].values()]


def _tool_messages(state):
    return [m for m in state["messages"] if isinstance(m, ToolMessage)]


# --- batch ------------------------------------------------------------------------------


def test_normal_batch_run(settings, retail_model, tmp_path, kitchen_files):
    h = Harness(settings, retail_model, tmp_path,
                [call("run_parsing", 1), call("run_eda", 2), call("run_synthesis", 3),
                 "Done: 7 issues, report written."], eda=[_note_code("region")])  # fmt: skip
    state = h.batch(kitchen_files)
    assert state["tool_history"] == ["run_parsing", "run_eda", "run_synthesis"]
    assert state["stop_reason"] == "task_complete" and state["tool_calls"] == 3
    assert state["raw_files"] is None and set(state["parsed_data_ref"]) == {"sales", "products",
                                                                            "stores"}  # fmt: skip
    assert len(state["findings"]) == 7  # 6 ground-truth issues + the EDA note
    assert Path(state["report_ref"]).name == "report.md" and "(partial)" not in Path(
        state["report_ref"]).read_text()  # fmt: skip
    assert len(ReviewQueue(h.workspace("cust-418")).drafts()) == 2
    assert all(f.remaining == 0 for f in h.fakes.values())


def test_synthesis_too_early_is_rejected_then_completes(settings, retail_model, tmp_path,
                                                        kitchen_files):  # fmt: skip
    h = Harness(settings, retail_model, tmp_path,
                [call("run_synthesis", 1), call("run_parsing", 2), call("run_eda", 3),
                 call("run_synthesis", 4), "Done."], eda=[_note_code("region")])  # fmt: skip
    state = h.batch(kitchen_files)
    assert state["tool_history"] == ["run_synthesis (rejected)", "run_parsing", "run_eda",
                                     "run_synthesis"]  # fmt: skip
    assert state["stop_reason"] == "task_complete"
    rejection = _tool_messages(state)[0]
    assert rejection.status == "error" and "required checks remain: transaction_date:present" in (
        rejection.content)  # fmt: skip


def test_batch_requires_one_eda_run_before_synthesis(settings, retail_model, tmp_path,
                                                     kitchen_files):  # fmt: skip
    h = Harness(settings, retail_model, tmp_path,
                [call("run_parsing", 1), call("run_synthesis", 2), call("run_eda", 3),
                 call("run_synthesis", 4), "Done."], eda=[_note_code("region")])  # fmt: skip
    state = h.batch(kitchen_files)
    assert state["tool_history"] == ["run_parsing", "run_synthesis (rejected)", "run_eda",
                                     "run_synthesis"]  # fmt: skip
    assert "run_eda must run at least once" in _tool_messages(state)[1].content
    assert state["stop_reason"] == "task_complete"


def test_tool_call_cap_returns_partial_report(settings, retail_model, tmp_path, kitchen_files):
    script = [call("run_parsing", 0)] + [call("run_eda", n, focus=f"pass {n}") for n in range(1, 7)]
    h = Harness(settings, retail_model, tmp_path, script,
                eda=[_note_code(f"note_{n}") for n in range(1, 6)])  # fmt: skip
    state = h.batch(kitchen_files)
    assert state["tool_history"] == ["run_parsing"] + ["run_eda"] * 5
    assert state["tool_calls"] == 6 and state["stop_reason"] == "tool_call_cap"
    report = Path(state["report_ref"]).read_text()
    assert "(partial)" in report and "Stopped early: tool_call_cap" in report
    assert "note_5" in report  # findings so far are in the partial report
    assert ReviewQueue(h.workspace("cust-418")).drafts() == []  # no drafts on a hard stop
    last_tool, final = state["messages"][-2:]
    assert (
        isinstance(last_tool, ToolMessage) and "stopped early (tool_call_cap)" in last_tool.content
    )
    assert isinstance(final, AIMessage) and "tool_call_cap" in final.content
    assert h.fakes["eda_coder"].remaining == 0 and h.fakes["synthesis"].remaining == 1


def test_same_tool_twice_without_change_stops(settings, retail_model, tmp_path, kitchen_files):
    h = Harness(settings, retail_model, tmp_path,
                [call("run_parsing", 1), call("run_eda", 2), call("run_eda", 3)],
                eda=[EMPTY_CODE])  # fmt: skip
    state = h.batch(kitchen_files)
    assert state["tool_history"] == ["run_parsing", "run_eda"]
    assert state["stop_reason"] == "no_progress"
    assert "(partial)" in Path(state["report_ref"]).read_text()


def test_token_budget_stops(settings, retail_model, tmp_path, kitchen_files):
    budget = settings.limits.batch.token_budget
    h = Harness(settings, retail_model, tmp_path, [call("run_parsing", 1), call("run_eda", 2)],
                tokens_per_call=budget // 2 + 1)  # fmt: skip
    state = h.batch(kitchen_files)
    assert state["tool_history"] == ["run_parsing"] and state["stop_reason"] == "token_budget"


def test_wall_clock_budget_stops(settings, retail_model, tmp_path, kitchen_files):
    times = itertools.chain([0.0, 1.0], itertools.repeat(301.0))
    h = Harness(settings, retail_model, tmp_path, [call("run_parsing", 1), call("run_eda", 2)],
                clock=lambda: next(times))  # fmt: skip
    state = h.batch(kitchen_files)
    assert state["tool_history"] == ["run_parsing"] and state["stop_reason"] == "time_budget"


def test_batch_reply_without_report_ends_partial(settings, retail_model, tmp_path, kitchen_files):
    h = Harness(settings, retail_model, tmp_path, ["All looks fine."])
    state = h.batch(kitchen_files)
    assert state["tool_history"] == [] and state["stop_reason"] == "ended_without_report"
    assert "(partial)" in Path(state["report_ref"]).read_text()


def test_only_the_first_of_parallel_tool_calls_runs(settings, retail_model, tmp_path,
                                                    kitchen_files):  # fmt: skip
    both = AIMessage(content="", tool_calls=[
        {"name": "run_parsing", "args": {}, "id": "call_a"},
        {"name": "run_eda", "args": {}, "id": "call_b"}])  # fmt: skip
    h = Harness(settings, retail_model, tmp_path, [both, "stop"])
    state = h.batch(kitchen_files)
    assert state["tool_history"] == ["run_parsing"]
    assert [m.tool_call_id for m in _tool_messages(state)] == ["call_a"]


# --- chat -------------------------------------------------------------------------------


def test_chat_turn_after_parsing_skips_parsing(settings, retail_model, tmp_path, kitchen_files):
    with open_checkpointer(tmp_path / "workspace" / "checkpoints.sqlite") as saver:
        h = Harness(settings, retail_model, tmp_path,
                    [call("run_parsing", 1), "Parsed: 6 issues found.",
                     "37 prices look like they were entered in cents.",
                     call("run_eda", 2, focus="prices by store"), "Prices are fine elsewhere."],
                    eda=[_note_code("unit_price_by_store")], checkpointer=saver)  # fmt: skip
        first = h.chat("Please check these files", uploads=kitchen_files)
        assert first["tool_history"] == ["run_parsing"] and first["stop_reason"] == "answered"
        second = h.chat("How many prices are wrong?")
        assert second["tool_history"] == [] and second["stop_reason"] == "answered"
        assert second["raw_files"] is None and second["tool_calls"] == 0
        assert all(isinstance(f, Finding) for f in second["findings"])
        prompt = h.fakes["orchestrator"].calls[2][0].content
        assert "Raw files pending: 0" in prompt and "unit_price implausible_value (37)" in prompt
        third = h.chat("Check prices by store")
        assert third["tool_history"] == ["run_eda"] and third["stop_reason"] == "answered"
        assert len(third["findings"]) == 7


def test_chat_without_data_asks_the_user(settings, retail_model, tmp_path):
    with open_checkpointer(tmp_path / "workspace" / "checkpoints.sqlite") as saver:
        h = Harness(settings, retail_model, tmp_path,
                    ["Please upload your sales, product and store files first."],
                    checkpointer=saver)  # fmt: skip
        state = h.chat("What is wrong with my data?")
    assert state["tool_history"] == [] and state["stop_reason"] == "needs_user"
    assert _tool_messages(state) == [] and state.get("report_ref") is None


def test_tool_error_returns_to_the_orchestrator(settings, retail_model, tmp_path):
    with open_checkpointer(tmp_path / "workspace" / "checkpoints.sqlite") as saver:
        script = [call("run_parsing", 1), "Please upload files first."]
        h = Harness(settings, retail_model, tmp_path, script, checkpointer=saver)
        state = h.chat("Parse my files")
    (error,) = _tool_messages(state)
    assert error.status == "error" and "no raw files are pending" in error.content
    assert state["tool_history"] == ["run_parsing"] and state["stop_reason"] == "needs_user"
