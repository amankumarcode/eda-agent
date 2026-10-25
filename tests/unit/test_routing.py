from langchain_core.messages import AIMessage, HumanMessage

from dva.orchestrator.graph import keep_first_tool_call
from dva.orchestrator.routing import decide, finish_reason, state_hash, synthesis_block_reason
from dva.state import Finding
from fakes import tool_call


def _state(**overrides):
    state = {
        "mode": "batch", "required_checks": ["a:present"], "checks_completed": ["a:present"],
        "tool_history": ["run_parsing", "run_eda"], "tool_calls": 2, "tokens_used": 0,
        "started_at": 0.0, "last_tool": "run_eda", "last_state_hash": None,
        "parsed_data_ref": {"sales": "x"}, "findings": [], "messages": [],
    }  # fmt: skip
    return {**state, **overrides}


def test_hash_ignores_messages_and_turn_fields_but_sees_findings(settings):
    base = _state()
    same = _state(messages=[HumanMessage("hi")], tool_calls=5, tokens_used=99, started_at=7.0)
    assert state_hash(base) == state_hash(same)
    finding = Finding(field="a", issue_type="other", severity="low", detail="d", count=1,
                      source="eda")  # fmt: skip
    assert state_hash(base) != state_hash(_state(findings=[finding]))


def test_synthesis_gate(settings):
    assert synthesis_block_reason(_state()) is None
    assert "required checks remain: b:join" in synthesis_block_reason(
        _state(required_checks=["a:present", "b:join"]))  # fmt: skip
    assert "run_eda must run" in synthesis_block_reason(_state(tool_history=["run_parsing"]))
    assert synthesis_block_reason(_state(mode="chat", tool_history=[])) is None


def test_hard_stops_in_order(settings):
    limits = settings.limits.batch
    message = tool_call("run_eda")
    over_time = decide(_state(tool_calls=99, tokens_used=10**9), message, limits, now=10_000)
    assert over_time["stop_reason"] == "time_budget"
    assert decide(_state(tool_calls=99, tokens_used=10**9), message, limits, 1)["stop_reason"] == (
        "token_budget")  # fmt: skip
    assert decide(_state(tool_calls=6), message, limits, 1)["stop_reason"] == "tool_call_cap"
    frozen = _state()
    frozen["last_state_hash"] = state_hash(frozen)
    assert decide(frozen, message, limits, 1)["stop_reason"] == "no_progress"
    assert decide(frozen, tool_call("run_synthesis"), limits, 1)["next_step"] == "tools"


def test_decide_counts_and_records_calls(settings):
    update = decide(_state(), tool_call("run_synthesis"), settings.limits.batch, 1)
    assert update["next_step"] == "tools" and update["tool_calls"] == 3
    assert update["tool_history"][-1] == "run_synthesis" and update["last_tool"] == "run_synthesis"
    blocked = decide(_state(tool_history=[]), tool_call("run_synthesis"), settings.limits.batch, 1)
    assert blocked["next_step"] == "reject_synthesis"
    assert blocked["tool_history"] == ["run_synthesis (rejected)"] and blocked["tool_calls"] == 3


def test_finish_reasons():
    assert finish_reason(_state(tool_history=["run_synthesis"], report_ref="r.md")) == (
        "end", "task_complete")  # fmt: skip
    assert finish_reason(_state()) == ("stop_partial", "ended_without_report")
    assert finish_reason(_state(mode="chat")) == ("end", "answered")
    assert finish_reason(_state(mode="chat", parsed_data_ref={}, raw_files=None)) == (
        "end", "needs_user")  # fmt: skip


def test_keep_first_tool_call_drops_extra_tool_use_blocks():
    message = AIMessage(
        content=[{"type": "text", "text": "ok"},
                 {"type": "tool_use", "id": "a", "name": "run_parsing", "input": {}},
                 {"type": "tool_use", "id": "b", "name": "run_eda", "input": {}}],
        tool_calls=[{"name": "run_parsing", "args": {}, "id": "a"},
                    {"name": "run_eda", "args": {}, "id": "b"}],
    )  # fmt: skip
    kept = keep_first_tool_call(message)
    assert [c["id"] for c in kept.tool_calls] == ["a"]
    assert [b.get("id") for b in kept.content if isinstance(b, dict)] == [None, "a"]
