import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from fakes import FakeChatModel, ScriptExhaustedError, tool_call


class Mapping(BaseModel):
    column: str
    field: str | None


@tool
def run_parsing() -> str:
    """Parse the uploaded files."""
    return "ok"


def test_returns_queued_text_and_records_calls():
    model = FakeChatModel(["first", AIMessage(content="second")])
    assert model.invoke("hello").content == "first"
    assert model.invoke([HumanMessage(content="again")]).content == "second"
    assert [m[0].content for m in model.calls] == ["hello", "again"]
    assert model.remaining == 0


def test_tool_calls_through_bind_tools_share_the_script():
    model = FakeChatModel([tool_call("run_parsing"), "all done"])
    bound = model.bind_tools([run_parsing])
    first = bound.invoke("go")
    assert first.tool_calls[0]["name"] == "run_parsing"
    assert first.tool_calls[0]["id"] == "call_run_parsing"
    assert model.bound_tools == ["run_parsing"]
    assert bound.invoke("next").content == "all done"


def test_structured_output_parses_models_and_dicts():
    model = FakeChatModel([Mapping(column="Qty", field="quantity"), {"column": "X", "field": None}])
    runnable = model.with_structured_output(Mapping)
    assert runnable.invoke("map") == Mapping(column="Qty", field="quantity")
    assert runnable.invoke("map") == Mapping(column="X", field=None)


def test_structured_output_reports_validation_errors_with_include_raw():
    model = FakeChatModel([{"column": "Qty"}])
    result = model.with_structured_output(Mapping, include_raw=True).invoke("map")
    assert result["parsed"] is None and result["parsing_error"] is not None
    assert isinstance(result["raw"], AIMessage)


def test_exceptions_are_raised_and_exhaustion_is_loud():
    model = FakeChatModel([RuntimeError("boom")])
    with pytest.raises(RuntimeError, match="boom"):
        model.invoke("x")
    with pytest.raises(ScriptExhaustedError):
        model.invoke("y")


def test_token_usage_is_reported():
    model = FakeChatModel(["hi"], tokens_per_call=100)
    assert model.invoke("x").usage_metadata["total_tokens"] == 100


def test_queue_appends():
    model = FakeChatModel()
    model.queue("later")
    assert model.invoke("x").content == "later"
