from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from dva.llm import StructuredOutputError, create_chat_model, invoke_structured
from fakes import FakeChatModel


class Answer(BaseModel):
    value: int


def test_create_chat_model_sends_no_temperature(settings):
    with patch("dva.llm.init_chat_model") as init:
        create_chat_model(settings, "column_mapper")
    init.assert_called_once_with(
        "claude-haiku-5-5", model_provider="anthropic", max_tokens=8000, effort="low"
    )


def test_create_chat_model_passes_temperature_when_configured(settings):
    llm = settings.llm.model_copy(update={"temperature": 0.0})
    with patch("dva.llm.init_chat_model") as init:
        create_chat_model(settings.model_copy(update={"llm": llm}), "eda_coder")
    assert init.call_args.kwargs["temperature"] == 0.0


def test_valid_first_answer():
    model = FakeChatModel([Answer(value=3)], tokens_per_call=10)
    result = invoke_structured(model, Answer, [HumanMessage("q")], retries=1)
    assert result.parsed == Answer(value=3)
    assert (result.attempts, result.first_pass_valid, result.tokens_used) == (1, True, 10)


def test_schema_error_is_retried_with_the_error_message():
    model = FakeChatModel([{"value": "not a number"}, {"value": 4}], tokens_per_call=10)
    result = invoke_structured(model, Answer, [HumanMessage("q")], retries=1)
    assert result.parsed.value == 4
    assert (result.attempts, result.first_pass_valid, result.tokens_used) == (2, False, 20)
    retry_prompt = model.calls[1][-1].content
    assert "invalid" in retry_prompt and "value" in retry_prompt


def test_semantic_validator_triggers_retry():
    def must_be_even(answer: Answer) -> None:
        if answer.value % 2:
            raise ValueError("value must be even")

    model = FakeChatModel([{"value": 3}, {"value": 8}])
    result = invoke_structured(model, Answer, [HumanMessage("q")], retries=1,
                               validate=must_be_even)  # fmt: skip
    assert result.parsed.value == 8
    assert "value must be even" in model.calls[1][-1].content


def test_second_failure_raises_clear_error():
    model = FakeChatModel([{"value": "x"}, {"wrong": 1}])
    with pytest.raises(StructuredOutputError, match="Answer: no valid output after 2 attempts"):
        invoke_structured(model, Answer, [HumanMessage("q")], retries=1)
    assert model.remaining == 0
