"""A scripted fake chat model for tests. It never calls a real LLM or the network.

Script items are consumed in order, one per model call:

- ``AIMessage``: returned as is (use ``tool_calls=[...]`` for tool calls).
- ``str``: returned as an ``AIMessage`` with that text.
- ``BaseModel`` instance: the parsed result of a structured-output call.
- ``dict``: validated against the structured-output schema; a validation failure is
  reported the way LangChain does with ``include_raw=True`` (``parsing_error`` set).
- ``Exception``: raised from the call.

Every call's input messages are recorded in ``calls`` so tests can assert on prompts.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, PrivateAttr, ValidationError

ScriptItem = AIMessage | str | BaseModel | dict | Exception


class ScriptExhaustedError(AssertionError):
    """The code under test made more model calls than the script provides."""


class FakeChatModel(BaseChatModel):
    """Returns queued responses; ``bind_tools`` and ``with_structured_output`` share the queue."""

    tokens_per_call: int = 0
    _script: list[ScriptItem] = PrivateAttr(default_factory=list)
    _calls: list[list[BaseMessage]] = PrivateAttr(default_factory=list)
    _bound_tools: list[str] = PrivateAttr(default_factory=list)

    def __init__(self, script: Sequence[ScriptItem] = (), **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._script = list(script)

    @property
    def _llm_type(self) -> str:
        return "fake-scripted"

    @property
    def calls(self) -> list[list[BaseMessage]]:
        return self._calls

    @property
    def bound_tools(self) -> list[str]:
        return self._bound_tools

    @property
    def remaining(self) -> int:
        return len(self._script)

    def queue(self, *items: ScriptItem) -> None:
        self._script.extend(items)

    def _next(self, messages: list[BaseMessage]) -> ScriptItem:
        self._calls.append(list(messages))
        if not self._script:
            raise ScriptExhaustedError(f"script exhausted after {len(self._calls) - 1} calls")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def _with_usage(self, message: AIMessage) -> AIMessage:
        if self.tokens_per_call and message.usage_metadata is None:
            half = self.tokens_per_call // 2
            usage = {"input_tokens": half, "output_tokens": self.tokens_per_call - half,
                     "total_tokens": self.tokens_per_call}  # fmt: skip
            message = message.model_copy(update={"usage_metadata": usage})
        return message

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        item = self._next(messages)
        if isinstance(item, str):
            item = AIMessage(content=item)
        if not isinstance(item, AIMessage):
            raise TypeError(f"plain call got a structured script item: {type(item).__name__}")
        return ChatResult(generations=[ChatGeneration(message=self._with_usage(item))])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> FakeChatModel:
        """Record the tool names and return self, so the shared script keeps driving it."""
        self._bound_tools = [getattr(t, "name", getattr(t, "__name__", str(t))) for t in tools]
        return self

    def with_structured_output(
        self, schema: Any, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable:
        """Return a runnable that pops the next item and parses it against ``schema``."""

        def run(messages: Any) -> Any:
            result = self._structured(schema, _as_messages(messages))
            if include_raw:
                return result
            if result["parsing_error"] is not None:
                raise result["parsing_error"]
            return result["parsed"]

        return RunnableLambda(run)

    def _structured(self, schema: type[BaseModel], messages: list[BaseMessage]) -> dict:
        item = self._next(messages)
        if isinstance(item, BaseModel):
            payload = item.model_dump(mode="json")
        elif isinstance(item, dict):
            payload = item
        else:
            raise TypeError(f"structured call got a plain script item: {type(item).__name__}")
        raw = self._with_usage(AIMessage(content=json.dumps(payload)))
        try:
            return {"raw": raw, "parsed": schema.model_validate(payload), "parsing_error": None}
        except ValidationError as exc:
            return {"raw": raw, "parsed": None, "parsing_error": exc}


def _as_messages(value: Any) -> list[BaseMessage]:
    if hasattr(value, "to_messages"):
        return value.to_messages()
    if isinstance(value, str):
        return [HumanMessage(content=value)]
    return list(value)


def tool_call(
    name: str, args: dict[str, Any] | None = None, call_id: str | None = None
) -> AIMessage:
    """Build an AIMessage that asks for one tool call."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args or {}, "id": call_id or f"call_{name}"}],
    )
