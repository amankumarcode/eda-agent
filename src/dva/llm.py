"""Chat model factory and structured-output calls with one validation retry."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Generic, TypeVar

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, ConfigDict

from dva.config import Role, Settings

T = TypeVar("T", bound=BaseModel)

RETRY_PROMPT = (
    "Your previous output was invalid:\n{error}\n"
    "Return a corrected output that satisfies the schema and the instructions."
)


class StructuredOutputError(RuntimeError):
    """The model did not produce valid structured output within the allowed attempts."""


class StructuredResult(BaseModel, Generic[T]):
    """A validated structured output plus bookkeeping for budgets and evals."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    parsed: T
    attempts: int
    first_pass_valid: bool
    tokens_used: int


def create_chat_model(settings: Settings, role: Role) -> BaseChatModel:
    """Create the chat model configured for a role with LangChain's init_chat_model.

    Reads: settings.models and settings.llm. Writes nothing. Sends no sampling parameter
    unless settings.llm.temperature is set.
    """
    spec = settings.models.for_role(role)
    kwargs: dict = {"max_tokens": spec.max_tokens}
    if settings.llm.temperature is not None:
        kwargs["temperature"] = settings.llm.temperature
    return init_chat_model(spec.model, model_provider=spec.provider, **kwargs)


def invoke_structured(
    model: BaseChatModel,
    schema: type[T],
    messages: Sequence[BaseMessage],
    *,
    retries: int,
    method: str = "json_schema",
    validate: Callable[[T], None] | None = None,
) -> StructuredResult[T]:
    """Call the model for structured output; on a validation failure, retry with the error.

    ``validate`` may raise ValueError for semantic problems the schema cannot express.
    Returns: the parsed result. Raises StructuredOutputError after ``retries`` retries.
    Writes nothing.
    """
    runnable = model.with_structured_output(schema, method=method, include_raw=True)
    history, tokens, error = list(messages), 0, ""
    for attempt in range(1, retries + 2):
        result = runnable.invoke(history)
        raw: AIMessage | None = result.get("raw")
        tokens += _total_tokens(raw)
        error = _validation_error(result, validate)
        if not error:
            return StructuredResult(parsed=result["parsed"], attempts=attempt,
                                    first_pass_valid=attempt == 1, tokens_used=tokens)  # fmt: skip
        history += [AIMessage(content=_text(raw)), HumanMessage(RETRY_PROMPT.format(error=error))]
    raise StructuredOutputError(
        f"{schema.__name__}: no valid output after {retries + 1} attempts; last error: {error}"
    )


def _validation_error(result: dict, validate: Callable | None) -> str:
    if result.get("parsing_error") is not None:
        return str(result["parsing_error"])
    if result.get("parsed") is None:
        return "no structured output was returned"
    if validate is not None:
        try:
            validate(result["parsed"])
        except ValueError as exc:
            return str(exc)
    return ""


def _total_tokens(raw: AIMessage | None) -> int:
    usage = getattr(raw, "usage_metadata", None) or {}
    return int(usage.get("total_tokens", 0))


def _text(raw: AIMessage | None) -> str:
    if raw is None:
        return ""
    return raw.content if isinstance(raw.content, str) else str(raw.content)
