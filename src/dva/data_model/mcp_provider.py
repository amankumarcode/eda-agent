"""Read data models from an MCP server exposing get_data_model and lookup_field.

The server returns the same structure as a YAML data model file, so it is validated by
the same code as the YAML provider. Calls are synchronous wrappers around the async MCP
client; do not call them from inside a running event loop.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from mcp import Client, StdioServerParameters

from dva.data_model.models import DataModel, FieldSpec
from dva.data_model.provider import DataModelNotFoundError, DataModelProvider
from dva.data_model.yaml_provider import data_model_from_dict

MODEL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class DataModelProviderError(RuntimeError):
    """The MCP server could not be reached or returned something unusable."""


class McpDataModelProvider(DataModelProvider):
    """Data models from an MCP server: a URL, stdio launch parameters, or (in tests) an
    in-process server object."""

    def __init__(self, server: Any, timeout_seconds: float = 30.0) -> None:
        self._server = server
        self._timeout = timeout_seconds

    def get_model(self, name: str) -> DataModel:
        """Call get_data_model and validate the result. Reads: the server. Writes nothing."""
        if not MODEL_NAME_PATTERN.match(name):
            raise DataModelNotFoundError(f"invalid data model name: {name!r}")
        raw = self._call("get_data_model", {"name": name})
        model = data_model_from_dict(raw)
        if model.name != name:
            raise DataModelProviderError(f"server returned model {model.name!r} for {name!r}")
        return model

    def lookup_field(self, data_model: str, field: str) -> list[FieldSpec]:
        """Call lookup_field, which returns {"matches": [{table, field, spec}, ...]}."""
        result = self._call("lookup_field", {"data_model": data_model, "field": field})
        return [FieldSpec(name=e["field"], table=e["table"], **e["spec"])
                for e in result["matches"]]  # fmt: skip

    def _call(self, tool: str, arguments: dict[str, Any]) -> Any:
        try:
            return asyncio.run(self._call_async(tool, arguments))
        except (DataModelNotFoundError, DataModelProviderError):
            raise
        except Exception as exc:  # transport and protocol errors from the SDK
            raise DataModelProviderError(f"MCP call {tool} failed: {exc}") from exc

    async def _call_async(self, tool: str, arguments: dict[str, Any]) -> Any:
        async with Client(self._server, read_timeout_seconds=self._timeout) as client:
            result = await client.call_tool(tool, arguments)
        text = "".join(getattr(block, "text", "") for block in result.content)
        if result.is_error:
            raise DataModelNotFoundError(f"MCP server: {text}")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise DataModelProviderError(f"MCP {tool} did not return JSON") from exc


def stdio_parameters(command: str, args: list[str], cwd: str) -> StdioServerParameters:
    """Launch parameters for an MCP server spoken to over stdin/stdout."""
    return StdioServerParameters(command=command, args=args, cwd=cwd)
