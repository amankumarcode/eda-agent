"""DataModelProvider interface and the factory that picks one from settings."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from dva.data_model.models import DataModel

if TYPE_CHECKING:
    from dva.config import Settings


class DataModelNotFoundError(LookupError):
    """Raised when a provider has no data model with the requested name."""


class DataModelProvider(ABC):
    """Source of data models, such as YAML files or an MCP server."""

    @abstractmethod
    def get_model(self, name: str) -> DataModel:
        """Return the named data model or raise DataModelNotFoundError."""


def make_provider(settings: Settings) -> DataModelProvider:
    """Build the provider selected by settings.data_model.provider.

    Reads: settings only. Writes nothing.
    """
    if settings.data_model.provider == "yaml":
        from dva.data_model.yaml_provider import YamlDataModelProvider

        return YamlDataModelProvider(settings.paths.data_model_dir)
    from dva.data_model.mcp_provider import McpDataModelProvider, stdio_parameters

    mcp = settings.data_model.mcp
    assert mcp is not None  # guaranteed by DataModelSettings validation
    server = mcp.url or stdio_parameters(mcp.command or sys.executable, list(mcp.args),
                                         str(settings.paths.project_root))  # fmt: skip
    return McpDataModelProvider(server, timeout_seconds=mcp.timeout_seconds)
