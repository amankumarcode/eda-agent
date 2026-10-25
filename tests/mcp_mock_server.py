"""Mock MCP server that serves data models from YAML files, for tests and local runs.

Tools:
- get_data_model(name): the data model as a dict (the same structure as the YAML file)
- lookup_field(data_model, field): {"matches": [...]}, every table's spec for one field

Errors are raised as ToolError so their message reaches the client (other exceptions are
reported only as "Error executing tool").

Run over stdio: python tests/mcp_mock_server.py --data-model-dir config/data_model
Built with the official mcp Python SDK (2.x, where FastMCP is now MCPServer).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _load(directory: Path, name: str) -> dict[str, Any]:
    if not NAME_PATTERN.match(name):
        raise ToolError(f"invalid data model name: {name!r}")
    path = directory / f"{name}.yaml"
    if not path.is_file():
        raise ToolError(f"data model {name!r} not found")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_server(directory: Path) -> MCPServer:
    """An MCP server exposing get_data_model and lookup_field over ``directory``."""
    server = MCPServer("dva-mock-data-models")

    @server.tool()
    def get_data_model(name: str) -> dict[str, Any]:
        """Return the named data model: tables, fields and cross-field rules."""
        return _load(directory, name)

    @server.tool()
    def lookup_field(data_model: str, field: str) -> dict[str, Any]:
        """Return the spec of one field in every table that defines it."""
        raw = _load(directory, data_model)
        matches = [{"table": table, "field": field, "spec": body["fields"][field]}
                   for table, body in raw["tables"].items() if field in body["fields"]]  # fmt: skip
        return {"matches": matches}

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-model-dir", type=Path, default=Path("config/data_model"))
    args = parser.parse_args()
    build_server(args.data_model_dir.resolve()).run("stdio")


if __name__ == "__main__":
    main()
