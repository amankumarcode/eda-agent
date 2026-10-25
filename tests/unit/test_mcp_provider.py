import pytest
from mcp.server.mcpserver import MCPServer
from pydantic import ValidationError

from dva.config import DataModelSettings
from dva.data_model import DataModelNotFoundError, make_provider
from dva.data_model.mcp_provider import DataModelProviderError, McpDataModelProvider
from mcp_mock_server import build_server


@pytest.fixture
def in_process(settings):
    return McpDataModelProvider(build_server(settings.paths.data_model_dir))


def _mcp_settings(settings, **mcp_overrides):
    mcp = settings.data_model.mcp.model_copy(update=mcp_overrides)
    data_model = settings.data_model.model_copy(update={"provider": "mcp", "mcp": mcp})
    return settings.model_copy(update={"data_model": data_model})


def test_mcp_returns_the_same_model_as_yaml_in_process(in_process, retail_model):
    model = in_process.get_model("retail_sales")
    assert model == retail_model
    assert model.required_checks() == retail_model.required_checks()


def test_mcp_returns_the_same_model_over_stdio(settings, retail_model):
    provider = make_provider(_mcp_settings(settings))
    assert isinstance(provider, McpDataModelProvider)
    assert provider.get_model("retail_sales") == retail_model


def test_lookup_field_returns_each_tables_spec(in_process):
    specs = in_process.lookup_field("retail_sales", "sku")
    assert [(s.table, s.join) for s in specs] == [("sales", "products.sku"), ("products", None)]
    assert in_process.lookup_field("retail_sales", "nope") == []


def test_unknown_and_invalid_names(in_process):
    with pytest.raises(DataModelNotFoundError, match="not found"):
        in_process.get_model("no_such_model")
    with pytest.raises(DataModelNotFoundError, match="invalid data model name"):
        in_process.get_model("../settings")


def test_server_returning_a_different_model_is_rejected(settings, project_root):
    import yaml

    server = MCPServer("wrong")

    @server.tool()
    def get_data_model(name: str) -> dict:
        """Always returns retail_sales, whatever is asked."""
        return yaml.safe_load((project_root / "config/data_model/retail_sales.yaml").read_text())

    with pytest.raises(DataModelProviderError, match="returned model 'retail_sales'"):
        McpDataModelProvider(server).get_model("other_model")


def test_unreachable_server_raises_provider_error(settings):
    provider = make_provider(_mcp_settings(settings, command="/nonexistent/server-binary"))
    with pytest.raises(DataModelProviderError, match="MCP call get_data_model failed"):
        provider.get_model("retail_sales")


def test_mcp_provider_requires_its_settings():
    with pytest.raises(ValidationError, match="data_model.mcp is required"):
        DataModelSettings(provider="mcp", default="retail_sales")
    assert DataModelSettings(provider="yaml", default="retail_sales").mcp is None


def test_url_setting_is_passed_through(settings):
    provider = make_provider(_mcp_settings(settings, url="http://localhost:9/mcp"))
    assert provider._server == "http://localhost:9/mcp"  # no connection is made here
