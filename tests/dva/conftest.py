"""Shared fixtures for the DVA test suite."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from dva.config import Settings, load_settings
from dva.data_model import DataModel, make_provider

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def settings() -> Settings:
    return load_settings(SETTINGS_PATH, load_env=False)


@pytest.fixture(scope="session")
def retail_model(settings: Settings) -> DataModel:
    return make_provider(settings).get_model("retail_sales")


@pytest.fixture(scope="session")
def fixtures_dir(settings: Settings) -> Path:
    return settings.paths.fixtures_dir


@pytest.fixture(scope="session")
def generator() -> ModuleType:
    """The fixture generator script, imported as a module."""
    name = "generate_fixtures"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
