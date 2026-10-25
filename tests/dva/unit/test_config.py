import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from dva.config import load_settings, resolve_config_path, session_logger


def test_settings_load_with_roles_and_limits(settings):
    assert settings.models.orchestrator.model == "claude-sonnet-5-5"
    assert settings.models.column_mapper.model == "claude-haiku-5-5"
    assert settings.models.eda_coder.model == "claude-opus-5-5"
    assert settings.models.synthesis.model == "claude-sonnet-5-5"
    assert settings.models.for_role("eval_matcher").model == "claude-haiku-5-5"
    assert settings.llm.temperature is None
    assert settings.llm.structured_output_method == "json_schema"
    assert settings.limits.batch.max_tool_calls == 6
    assert settings.limits.chat.max_tool_calls == 4
    assert settings.limits.batch.wall_clock_seconds == 300
    assert settings.limits.chat.wall_clock_seconds == 120
    assert settings.sandbox.timeout_seconds == 60
    assert settings.sandbox.memory_limit_mb == 1024
    assert settings.sandbox.output_char_limit == 4000
    assert "pandas" in settings.sandbox.allowed_imports
    assert settings.data_model.provider == "yaml"


def test_paths_are_absolute_against_project_root(settings, project_root):
    assert settings.paths.workspace_root == project_root / "workspace"
    assert settings.paths.checkpoint_db == project_root / "workspace" / "checkpoints.sqlite"
    assert settings.paths.data_model_dir == project_root / "config" / "data_model"
    assert all(Path(p).is_absolute() for p in settings.paths.model_dump().values())


def test_settings_are_immutable(settings):
    with pytest.raises(ValidationError):
        settings.limits.batch.max_tool_calls = 99


def _write_settings(tmp_path, project_root, mutate):
    raw = yaml.safe_load((project_root / "config" / "settings.yaml").read_text())
    mutate(raw)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    path = config_dir / "settings.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_unknown_keys_are_rejected(tmp_path, project_root):
    path = _write_settings(tmp_path, project_root, lambda raw: raw["limits"].update(extra=1))
    with pytest.raises(ValidationError):
        load_settings(path, load_env=False)


def test_invalid_limit_is_rejected(tmp_path, project_root):
    def zero_cap(raw):
        raw["limits"]["batch"]["max_tool_calls"] = 0

    path = _write_settings(tmp_path, project_root, zero_cap)
    with pytest.raises(ValidationError):
        load_settings(path, load_env=False)


def test_env_var_selects_config_and_dotenv_is_loaded(tmp_path, project_root, monkeypatch):
    path = _write_settings(tmp_path, project_root, lambda raw: None)
    (tmp_path / ".env").write_text("DVA_TEST_ONLY_VAR=from-dotenv\n")
    monkeypatch.setenv("DVA_CONFIG", str(path))
    monkeypatch.delenv("DVA_TEST_ONLY_VAR", raising=False)
    assert resolve_config_path() == path.resolve()
    settings = load_settings()
    assert settings.paths.workspace_root == tmp_path / "workspace"
    assert os.environ["DVA_TEST_ONLY_VAR"] == "from-dotenv"
    monkeypatch.delenv("DVA_TEST_ONLY_VAR")


def test_settings_hold_no_secrets(settings):
    dumped = str(settings.model_dump()).lower()
    assert "api_key" not in dumped and "sk-ant" not in dumped


def test_session_logger_stamps_context(caplog):
    logger = session_logger("dva.test", "cust-1", "parsing")
    with caplog.at_level("INFO", logger="dva.test"):
        logger.info("hello")
    record = caplog.records[-1]
    assert record.session_id == "cust-1" and record.phase == "parsing"
