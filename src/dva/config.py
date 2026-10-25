"""Load settings from config/settings.yaml and secrets from .env.

Settings never contain secrets: API keys stay in the environment, where the
provider packages read them.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

CONFIG_ENV_VAR = "DVA_CONFIG"
DEFAULT_CONFIG_PATH = Path("config/settings.yaml")
LOG_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s [session=%(session_id)s phase=%(phase)s] %(message)s"
)

Role = Literal["orchestrator", "column_mapper", "eda_coder", "synthesis", "eval_matcher"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelSettings(_Strict):
    provider: str
    model: str
    max_tokens: PositiveInt


class ModelsSettings(_Strict):
    orchestrator: ModelSettings
    column_mapper: ModelSettings
    eda_coder: ModelSettings
    synthesis: ModelSettings
    eval_matcher: ModelSettings

    def for_role(self, role: Role) -> ModelSettings:
        """Return the model settings for one role."""
        return getattr(self, role)


class LlmSettings(_Strict):
    temperature: float | None = None
    structured_output_method: Literal["json_schema", "function_calling"] = "json_schema"
    structured_output_retries: int = Field(ge=0, le=3)


class RunLimits(_Strict):
    max_tool_calls: PositiveInt
    wall_clock_seconds: PositiveInt
    token_budget: PositiveInt


class LimitsSettings(_Strict):
    batch: RunLimits
    chat: RunLimits


class SandboxSettings(_Strict):
    timeout_seconds: PositiveInt
    memory_limit_mb: PositiveInt
    output_char_limit: PositiveInt
    allowed_imports: tuple[str, ...]
    blocked_calls: tuple[str, ...]
    blocked_modules: tuple[str, ...]


class DataModelSettings(_Strict):
    provider: Literal["yaml", "mcp"]
    default: str


class PathSettings(_Strict):
    workspace_root: Path
    data_model_dir: Path
    checkpoint_db: Path
    fixtures_dir: Path
    eval_results_dir: Path


class LoggingSettings(_Strict):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class Settings(_Strict):
    models: ModelsSettings
    llm: LlmSettings
    limits: LimitsSettings
    sandbox: SandboxSettings
    data_model: DataModelSettings
    paths: PathSettings
    logging: LoggingSettings


def resolve_config_path(path: Path | str | None = None) -> Path:
    """Return the settings file to use.

    Reads: the explicit ``path``, else the DVA_CONFIG env var, else config/settings.yaml
    relative to the current directory. Writes nothing.
    """
    chosen = path or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH
    return Path(chosen).expanduser().resolve()


def load_settings(path: Path | str | None = None, *, load_env: bool = True) -> Settings:
    """Load and validate settings.

    Reads: .env (into the process environment, without overriding existing variables)
    and the settings YAML. Returns: validated Settings with paths made absolute against
    the project root (the parent of the config directory). Writes nothing.
    """
    config_path = resolve_config_path(path)
    if load_env:
        load_dotenv(config_path.parent.parent / ".env", override=False)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = Settings.model_validate(raw)
    return _with_absolute_paths(settings, config_path.parent.parent)


def _with_absolute_paths(settings: Settings, project_root: Path) -> Settings:
    resolved = {
        name: (value if value.is_absolute() else (project_root / value).resolve())
        for name, value in settings.paths.model_dump().items()
    }
    return settings.model_copy(update={"paths": PathSettings(**resolved)})


def configure_logging(settings: Settings) -> None:
    """Configure root logging with session_id and phase on every line.

    Reads: settings.logging.level. Writes nothing to disk.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.addFilter(_DefaultContextFilter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.logging.level)


def session_logger(name: str, session_id: str, phase: str) -> logging.LoggerAdapter:
    """Return a logger that stamps session_id and phase on each record."""
    return logging.LoggerAdapter(
        logging.getLogger(name), {"session_id": session_id, "phase": phase}
    )


class _DefaultContextFilter(logging.Filter):
    """Fill session_id/phase for records logged outside a session logger."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.session_id = getattr(record, "session_id", "-")
        record.phase = getattr(record, "phase", "-")
        return True
