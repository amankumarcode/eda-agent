"""Read data models from config/data_model/<name>.yaml."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from dva.data_model.models import CrossFieldRule, DataModel, FieldSpec, TableSpec
from dva.data_model.provider import DataModelNotFoundError, DataModelProvider

MODEL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class YamlDataModelProvider(DataModelProvider):
    """Loads one YAML file per data model from a directory."""

    def __init__(self, directory: Path) -> None:
        self._directory = Path(directory)

    def get_model(self, name: str) -> DataModel:
        """Read: <directory>/<name>.yaml. Returns: the validated DataModel. Writes nothing."""
        if not MODEL_NAME_PATTERN.match(name):
            raise DataModelNotFoundError(f"invalid data model name: {name!r}")
        path = self._directory / f"{name}.yaml"
        if not path.is_file():
            raise DataModelNotFoundError(f"no data model {name!r} in {self._directory}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        model = data_model_from_dict(raw)
        if model.name != name:
            raise ValueError(f"{path} declares name {model.name!r}, expected {name!r}")
        return model


def data_model_from_dict(raw: dict[str, Any]) -> DataModel:
    """Build a DataModel from the YAML structure (tables -> fields -> spec)."""
    tables = {
        table_name: TableSpec(
            name=table_name,
            fields={
                field_name: FieldSpec(name=field_name, table=table_name, **spec)
                for field_name, spec in table["fields"].items()
            },
        )
        for table_name, table in raw["tables"].items()
    }
    rules = tuple(CrossFieldRule.parse(rule) for rule in raw.get("cross_field_rules") or [])
    return DataModel(name=raw["name"], tables=tables, cross_field_rules=rules)
