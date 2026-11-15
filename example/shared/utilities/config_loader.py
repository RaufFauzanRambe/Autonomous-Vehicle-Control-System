"""Configuration loading helpers shared by every example module.

Conventions used across the project:

* YAML (PyYAML) is the default format for module configuration files.
* JSON is used for maps, schemas and scenario definitions.
* Paths inside a configuration file are interpreted relative to the
  module directory that owns the file (each ``main.py`` resolves them).
* ``deep_update`` lets a module overlay CLI-provided overrides on top of
  file defaults without mutating the original dictionaries.

Extension point: replace the plain ``dict`` return values with a
validated ``pydantic``/``dataclass`` configuration object if stricter
schema checking is required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file into a plain dictionary.

    Args:
        path: Path to a ``.yaml`` / ``.yml`` file.

    Returns:
        Parsed configuration dictionary. An empty file yields ``{}``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load a JSON file into a plain dictionary.

    Args:
        path: Path to a ``.json`` file.

    Returns:
        Parsed JSON content as a dictionary.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` and return a new dictionary.

    Nested dictionaries are merged key by key; any other type in
    ``override`` (scalar, list, ``None``) replaces the value in ``base``.
    Neither input dictionary is mutated.

    Args:
        base: Default configuration dictionary.
        override: Overrides, e.g. values parsed from CLI arguments.

    Returns:
        A new merged dictionary.

    Example:
        >>> deep_update({"a": {"b": 1}}, {"a": {"c": 2}})
        {'a': {'b': 1, 'c': 2}}
    """
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged
