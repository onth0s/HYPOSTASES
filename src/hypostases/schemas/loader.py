"""HYPOSTASES Schemas — YAML Schema Loaders & Managers.

Spec Ref: Part I §2.1, Part IV §6.
Ground truth YAML definitions live in /schema/.
This module provides machine-readable access to schemas and invariants.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from hypostases.utils import find_project_root


def get_schema_dir() -> Path:
    """Locates the /schema/ directory relative to package root or project root."""
    project_root = find_project_root()
    schema_path = project_root / "schema"
    if schema_path.is_dir() and (schema_path / "schema_v1.yaml").exists():
        return schema_path
    raise FileNotFoundError("Could not locate ground-truth /schema/ directory")


@lru_cache(maxsize=16)
def load_yaml(filename: str) -> dict:
    """Loads and parses a YAML schema file from /schema/."""
    schema_dir = get_schema_dir()
    filepath = schema_dir / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Schema file not found: {filepath}")

    with open(filepath, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_schema(name: str = "schema_v1") -> dict:
    """Loads a named schema definition (defaults to schema_v1.yaml)."""
    filename = name if name.endswith(".yaml") else f"{name}.yaml"
    return load_yaml(filename)


def load_invariants() -> dict:
    """Loads cross-cutting constraints from invariants.yaml."""
    return load_yaml("invariants.yaml")


def load_components() -> dict:
    """Loads component taxonomy from components.yaml."""
    return load_yaml("components.yaml")


def load_time_model() -> dict:
    """Loads time tier definitions from time_model.yaml."""
    return load_yaml("time_model.yaml")


def load_update_dynamics() -> dict:
    """Loads update dynamics definitions from update_dynamics.yaml."""
    return load_yaml("update_dynamics.yaml")


def load_memory_preset(preset_filename: str = "memory/memory_preset.yaml") -> dict:
    """Loads memory architecture configuration preset."""
    return load_yaml(preset_filename)


def load_skill_artifact_schema(schema_filename: str = "memory/skill_artifact_schema.yaml") -> dict:
    """Loads skill artifact schema definition."""
    return load_yaml(schema_filename)


def load_planning_config(config_filename: str = "planning_config.yaml") -> dict:
    """Loads explicit planning layer configuration."""
    return load_yaml(config_filename)


def load_hierarchical_world_model_config(
    config_filename: str = "hierarchical_world_model_config.yaml",
) -> dict:
    """Loads hierarchical world model and conceptual spaces configuration."""
    return load_yaml(config_filename)
