"""Tests for the chess experiment config loader and precedence resolution.

Covers Rule 006 data-driven YAML defaults and the CLI > checkpoint > YAML >
defaults precedence used by ``hypostases train --chess``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hypostases.plugins.domains.chess.chess_config import (
    load_chess_experiment_config,
    resolve_chess_training_config,
)


def test_loader_reads_plugin_config() -> None:
    """The loader must resolve chess_experiment_config.yaml next to the plugin."""
    cfg = load_chess_experiment_config()
    assert cfg.get("experiment_id") == "chess_dual_testing_grounds_v1"
    assert cfg["training"]["generations"] == 48
    assert cfg["training"]["games_per_generation"] == 32
    assert cfg["ground_a"]["snapshot_interval_k"] == 8
    assert cfg["ground_a"]["parallel_workers"] == 20


def test_loader_missing_path_returns_empty(tmp_path: Path) -> None:
    assert load_chess_experiment_config(tmp_path / "missing.yaml") == {}


def test_resolve_precedence_cli_over_yaml_over_defaults() -> None:
    """CLI > YAML > defaults precedence."""
    raw: dict[str, Any] = {
        "training": {"generations": 48, "games_per_generation": 32},
        "ground_a": {"snapshot_interval_k": 8, "parallel_workers": 20},
        "pre_registration": {"seeds": [42, 69]},
    }

    resolved = resolve_chess_training_config(raw_config=raw)
    assert resolved["total_generations"] == 48
    assert resolved["games_per_generation"] == 32
    assert resolved["snapshot_interval_k"] == 8
    assert resolved["seed"] == 42
    assert resolved["workers"] == 20

    cli = {"gens": 10, "games": 4, "snapshot_interval": 2, "seed": 7, "workers": 3}
    overridden = resolve_chess_training_config(cli=cli, raw_config=raw)
    assert overridden["total_generations"] == 10
    assert overridden["games_per_generation"] == 4
    assert overridden["snapshot_interval_k"] == 2
    assert overridden["seed"] == 7
    assert overridden["workers"] == 3


def test_resolve_falls_back_to_defaults_without_yaml() -> None:
    """With no YAML, trainer defaults are used."""
    resolved = resolve_chess_training_config()
    assert resolved["total_generations"] == 20
    assert resolved["games_per_generation"] == 15
    assert resolved["workers"] == 20
    assert resolved["learning_rate"] == 0.01
    assert resolved["beta_efe"] == 0.05
