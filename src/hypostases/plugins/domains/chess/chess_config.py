"""Chess experiment configuration loader (Rule 006 data-driven YAML).

Loads ``chess_experiment_config.yaml`` (pre-registered experiment parameters) from the
plugin directory so it is independent of the process working directory. The ``train``
CLI path resolves its defaults from this file with precedence
``CLI > checkpoint > YAML > trainer defaults``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent / "chess_experiment_config.yaml"


def load_chess_experiment_config(
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Loads the chess experiment configuration from YAML.

    Args:
        config_path: Optional explicit path. When omitted, resolves
            ``chess_experiment_config.yaml`` next to this module.

    Returns:
        The parsed configuration dictionary. Returns an empty dict when the file
        cannot be located, so callers fall back to their own defaults.
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_chess_training_config(
    cli: dict[str, Any] | None = None,
    loaded_checkpoint: dict[str, Any] | None = None,
    raw_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolves effective training parameters under CLI > checkpoint > YAML > defaults.

    Args:
        cli: Explicit CLI-provided values (only keys the user actually set).
        loaded_checkpoint: Checkpoint-restored agent state on resume
            (``theta_meta``, ``temperature``, ``beta_efe``, ``target_generations``).
        raw_config: Parsed chess_experiment_config.yaml (empty dict if unavailable).

    Returns:
        A flat effective configuration used by ``run_chess_training``.
    """
    cli = cli or {}
    ckpt = loaded_checkpoint or {}
    cfg = raw_config or {}
    t_cfg = cfg.get("training", {})
    g_a_cfg = cfg.get("ground_a", {})
    pre_reg = cfg.get("pre_registration", {})

    def pick(cli_key: str, yaml_val: Any, default: Any) -> Any:
        if cli.get(cli_key) is not None:
            return cli[cli_key]
        if yaml_val is not None:
            return yaml_val
        return default

    seeds = pre_reg.get("seeds") or [42]

    return {
        "total_generations": int(pick("gens", t_cfg.get("generations"), 20)),
        "games_per_generation": int(pick("games", t_cfg.get("games_per_generation"), 15)),
        "snapshot_interval_k": int(
            pick(
                "snapshot_interval",
                t_cfg.get("snapshot_interval_k", g_a_cfg.get("snapshot_interval_k")),
                5,
            )
        ),
        "seed": int(pick("seed", seeds[0], 42)),
        "workers": int(pick("workers", g_a_cfg.get("parallel_workers"), 20)),
        "learning_rate": float(t_cfg.get("learning_rate", 0.01)),
        "beta_efe": float(t_cfg.get("efe_beta", 0.05)),
        "initial_temperature": float(t_cfg.get("policy_temperature", 0.8)),
        "value_gamma": float(t_cfg.get("value_gamma", 0.97)),
        "min_temperature": float(t_cfg.get("min_temperature", 0.20)),
        "max_moves_training": int(t_cfg.get("max_moves_training", 400)),
        "early_adjudication_material": float(t_cfg.get("early_adjudication_material", 15.0)),
        "initial_priors": t_cfg.get("initial_priors", "random"),
        "curriculum_probability": float(t_cfg.get("curriculum_probability", 0.25)),
        "nnue_epochs": int(t_cfg.get("nnue_epochs", 30)),
        "nnue_learning_rate": float(t_cfg.get("nnue_learning_rate", 0.003)),
        "replay_capacity": int(t_cfg.get("replay_capacity", 6000)),
        "search_depth": int(t_cfg.get("search_depth", 3)),
        "resign_value_threshold": float(t_cfg.get("resign_value_threshold", -3.0)),
        "resign_confirm_moves": int(t_cfg.get("resign_confirm_moves", 6)),
        "adjudicate_bare_king_requires_mate": bool(
            t_cfg.get("adjudicate_bare_king_requires_mate", True)
        ),
        "ground_a_games_per_pair": int(g_a_cfg.get("games_per_pair", 8)),
        "ground_a_max_moves_eval": int(g_a_cfg.get("max_moves_eval", 120)),
        "ground_a_eval_temperature": (
            float(g_a_cfg["eval_temperature"])
            if g_a_cfg.get("eval_temperature") is not None
            else None
        ),
        "ground_a_search_depth": int(g_a_cfg.get("search_depth", 3)),
        "resumed": ckpt.get("_resumed", False),
    }
