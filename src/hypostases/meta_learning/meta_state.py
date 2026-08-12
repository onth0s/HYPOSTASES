"""Meta-parameter state representation with Rule 011 dual persistence."""

import os
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class MetaParameterVector:
    """Represents theta_meta: cognitive hyperparameters, learning rates, EFE weights, and modulators.

    Complies with Rule 011:
    (1) In-memory tuple projection within c.m_procedural
    (2) Persistent human-readable YAML serialization as default state snapshot format
    """

    learning_rate: float = 0.01
    mood_decay_rate: float = 0.1  # Rule 004
    rollout_depth: int = 4
    particle_count: int = 16
    efe_beta: float = 0.5
    kmp_k: int = 4  # Rule 008
    peft_gamma_qkv: float = 1.0
    peft_gamma_out: float = 1.0
    peft_gamma_mlp: float = 1.0
    version: int = 1
    extra_params: dict[str, Any] = field(default_factory=dict)

    def to_procedural_tuple(self) -> tuple[float, float, int, int, float, int]:
        """Projects meta-parameters into in-memory procedural memory tuple c.m_procedural."""
        return (
            self.learning_rate,
            self.mood_decay_rate,
            self.rollout_depth,
            self.particle_count,
            self.efe_beta,
            self.kmp_k,
        )

    @classmethod
    def from_procedural_tuple(
        cls, tup: tuple[float, float, int, int, float, int]
    ) -> "MetaParameterVector":
        """Reconstructs MetaParameterVector from in-memory procedural memory tuple."""
        return cls(
            learning_rate=tup[0],
            mood_decay_rate=tup[1],
            rollout_depth=tup[2],
            particle_count=tup[3],
            efe_beta=tup[4],
            kmp_k=tup[5],
        )

    def to_yaml_dict(self) -> dict[str, Any]:
        """Converts meta-parameters to dictionary suitable for YAML snapshot serialization."""
        return {
            "meta_parameters_snapshot": {
                "version": self.version,
                "theta_meta": {
                    "learning_rate": self.learning_rate,
                    "mood_decay_rate": self.mood_decay_rate,
                    "rollout_depth": self.rollout_depth,
                    "particle_count": self.particle_count,
                    "efe_beta": self.efe_beta,
                    "kmp_k": self.kmp_k,
                    "peft_modulators": {
                        "gamma_qkv": self.peft_gamma_qkv,
                        "gamma_out": self.peft_gamma_out,
                        "gamma_mlp": self.peft_gamma_mlp,
                    },
                    "extra_params": self.extra_params,
                },
            }
        }

    def save_yaml(self, file_path: str) -> None:
        """Serializes meta-parameters to human-readable YAML snapshot file (Rule 011)."""
        data = self.to_yaml_dict()
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load_yaml(cls, file_path: str) -> "MetaParameterVector":
        """Loads meta-parameters from human-readable YAML snapshot file (Rule 011)."""
        with open(file_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        snapshot = data.get("meta_parameters_snapshot", {})
        tm = snapshot.get("theta_meta", {})
        peft = tm.get("peft_modulators", {})

        return cls(
            learning_rate=float(tm.get("learning_rate", 0.01)),
            mood_decay_rate=float(tm.get("mood_decay_rate", 0.1)),
            rollout_depth=int(tm.get("rollout_depth", 4)),
            particle_count=int(tm.get("particle_count", 16)),
            efe_beta=float(tm.get("efe_beta", 0.5)),
            kmp_k=int(tm.get("kmp_k", 4)),
            peft_gamma_qkv=float(peft.get("gamma_qkv", 1.0)),
            peft_gamma_out=float(peft.get("gamma_out", 1.0)),
            peft_gamma_mlp=float(peft.get("gamma_mlp", 1.0)),
            version=int(snapshot.get("version", 1)),
            extra_params=tm.get("extra_params", {}),
        )
