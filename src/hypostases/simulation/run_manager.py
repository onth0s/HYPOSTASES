"""HYPOSTASES Unified Run Manager.

Manages timestamped run bundles under exports/runs/run_YYYYMMDD_HHMMSS/.
All training artifacts (checkpoints, NNUE weights, PGN games, telemetry)
are isolated per-run. Atomic file writes prevent checkpoint corruption on
mid-training exits. Rule 006 & Rule 011 compliant.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class UnifiedRunManager:
    """Manages creation, persistence, and resume of unified run bundles."""

    STATUS_RUNNING = "running"
    STATUS_COMPLETED = "completed"
    STATUS_INTERRUPTED = "interrupted"

    LATEST_POINTER = "latest"

    def __init__(self, base_dir: str | Path = "exports/runs") -> None:
        self.base_dir = Path(base_dir)

    # ------------------------------------------------------------------
    # Run directory management
    # ------------------------------------------------------------------

    def create_run(self, run_id: str | None = None) -> Path:
        """Creates a new timestamped run bundle directory.

        Returns the resolved run directory path.
        """
        if run_id is None:
            ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            run_id = f"run_{ts}"

        run_dir = self.base_dir / run_id
        for subdir in ("nnue", "pgn/self_play", "pgn/ground_a", "pgn/ground_b", "telemetry"):
            (run_dir / subdir).mkdir(parents=True, exist_ok=True)

        self._update_latest_pointer(run_dir)
        return run_dir

    def get_nth_latest_run(self, n: int = 1) -> Path | None:
        """Returns the Nth most recent run directory by ISO timestamp (1-based index).

        Args:
            n: 1 = latest, 2 = second latest, etc.

        Returns:
            Path to the Nth latest run directory, or None if fewer than N runs exist.
        """
        if not self.base_dir.exists():
            return None

        run_dirs = sorted(
            [
                d
                for d in self.base_dir.iterdir()
                if d.is_dir() and d.name.startswith("run_") and d.name != self.LATEST_POINTER
            ],
            key=lambda d: d.name,
            reverse=True,
        )

        if len(run_dirs) < n:
            return None
        return run_dirs[n - 1]

    def get_latest_run(self) -> Path | None:
        """Returns the latest run directory."""
        return self.get_nth_latest_run(n=1)

    def list_runs(self) -> list[Path]:
        """Returns all run directories sorted newest-first."""
        if not self.base_dir.exists():
            return []
        return sorted(
            [
                d
                for d in self.base_dir.iterdir()
                if d.is_dir() and d.name.startswith("run_") and d.name != self.LATEST_POINTER
            ],
            key=lambda d: d.name,
            reverse=True,
        )

    def _update_latest_pointer(self, run_dir: Path) -> None:
        """Updates the `latest` pointer to the given run directory."""
        latest_path = self.base_dir / self.LATEST_POINTER
        # Write the run ID as plain text for cross-platform portability
        self.base_dir.mkdir(parents=True, exist_ok=True)
        latest_path.write_text(run_dir.name, encoding="utf-8")

    # ------------------------------------------------------------------
    # Atomic file writes
    # ------------------------------------------------------------------

    def _atomic_write_json(self, target: Path, data: dict[str, Any]) -> None:
        """Writes JSON atomically via a .tmp file -> os.replace."""
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, target)

    def _atomic_write_yaml(self, target: Path, data: dict[str, Any]) -> None:
        """Writes YAML atomically via a .tmp file -> os.replace."""
        tmp = target.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        os.replace(tmp, target)

    # ------------------------------------------------------------------
    # Save & load run metadata
    # ------------------------------------------------------------------

    def save_run_metadata(
        self,
        run_dir: Path,
        *,
        status: str,
        target_generations: int,
        last_completed_gen: int,
        total_games_played: int = 0,
        interrupted_at_gen: int | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
        interrupted_at: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Saves run_metadata.json atomically."""
        metadata: dict[str, Any] = {
            "status": status,
            "target_generations": target_generations,
            "last_completed_gen": last_completed_gen,
            "total_games_played": total_games_played,
        }
        if started_at:
            metadata["started_at"] = started_at
        if completed_at:
            metadata["completed_at"] = completed_at
        if interrupted_at_gen is not None:
            metadata["interrupted_at_gen"] = interrupted_at_gen
        if interrupted_at:
            metadata["interrupted_at"] = interrupted_at
        if extra:
            metadata.update(extra)
        self._atomic_write_json(run_dir / "run_metadata.json", metadata)

    def load_run_metadata(self, run_dir: Path) -> dict[str, Any]:
        """Loads run_metadata.json from a run directory."""
        meta_path = run_dir / "run_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No run_metadata.json found in {run_dir}")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Save & load checkpoint state (agent θ_meta + telemetry)
    # ------------------------------------------------------------------

    def save_checkpoint_state(
        self,
        run_dir: Path,
        *,
        gen_idx: int,
        theta_meta: list[float],
        temperature: float,
        beta_efe: float,
        telemetry: dict[int, dict[str, Any]],
    ) -> None:
        """Saves checkpoint_state.yaml atomically (Rule 006 & Rule 011)."""
        data: dict[str, Any] = {
            "last_saved_gen": gen_idx,
            "agent": {
                "theta_meta": [float(v) for v in theta_meta],
                "temperature": float(temperature),
                "beta_efe": float(beta_efe),
            },
            "telemetry": {str(g): v for g, v in telemetry.items()},
        }
        self._atomic_write_yaml(run_dir / "checkpoint_state.yaml", data)

    def load_checkpoint_state(self, run_dir: Path) -> dict[str, Any]:
        """Loads checkpoint_state.yaml from a run directory."""
        state_path = run_dir / "checkpoint_state.yaml"
        if not state_path.exists():
            raise FileNotFoundError(f"No checkpoint_state.yaml found in {run_dir}")
        with open(state_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Save & load NNUE weights
    # ------------------------------------------------------------------

    def save_nnue_weights(
        self,
        run_dir: Path,
        weights: dict[str, np.ndarray],
        tag: str = "latest",
    ) -> None:
        """Saves NNUE weights as a .npz archive atomically."""
        target = run_dir / "nnue" / f"nnue_{tag}.npz"
        tmp = target.with_suffix(".tmp.npz")
        np.savez(tmp, **weights)
        os.replace(tmp, target)

    def load_nnue_weights(self, run_dir: Path, tag: str = "latest") -> dict[str, np.ndarray]:
        """Loads NNUE weights from a .npz archive in a run directory."""
        target = run_dir / "nnue" / f"nnue_{tag}.npz"
        if not target.exists():
            raise FileNotFoundError(f"NNUE weights not found: {target}")
        data = np.load(target)
        return dict(data)

    # ------------------------------------------------------------------
    # Save & load snapshots index
    # ------------------------------------------------------------------

    def save_snapshots_index(
        self,
        run_dir: Path,
        snapshots: list[dict[str, Any]],
    ) -> None:
        """Saves a JSON index of snapshot metadata (theta_meta, temperature, generation)."""
        self._atomic_write_json(
            run_dir / "snapshots_index.json",
            {"snapshots": snapshots},
        )

    def load_snapshots_index(self, run_dir: Path) -> list[dict[str, Any]]:
        """Loads snapshot index from a run directory."""
        path = run_dir / "snapshots_index.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("snapshots", [])

    # ------------------------------------------------------------------
    # Convenience: full checkpoint save (atomic bundle)
    # ------------------------------------------------------------------

    def save_full_checkpoint(
        self,
        run_dir: Path,
        *,
        status: str,
        target_generations: int,
        last_completed_gen: int,
        total_games_played: int,
        theta_meta: list[float],
        temperature: float,
        beta_efe: float,
        nnue_weights: dict[str, np.ndarray],
        telemetry: dict[int, dict[str, Any]],
        snapshots_meta: list[dict[str, Any]],
        started_at: str | None = None,
        interrupted_at_gen: int | None = None,
        interrupted_at: str | None = None,
        completed_at: str | None = None,
    ) -> None:
        """Atomically saves all checkpoint components together."""
        self.save_nnue_weights(run_dir, nnue_weights, tag="latest")
        self.save_checkpoint_state(
            run_dir,
            gen_idx=last_completed_gen,
            theta_meta=theta_meta,
            temperature=temperature,
            beta_efe=beta_efe,
            telemetry=telemetry,
        )
        self.save_snapshots_index(run_dir, snapshots_meta)
        self.save_run_metadata(
            run_dir,
            status=status,
            target_generations=target_generations,
            last_completed_gen=last_completed_gen,
            total_games_played=total_games_played,
            interrupted_at_gen=interrupted_at_gen,
            started_at=started_at,
            interrupted_at=interrupted_at,
            completed_at=completed_at,
        )
        # Update latest pointer on every save
        self._update_latest_pointer(run_dir)

    # ------------------------------------------------------------------
    # Cleanup (legacy exports)
    # ------------------------------------------------------------------

    def migrate_legacy_pgn_dirs(
        self,
        legacy_dirs: list[str | Path] | None = None,
    ) -> list[str]:
        """Removes legacy PGN directories if they exist and are outside unified runs."""
        removed: list[str] = []
        legacy = legacy_dirs or ["exports/pgn"]
        for path in legacy:
            p = Path(path)
            if p.exists() and not str(p).startswith(str(self.base_dir)):
                shutil.rmtree(p, ignore_errors=True)
                removed.append(str(p))
        return removed
