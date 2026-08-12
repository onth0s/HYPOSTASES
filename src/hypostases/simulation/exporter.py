"""RunExporter module for compartmentalizing simulation run exports and state checkpoints.

Rule 011 & Rule 006 compliant.
"""

import os
from datetime import UTC, datetime
from typing import Any

import yaml

from hypostases.meta_learning.meta_state import MetaParameterVector


class RunExporter:
    """Manages isolated run directories under exports/runs/<run_id>/."""

    def __init__(self, run_id: str, base_dir: str = "exports/runs") -> None:
        self.run_id = run_id
        self.run_dir = os.path.join(base_dir, run_id)
        self.checkpoints_dir = os.path.join(self.run_dir, "checkpoints")
        self.final_export_dir = os.path.join(self.run_dir, "final_export")

        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.final_export_dir, exist_ok=True)

    def initialize_run(self, scenario_name: str, seed: int, agent_names: list[str]) -> str:
        """Creates run directory and writes run_manifest.yaml."""
        manifest_data = {
            "run_manifest": {
                "run_id": self.run_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "scenario_name": scenario_name,
                "seed": seed,
                "agents": agent_names,
                "engine_version": "0.4.0",
            }
        }
        manifest_path = os.path.join(self.run_dir, "run_manifest.yaml")
        with open(manifest_path, "w", encoding="utf-8") as f:
            yaml.dump(manifest_data, f, default_flow_style=False, sort_keys=False)
        return manifest_path

    def save_checkpoint(self, tick: int, agent_meta_params: dict[str, MetaParameterVector]) -> str:
        """Serializes meta-parameters for all agents at a specific tick (Rule 011)."""
        tick_dir = os.path.join(self.checkpoints_dir, f"tick_{tick:06d}")
        os.makedirs(tick_dir, exist_ok=True)

        for agent_name, meta_params in agent_meta_params.items():
            snapshot_path = os.path.join(tick_dir, f"{agent_name}_meta.yaml")
            meta_params.save_yaml(snapshot_path)

        return tick_dir

    def finalize_run(self, summary_metrics: dict[str, Any]) -> str:
        """Writes final run summary to final_export/run_summary.yaml."""
        summary_data = {
            "run_summary": {
                "run_id": self.run_id,
                "completed_at": datetime.now(UTC).isoformat(),
                "metrics": summary_metrics,
            }
        }
        summary_path = os.path.join(self.final_export_dir, "run_summary.yaml")
        with open(summary_path, "w", encoding="utf-8") as f:
            yaml.dump(summary_data, f, default_flow_style=False, sort_keys=False)
        return summary_path
