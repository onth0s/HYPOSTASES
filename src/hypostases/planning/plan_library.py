"""HYPOSTASES Planning — Strategy & Skill Library.

Spec Ref: docs/WAVE_2_FRONT_02/front_02_explicit_planning_layer_spec.md
Synthesizes Voyager (Wang et al.) skill library, GOAP 2003 (Orkin) A* template matching,
and AdaPlanner (Sun et al.) skill acquisition and filtering.

Rule 006 Compliance: YAML serialization format stored in schema/plans/.
Rule 007 Assessment: Latency monitoring for YAML persistence performance tax.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hypostases.planning.plan_types import Plan, PlanNode, PlanStatus
from hypostases.schemas.loader import get_schema_dir


@dataclass
class PlanTemplate:
    """Archived reusable strategy template (Voyager Synthesis)."""

    template_id: str
    goal_name: str
    prerequisite_state: dict[str, Any] = field(default_factory=dict)
    expected_effects: dict[str, Any] = field(default_factory=dict)
    node_specs: list[dict[str, Any]] = field(default_factory=list)
    average_utility_gain: float = 0.0
    times_used: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class PlanLibrary:
    """Persistent strategy template indexing, matching, and YAML storage manager."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        library_cfg = self.config.get("library", {})
        self.storage_dir_name = library_cfg.get("storage_directory", "schema/plans")
        self.min_utility_threshold = library_cfg.get("min_utility_gain_threshold", 0.2)
        self.warn_latency_ms = self.config.get("performance_monitoring", {}).get(
            "warn_if_yaml_latency_ms_exceeds", 50.0
        )

        self.templates: dict[str, PlanTemplate] = {}
        self.storage_path: Path | None = None
        self._init_storage()

    def _init_storage(self) -> None:
        """Initializes storage directory in schema/plans/."""
        try:
            schema_dir = get_schema_dir()
            self.storage_path = schema_dir / "plans"
        except FileNotFoundError:
            self.storage_path = Path("schema/plans")

        self.storage_path.mkdir(parents=True, exist_ok=True)

    def match_template(self, goal_name: str, current_state: dict[str, Any]) -> PlanTemplate | None:
        """GOAP 2003 A* Goal-Oriented Template Matching.

        Finds best matching PlanTemplate for given goal and current state prerequisites.
        """
        candidates: list[PlanTemplate] = []
        for template in self.templates.values():
            if template.goal_name == goal_name:
                # Verify prerequisite state compatibility
                prereqs = template.prerequisite_state
                match = True
                for k, v in prereqs.items():
                    if k in current_state and current_state[k] != v:
                        match = False
                        break
                if match:
                    candidates.append(template)

        if not candidates:
            return None

        # Return template with highest average utility gain
        return max(candidates, key=lambda t: t.average_utility_gain)

    def instantiate_plan(self, template: PlanTemplate, plan_id: str) -> Plan:
        """Instantiates executable Plan object from a PlanTemplate."""
        nodes: list[PlanNode] = []
        for spec in template.node_specs:
            node = PlanNode(
                node_id=spec.get("node_id", f"node_{len(nodes) + 1}"),
                action_name=spec.get("action_name", "UNKNOWN"),
                action_params=spec.get("action_params", {}),
                preconditions=spec.get("preconditions", {}),
                effects=spec.get("effects", {}),
                expected_utility_delta=spec.get("expected_utility_delta", 0.0),
            )
            nodes.append(node)

        template.times_used += 1

        return Plan(
            plan_id=plan_id,
            goal_name=template.goal_name,
            goal_params={},
            nodes=nodes,
            status=PlanStatus.PLANNED,
            metadata={"source_template": template.template_id},
        )

    def discover_and_archive_skill(
        self, plan: Plan, utility_gain: float, prerequisite_state: dict[str, Any]
    ) -> PlanTemplate | None:
        """AdaPlanner §3.3 Skill Discovery & Filtering.

        Archives successful plan executions iff net utility gain exceeds min_utility_threshold.
        """
        if utility_gain < self.min_utility_threshold or not plan.nodes:
            return None

        template_id = f"template_{plan.goal_name}_{len(self.templates) + 1}"
        node_specs = []
        for n in plan.nodes:
            node_specs.append(
                {
                    "node_id": n.node_id,
                    "action_name": n.action_name,
                    "action_params": n.action_params,
                    "preconditions": n.preconditions,
                    "effects": n.effects,
                    "expected_utility_delta": n.expected_utility_delta,
                }
            )

        template = PlanTemplate(
            template_id=template_id,
            goal_name=plan.goal_name,
            prerequisite_state=prerequisite_state,
            expected_effects=plan.nodes[-1].effects if plan.nodes else {},
            node_specs=node_specs,
            average_utility_gain=utility_gain,
            times_used=1,
        )

        self.templates[template_id] = template
        self.save_template_to_yaml(template)
        return template

    def save_template_to_yaml(self, template: PlanTemplate) -> Path:
        """Rule 006 / Rule 007 YAML Serialization with latency monitoring."""
        start_t = time.perf_counter()

        if self.storage_path is None:
            self._init_storage()

        filepath = self.storage_path / f"{template.template_id}.yaml"
        data = asdict(template)

        with open(filepath, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        # Rule 007 Performance Assessment Check
        if elapsed_ms > self.warn_latency_ms:
            print(
                f"[Rule 007 Performance Alert] YAML serialization of plan template {template.template_id} "
                f"took {elapsed_ms:.2f}ms (threshold: {self.warn_latency_ms}ms). "
                f"Consider prompting User for binary IPC compression."
            )

        return filepath

    def load_templates_from_yaml(self) -> int:
        """Loads all YAML plan templates from storage directory with latency tracking."""
        start_t = time.perf_counter()

        if self.storage_path is None or not self.storage_path.exists():
            return 0

        count = 0
        for filepath in self.storage_path.glob("*.yaml"):
            try:
                with open(filepath, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, dict) and "template_id" in data:
                        template = PlanTemplate(**data)
                        self.templates[template.template_id] = template
                        count += 1
            except Exception as e:
                print(f"Warning: Failed to load plan template from {filepath}: {e}")

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        if elapsed_ms > self.warn_latency_ms:
            print(
                f"[Rule 007 Performance Alert] YAML template loading took {elapsed_ms:.2f}ms "
                f"(threshold: {self.warn_latency_ms}ms)."
            )

        return count
