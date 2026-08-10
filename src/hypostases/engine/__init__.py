"""HYPOSTASES Engine Package — Core simulation engine (v4 target)."""

from hypostases.engine import constants
from hypostases.engine.dynamics import (
    evolve,
    evolve_rb,
    feedback,
    goal_probs,
    pi_decision,
    step_env,
)
from hypostases.engine.likelihood import action_likelihood, expected_action_type, predict_amount
from hypostases.engine.types import (
    N_K,
    Action,
    ActionType,
    Agent,
    AgentState,
    Characteristics,
    DeltaLog,
    FeedbackDelta,
    GoalCategory,
    GoalHierarchy,
    K,
    PowerExternal,
    WorldModel,
)

__all__ = [
    "N_K",
    "Action",
    "ActionType",
    "Agent",
    "AgentState",
    "Characteristics",
    "DeltaLog",
    "FeedbackDelta",
    "GoalCategory",
    "GoalHierarchy",
    "K",
    "PowerExternal",
    "WorldModel",
    "action_likelihood",
    "constants",
    "evolve",
    "evolve_rb",
    "expected_action_type",
    "feedback",
    "goal_probs",
    "pi_decision",
    "predict_amount",
    "step_env",
]
