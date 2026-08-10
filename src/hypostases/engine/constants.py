"""HYPOSTASES Engine — Named Constants.

All numeric parameters that govern engine behavior and inference.
Importing from this module is the sole authoritative source for these values.
"""

from __future__ import annotations

from typing import Final

import numpy as np

# --- Softmax / Policy ---
SOFTMAX_EPSILON: Final[float] = 1e-3
TEMPERATURE_OFFSET: Final[float] = 0.15

# --- STATUS reserve sensitivity (schema_v1.yaml §12.5) ---
STATUS_COUPLING: Final[float] = 0.08
STATUS_RESERVE_THRESHOLD: Final[float] = 5.0

# --- Action costs (per K goal category: [SURVIVAL, ACQUISITION, RELATIONAL, STATUS]) ---
ACTION_COSTS: Final[np.ndarray] = np.array([5.0, 5.0, 3.0, 1.0])

# --- Feedback coefficients ---
REQUEST_MOOD_PENALTY: Final[float] = 0.1
SHARE_MOOD_BONUS: Final[float] = 0.05
WITHDRAW_MOOD_PENALTY: Final[float] = 0.03
SHARE_SOCIAL_GAIN: Final[float] = 0.3
REQUEST_SOCIAL_COST: Final[float] = 0.02
WITHDRAW_SOCIAL_COST: Final[float] = 0.01
RELATIONAL_U_GAIN: Final[float] = 0.1

# --- World Model updates ---
WORLD_MU_GAIN: Final[float] = 0.2
WORLD_REPLENISH_GAIN: Final[float] = 0.05
WORLD_SIGMA2_UPDATE_GAIN: Final[float] = 0.05
SIGMA2_MIN: Final[float] = 1e-4
PEER_BELIEF_ALPHA: Final[float] = 0.3

# --- Inference ---
LIKELIHOOD_MIN: Final[float] = 1e-12
ROUGHEN_RESERVE_SD: Final[float] = 0.3

# --- Rao-Blackwellization: Kalman world-model noise parameters (Phase 4) ---
# Q: process noise injected per tick — calibrated from WORLD_SIGMA2_UPDATE_GAIN * typical |surprise|
KALMAN_PROCESS_NOISE_Q: Final[float] = 0.1
# R: observation noise variance — calibrated from WORLD_MU_GAIN / WORLD_SIGMA2_UPDATE_GAIN ratio
KALMAN_OBS_NOISE_R: Final[float] = 2.0

# --- Defaults & Calibration ---
DEFAULT_XI: Final[np.ndarray] = np.array([0.2, 0.2, 0.2, 0.2])
DEFAULT_POOL_INIT: Final[float] = 10.0
MOOD_DECAY_RATE: Final[float] = 0.1
SIGMA2_MAX: Final[float] = 20.0
WITHDRAW_FEE: Final[float] = 0.2
WITHDRAW_DEGRADE: Final[float] = 0.5
