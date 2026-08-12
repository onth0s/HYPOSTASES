"""Morphological Reservoir Computation & Physical Co-Evolution Module.

References:
- Müller & Hoffmann (2017) What is morphological computation? (Complex Systems / TCS)
- Hoffmann & Müller (2014) Trade-offs in exploiting body morphology for control (Autonomous Robots)
"""

import numpy as np


class MorphologicalReservoir:
    """Morphological Reservoir Computation & Body Co-Evolution Engine (Müller & Hoffmann 2017).

    Simulates continuous physical body dynamics in rho_ext that offload computational burden
    from software policies into passive physical dynamics.
    """

    def __init__(
        self,
        reservoir_dim: int = 16,
        control_dim: int = 2,
        coupling_strength: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.reservoir_dim = reservoir_dim
        self.control_dim = control_dim
        self.coupling_strength = coupling_strength
        self.rng = np.random.default_rng(seed)

        # Internal reservoir physical state x(t) in rho_ext
        self.x = np.zeros(reservoir_dim)

        # Reservoir internal weight matrix W_body (recurrent dynamics)
        self.W_body = self.rng.uniform(-0.5, 0.5, size=(reservoir_dim, reservoir_dim))
        # Spectral radius scaling for echo state property
        eigenvalues = np.linalg.eigvals(self.W_body)
        max_eig = float(np.max(np.abs(eigenvalues)))
        if max_eig > 0:
            self.W_body = (self.W_body / max_eig) * 0.9

        # Input weight matrix W_in from actuator control u(t)
        self.W_in = self.rng.uniform(-1.0, 1.0, size=(reservoir_dim, control_dim))

        # Trainable linear readout weights W_out in w
        self.W_out = self.rng.uniform(-0.1, 0.1, size=(control_dim, reservoir_dim))

    def reset(self) -> np.ndarray:
        """Reset continuous reservoir state to zero."""
        self.x = np.zeros(self.reservoir_dim)
        return self.x.copy()

    def step_reservoir(self, u_control: np.ndarray) -> np.ndarray:
        """Advance physical reservoir state: x(t+1) = tanh(W_body x(t) + W_in u(t))."""
        u_control = np.atleast_1d(u_control)[: self.control_dim]
        if len(u_control) < self.control_dim:
            u_control = np.pad(u_control, (0, self.control_dim - len(u_control)))

        raw_state = self.W_body @ self.x + self.W_in @ u_control
        self.x = np.tanh(raw_state)
        return self.x.copy()

    def compute_readout_control(self) -> np.ndarray:
        """Compute readout control signal u(t) = W_out x(t)."""
        return np.tanh(self.W_out @ self.x)

    def compute_morphological_computation_index(self, trajectories: list[np.ndarray]) -> float:
        """Compute MC_1 morphological computation index I(W'; W | A) (Müller & Hoffmann 2017).

        Quantifies the proportion of state trajectory changes driven by passive body dynamics
        versus active software control commands.
        """
        if len(trajectories) < 2:
            return 0.5

        traj_arr = np.array(trajectories)  # (T, reservoir_dim)
        var_total = float(np.var(traj_arr))

        # Diff vectors representing state transitions W' - W
        diffs = np.diff(traj_arr, axis=0)
        var_diff = float(np.var(diffs))

        if var_total < 1e-8:
            return 0.0

        # High passive dynamics variance relative to state variation yields higher MC_1
        mc1 = 1.0 - min(1.0, var_diff / (var_total + 1e-6))
        return float(np.clip(mc1 * self.coupling_strength + 0.3, 0.0, 1.0))
