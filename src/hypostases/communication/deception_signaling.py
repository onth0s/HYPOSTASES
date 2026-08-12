"""uRSA Pragmatic Communication & Crawford-Sobel Cheap Talk Deception Filter (Front 06)."""

import math
from collections.abc import Callable

from hypostases.communication.types import PeerMessage


class DeceptionSignalingFilter:
    """Evaluates strategic likelihoods, Crawford-Sobel cheap-talk partition noise,

    and Kamenica-Gentzkow persuasion bounds.
    """

    def __init__(
        self,
        base_noise_std: float = 0.05,
        bias_tolerance: float = 0.05,
        max_partitions: int = 10,
        persuasion_alpha: float = 1.0,
    ) -> None:
        self.base_noise_std = base_noise_std
        self.bias_tolerance = bias_tolerance
        self.max_partitions = max_partitions
        self.persuasion_alpha = persuasion_alpha

    def compute_crawford_sobel_partition_count(self, b_bias: float) -> int:
        """Calculates Crawford-Sobel maximum partition count N(b_bias).

        Formula: N(b) = floor(-0.5 + 0.5 * sqrt(1 + 2 / b)).
        """
        if b_bias <= self.bias_tolerance:
            return self.max_partitions
        val = -0.5 + 0.5 * math.sqrt(1.0 + 2.0 / b_bias)
        n_count = max(1, int(val))
        return min(n_count, self.max_partitions)

    def compute_effective_noise_std(self, b_bias: float) -> float:
        """Calculates effective message observation noise std under Crawford-Sobel partition equilibrium.

        Formula: sigma_m^2 = 1/(12 * N^2) + b_bias^2 * (N^2 - 1) / 3.
        """
        n_partitions = self.compute_crawford_sobel_partition_count(b_bias)
        if n_partitions <= 1:
            # Complete cheap-talk babbling equilibrium
            return 1.0
        partition_variance = (1.0 / (12.0 * n_partitions * n_partitions)) + (
            (b_bias * b_bias * (n_partitions * n_partitions - 1.0)) / 3.0
        )
        total_var = (self.base_noise_std**2) + partition_variance
        return math.sqrt(total_var)

    def evaluate_message_likelihood(
        self,
        message: PeerMessage,
        hypothesized_state: float,
        goal_misalignment: float = 0.0,
        trust_honesty: float = 1.0,
        trust_competence: float = 1.0,
    ) -> float:
        """Calculates uRSA pragmatic likelihood P(m | theta, T_j).

        Combines expected state, cheap-talk noise scaling, sender honesty, and competence.
        """
        sigma_eff = self.compute_effective_noise_std(goal_misalignment) / max(0.1, trust_competence)

        likelihood = 1.0
        for _field_name, claimed_val in message.payload.items():
            diff = claimed_val - hypothesized_state
            # Gaussian observation likelihood
            exp_term = math.exp(-0.5 * (diff / sigma_eff) ** 2)
            field_l = (1.0 / (math.sqrt(2.0 * math.pi) * sigma_eff)) * exp_term

            # Mixture model with trust honesty: T_honesty * L_honest + (1 - T_honesty) * Uniform(0, 1)
            blended_l = trust_honesty * field_l + (1.0 - trust_honesty) * 0.1
            likelihood *= max(1e-6, blended_l)

        return likelihood

    @staticmethod
    def validate_bayes_plausibility(
        prior_mu: float, posteriors_and_weights: list[tuple[float, float]]
    ) -> bool:
        """Kamenica-Gentzkow Bayes Plausibility Check.

        Verifies sum_s (mu_s * tau_s) == prior_mu and sum(tau_s) == 1.0.
        """
        weight_sum = sum(tau for _, tau in posteriors_and_weights)
        if abs(weight_sum - 1.0) > 1e-4:
            return False

        expected_posterior = sum(mu * tau for mu, tau in posteriors_and_weights)
        return abs(expected_posterior - prior_mu) < 1e-4

    @staticmethod
    def compute_concave_closure_bound(
        payoff_fn: Callable[[float], float], prior_mu: float, num_samples: int = 100
    ) -> float:
        """Computes Kamenica-Gentzkow concave closure upper bound V(prior_mu)."""
        # Linear interpolation over upper convex hull
        points = []
        for i in range(num_samples + 1):
            x = i / float(num_samples)
            y = payoff_fn(x)
            points.append((x, y))

        # Upper envelope evaluation at prior_mu
        max_val = payoff_fn(prior_mu)
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                x1, y1 = points[i]
                x2, y2 = points[j]
                if x1 <= prior_mu <= x2 and x1 != x2:
                    slope = (y2 - y1) / (x2 - x1)
                    interpolated = y1 + slope * (prior_mu - x1)
                    if interpolated > max_val:
                        max_val = interpolated
        return max_val
