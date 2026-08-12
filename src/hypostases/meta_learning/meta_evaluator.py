"""Meta-Performance Evaluator for tracking cognitive sample efficiency and compute cost."""

from dataclasses import dataclass


@dataclass
class MetaPerformanceMetric:
    utility_gain: float
    belief_variance_reduction: float
    compute_cost: float


class MetaEvaluator:
    """Evaluates meta-learning objective R_meta = delta_g + lambda_info * delta_sigma^2 - lambda_cost * C_compute."""

    def __init__(
        self,
        lambda_info: float = 0.5,
        lambda_cost: float = 0.01,
    ) -> None:
        self.lambda_info = lambda_info
        self.lambda_cost = lambda_cost
        self.history: list[MetaPerformanceMetric] = []

    def record_step(
        self,
        utility_gain: float,
        belief_variance_reduction: float,
        compute_cost: float,
    ) -> float:
        metric = MetaPerformanceMetric(
            utility_gain=utility_gain,
            belief_variance_reduction=belief_variance_reduction,
            compute_cost=compute_cost,
        )
        self.history.append(metric)
        return self.compute_meta_reward(metric)

    def compute_meta_reward(self, metric: MetaPerformanceMetric) -> float:
        return (
            metric.utility_gain
            + (self.lambda_info * metric.belief_variance_reduction)
            - (self.lambda_cost * metric.compute_cost)
        )

    def get_average_meta_reward(self, window: int = 10) -> float:
        if not self.history:
            return 0.0
        sub_history = self.history[-window:]
        rewards = [self.compute_meta_reward(m) for m in sub_history]
        return sum(rewards) / len(rewards)
