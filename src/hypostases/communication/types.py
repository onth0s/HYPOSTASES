"""Dataclasses and types for Communication as Bayesian Evidence (Front 06)."""

from dataclasses import dataclass, field


@dataclass
class PeerMessage:
    """Represents a probabilistic observation message from a peer agent."""

    sender_id: str
    receiver_id: str
    payload: dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0
    declared_variance: float = 0.05
    hypothesis_claims: dict[str, float] = field(default_factory=dict)
    provenance_chain: list[str] = field(default_factory=list)

    def message_hash(self) -> str:
        """Generates a signature for message deduplication across network topologies."""
        sorted_payload = sorted(self.payload.items())
        sorted_hypotheses = sorted(self.hypothesis_claims.items())
        chain_str = "->".join(self.provenance_chain)
        return f"{self.sender_id}:{chain_str}:{sorted_payload}:{sorted_hypotheses}"


@dataclass
class TrustProfile:
    """Sabater-Sierra ReGreT & Beta-binomial dual-aspect trust representation."""

    peer_id: str
    alpha_honesty: float = 2.0
    beta_honesty: float = 1.0
    competence_variance: float = 0.1
    direct_interactions: int = 0
    witness_interactions: int = 0

    def expected_honesty(self) -> float:
        """Returns expected honesty probability E[T_honesty] = alpha / (alpha + beta)."""
        total = self.alpha_honesty + self.beta_honesty
        if total <= 0.0:
            return 0.5
        return self.alpha_honesty / total

    def update_honesty(self, verified_truthful: bool, weight: float = 1.0) -> None:
        """Updates Beta distribution parameters based on interaction outcomes."""
        if verified_truthful:
            self.alpha_honesty += weight
        else:
            self.beta_honesty += weight
        self.direct_interactions += 1


@dataclass
class SubjectiveOpinion:
    """Jøsang Subjective Logic Opinion tuple omega = (b, d, u, a)."""

    b: float  # belief
    d: float  # disbelief
    u: float  # uncertainty
    a: float = 0.5  # base rate / relative atomicity

    def __post_init__(self) -> None:
        self.normalize()

    def normalize(self) -> None:
        """Ensures b + d + u = 1 within floating point precision."""
        self.b = max(0.0, self.b)
        self.d = max(0.0, self.d)
        self.u = max(0.0, self.u)
        total = self.b + self.d + self.u
        if total > 0.0 and abs(total - 1.0) > 1e-6:
            self.b /= total
            self.d /= total
            self.u /= total

    @property
    def is_valid(self) -> bool:
        """Checks normalization constraint b + d + u = 1."""
        return (
            self.b >= 0.0
            and self.d >= 0.0
            and self.u >= 0.0
            and abs((self.b + self.d + self.u) - 1.0) < 1e-4
        )

    def expected_probability(self) -> float:
        """Calculates expected probability P = b + a * u."""
        return self.b + self.a * self.u

    def consensus(self, other: "SubjectiveOpinion") -> "SubjectiveOpinion":
        """Jøsang Consensus Operator (omega_A (+) omega_B).

        Fuses two independent subjective opinions about the same state variable.
        """
        u_a, u_b = self.u, other.u
        b_a, b_b = self.b, other.b
        d_a, d_b = self.d, other.d
        denom = u_a + u_b - u_a * u_b
        if abs(denom) < 1e-9:
            return SubjectiveOpinion(b=0.5, d=0.5, u=0.0, a=self.a)

        b_res = (b_a * u_b + b_b * u_a) / denom
        d_res = (d_a * u_b + d_b * u_a) / denom
        u_res = (u_a * u_b) / denom
        return SubjectiveOpinion(b=b_res, d=d_res, u=u_res, a=self.a)

    def discount(self, other: "SubjectiveOpinion") -> "SubjectiveOpinion":
        """Jøsang Discounting Operator (omega_A (x) omega_B).

        Applies trust discounting along indirect reporting paths A -> B.
        """
        b_a, u_a = self.b, self.u
        b_b, d_b, u_b = other.b, other.d, other.u

        b_res = b_a * b_b
        d_res = b_a * d_b
        u_res = u_a + b_a * u_b
        return SubjectiveOpinion(b=b_res, d=d_res, u=u_res, a=other.a)


@dataclass
class BayesianBeliefState:
    """Continuous state vector belief representation."""

    state_means: dict[str, float] = field(default_factory=dict)
    state_variances: dict[str, float] = field(default_factory=dict)
    peer_beliefs: dict[str, float] = field(default_factory=dict)


@dataclass
class DiscreteHypothesisPosterior:
    """Discrete hypothesis space posterior distribution P(H_k | m).

    Enables native integration with Front 11 Abductive Reasoning.
    """

    hypothesis_probabilities: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.normalize()

    def normalize(self) -> None:
        """Normalizes hypothesis probabilities so sum(P(H_k)) = 1.0."""
        if not self.hypothesis_probabilities:
            return
        total = sum(max(0.0, v) for v in self.hypothesis_probabilities.values())
        if total > 0.0:
            for k in self.hypothesis_probabilities:
                self.hypothesis_probabilities[k] = (
                    max(0.0, self.hypothesis_probabilities[k]) / total
                )
        else:
            uniform = 1.0 / len(self.hypothesis_probabilities)
            for k in self.hypothesis_probabilities:
                self.hypothesis_probabilities[k] = uniform
