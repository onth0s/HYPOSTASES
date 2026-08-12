"""Bayesian Communication Engine for Dual Posterior Updates (Front 06)."""

from hypostases.communication.deception_signaling import DeceptionSignalingFilter
from hypostases.communication.trust_reputation import TrustReputationEngine
from hypostases.communication.types import (
    BayesianBeliefState,
    DiscreteHypothesisPosterior,
    PeerMessage,
)
from hypostases.engine.types import AgentState


class BayesianCommunicationEngine:
    """Master engine evaluating peer messages, updating continuous state posteriors

    and discrete hypothesis space posteriors (Front 11 integration ready).
    """

    def __init__(
        self,
        trust_engine: TrustReputationEngine | None = None,
        deception_filter: DeceptionSignalingFilter | None = None,
    ) -> None:
        self.trust_engine = trust_engine or TrustReputationEngine()
        self.deception_filter = deception_filter or DeceptionSignalingFilter()

    def process_incoming_message(
        self,
        message: PeerMessage,
        agent: AgentState,
        hypothesis_posterior: DiscreteHypothesisPosterior | None = None,
        goal_misalignment: float = 0.0,
    ) -> tuple[BayesianBeliefState, DiscreteHypothesisPosterior]:
        """Processes an incoming peer message and updates dual belief posteriors."""
        # 1. Acemoglu deduplication check
        if self.trust_engine.is_duplicate_message(message):
            # Return current belief state without double-counting
            current_belief = BayesianBeliefState(
                peer_beliefs=agent.w.peer_beliefs.copy(),
            )
            h_post = hypothesis_posterior or DiscreteHypothesisPosterior()
            return current_belief, h_post

        # 2. Retrieve sender trust profile
        trust_profile = self.trust_engine.get_trust_profile(message.sender_id)
        expected_honesty = trust_profile.expected_honesty()

        # 3. Continuous state posterior update (Gaussian conjugate Bayes update)
        new_peer_beliefs = agent.w.peer_beliefs.copy()
        state_means: dict[str, float] = {}
        state_variances: dict[str, float] = {}

        for field_name, val in message.payload.items():
            prev_val = new_peer_beliefs.get(field_name, val)

            # Evaluate uRSA pragmatic likelihood
            l_val = self.deception_filter.evaluate_message_likelihood(
                message=message,
                hypothesized_state=val,
                goal_misalignment=goal_misalignment,
                trust_honesty=expected_honesty,
                trust_competence=1.0,
            )

            # Bayesian update weighting based on expected honesty & likelihood
            weight = expected_honesty * min(1.0, max(0.0, l_val))
            updated_val = weight * val + (1.0 - weight) * prev_val
            new_peer_beliefs[field_name] = updated_val

            state_means[field_name] = updated_val
            state_variances[field_name] = (1.0 - weight) * message.declared_variance

        # Synchronize with agent.w.peer_beliefs
        agent.w.peer_beliefs.update(new_peer_beliefs)

        # 4. Discrete Hypothesis Space posterior update P(H_k | m)
        if hypothesis_posterior is None or not hypothesis_posterior.hypothesis_probabilities:
            h_post = DiscreteHypothesisPosterior(
                hypothesis_probabilities={"H1": 0.3333, "H2": 0.3333, "H3": 0.3334}
            )
        else:
            h_post = hypothesis_posterior

        new_h_probs: dict[str, float] = {}
        for h_id, prior_prob in h_post.hypothesis_probabilities.items():
            # Likelihood of message given hypothesis
            hyp_state = 1.0 if h_id == "H1" else (0.5 if h_id == "H2" else 0.0)
            lh = self.deception_filter.evaluate_message_likelihood(
                message=message,
                hypothesized_state=hyp_state,
                goal_misalignment=goal_misalignment,
                trust_honesty=expected_honesty,
            )
            new_h_probs[h_id] = lh * prior_prob

        updated_h_post = DiscreteHypothesisPosterior(hypothesis_probabilities=new_h_probs)
        updated_h_post.normalize()

        # Front 11 AbductiveEngine integration hook
        if hasattr(agent.w, "abductive_engine") and agent.w.abductive_engine is not None:
            agent.w.abductive_engine.update_from_peer_message(
                message_content=str(message.payload),
                sender_id=message.sender_id,
                trust_score=expected_honesty,
            )

        belief_state = BayesianBeliefState(
            state_means=state_means,
            state_variances=state_variances,
            peer_beliefs=new_peer_beliefs,
        )

        return belief_state, updated_h_post

    @staticmethod
    def evaluate_acemoglu_additive_decision(private_belief: float, social_belief: float) -> int:
        """Acemoglu Additive Decision Decomposition (x_n = 1 iff p_private + p_social > 1)."""
        return 1 if (private_belief + social_belief) > 1.0 else 0
