"""Sabater-Sierra ReGreT Multi-Trust Engine & Acemoglu Network Deduplication (Front 06)."""

from collections import deque

from hypostases.communication.types import PeerMessage, SubjectiveOpinion, TrustProfile


class TrustReputationEngine:
    """Manages peer trust profiles, Subjective Logic evidence fusion,

    Sabater-Sierra witness reputation routing, and Acemoglu message deduplication.
    """

    def __init__(
        self,
        alpha_0: float = 2.0,
        beta_0: float = 1.0,
        trust_decay_rate: float = 0.05,
        provenance_window: int = 20,
    ) -> None:
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.trust_decay_rate = trust_decay_rate
        self.provenance_window = provenance_window

        self.trust_profiles: dict[str, TrustProfile] = {}
        self.peer_opinions: dict[str, SubjectiveOpinion] = {}
        self.seen_message_hashes: deque[str] = deque(maxlen=provenance_window)

    def get_trust_profile(self, peer_id: str) -> TrustProfile:
        """Retrieves or initializes a peer trust profile."""
        if peer_id not in self.trust_profiles:
            self.trust_profiles[peer_id] = TrustProfile(
                peer_id=peer_id,
                alpha_honesty=self.alpha_0,
                beta_honesty=self.beta_0,
            )
        return self.trust_profiles[peer_id]

    def update_direct_experience(
        self, peer_id: str, verified_truthful: bool, weight: float = 1.0
    ) -> TrustProfile:
        """Updates Beta-binomial parameters for direct interaction outcome."""
        profile = self.get_trust_profile(peer_id)
        profile.update_honesty(verified_truthful, weight)

        # Update Subjective Opinion tuple
        b = max(0.0, (profile.alpha_honesty - 1.0) / (profile.alpha_honesty + profile.beta_honesty))
        d = max(0.0, (profile.beta_honesty - 1.0) / (profile.alpha_honesty + profile.beta_honesty))
        u = max(0.0, 2.0 / (profile.alpha_honesty + profile.beta_honesty))
        opinion = SubjectiveOpinion(b=b, d=d, u=u)
        self.peer_opinions[peer_id] = opinion

        return profile

    def aggregate_witness_reputation(
        self, target_peer_id: str, witness_opinions: list[tuple[str, SubjectiveOpinion]]
    ) -> SubjectiveOpinion:
        """Sabater-Sierra ReGreT Witness Reputation Aggregation.

        Applies Jøsang discounting (x) using witness trust and consensus (+) across witnesses.
        """
        if not witness_opinions:
            return SubjectiveOpinion(b=0.0, d=0.0, u=1.0)

        accumulated_opinion: SubjectiveOpinion | None = None
        for witness_id, witness_rep in witness_opinions:
            witness_trust = self.get_trust_profile(witness_id)
            w_h = witness_trust.expected_honesty()

            # Witness trust opinion
            trust_op = SubjectiveOpinion(b=w_h, d=1.0 - w_h, u=0.0)

            # Discount witness opinion using evaluator's trust in witness
            discounted = trust_op.discount(witness_rep)

            if accumulated_opinion is None:
                accumulated_opinion = discounted
            else:
                accumulated_opinion = accumulated_opinion.consensus(discounted)

        return (
            accumulated_opinion if accumulated_opinion else SubjectiveOpinion(b=0.0, d=0.0, u=1.0)
        )

    def is_duplicate_message(self, message: PeerMessage) -> bool:
        """Acemoglu Network Deduplication Filter.

        Prevents double-counting of correlated signals across cyclic network topologies.
        """
        msg_hash = message.message_hash()
        if msg_hash in self.seen_message_hashes:
            return True
        self.seen_message_hashes.append(msg_hash)
        return False

    def apply_epoch_decay(self) -> None:
        """Applies baseline epoch decay regularization to peer trust parameters."""
        for profile in self.trust_profiles.values():
            profile.alpha_honesty = 1.0 + (profile.alpha_honesty - 1.0) * (
                1.0 - self.trust_decay_rate
            )
            profile.beta_honesty = 1.0 + (profile.beta_honesty - 1.0) * (
                1.0 - self.trust_decay_rate
            )
