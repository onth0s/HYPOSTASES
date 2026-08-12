"""Unit & Integration Test Suite for Wave 5 Front 14 Natural Language Symbolic Compression.

Spec Ref: docs/WAVE_5_FRONT_14/front_14_natural_language_symbolic_compression_spec.md
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from hypostases.natural_language_compression import (
    CommunicativeLanguageSymbolismRouter,
    NaturalLanguageGovernanceProtocol,
    SymbolicAbductionInterface,
    SymbolicCompressionEngine,
    SymbolicMappingTransferLayer,
    VisualEpistemicDualityMapper,
)
from hypostases.schemas.loader import load_natural_language_compression_config


def test_schema_loader() -> None:
    """Verifies YAML schema loader for Front 14 config."""
    cfg = load_natural_language_compression_config()
    assert cfg is not None
    assert cfg.get("version") == "1.0"
    assert cfg.get("efe_mode") is True
    assert "mdl_config" in cfg
    assert "vocabulary_config" in cfg


def test_visual_epistemic_duality_mapper() -> None:
    """Verifies Giaquinto (2007) Visual-Epistemic Duality Mapper."""
    cfg = load_natural_language_compression_config()
    mapper = VisualEpistemicDualityMapper(cfg)

    state_vec = [0.1, 0.5, -0.3, 0.8, 0.2, -0.1, 0.4, 0.7]
    symbol_ids = mapper.encode_spatial_to_symbolic(state_vec)
    assert len(symbol_ids) == 4

    reconstructed = mapper.decode_symbolic_to_spatial(symbol_ids)
    assert reconstructed.shape == (8,)

    inv_passed = mapper.verify_topological_invariance(state_vec)
    assert inv_passed is True


def test_symbolic_mapping_transfer_layer() -> None:
    """Verifies Feng & Lu (ACL 2023) Symbolic Mapping Layer & refdis score."""
    cfg = load_natural_language_compression_config()
    layer = SymbolicMappingTransferLayer(cfg)

    obs = [0.5, -0.2, 0.8, 0.1, -0.4, 0.3, 0.9, 0.0]
    p = layer.get_relevance_probabilities(obs)
    assert p.shape == (512,)
    assert (p >= 0.0).all() and (p <= 1.0).all()

    word_bank = layer.sample_word_bank(obs)
    assert len(word_bank) > 0

    refdis = layer.compute_referential_disentanglement({1: 10, 2: 5, 3: 1})
    assert 0.0 <= refdis <= 1.0


def test_communicative_language_symbolism_router() -> None:
    """Verifies Pei et al. (ICML 2026) CLSR Router & token bound."""
    cfg = load_natural_language_compression_config()
    router = CommunicativeLanguageSymbolismRouter(cfg)

    i_req = router.compute_required_information(uncertainty_entropy=2.0, target_acc=0.95)
    assert i_req >= 0.0

    min_tokens = router.compute_min_token_bound(uncertainty_entropy=2.0, target_acc=0.95)
    assert min_tokens >= 0.0

    allocated, protocol = router.route_token_allocation(query_uncertainty=0.1)
    assert protocol == "DIRECT_SYMBOLIC_EXECUTION"
    assert allocated == 1

    allocated_mult, protocol_mult = router.route_token_allocation(query_uncertainty=3.5)
    assert protocol_mult in ["MULTI_LSF_AGGREGATION", "EXTENDED_REASONING_PROTOCOL"]
    assert allocated_mult > 1


def test_symbolic_compression_engine() -> None:
    """Verifies core engine state compression, EFE mode, and dual persistence."""
    engine = SymbolicCompressionEngine()

    state_tuple = {
        "c": [0.5, 0.2, 0.1, 0.8, 0.3, 0.4, 0.9, 0.0],
        "w": [0.1, 0.6, 0.2, 0.4, 0.5, 0.7, 0.3, 0.2],
        "g": [0.8, 0.1, 0.7, 0.2, 0.9, 0.0, 0.6, 0.4],
        "rho_ext": [1.0, 1.0, 0.5, 0.5, 1.0, 0.8, 0.9, 0.7],
    }

    token_ids, code_len, distortion = engine.compress_state(state_tuple)
    assert len(token_ids) > 0
    assert code_len > 0.0

    mdl_loss = engine.compute_mdl_loss(code_len, distortion)
    assert mdl_loss >= code_len

    efe = engine.compute_expected_free_energy(
        pragmatic_risk=0.5, epistemic_info_gain=1.2, ambiguity=0.2
    )
    assert efe == pytest.approx(0.5 - 1.2 + 0.2)

    # Multi-agent message creation
    msg = engine.create_symbolic_message("agent_1", "agent_2", state_tuple)
    assert msg.sender_id == "agent_1"
    assert msg.recipient_id == "agent_2"
    assert len(msg.token_ids) > 0
    assert msg.checksum >= 0

    # Rule 011 Dual Persistence
    with tempfile.TemporaryDirectory() as tmp_dir:
        snapshot_path = Path(tmp_dir) / "snapshot.yaml"
        engine.save_snapshot_yaml(snapshot_path)
        assert snapshot_path.exists()

        engine_new = SymbolicCompressionEngine()
        engine_new.load_snapshot_yaml(snapshot_path)
        assert engine_new.lambda_mdl == engine.lambda_mdl


def test_interfaces_integration() -> None:
    """Verifies Front 11 Abduction and Front 05 Governance interfaces."""
    engine = SymbolicCompressionEngine()

    # Front 11 Abduction Interface
    abduction = SymbolicAbductionInterface(engine)
    hyp_dict = {"id": "H1", "cause": "RESOURC_SCARCITY", "confidence": 0.85}
    token_str = abduction.format_hypothesis_as_symbolic_token_stream(hyp_dict)
    assert "H1" in token_str
    assert "RESOURC_SCARCITY" in token_str

    surprisal = abduction.compute_hypothesis_surprisal(token_str)
    assert surprisal > 0.0

    # Front 05 Governance Interface
    governance = NaturalLanguageGovernanceProtocol(engine)
    treaty = {"treaty_id": "T1", "rules": ["NO_FREE_RIDER", "CAP_EXTRACTION"], "punish_cost": 2.5}
    serialized = governance.serialize_treaty_protocol(treaty)
    assert "T1" in serialized
    assert "NO_FREE_RIDER" in serialized

    deserialized = governance.deserialize_treaty_protocol(serialized)
    assert deserialized["treaty_id"] == "T1"
    assert "NO_FREE_RIDER" in deserialized["rules"]
    assert deserialized["punish_cost"] == 2.5
