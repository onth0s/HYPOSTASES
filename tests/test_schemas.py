"""Tests for HYPOSTASES Schema Loaders."""

from hypostases.schemas import (
    get_schema_dir,
    load_components,
    load_invariants,
    load_schema,
    load_time_model,
    load_update_dynamics,
)


def test_schema_dir_resolution():
    schema_dir = get_schema_dir()
    assert schema_dir.is_dir()
    assert (schema_dir / "schema_v1.yaml").exists()


def test_load_schema_v1():
    data = load_schema("schema_v1")
    assert "domain" in data
    assert "goal_categories" in data
    assert data["_meta"]["spec_version"] == "v4"


def test_load_invariants():
    data = load_invariants()
    assert "constraints" in data
    assert data["_meta"]["spec_version"] == "v4"
    assert "memory_decay_application" in data["constraints"]
    assert "time_budget_lifecycle" in data["constraints"]
    assert "peer_beliefs_update" in data["constraints"]


def test_load_components():
    data = load_components()
    assert isinstance(data, dict)


def test_load_time_model():
    data = load_time_model()
    assert isinstance(data, dict)


def test_load_update_dynamics():
    data = load_update_dynamics()
    assert isinstance(data, dict)


def test_schema_completeness_audit():
    from hypostases.schemas import assert_schema_completeness

    # This should pass without raising InvariantViolationError
    assert_schema_completeness()
