"""Tests for the engine-neutral worker role registry."""

from __future__ import annotations

from lilbee.config_meta import MODEL_ROLE_FIELDS
from lilbee.core.config import Config
from lilbee.providers.roles import (
    MODEL_FIELD_TO_ROLE,
    REPLICATED_ROLES,
    ROLE_REGISTRY,
    WorkerRole,
)


def test_registry_covers_every_worker_role() -> None:
    # A new role must get a registry entry, or the scattered planning/placement
    # tuples silently drop it.
    assert set(ROLE_REGISTRY) == set(WorkerRole)
    assert all(role is info.role for role, info in ROLE_REGISTRY.items())


def test_registry_config_fields_and_knobs_are_real_config_attributes() -> None:
    # Guards "add a role, forget to wire its cfg field/knob": every config_field
    # and every replica_knob must name an actual Config attribute.
    for info in ROLE_REGISTRY.values():
        assert info.config_field in Config.model_fields
        if info.replica_knob is not None:
            assert info.replica_knob in Config.model_fields


def test_replicated_roles_have_a_knob_and_match_the_flag() -> None:
    # ``replicated`` and "has a replica_knob" are the same fact by construction;
    # resolve_replica_count relies on that equivalence.
    for info in ROLE_REGISTRY.values():
        assert info.replicated == (info.replica_knob is not None)
    assert set(REPLICATED_ROLES) == {r for r, i in ROLE_REGISTRY.items() if i.replicated}


def test_model_field_to_role_covers_every_model_role_field() -> None:
    # Every writable model-role config field must map to exactly one worker role,
    # so a settings change can reload the right server.
    assert set(MODEL_FIELD_TO_ROLE) == set(MODEL_ROLE_FIELDS)


def test_model_field_to_role_maps_to_distinct_roles() -> None:
    roles = list(MODEL_FIELD_TO_ROLE.values())
    assert sorted(roles) == sorted(set(roles))
    assert set(roles) == {
        WorkerRole.CHAT,
        WorkerRole.EMBED,
        WorkerRole.RERANK,
        WorkerRole.VISION,
    }
