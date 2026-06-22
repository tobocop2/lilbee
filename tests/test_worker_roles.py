"""Tests for the engine-neutral worker role map."""

from __future__ import annotations

from lilbee.config_meta import MODEL_ROLE_FIELDS
from lilbee.providers.roles import MODEL_FIELD_TO_ROLE, WorkerRole


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
