import pytest
from pydantic import ValidationError

from lilbee.core.config.model import Config
from lilbee.providers.fleet.placement_spec import PlacementSpec, RolePlacement
from lilbee.providers.roles import WorkerRole


def test_parses_json_string_to_spec():
    cfg = Config(placement='{"chat": {"devices": [0, 1]}}')
    assert isinstance(cfg.placement, PlacementSpec)
    assert cfg.placement.roles[WorkerRole.CHAT].devices == (0, 1)


def test_accepts_spec_object():
    spec = PlacementSpec({WorkerRole.EMBED: RolePlacement(devices=(0,))})
    assert Config(placement=spec).placement == spec


def test_blank_is_none():
    assert Config(placement="").placement is None
    assert Config(placement=None).placement is None


def test_invalid_json_raises():
    with pytest.raises(ValidationError):
        Config(placement="{nope")


def test_placement_is_writable_not_public():
    from lilbee.config_meta import PUBLIC_CONFIG_FIELDS, WRITABLE_CONFIG_FIELDS

    assert "placement" in WRITABLE_CONFIG_FIELDS
    assert "placement" not in PUBLIC_CONFIG_FIELDS


def test_unexpected_type_raises():
    """A non-string, non-PlacementSpec value must fail validation."""
    with pytest.raises(ValidationError):
        Config(placement=42)
