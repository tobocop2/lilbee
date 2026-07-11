"""The persistent-engine settings exist, default off, and round-trip every surface."""

from __future__ import annotations

import pytest

from lilbee.app.settings import apply_settings_update, list_settings
from lilbee.app.settings_map import SETTINGS_MAP
from lilbee.config_meta import WRITABLE_CONFIG_FIELDS
from lilbee.core.config import cfg


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path):
    """Point cfg at a per-test data root and restore the full snapshot."""
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    yield
    for field_name in type(snapshot).model_fields:
        setattr(cfg, field_name, getattr(snapshot, field_name))


def test_defaults_are_on_demand():
    """Off by default: today's teardown-on-quit behavior is the shipped default."""
    assert cfg.model_fields["keep_engine_warm"].default is False
    assert cfg.model_fields["engine_idle_ttl_minutes"].default == 5


@pytest.mark.parametrize("name", ["keep_engine_warm", "engine_idle_ttl_minutes"])
def test_exposed_in_the_settings_map(name):
    """The TUI settings screen renders from SETTINGS_MAP."""
    assert name in SETTINGS_MAP
    assert SETTINGS_MAP[name].help_text


@pytest.mark.parametrize("name", ["keep_engine_warm", "engine_idle_ttl_minutes"])
def test_writable_for_http_and_mcp(name):
    """HTTP and MCP writes are gated on the derived writable set."""
    assert name in WRITABLE_CONFIG_FIELDS


def test_round_trip_through_apply_settings_update():
    """The choke point every surface uses accepts and persists both fields."""
    apply_settings_update({"keep_engine_warm": True, "engine_idle_ttl_minutes": 15})
    listed = {s.key: s.value for s in list_settings()}
    assert listed["keep_engine_warm"] is True
    assert listed["engine_idle_ttl_minutes"] == 15


def test_negative_ttl_is_rejected():
    """A negative idle ttl has no meaning; the update must refuse it."""
    with pytest.raises(ValueError, match="engine_idle_ttl_minutes"):
        apply_settings_update({"engine_idle_ttl_minutes": -5})
