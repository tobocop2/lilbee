"""Clearing a saved placement works on a host where no engine is installed."""

from __future__ import annotations

import pytest

from lilbee.app import placement as app_placement
from lilbee.core import settings
from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.placement_spec import PlacementSpec, RolePlacement
from lilbee.providers.roles import WorkerRole
from lilbee.server import handlers

_SPEC = PlacementSpec({WorkerRole.CHAT: RolePlacement(devices=(0,))})
_OTHER_SPEC = PlacementSpec({WorkerRole.CHAT: RolePlacement(devices=(1,))})


def _planner_fails_with_os_error(spec, reading):
    raise OSError("the engine binary cannot be read")


@pytest.fixture
def no_engine_host(monkeypatch, tmp_path):
    """A data root holding a saved placement, with no engine installed."""
    monkeypatch.setattr(cfg, "data_root", tmp_path)
    monkeypatch.setattr("lilbee.app.engine_choice.installed_engines", dict)
    monkeypatch.setattr(app_placement, "peek_services", lambda: None)
    settings.update_values(tmp_path, {"placement": _SPEC.to_json()})
    settings.sync_from_resolver(cfg, ["placement"])
    assert cfg.placement == _SPEC.to_json()
    return tmp_path


def test_clear_deletes_the_saved_placement_with_no_engine(no_engine_host):
    view = app_placement.set_placement(None)
    assert "placement" not in settings.load(no_engine_host)
    assert cfg.placement is None
    assert view.manual is False
    assert view.spec_json is None
    assert view.roles == ()
    skipped = {skip.role: skip for skip in view.skipped_no_engine}
    assert skipped[WorkerRole.CHAT].model == cfg.chat_model
    assert skipped[WorkerRole.EMBED].model == cfg.embedding_model
    assert all(skip.notice for skip in skipped.values())


def test_the_handler_clears_with_no_engine(no_engine_host):
    import asyncio

    response = asyncio.run(handlers.placement_clear())
    assert "placement" not in settings.load(no_engine_host)
    assert response.manual is False


def test_setting_a_spec_with_no_engine_still_refuses_and_writes_nothing(no_engine_host):
    with pytest.raises(ProviderError, match="No engine is assigned to any role"):
        app_placement.set_placement(_OTHER_SPEC)
    assert settings.load(no_engine_host)["placement"] == _SPEC.to_json()
    assert cfg.placement == _SPEC.to_json()


def test_clear_goes_ahead_when_the_planner_raises_os_error(no_engine_host, monkeypatch):
    monkeypatch.setattr(app_placement, "_resolve", _planner_fails_with_os_error)
    view = app_placement.set_placement(None)
    assert "placement" not in settings.load(no_engine_host)
    assert view.roles == ()


def test_setting_a_spec_refuses_when_the_planner_raises_os_error(no_engine_host, monkeypatch):
    monkeypatch.setattr(app_placement, "_resolve", _planner_fails_with_os_error)
    with pytest.raises(OSError, match="cannot be read"):
        app_placement.set_placement(_OTHER_SPEC)
    assert settings.load(no_engine_host)["placement"] == _SPEC.to_json()
