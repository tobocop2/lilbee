"""Showing and previewing a placement answer on a host where no engine is installed."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from lilbee.app import placement as app_placement
from lilbee.cli.placement import placement_app
from lilbee.core import settings
from lilbee.core.config import cfg
from lilbee.mcp_server import get_placement_tool, preview_placement_tool
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.placement_spec import PlacementError, PlacementSpec, RolePlacement
from lilbee.providers.roles import WorkerRole
from lilbee.server.routes.placement import placement_preview_route, placement_route
from tests._litestar_test_client import create_test_client

_SPEC = PlacementSpec({WorkerRole.CHAT: RolePlacement(devices=(0,))})
_NO_ENGINE = "No engine is assigned to any role"


def _unreadable_spec():
    raise PlacementError("the saved spec is unreadable")


@pytest.fixture
def no_engine_host(monkeypatch, tmp_path):
    """A data root with no engine installed and no saved placement."""
    monkeypatch.setattr(cfg, "data_root", tmp_path)
    monkeypatch.setattr(cfg, "placement", None)
    monkeypatch.setattr("lilbee.app.engine_choice.installed_engines", dict)
    monkeypatch.setattr(app_placement, "peek_services", lambda: None)
    return tmp_path


@pytest.fixture
def saved_spec(no_engine_host):
    """The same host holding a saved placement."""
    settings.update_values(no_engine_host, {"placement": _SPEC.to_json()})
    settings.sync_from_resolver(cfg, ["placement"])
    assert cfg.placement == _SPEC.to_json()


def test_show_with_a_saved_spec_reports_it_and_the_missing_engines(saved_spec):
    view = app_placement.get_placement()
    assert view.spec_json == _SPEC.to_json()
    assert view.manual is True
    assert view.rejected_spec_json is None
    assert view.refusal is None
    assert view.roles == ()
    assert view.gpus == ()
    skipped = {skip.role: skip for skip in view.skipped_no_engine}
    assert skipped[WorkerRole.CHAT].model == cfg.chat_model
    assert all(skip.notice for skip in skipped.values())


def test_show_with_no_saved_spec_reports_none(no_engine_host):
    view = app_placement.get_placement()
    assert view.spec_json is None
    assert view.manual is False
    assert view.rejected_spec_json is None
    assert view.refusal is None
    assert view.skipped_no_engine


def test_show_with_an_unreadable_saved_spec_keeps_the_text_and_the_reason(
    no_engine_host, monkeypatch
):
    monkeypatch.setattr(cfg, "placement", '{"chat": {"devices": [0]}}')
    monkeypatch.setattr(app_placement, "_active_spec", _unreadable_spec)
    view = app_placement.get_placement()
    assert view.rejected_spec_json == '{"chat": {"devices": [0]}}'
    assert view.refusal == "the saved spec is unreadable"
    assert view.spec_json is None
    assert view.manual is False


def test_preview_with_no_engine_refuses_with_the_no_engine_message(no_engine_host):
    with pytest.raises(ProviderError, match=_NO_ENGINE):
        app_placement.preview_placement(_SPEC)


def test_the_cli_shows_the_saved_spec(saved_spec, tmp_path):
    result = CliRunner().invoke(placement_app, ["show", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "Placement (manual)" in result.output


def test_the_cli_json_show_parses(saved_spec, tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "json_mode", True)
    result = CliRunner().invoke(placement_app, ["show", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["spec_json"] == _SPEC.to_json()


def test_the_cli_preview_refuses_without_a_traceback(no_engine_host):
    result = CliRunner().invoke(placement_app, ["preview"])
    assert result.exit_code == 1
    assert _NO_ENGINE in " ".join(result.output.split())
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_the_rest_route_shows_the_saved_spec(saved_spec):
    with create_test_client(route_handlers=[placement_route]) as client:
        response = client.get("/api/placement")
    assert response.status_code == 200
    assert response.json()["spec_json"] == _SPEC.to_json()


def test_the_rest_preview_refuses_with_the_no_engine_message(no_engine_host):
    with create_test_client(route_handlers=[placement_preview_route]) as client:
        response = client.post("/api/placement/preview", json={})
    assert response.status_code == 503
    assert _NO_ENGINE in response.json()["detail"]


def test_the_mcp_tool_shows_the_saved_spec(saved_spec):
    assert get_placement_tool()["spec_json"] == _SPEC.to_json()


def test_the_mcp_preview_refuses_with_the_no_engine_message(no_engine_host):
    assert _NO_ENGINE in preview_placement_tool()["error"]
