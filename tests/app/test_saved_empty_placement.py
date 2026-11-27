"""A saved placement of ``{}`` reads as refused, with the text the set path gives.

The planner is the real one, on an engine that has a card and a chat model. The set and
preview paths refuse ``{}`` because it has no entry for the chat model; the read paths
(the service, the CLI, the HTTP route, the MCP tool) give the same refusal. The CLI
follows it with the clear hint, as it does for every refusal.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lilbee.app import placement as app_placement
from lilbee.app.engine_choice import BookReading
from lilbee.cli.placement import placement_app
from lilbee.core import settings
from lilbee.core.config import cfg
from lilbee.mcp_server import get_placement_tool
from lilbee.providers import engine_params
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet.placement_spec import PlacementError, PlacementSpec
from lilbee.providers.roles import WorkerRole
from lilbee.server.routes.placement import placement_route
from tests._fleet_engines import CardEngine, Role, cards_of, engines_for_all_roles, prepare_host
from tests._litestar_test_client import create_test_client

GIB = 1 << 30
NO_ENTRY_FOR_CHAT = (
    "chat has a model but the placement has no entry for it. "
    "Add an entry for chat with devices or a target, or clear the placement."
)
CLI = CliRunner()


def _not_installed(ref: str) -> Path:
    raise ProviderError(f"model {ref} is not installed", kind=ProviderErrorKind.NOT_FOUND)


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A host with an engine, a card and a chat model, and ``{}`` saved as the placement."""
    roles = (Role(WorkerRole.CHAT, 1),)
    prepare_host(monkeypatch, tmp_path, roles, 64 * GIB)
    engine = CardEngine("engine-a", tmp_path / "engine-server", roles, cards_of((8,)))
    reading = BookReading(adapters=engines_for_all_roles(engine), notices={})
    monkeypatch.setattr(app_placement, "_planning_engines", lambda: reading)
    monkeypatch.setattr(app_placement, "peek_services", lambda: None)
    monkeypatch.setattr(cfg, "data_root", tmp_path)
    settings.update_values(tmp_path, {"placement": "{}"})
    monkeypatch.setattr(cfg, "placement", "{}")
    return tmp_path


def _assert_refused(view: app_placement.PlacementView) -> None:
    assert view.manual is False
    assert view.spec_json is None
    assert view.rejected_spec_json == "{}"
    assert view.refusal is not None
    assert NO_ENTRY_FOR_CHAT in view.refusal
    assert view.refusal_kind is app_placement.PlacementRefusal.REFUSED


@pytest.mark.usefixtures("host")
def test_the_planner_gives_the_refusal_the_set_path_raises() -> None:
    with pytest.raises(PlacementError) as raised:
        app_placement.preview_placement(PlacementSpec.from_json("{}"))
    assert str(raised.value) == NO_ENTRY_FOR_CHAT
    _assert_refused(app_placement.get_placement())


@pytest.mark.usefixtures("host")
def test_with_no_model_installed_the_read_gives_the_text_the_preview_gives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine_params, "resolve_model_path", _not_installed)
    previewed = app_placement.preview_placement(PlacementSpec.from_json("{}"))
    view = app_placement.get_placement()
    assert previewed.refusal == app_placement.SAVED_PLACEMENT_PLACES_NO_ROLE
    assert view.refusal == previewed.refusal
    assert view.refusal_kind is previewed.refusal_kind
    assert view.manual is False


def test_the_cli_shows_the_refusal_before_the_clear_hint(host: Path) -> None:
    result = CLI.invoke(placement_app, ["show", "--data-dir", str(host)])
    assert result.exit_code == 0
    shown = " ".join(result.output.split())
    assert NO_ENTRY_FOR_CHAT in shown
    assert shown.index(NO_ENTRY_FOR_CHAT) < shown.index("placement clear")


@pytest.mark.usefixtures("host")
def test_the_http_route_gives_the_refusal() -> None:
    with create_test_client(route_handlers=[placement_route]) as client:
        response = client.get("/api/placement")
    assert response.status_code == 200
    body = response.json()
    assert body["manual"] is False
    assert body["rejected_spec_json"] == "{}"
    assert NO_ENTRY_FOR_CHAT in body["refusal"]


@pytest.mark.usefixtures("host")
def test_the_mcp_tool_gives_the_refusal() -> None:
    body = get_placement_tool()
    assert body["manual"] is False
    assert NO_ENTRY_FOR_CHAT in body["refusal"]
