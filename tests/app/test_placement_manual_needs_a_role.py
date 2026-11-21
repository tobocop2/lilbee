"""A saved placement counts as manual only when it places a role; otherwise the view says why.

The planner is the real one, on an engine that has a card and models that are or are not
installed. Every surface reads the same view: the service, the CLI, the MCP tool and the
HTTP route.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lilbee.app import placement as app_placement
from lilbee.app.engine_choice import BookReading
from lilbee.cli.placement import placement_app
from lilbee.core import settings
from lilbee.core.config import cfg
from lilbee.mcp_server import get_placement_tool, preview_placement_tool, set_placement_tool
from lilbee.providers import engine_params
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet.placement import InstancePlan
from lilbee.providers.fleet.placement_spec import PlacementError, PlacementSpec, RolePlacement
from lilbee.providers.fleet.planning import ResolvedPlacement
from lilbee.providers.roles import WorkerRole
from lilbee.server.routes.placement import placement_preview_route, placement_route
from tests._fleet_engines import CardEngine, Role, cards_of, engines_for_all_roles, prepare_host
from tests._litestar_test_client import create_test_client

GIB = 1 << 30
SPEC = PlacementSpec({WorkerRole.CHAT: RolePlacement(devices=(0,))})
WAITS_FOR_A_MODEL = (
    "The saved placement is kept but not used: no model is installed for any role it names. "
    "It applies once a model is installed."
)
PLACES_NO_ROLE = "The saved placement is kept but not used: it places no role on this machine."
NO_ENTRY_FOR_CHAT = "chat has a model but the placement has no entry for it"
CLI = CliRunner()


class _NamedEngine:
    """An engine that only has to be named by the view of a placed role."""

    name = "engine-a"


def _not_installed(ref: str) -> Path:
    raise ProviderError(f"model {ref} is not installed", kind=ProviderErrorKind.NOT_FOUND)


@pytest.fixture
def engine_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A host with an engine and a card, a chat model configured, and the spec saved."""
    roles = (Role(WorkerRole.CHAT, 1),)
    prepare_host(monkeypatch, tmp_path, roles, 64 * GIB)
    engine = CardEngine("engine-a", tmp_path / "engine-server", roles, cards_of((8,)))
    reading = BookReading(adapters=engines_for_all_roles(engine), notices={})
    monkeypatch.setattr(app_placement, "_planning_engines", lambda: reading)
    monkeypatch.setattr(app_placement, "peek_services", lambda: None)
    monkeypatch.setattr(cfg, "data_root", tmp_path)
    settings.update_values(tmp_path, {"placement": SPEC.to_json()})
    monkeypatch.setattr(cfg, "placement", SPEC.to_json())
    return tmp_path


@pytest.fixture
def no_model_installed(engine_host: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The configured chat model is not on this host."""
    monkeypatch.setattr(engine_params, "resolve_model_path", _not_installed)


def _assert_waits(view: app_placement.PlacementView) -> None:
    assert view.manual is False
    assert view.spec_json is None
    assert view.rejected_spec_json == SPEC.to_json()
    assert view.refusal == WAITS_FOR_A_MODEL
    assert view.refusal_kind is app_placement.PlacementRefusal.WAITING_FOR_MODEL
    assert view.roles == ()
    assert [skip.role for skip in view.skipped_not_installed] == [WorkerRole.CHAT]


@pytest.mark.usefixtures("no_model_installed")
def test_show_with_no_model_installed_reports_auto_and_why() -> None:
    _assert_waits(app_placement.get_placement())


@pytest.mark.usefixtures("no_model_installed")
def test_preview_with_no_model_installed_reports_auto_and_why() -> None:
    _assert_waits(app_placement.preview_placement(SPEC))


@pytest.mark.usefixtures("no_model_installed")
def test_set_with_no_model_installed_saves_the_spec_and_reports_auto_and_why() -> None:
    _assert_waits(app_placement.set_placement(SPEC))
    assert cfg.placement == SPEC.to_json()


@pytest.fixture
def show_args(engine_host: Path) -> list[str]:
    """The CLI arguments that read the host's saved config."""
    return ["show", "--data-dir", str(engine_host)]


@pytest.mark.usefixtures("no_model_installed")
def test_the_cli_says_auto_and_why(show_args: list[str]) -> None:
    result = CLI.invoke(placement_app, show_args)
    assert result.exit_code == 0
    shown = " ".join(result.output.split())
    assert "Placement (auto)" in shown
    assert "Placement (manual)" not in shown
    assert WAITS_FOR_A_MODEL in shown
    assert "placement clear" not in shown


@pytest.mark.usefixtures("no_model_installed")
def test_the_cli_json_carries_the_reason_and_keeps_every_field(
    monkeypatch: pytest.MonkeyPatch, show_args: list[str]
) -> None:
    monkeypatch.setattr(cfg, "json_mode", True)
    result = CLI.invoke(placement_app, show_args)
    assert result.exit_code == 0
    body = json.loads(result.stdout)
    assert body["manual"] is False
    assert body["spec_json"] is None
    assert body["rejected_spec_json"] == SPEC.to_json()
    assert body["refusal"] == WAITS_FOR_A_MODEL
    assert body["roles"] == []
    assert body["skipped_not_installed"] == [{"role": "chat", "model": cfg.chat_model}]
    assert {"gpus", "unplaceable", "engine_backend", "hidden_gpus"} <= body.keys()


@pytest.mark.usefixtures("no_model_installed")
def test_the_http_route_says_auto_and_why() -> None:
    with create_test_client(route_handlers=[placement_route]) as client:
        response = client.get("/api/placement")
    assert response.status_code == 200
    body = response.json()
    assert body["manual"] is False
    assert body["spec_json"] is None
    assert body["rejected_spec_json"] == SPEC.to_json()
    assert body["refusal"] == WAITS_FOR_A_MODEL


@pytest.mark.usefixtures("no_model_installed")
def test_the_mcp_tool_says_auto_and_why() -> None:
    body = get_placement_tool()
    assert body["manual"] is False
    assert body["spec_json"] is None
    assert body["rejected_spec_json"] == SPEC.to_json()
    assert body["refusal"] == WAITS_FOR_A_MODEL


@pytest.mark.usefixtures("no_model_installed")
def test_the_mcp_set_tool_says_auto_and_why() -> None:
    body = set_placement_tool(json.loads(SPEC.to_json()))
    assert body["manual"] is False
    assert body["refusal"] == WAITS_FOR_A_MODEL


@pytest.mark.usefixtures("engine_host")
def test_a_spec_that_places_a_role_is_manual_with_no_refusal() -> None:
    view = app_placement.get_placement()
    assert view.manual is True
    assert view.spec_json == SPEC.to_json()
    assert view.rejected_spec_json is None
    assert view.refusal is None
    assert [role.role for role in view.roles] == [WorkerRole.CHAT]


@pytest.mark.usefixtures("engine_host")
def test_preview_of_auto_is_not_manual_and_carries_no_spec() -> None:
    view = app_placement.preview_placement(None)
    assert view.manual is False
    assert view.spec_json is None
    assert view.rejected_spec_json is None
    assert view.refusal is None


@pytest.mark.usefixtures("engine_host")
def test_preview_of_an_empty_spec_with_a_model_refuses_for_the_missing_entry() -> None:
    with pytest.raises(PlacementError, match=NO_ENTRY_FOR_CHAT):
        app_placement.preview_placement(PlacementSpec({}))


@pytest.mark.usefixtures("engine_host")
def test_the_mcp_preview_of_an_empty_spec_with_a_model_refuses_like_the_library() -> None:
    body = preview_placement_tool({})
    assert body.keys() == {"error"}
    assert NO_ENTRY_FOR_CHAT in body["error"]


@pytest.mark.usefixtures("engine_host")
def test_the_http_preview_of_an_empty_spec_with_a_model_refuses_the_same_way() -> None:
    with create_test_client(route_handlers=[placement_preview_route]) as client:
        response = client.post("/api/placement/preview", json={"spec": {}})
    assert response.status_code == 422
    assert NO_ENTRY_FOR_CHAT in response.json()["detail"]


def test_the_cli_preview_of_an_empty_spec_with_a_model_refuses_the_same_way(
    engine_host: Path,
) -> None:
    result = CLI.invoke(placement_app, ["preview", "--spec", "{}", "--data-dir", str(engine_host)])
    assert result.exit_code == 1
    assert NO_ENTRY_FOR_CHAT in " ".join(result.output.split())


@pytest.mark.usefixtures("no_model_installed")
def test_preview_of_an_empty_spec_with_no_model_refuses_because_it_names_no_role() -> None:
    view = app_placement.preview_placement(PlacementSpec({}))
    assert view.manual is False
    assert view.refusal == PLACES_NO_ROLE
    assert view.rejected_spec_json == "{}"


@pytest.mark.usefixtures("no_model_installed")
def test_the_mcp_preview_of_an_empty_spec_with_no_model_refuses_like_the_library() -> None:
    body = preview_placement_tool({})
    assert body["manual"] is False
    assert body["refusal"] == PLACES_NO_ROLE
    assert body["rejected_spec_json"] == "{}"


@pytest.mark.usefixtures("no_model_installed")
def test_the_http_preview_of_an_empty_spec_with_no_model_refuses_the_same_way() -> None:
    with create_test_client(route_handlers=[placement_preview_route]) as client:
        response = client.post("/api/placement/preview", json={"spec": {}})
    assert response.status_code == 200
    assert response.json()["refusal"] == PLACES_NO_ROLE


@pytest.mark.usefixtures("no_model_installed")
def test_the_cli_preview_of_an_empty_spec_with_no_model_refuses_the_same_way(
    engine_host: Path,
) -> None:
    result = CLI.invoke(placement_app, ["preview", "--spec", "{}", "--data-dir", str(engine_host)])
    assert result.exit_code == 0
    assert PLACES_NO_ROLE in " ".join(result.output.split())


@pytest.mark.usefixtures("engine_host")
def test_the_mcp_preview_with_no_spec_is_auto_with_no_refusal() -> None:
    body = preview_placement_tool(None)
    assert body["refusal"] is None
    assert body["rejected_spec_json"] is None


@pytest.mark.usefixtures("engine_host")
def test_set_and_clear_keep_their_views() -> None:
    saved = app_placement.set_placement(SPEC)
    assert saved.manual is True
    assert saved.spec_json == SPEC.to_json()
    cleared = app_placement.set_placement(None)
    assert cleared.manual is False
    assert cleared.refusal is None
    assert cleared.rejected_spec_json is None


def test_a_spec_that_places_nothing_for_another_reason_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty = ResolvedPlacement(devices=(), instances=(), unplaceable_roles=(), model_refs={})
    monkeypatch.setattr(app_placement, "resolve_placement_plan", lambda spec, **_kw: empty)
    monkeypatch.setattr(cfg, "placement", SPEC.to_json())
    view = app_placement._read_placement(BookReading({}, {}))
    assert view.manual is False
    assert view.spec_json is None
    assert view.refusal == PLACES_NO_ROLE
    assert view.refusal_kind is app_placement.PlacementRefusal.REFUSED
    assert view.rejected_spec_json == SPEC.to_json()


def test_a_spec_that_places_a_role_beside_a_missing_model_stays_manual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial = ResolvedPlacement(
        devices=(),
        instances=(InstancePlan(role=WorkerRole.EMBED, devices=(0,), tensor_split=None),),
        unplaceable_roles=(),
        model_refs={WorkerRole.EMBED: "org/embed.gguf"},
        skipped_not_installed={WorkerRole.CHAT: "org/chat.gguf"},
    )
    monkeypatch.setattr(app_placement, "resolve_placement_plan", lambda spec, **_kw: partial)
    monkeypatch.setattr(cfg, "placement", SPEC.to_json())
    reading = BookReading({WorkerRole.EMBED: _NamedEngine()}, {})  # type: ignore[dict-item]
    view = app_placement._read_placement(reading)
    assert view.manual is True
    assert view.spec_json == SPEC.to_json()
    assert view.refusal is None
    assert [skip.role for skip in view.skipped_not_installed] == [WorkerRole.CHAT]
