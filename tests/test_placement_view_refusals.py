"""The placement view lists each role the launch refuses, with the reason it gives."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lilbee import mcp_server
from lilbee.app import placement as app_placement
from lilbee.app.engine_choice import BookReading
from lilbee.cli.placement import placement_app
from lilbee.core.config import cfg
from lilbee.providers import engine_params
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.devices import DeviceProbe, FleetDevice
from lilbee.providers.fleet.engines.mistral import MistralAdapter
from lilbee.providers.fleet.placement_spec import PlacementSpec
from lilbee.providers.roles import WorkerRole
from lilbee.server.routes.placement import placement_preview_route, placement_route
from tests._fleet_engines import (
    CardEngine,
    Role,
    cards_of,
    engines_for_all_roles,
    prepare_host,
)
from tests._litestar_test_client import create_test_client
from tests._stated_sizes import state_size

GIB = 1 << 30
ROLES = (Role(WorkerRole.CHAT, 10), Role(WorkerRole.EMBED, 6))
LOW_WINDOW = 512
FLOOR_TEXT = "backs only a 512-token context"
PINNED_CHAT = '{"chat": {"devices": [0]}, "embed": {"devices": [0]}}'


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> CardEngine:
    """A 24 GiB card, a chat and an embed model, and a chat window below the floor."""
    prepare_host(monkeypatch, tmp_path, ROLES, 64 * GIB)
    monkeypatch.setattr(planning, "_role_ctx", lambda *_args, **_kwargs: LOW_WINDOW)
    engine = CardEngine("engine-a", tmp_path / "engine-server", ROLES, cards_of((24,)), spans=True)
    reading = BookReading(adapters=engines_for_all_roles(engine), notices={})
    monkeypatch.setattr(app_placement, "_planning_engines", lambda: reading)
    return engine


class TestAChatBelowTheFloor:
    def test_the_launch_refuses_it_with_the_floor_reason(self, host: CardEngine) -> None:
        plan = planning.plan_all_launches(engines=engines_for_all_roles(host))
        assert FLOOR_TEXT in plan.refused_roles[WorkerRole.CHAT]

    def test_the_auto_view_lists_it_as_refused_and_drops_its_row(self, host: CardEngine) -> None:
        view = app_placement.get_placement()
        assert view.refusal is not None
        assert FLOOR_TEXT in view.refusal
        assert [r.role for r in view.roles] == [WorkerRole.EMBED]

    def test_a_manual_view_keeps_its_spec_and_lists_the_refusal(
        self, host: CardEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "placement", PINNED_CHAT)
        view = app_placement.get_placement()
        assert view.manual is True
        assert view.spec_json is not None
        assert view.rejected_spec_json is None
        assert view.refusal is not None
        assert FLOOR_TEXT in view.refusal
        assert [r.role for r in view.roles] == [WorkerRole.EMBED]

    def test_the_preview_lists_it_as_refused(self, host: CardEngine) -> None:
        view = app_placement.preview_placement(PlacementSpec.from_json(PINNED_CHAT))
        assert view.refusal is not None
        assert FLOOR_TEXT in view.refusal

    def test_a_window_that_holds_the_floor_is_not_refused(
        self, host: CardEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(planning, "_role_ctx", lambda *_args, **_kwargs: 8192)
        view = app_placement.get_placement()
        assert view.refusal is None
        assert {r.role for r in view.roles} == {WorkerRole.CHAT, WorkerRole.EMBED}


class TestEverySurfaceListsTheRefusal:
    def test_rest_get_carries_the_refusal(self, host: CardEngine) -> None:
        with create_test_client([placement_route]) as client:
            body = client.get("/api/placement").json()
        assert FLOOR_TEXT in body["refusal"]
        assert [r["role"] for r in body["roles"]] == ["embed"]

    def test_rest_preview_carries_the_refusal(self, host: CardEngine) -> None:
        with create_test_client([placement_preview_route]) as client:
            body = client.post("/api/placement/preview", json={"spec": None}).json()
        assert FLOOR_TEXT in body["refusal"]

    def test_the_cli_prints_the_refusal(self, host: CardEngine) -> None:
        result = CliRunner().invoke(placement_app, ["show"])
        assert result.exit_code == 0
        assert FLOOR_TEXT in " ".join(result.output.split())

    def test_the_cli_json_carries_the_refusal(
        self, host: CardEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "json_mode", True)
        result = CliRunner().invoke(placement_app, ["show"])
        assert result.exit_code == 0
        assert FLOOR_TEXT in json.loads(result.output)["refusal"]

    def test_mcp_carries_the_refusal(self, host: CardEngine) -> None:
        body = mcp_server.get_placement_tool()
        assert FLOOR_TEXT in body["refusal"]


class _FolderEngine(MistralAdapter):
    """The real folder engine, on a host with one known binary and known cards."""

    def __init__(self, binary_path: Path, devices: list[FleetDevice]) -> None:
        self._binary = binary_path
        self._devices = devices

    def binary(self) -> Path:
        return self._binary

    def discover_devices(self, binary: Path) -> DeviceProbe:
        return DeviceProbe(devices=self._devices, output="")


@pytest.mark.usefixtures("stated_sizes")
class TestAChatPastItsCard:
    """The card refusal of the launch, which the plan alone calls a tight placement."""

    @pytest.fixture
    def engine(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _FolderEngine:
        prepare_host(monkeypatch, tmp_path, (Role(WorkerRole.CHAT, 22),), 256 * GIB)
        folder = tmp_path / "chat-folder"
        folder.mkdir()
        state_size(folder / "model.safetensors", 22 * GIB)
        (folder / "config.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(engine_params, "resolve_model_path", lambda _ref: folder)
        (tmp_path / "engine-server").write_text("", encoding="utf-8")
        engine = _FolderEngine(tmp_path / "engine-server", cards_of((24,)))
        reading = BookReading(adapters=engines_for_all_roles(engine), notices={})
        monkeypatch.setattr(app_placement, "_planning_engines", lambda: reading)
        return engine

    def test_the_view_lists_it_with_the_reason_the_launch_gives(
        self, engine: _FolderEngine
    ) -> None:
        launch = planning.plan_all_launches(engines=engines_for_all_roles(engine))
        view = app_placement.get_placement()
        assert launch.refused_roles[WorkerRole.CHAT].startswith("The chat model")
        assert view.refusal == launch.refused_roles[WorkerRole.CHAT]
        assert view.roles == ()
