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


def _folder_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, devices: list[FleetDevice]
) -> _FolderEngine:
    """A host whose chat model is a 22 GiB folder, on *devices*."""
    prepare_host(monkeypatch, tmp_path, (Role(WorkerRole.CHAT, 22),), 256 * GIB)
    folder = tmp_path / "chat-folder"
    folder.mkdir()
    state_size(folder / "model.safetensors", 22 * GIB)
    (folder / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(engine_params, "resolve_model_path", lambda _ref: folder)
    (tmp_path / "engine-server").write_text("", encoding="utf-8")
    engine = _FolderEngine(tmp_path / "engine-server", devices)
    reading = BookReading(adapters=engines_for_all_roles(engine), notices={})
    monkeypatch.setattr(app_placement, "_planning_engines", lambda: reading)
    return engine


@pytest.mark.usefixtures("stated_sizes")
class TestAChatPastItsCard:
    """The card refusal of the launch, which the plan alone calls a tight placement."""

    @pytest.fixture
    def engine(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _FolderEngine:
        return _folder_host(monkeypatch, tmp_path, cards_of((24,)))

    def test_the_view_lists_it_with_the_reason_the_launch_gives(
        self, engine: _FolderEngine
    ) -> None:
        launch = planning.plan_all_launches(engines=engines_for_all_roles(engine))
        view = app_placement.get_placement()
        assert launch.refused_roles[WorkerRole.CHAT].startswith("The chat model")
        assert view.refusal == launch.refused_roles[WorkerRole.CHAT]
        assert view.roles == ()


class TestTheRefusalText:
    """The view line is the launch reason, joined with a spec refusal when there is one."""

    def test_the_line_is_the_launch_reason_alone(self, host: CardEngine) -> None:
        launch = planning.plan_all_launches(engines=engines_for_all_roles(host))
        view = app_placement.get_placement()
        assert view.refusal == launch.refused_roles[WorkerRole.CHAT]

    def test_a_spec_refusal_and_the_launch_reason_share_one_line(self, host: CardEngine) -> None:
        reading = app_placement._planning_engines()
        resolved = app_placement._resolve(None, reading)
        launch = planning.plan_all_launches(engines=engines_for_all_roles(host))
        view = app_placement._view(
            resolved, reading.adapters, manual=False, spec_json=None, refusal="The spec is off."
        )
        assert view.refusal == f"The spec is off. {launch.refused_roles[WorkerRole.CHAT]}"

    def test_a_spec_refusal_stands_alone_without_a_launch_refusal(
        self, host: CardEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(planning, "_role_ctx", lambda *_args, **_kwargs: 8192)
        reading = app_placement._planning_engines()
        resolved = app_placement._resolve(None, reading)
        view = app_placement._view(
            resolved, reading.adapters, manual=False, spec_json=None, refusal="The spec is off."
        )
        assert view.refusal == "The spec is off."


class TestTheResolvedPlan:
    def test_the_refused_role_leaves_the_instances_and_is_reported(self, host: CardEngine) -> None:
        resolved = planning.resolve_placement_plan(None, engines=engines_for_all_roles(host))
        assert [inst.role for inst in resolved.instances] == [WorkerRole.EMBED]
        assert list(resolved.unlaunched_roles) == [WorkerRole.CHAT]
        assert FLOOR_TEXT in resolved.unlaunched_roles[WorkerRole.CHAT]

    def test_a_window_that_holds_the_floor_reports_nothing(
        self, host: CardEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(planning, "_role_ctx", lambda *_args, **_kwargs: 8192)
        resolved = planning.resolve_placement_plan(None, engines=engines_for_all_roles(host))
        assert resolved.unlaunched_roles == {}
        assert {inst.role for inst in resolved.instances} == {WorkerRole.CHAT, WorkerRole.EMBED}

    def test_the_plan_that_launches_and_the_plan_that_shows_agree(self, host: CardEngine) -> None:
        engines = engines_for_all_roles(host)
        launch = planning.plan_all_launches(engines=engines)
        resolved = planning.resolve_placement_plan(None, engines=engines)
        assert resolved.unlaunched_roles == launch.refused_roles
        assert {inst.role for inst in resolved.instances} == {ln.role for ln in launch.launches}


class TestTheCapacityTheViewChecks:
    """Auto packing skips a shared-memory card, but a pin still names it."""

    @pytest.fixture
    def engine(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> CardEngine:
        prepare_host(monkeypatch, tmp_path, ROLES, 64 * GIB)
        devices = [*cards_of((24,)), FleetDevice("CUDA", 1, "igpu", 16 * GIB, 16 * GIB, True)]
        return CardEngine("engine-a", tmp_path / "engine-server", ROLES, devices, spans=True)

    def _capacity_of_the_check(
        self, engine: CardEngine, monkeypatch: pytest.MonkeyPatch, spec: PlacementSpec | None
    ) -> dict[int, int]:
        seen: dict[int, int] = {}
        real = planning._launches_and_refusals

        def spy(*args: object, **kwargs: object) -> object:
            seen.update(kwargs["capacity"])  # type: ignore[arg-type]
            return real(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(planning, "_launches_and_refusals", spy)
        planning.resolve_placement_plan(spec, engines=engines_for_all_roles(engine))
        return seen

    def test_a_pinned_shared_memory_card_keeps_its_capacity(
        self, engine: CardEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spec = PlacementSpec.from_json('{"chat": {"devices": [1]}, "embed": {"devices": [0]}}')
        assert self._capacity_of_the_check(engine, monkeypatch, spec) == {
            0: 24 * GIB,
            1: 16 * GIB,
        }

    def test_an_unpinned_shared_memory_card_is_left_out(
        self, engine: CardEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert self._capacity_of_the_check(engine, monkeypatch, None) == {0: 24 * GIB}


class TestNamedCards:
    def test_no_spec_names_no_card(self) -> None:
        assert planning._named_cards(None) == set()

    def test_the_cards_a_spec_pins_a_role_to(self) -> None:
        spec = PlacementSpec.from_json('{"chat": {"devices": [1]}, "embed": {"devices": [0, 2]}}')
        assert planning._named_cards(spec) == {0, 1, 2}
