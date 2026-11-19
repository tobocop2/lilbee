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
CPU_EMBED = '{"chat": {"target": "auto"}, "embed": {"target": "cpu"}}'


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
    def test_the_launch_still_starts_the_other_roles(self, host: CardEngine) -> None:
        launch = planning.plan_all_launches(engines=engines_for_all_roles(host))
        assert [ln.role for ln in launch.launches] == [WorkerRole.EMBED]

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
        devices = [
            FleetDevice("CUDA", 0, "gpu0", 24 * GIB, 10 * GIB),
            FleetDevice("CUDA", 1, "igpu", 16 * GIB, 16 * GIB, True),
        ]
        return CardEngine("engine-a", tmp_path / "engine-server", ROLES, devices, spans=True)

    def _the_check_call(
        self, engine: CardEngine, monkeypatch: pytest.MonkeyPatch, spec: PlacementSpec | None
    ) -> dict[str, object]:
        seen: dict[str, object] = {}
        real = planning._launches_and_refusals

        def spy(*args: object, **kwargs: object) -> object:
            seen.update(kwargs)
            return real(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(planning, "_launches_and_refusals", spy)
        planning.resolve_placement_plan(spec, engines=engines_for_all_roles(engine))
        return seen

    def _capacity_of_the_check(
        self, engine: CardEngine, monkeypatch: pytest.MonkeyPatch, spec: PlacementSpec | None
    ) -> object:
        return self._the_check_call(engine, monkeypatch, spec)["capacity"]

    def test_the_check_reads_total_memory_on_a_box_in_use(
        self, engine: CardEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = self._the_check_call(engine, monkeypatch, None)
        assert seen["clean_box"] is False
        assert seen["capacity"] == {0: 24 * GIB}

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


@pytest.mark.usefixtures("stated_sizes")
class TestARefusedRoleIsNotTight:
    """A role the launch refuses is not placed, so no surface lists it with a shortfall."""

    @pytest.fixture
    def engine(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _FolderEngine:
        return _folder_host(monkeypatch, tmp_path, cards_of((24,)))

    def test_the_resolved_plan_has_the_refusal_and_no_shortfall(
        self, engine: _FolderEngine
    ) -> None:
        resolved = planning.resolve_placement_plan(None, engines=engines_for_all_roles(engine))
        assert list(resolved.unlaunched_roles) == [WorkerRole.CHAT]
        assert resolved.tight_roles == {}

    def test_the_view_has_the_refusal_and_no_tight_role(self, engine: _FolderEngine) -> None:
        view = app_placement.get_placement()
        assert view.refusal is not None
        assert view.tight == ()

    def test_the_cli_json_lists_no_tight_role(
        self, engine: _FolderEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "json_mode", True)
        result = CliRunner().invoke(placement_app, ["show"])
        assert result.exit_code == 0
        body = json.loads(result.output)
        assert body["refusal"]
        assert body["tight"] == []

    def test_rest_lists_no_tight_role(self, engine: _FolderEngine) -> None:
        with create_test_client([placement_route]) as client:
            body = client.get("/api/placement").json()
        assert body["refusal"]
        assert body["tight"] == []

    def test_rest_preview_lists_no_tight_role(self, engine: _FolderEngine) -> None:
        with create_test_client([placement_preview_route]) as client:
            body = client.post("/api/placement/preview", json={"spec": None}).json()
        assert body["refusal"]
        assert body["tight"] == []

    def test_mcp_lists_no_tight_role(self, engine: _FolderEngine) -> None:
        body = mcp_server.get_placement_tool()
        assert body["refusal"]
        assert body["tight"] == []


class TestAPinnedWindowWarnsOnce:
    """A num_ctx pin past the trained window is warned about once per placement read."""

    TRAINED = 2048
    WARNING = "was trained for"

    @pytest.fixture
    def pinned(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> CardEngine:
        prepare_host(monkeypatch, tmp_path, ROLES, 64 * GIB)
        monkeypatch.setattr(cfg, "num_ctx", 4096)
        monkeypatch.setattr(planning, "_known_chat_ceiling", lambda *_a, **_k: self.TRAINED)
        engine = CardEngine(
            "engine-a", tmp_path / "engine-server", ROLES, cards_of((24,)), spans=True
        )
        reading = BookReading(adapters=engines_for_all_roles(engine), notices={})
        monkeypatch.setattr(app_placement, "_planning_engines", lambda: reading)
        caplog.set_level("WARNING")
        return engine

    def _warnings(self, caplog: pytest.LogCaptureFixture) -> list[str]:
        return [r.getMessage() for r in caplog.records if self.WARNING in r.getMessage()]

    def test_the_view_prints_the_line_once(
        self, pinned: CardEngine, caplog: pytest.LogCaptureFixture
    ) -> None:
        view = app_placement.get_placement()
        assert view.refusal is None
        assert len(self._warnings(caplog)) == 1

    def test_the_line_names_the_pin_and_the_trained_window(
        self, pinned: CardEngine, caplog: pytest.LogCaptureFixture
    ) -> None:
        app_placement.get_placement()
        assert "num_ctx is set to 4096" in self._warnings(caplog)[0]
        assert "trained for 2048" in self._warnings(caplog)[0]

    def test_the_launch_prints_the_line_once(
        self, pinned: CardEngine, caplog: pytest.LogCaptureFixture
    ) -> None:
        planning.plan_all_launches(engines=engines_for_all_roles(pinned))
        assert len(self._warnings(caplog)) == 1

    def test_a_pin_inside_the_trained_window_prints_nothing(
        self, pinned: CardEngine, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "num_ctx", 1024)
        app_placement.get_placement()
        assert self._warnings(caplog) == []

    def test_a_launch_with_a_cpu_role_prints_the_line_once(
        self, pinned: CardEngine, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "placement", CPU_EMBED)
        plan = planning.plan_all_launches(engines=engines_for_all_roles(pinned))
        assert {ln.role for ln in plan.launches} == {WorkerRole.CHAT, WorkerRole.EMBED}
        assert len(self._warnings(caplog)) == 1

    def test_the_view_of_a_cpu_role_spec_prints_the_line_once(
        self, pinned: CardEngine, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "placement", CPU_EMBED)
        view = app_placement.get_placement()
        assert view.refusal is None
        assert {r.role for r in view.roles} == {WorkerRole.CHAT, WorkerRole.EMBED}
        assert len(self._warnings(caplog)) == 1

    def test_the_next_launch_warns_again_for_the_same_window(
        self, pinned: CardEngine, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "placement", CPU_EMBED)
        for _ in range(2):
            planning.plan_all_launches(engines=engines_for_all_roles(pinned))
        assert len(self._warnings(caplog)) == 2

    def test_a_second_window_warns_beside_the_first(
        self, pinned: CardEngine, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "placement", CPU_EMBED)
        planning.plan_all_launches(engines=engines_for_all_roles(pinned))
        monkeypatch.setattr(planning, "_known_chat_ceiling", lambda *_a, **_k: 1024)
        planning.plan_all_launches(engines=engines_for_all_roles(pinned))
        lines = self._warnings(caplog)
        assert len(lines) == 2
        assert "trained for 2048" in lines[0]
        assert "trained for 1024" in lines[1]

    def test_a_plan_scope_warns_once_per_window_across_nested_plans(
        self, pinned: CardEngine, caplog: pytest.LogCaptureFixture
    ) -> None:
        with planning._one_warning_per_plan():
            for _ in range(2):
                planning.plan_all_launches(engines=engines_for_all_roles(pinned))
        assert len(self._warnings(caplog)) == 1

    def test_a_scope_warns_again_for_another_model_or_another_pin(
        self, pinned: CardEngine, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with planning._one_warning_per_plan():
            planning._served_pinned_ctx(Path("/m/a.gguf"), None)
            planning._served_pinned_ctx(Path("/m/a.gguf"), None)
            assert len(self._warnings(caplog)) == 1
            planning._served_pinned_ctx(Path("/m/b.gguf"), None)
            assert len(self._warnings(caplog)) == 2
            monkeypatch.setattr(cfg, "num_ctx", 8192)
            planning._served_pinned_ctx(Path("/m/a.gguf"), None)
        assert len(self._warnings(caplog)) == 3

    def test_a_launch_outside_any_plan_warns_every_time(
        self, pinned: CardEngine, caplog: pytest.LogCaptureFixture
    ) -> None:
        planning._served_pinned_ctx(Path("/m/a.gguf"), None)
        planning._served_pinned_ctx(Path("/m/a.gguf"), None)
        assert len(self._warnings(caplog)) == 2

    def test_the_sizing_asks_for_the_ceiling_of_the_model_and_its_header(
        self, pinned: CardEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asked: list[tuple[Path, dict[str, str] | None]] = []

        def ceiling(path: Path, meta: dict[str, str] | None) -> int:
            asked.append((path, meta))
            return self.TRAINED

        monkeypatch.setattr(planning, "_known_chat_ceiling", ceiling)
        header = {"context_length": "2048"}
        planning._served_pinned_ctx(Path("/m/a.gguf"), header)
        assert asked == [(Path("/m/a.gguf"), header)]

    def test_the_line_reads_in_full(
        self, pinned: CardEngine, caplog: pytest.LogCaptureFixture
    ) -> None:
        planning._served_pinned_ctx(Path("/m/a.gguf"), None)
        assert self._warnings(caplog) == [
            "num_ctx is set to 4096 but a.gguf was trained for 2048, so 2048 is what will be "
            "served. Lower num_ctx to stop planning against a window this model does not have."
        ]

    def test_a_scope_warns_again_when_the_served_window_moves(
        self, pinned: CardEngine, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with planning._one_warning_per_plan():
            planning._served_pinned_ctx(Path("/m/a.gguf"), None)
            monkeypatch.setattr(planning, "_known_chat_ceiling", lambda *_a, **_k: 1024)
            planning._served_pinned_ctx(Path("/m/a.gguf"), None)
        assert len(self._warnings(caplog)) == 2

    def test_a_scope_prints_one_line_for_two_files_of_one_name(
        self, pinned: CardEngine, caplog: pytest.LogCaptureFixture
    ) -> None:
        with planning._one_warning_per_plan():
            planning._served_pinned_ctx(Path("/x/a.gguf"), None)
            planning._served_pinned_ctx(Path("/y/a.gguf"), None)
        assert len(self._warnings(caplog)) == 1


class TestTheLaunchWarningsWarnOncePerPlan:
    """The network-filesystem and no-NVLink lines print once per plan on every launch path."""

    NETWORK = "served from a network filesystem"
    PCIE = "without NVLink"

    def _host(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        cards: tuple[int, ...],
        chat_gib: float,
    ) -> CardEngine:
        roles = (Role(WorkerRole.CHAT, chat_gib), Role(WorkerRole.EMBED, 6))
        prepare_host(monkeypatch, tmp_path, roles, 64 * GIB)
        monkeypatch.setattr(cfg, "placement", CPU_EMBED)
        engine = CardEngine(
            "engine-a", tmp_path / "engine-server", roles, cards_of(cards), spans=True
        )
        reading = BookReading(adapters=engines_for_all_roles(engine), notices={})
        monkeypatch.setattr(app_placement, "_planning_engines", lambda: reading)
        caplog.set_level("WARNING")
        return engine

    @pytest.fixture
    def network(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> CardEngine:
        engine = self._host(monkeypatch, tmp_path, caplog, (24,), 10)
        monkeypatch.setattr(planning, "is_network_path", lambda _p: True)
        monkeypatch.setattr(planning.model_cache, "total_system_memory", lambda: 1000)
        return engine

    @pytest.fixture
    def split(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> CardEngine:
        engine = self._host(monkeypatch, tmp_path, caplog, (24, 24), 30)
        monkeypatch.setattr(planning, "host_lacks_nvlink", lambda: True)
        return engine

    def _lines(self, caplog: pytest.LogCaptureFixture, text: str) -> list[str]:
        return [r.getMessage() for r in caplog.records if text in r.getMessage()]

    def test_a_launch_prints_the_network_line_once(
        self, network: CardEngine, caplog: pytest.LogCaptureFixture
    ) -> None:
        plan = planning.plan_all_launches(engines=engines_for_all_roles(network))
        assert {ln.role for ln in plan.launches} == {WorkerRole.CHAT, WorkerRole.EMBED}
        assert len(self._lines(caplog, self.NETWORK)) == 1

    def test_the_view_prints_the_network_line_once(
        self, network: CardEngine, caplog: pytest.LogCaptureFixture
    ) -> None:
        view = app_placement.get_placement()
        assert {r.role for r in view.roles} == {WorkerRole.CHAT, WorkerRole.EMBED}
        assert len(self._lines(caplog, self.NETWORK)) == 1

    def test_the_next_launch_prints_the_network_line_again(
        self, network: CardEngine, caplog: pytest.LogCaptureFixture
    ) -> None:
        for _ in range(2):
            planning.plan_all_launches(engines=engines_for_all_roles(network))
        assert len(self._lines(caplog, self.NETWORK)) == 2

    def test_a_launch_prints_the_split_line_once(
        self, split: CardEngine, caplog: pytest.LogCaptureFixture
    ) -> None:
        plan = planning.plan_all_launches(engines=engines_for_all_roles(split))
        assert WorkerRole.CHAT in {ln.role for ln in plan.launches}
        assert len(self._lines(caplog, self.PCIE)) == 1

    def test_the_view_prints_the_split_line_once(
        self, split: CardEngine, caplog: pytest.LogCaptureFixture
    ) -> None:
        app_placement.get_placement()
        assert len(self._lines(caplog, self.PCIE)) == 1

    def test_the_next_launch_prints_the_split_line_again(
        self, split: CardEngine, caplog: pytest.LogCaptureFixture
    ) -> None:
        for _ in range(2):
            planning.plan_all_launches(engines=engines_for_all_roles(split))
        assert len(self._lines(caplog, self.PCIE)) == 2

    def test_a_scope_keeps_one_record_for_each_kind_of_line_and_each_key(self) -> None:
        kind = planning._LaunchWarning
        with planning._one_warning_per_plan():
            assert planning._first_warning(kind.NETWORK_FS, "m")
            assert not planning._first_warning(kind.NETWORK_FS, "m")
            assert planning._first_warning(kind.NETWORK_FS, "other")
            assert planning._first_warning(kind.PCIE_SPLIT, "m", (0, 1))
            assert not planning._first_warning(kind.PCIE_SPLIT, "m", (0, 1))
            assert planning._first_warning(kind.PCIE_SPLIT, "m", (0, 2))
            assert planning._first_warning(kind.PCIE_SPLIT, "m")

    def test_outside_a_plan_every_line_prints(self) -> None:
        kind = planning._LaunchWarning
        assert planning._first_warning(kind.NETWORK_FS, "m")
        assert planning._first_warning(kind.NETWORK_FS, "m")


class TestASharedMemoryHostBudget:
    """On a host whose only card is shared memory, the free-memory budget sizes the launch."""

    CHAT = (Role(WorkerRole.CHAT, 10),)

    def _engine(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, free: int
    ) -> tuple[CardEngine, list[FleetDevice]]:
        prepare_host(monkeypatch, tmp_path, self.CHAT, 64 * GIB)
        monkeypatch.setattr(planning.model_cache, "free_system_memory", lambda: free)
        devices = [FleetDevice("CUDA", 0, "igpu", 16 * GIB, 16 * GIB, True)]
        return CardEngine("engine-a", tmp_path / "engine-server", self.CHAT, devices), devices

    def _chat_slots(self, engine: CardEngine) -> int:
        plan = planning.plan_all_launches(engines=engines_for_all_roles(engine))
        (chat,) = [ln for ln in plan.launches if ln.role is WorkerRole.CHAT]
        return chat.slots

    def test_a_roomy_budget_serves_every_slot(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        engine, _ = self._engine(monkeypatch, tmp_path, 64 * GIB)
        assert self._chat_slots(engine) == planning._CHAT_SLOTS

    def test_a_budget_below_the_chat_footprint_serves_one_slot(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        engine, _ = self._engine(monkeypatch, tmp_path, 8 * GIB)
        assert self._chat_slots(engine) == 1

    def test_the_view_check_gets_the_budget_the_launch_gets(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        engine, devices = self._engine(monkeypatch, tmp_path, 8 * GIB)
        budgets: list[int | None] = []
        real = planning._launches_and_refusals

        def spy(*args: object, **kwargs: object) -> object:
            budgets.append(kwargs["unified_budget"])  # type: ignore[arg-type]
            return real(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(planning, "_launches_and_refusals", spy)
        engines = engines_for_all_roles(engine)
        planning.plan_all_launches(engines=engines)
        planning.resolve_placement_plan(None, engines=engines)
        expected = planning._unified_memory_budget(devices, engine)
        assert expected is not None
        assert budgets == [expected, expected]


class TestAnUnplaceableRoleCarriesTheLaunchReason:
    """A role the shared-memory budget refuses shows the sentence the launch logs."""

    BUDGET_ROLES = (Role(WorkerRole.CHAT, 40), Role(WorkerRole.EMBED, 1))
    BUDGET_TEXT = "does not fit available memory and will not be served"

    @pytest.fixture
    def engine(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> CardEngine:
        prepare_host(monkeypatch, tmp_path, self.BUDGET_ROLES, 16 * GIB)
        devices = [FleetDevice("CUDA", 0, "igpu", 16 * GIB, 16 * GIB, True)]
        engine = CardEngine("engine-a", tmp_path / "engine-server", self.BUDGET_ROLES, devices)
        reading = BookReading(adapters=engines_for_all_roles(engine), notices={})
        monkeypatch.setattr(app_placement, "_planning_engines", lambda: reading)
        return engine

    def _logged_reason(self, engine: CardEngine, caplog: pytest.LogCaptureFixture) -> str:
        with caplog.at_level("WARNING", logger=planning.__name__):
            planning.plan_all_launches(engines=engines_for_all_roles(engine))
        (reason,) = [r.getMessage() for r in caplog.records if self.BUDGET_TEXT in r.getMessage()]
        return reason

    def test_the_resolved_plan_carries_the_logged_sentence(
        self, engine: CardEngine, caplog: pytest.LogCaptureFixture
    ) -> None:
        logged = self._logged_reason(engine, caplog)
        resolved = planning.resolve_placement_plan(None, engines=engines_for_all_roles(engine))
        assert resolved.unplaceable_roles == (WorkerRole.CHAT,)
        assert resolved.unplaceable_reasons == {WorkerRole.CHAT: logged}

    def test_the_view_refusal_is_the_logged_sentence(
        self, engine: CardEngine, caplog: pytest.LogCaptureFixture
    ) -> None:
        logged = self._logged_reason(engine, caplog)
        view = app_placement.get_placement()
        assert view.unplaceable == (WorkerRole.CHAT,)
        assert view.refusal == logged

    def test_rest_carries_the_logged_sentence(
        self, engine: CardEngine, caplog: pytest.LogCaptureFixture
    ) -> None:
        logged = self._logged_reason(engine, caplog)
        with create_test_client([placement_route]) as client:
            body = client.get("/api/placement").json()
        assert body["unplaceable"] == ["chat"]
        assert body["refusal"] == logged

    def test_mcp_carries_the_logged_sentence(
        self, engine: CardEngine, caplog: pytest.LogCaptureFixture
    ) -> None:
        logged = self._logged_reason(engine, caplog)
        assert mcp_server.get_placement_tool()["refusal"] == logged

    def test_the_cli_prints_the_sentence_without_a_clear_instruction(
        self, engine: CardEngine
    ) -> None:
        result = CliRunner().invoke(placement_app, ["show"])
        shown = " ".join(result.output.split())
        assert self.BUDGET_TEXT in shown
        assert "placement clear" not in shown

    def test_a_role_that_fits_leaves_the_refusal_empty(
        self, engine: CardEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "chat_model", "")
        view = app_placement.get_placement()
        assert view.unplaceable == ()
        assert view.refusal is None
