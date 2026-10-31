"""Every caller of the planner reads the engine of each role from the engine book at the call."""

from __future__ import annotations

import logging

from lilbee.app import placement
from lilbee.app.engine_choice import BookReading, EngineBook
from lilbee.core.config import cfg
from lilbee.providers.fleet import planning
from lilbee.providers.fleet import provider as prov_mod
from lilbee.providers.fleet.provider import FleetProvider
from lilbee.providers.fleet.swap_manager import SwapManager
from lilbee.providers.roles import EngineBackend, WorkerRole
from tests.test_fleet_mixed_plan import _GIB, _host, _pair, _SizedEngine
from tests.test_fleet_swap_manager import _GROUP, _fake_response, _FakeProc, _launch
from tests.test_fleet_swap_manager import _patch_http as _patch_swap_http
from tests.test_fleet_swap_manager import _patch_spawn as _patch_swap_spawn


def _book(monkeypatch, chat, embed) -> dict[WorkerRole, _SizedEngine]:
    """Put chat on one adapter and embed on another in the engine book."""
    mapping = {WorkerRole.CHAT: chat, WorkerRole.EMBED: embed}
    monkeypatch.setattr(EngineBook, "read", lambda self: BookReading(dict(mapping), {}))
    return mapping


def _backend_by_engine(engine) -> EngineBackend:
    return EngineBackend.CUDA if engine.name == "engine-a" else EngineBackend.VULKAN


class TestTwoAdaptersOnePlan:
    def test_placement_and_the_launch_path_give_one_plan_and_the_same_cards(
        self, monkeypatch, tmp_path
    ) -> None:
        device = _host(monkeypatch, tmp_path, unified=False, total=24 * _GIB, cards=2)
        chat, embed = _pair(tmp_path, device, 8, 2)
        _book(monkeypatch, chat, embed)
        monkeypatch.setattr(placement, "cfg", cfg)
        provider = FleetProvider(engine=chat)

        view = placement.get_placement()
        plan = planning.plan_all_launches(engines=provider._engines())

        assert {role.role: role.engine for role in view.roles} == {
            WorkerRole.CHAT: "engine-a",
            WorkerRole.EMBED: "engine-b",
        }
        assert {role.role for role in view.roles} == {launch.role for launch in plan.launches}
        assert {
            launch.role: launch.env_overrides["CUDA_VISIBLE_DEVICES"] for launch in plan.launches
        } == {role.role: str(role.devices[0]) for role in view.roles}
        assert {role.role: role.devices for role in view.roles} == {
            WorkerRole.CHAT: (1,),
            WorkerRole.EMBED: (0,),
        }

    def test_the_read_path_places_a_role_by_its_own_adapters_figure(
        self, monkeypatch, tmp_path
    ) -> None:
        # Chat 8 GiB and embed 20 GiB do not fit one 24 GiB pool; two embeds of 8 would.
        device = _host(monkeypatch, tmp_path, unified=True, total=24 * _GIB)
        chat, embed = _pair(tmp_path, device, 8, 20)
        mapping = _book(monkeypatch, chat, embed)

        resolved = planning.resolve_placement_plan(None, engines=mapping)

        assert resolved.unplaceable_roles == (WorkerRole.CHAT,)
        assert {instance.role for instance in resolved.instances} == {WorkerRole.EMBED}

    def test_a_role_with_no_engine_does_not_stop_the_read(self, monkeypatch, tmp_path) -> None:
        device = _host(monkeypatch, tmp_path, unified=False, total=24 * _GIB)
        chat, _embed = _pair(tmp_path, device, 8, 2)
        resolved = planning.resolve_placement_plan(None, engines={WorkerRole.CHAT: chat})

        assert {instance.role for instance in resolved.instances} == {WorkerRole.CHAT}

    def test_the_book_leaves_out_a_role_with_no_engine(self, monkeypatch) -> None:
        from lilbee.providers.fleet.engines import choice

        monkeypatch.delitem(choice.ADAPTERS, choice.EngineName.MLX, raising=False)
        monkeypatch.setattr(cfg, "chat_model", "org/repo#mlx")
        found = EngineBook(cfg).adapters()
        assert WorkerRole.CHAT not in found
        assert WorkerRole.EMBED in found


class TestProviderReadsTheBook:
    def test_the_demand_check_gets_the_book_not_one_engine(self, monkeypatch, tmp_path) -> None:
        device = _host(monkeypatch, tmp_path, unified=False, total=24 * _GIB)
        chat, embed = _pair(tmp_path, device, 8, 2)
        mapping = _book(monkeypatch, chat, embed)
        seen: list[object] = []

        def record(engine, engines):
            seen.append(engines)
            return prov_mod._EngineDemand(set(), 0, {}, {})

        monkeypatch.setattr(prov_mod, "_placeable_demand", record)
        monkeypatch.setattr(prov_mod, "kernel_arbitrates_locks", lambda _d: True)
        provider = FleetProvider(engine=chat)
        monkeypatch.setattr(provider, "_acquire_in_dir", lambda *_a, **_k: False)
        provider._acquire_engine(tmp_path)

        assert seen == [mapping]

    def test_adopted_launches_are_logged_with_the_engine_of_their_role(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        device = _host(monkeypatch, tmp_path, unified=False, total=24 * _GIB)
        chat, embed = _pair(tmp_path, device, 8, 2)
        monkeypatch.setattr(planning, "engine_build_id", lambda: "b")
        monkeypatch.setattr(planning, "engine_backend", _backend_by_engine)
        monkeypatch.setattr(planning, "probed_devices", lambda engine: ())
        with caplog.at_level(logging.INFO, logger=planning.__name__):
            planning.log_role_launches(
                [_launch(WorkerRole.CHAT), _launch(WorkerRole.EMBED)],
                engines={WorkerRole.CHAT: chat, WorkerRole.EMBED: embed},
                owner_pid=7,
            )
        lines = [r.message for r in caplog.records if r.message.startswith("Adopted")]
        assert ["backend cuda" in line for line in lines] == [True, False]
        assert ["backend vulkan" in line for line in lines] == [False, True]

    def test_a_role_with_no_engine_now_logs_no_launch(self, monkeypatch, tmp_path, caplog) -> None:
        device = _host(monkeypatch, tmp_path, unified=False, total=24 * _GIB)
        chat, _embed = _pair(tmp_path, device, 8, 2)
        with caplog.at_level(logging.INFO, logger=planning.__name__):
            planning.log_role_launches([_launch(WorkerRole.EMBED)], engines={WorkerRole.CHAT: chat})
        assert not [r for r in caplog.records if "serving" in r.message]


class TestSwapManagerLogsTheRoleEngine:
    def test_each_launch_is_logged_with_the_engine_of_its_role(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        device = _host(monkeypatch, tmp_path, unified=False, total=24 * _GIB)
        chat, embed = _pair(tmp_path, device, 8, 2)
        monkeypatch.setattr(planning, "engine_build_id", lambda: "b")
        monkeypatch.setattr(planning, "engine_backend", _backend_by_engine)
        monkeypatch.setattr(planning, "probed_devices", lambda engine: ())
        _patch_swap_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_swap_http(monkeypatch, lambda _url: _fake_response(status=200))
        mapping = {WorkerRole.CHAT: chat, WorkerRole.EMBED: embed}
        with caplog.at_level(logging.INFO, logger=planning.__name__):
            SwapManager(tmp_path, _GROUP).start(
                [_launch(WorkerRole.CHAT), _launch(WorkerRole.EMBED)], engines=mapping
            )
        lines = [r.message for r in caplog.records if r.message.startswith("Launched")]
        assert ["backend cuda" in line for line in lines] == [True, False]
        assert ["backend vulkan" in line for line in lines] == [False, True]
