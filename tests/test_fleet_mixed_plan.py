"""A plan whose roles run on different engines shares one memory budget."""

from __future__ import annotations

import itertools
import json
import logging
import random
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from lilbee.core.config import cfg
from lilbee.core.config.enums import KvCacheType
from lilbee.providers import engine_params
from lilbee.providers.fleet import devices as devices_module
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.devices import DeviceProbe, FleetDevice
from lilbee.providers.fleet.engines.base import LaunchRequest
from lilbee.providers.fleet.engines.llama import LLAMA
from lilbee.providers.fleet.groups import SwapGroup, group_for
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind, MemoryRequest
from lilbee.providers.fleet.placement import ModelPlacementInput, plan_placement
from lilbee.providers.fleet.swap_config import build_swap_config
from lilbee.providers.roles import ROLE_REGISTRY, EngineBackend, WorkerRole
from tests._fleet_engines import (
    CardEngine,
    Role,
    cards_of,
    engines_for_all_roles,
    model_file,
    prepare_host,
)
from tests.conformance.fake_adapter import FakeAdapter
from tests.test_fleet_one_engine_cards import GOLDEN, SCENARIOS, run_scenario

_GIB = 1 << 30
_PLANNING_LOG = "lilbee.providers.fleet.planning"
_ALL_ROLES = frozenset({WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.VISION})


class _SizedEngine(FakeAdapter):
    """An engine that charges a fixed footprint for every model it loads."""

    served_roles = _ALL_ROLES

    def __init__(self, name: str, binary: Path, kind: MemoryKind, footprint: int, device) -> None:
        self.name = name
        self.memory_kind = kind
        self._binary = binary
        self._footprint = footprint
        self._device = device

    def __hash__(self) -> int:
        return hash(self.name)

    def binary(self) -> Path:
        return self._binary

    def discover_devices(self, binary: Path) -> DeviceProbe:
        return DeviceProbe(
            devices=list(self._device), output="", spoke_protocol=False, backend=EngineBackend.CUDA
        )

    def estimate_memory(self, request) -> MemoryEstimate:
        return MemoryEstimate(
            kind=self.memory_kind,
            device_bytes=self._footprint,
            host_bytes=0,
            shared_bytes=self._footprint,
            per_device=(self._footprint,),
            per_device_shared=(self._footprint,),
        )

    def model_meta(self, path: Path):
        return {"architecture": "fakearch", "context_length": "4096"}

    def check_readback(self, binary: Path) -> bool:
        return False

    def supports_fit_target(self, binary: Path) -> bool:
        return False

    def supports_shared_kv(self, binary: Path) -> bool:
        return False

    def runtime_env(self):
        return {}

    def swap_extras(self, launch):
        return {}


def _host(
    monkeypatch, tmp_path: Path, *, unified: bool, total: int, cards: int = 1
) -> list[FleetDevice]:
    """Patch one machine with *cards* devices of *total* bytes each."""
    model = tmp_path / "model.bin"
    model.write_bytes(b"x" * 4096)
    monkeypatch.setattr(engine_params, "resolve_model_path", lambda ref: model)
    planning.clear_read_device_cache()
    for module in (planning, planning.model_cache):
        monkeypatch.setattr(module, "total_system_memory", lambda: total)
        monkeypatch.setattr(module, "free_system_memory", lambda: total)
    monkeypatch.setattr(cfg, "system_memory_reserve_gb", 0)
    monkeypatch.setattr(cfg, "gpu_memory_fraction", 1.0)
    monkeypatch.setattr(cfg, "num_ctx", None)
    monkeypatch.setattr(cfg, "chat_model", "org/repo/chat.gguf")
    monkeypatch.setattr(cfg, "embedding_model", "org/repo/embed.gguf")
    monkeypatch.setattr(cfg, "reranker_model", "")
    monkeypatch.setattr(cfg, "vision_model", "")
    monkeypatch.setattr(cfg, "placement", None)
    return [
        FleetDevice("CUDA", index, f"gpu{index}", total, total, unified=unified)
        for index in range(cards)
    ]


def _pair(tmp_path: Path, device, chat_gib: int, other_gib: int, *, other_kind=MemoryKind.SIZED):
    """Two engines with their own binaries: one for chat, one for the other role."""
    first = tmp_path / "engine-a-server"
    second = tmp_path / "engine-b-server"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    chat = _SizedEngine("engine-a", first, MemoryKind.SIZED, chat_gib * _GIB, device)
    other = _SizedEngine("engine-b", second, other_kind, other_gib * _GIB, device)
    return chat, other


def _warnings(caplog) -> str:
    return " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


def _m0(model: str) -> str:
    return (
        f"{model} does not fit available memory and will not be served; "
        "free up memory or use a smaller model."
    )


def _m1(model: str, need: float, room: float, others: str) -> str:
    return (
        f"{model} needs {need:.1f} GiB, but the GPU with the most room has {room:.1f} GiB "
        f"left beside the other models ({others}), short by {max(need - room, 0):.1f} GiB, "
        f"so it will not be served. Use a smaller model for {model.split()[0]} or for one of "
        "those."
    )


def _m2(model: str, need: float, usable: float) -> str:
    return (
        f"{model} needs {need:.1f} GiB, more than any one GPU can hold ({usable:.1f} GiB "
        "usable), and no split across the GPUs holds it, so it will not be served. "
        "Use a smaller model."
    )


def _m3(model: str, gpus: int, engine: str) -> str:
    return (
        f"{model} will not be served: this host has {gpus} GPUs and {engine} cannot be "
        "told which one to use. Use another engine for this role."
    )


@pytest.mark.parametrize("unified", [True, False], ids=["unified", "one-gpu"])
class TestMixedPlanOnOneMemoryPool:
    def test_roles_that_fit_together_are_all_served_without_a_warning(
        self, monkeypatch, tmp_path, caplog, unified
    ) -> None:
        device = _host(monkeypatch, tmp_path, unified=unified, total=24 * _GIB)
        chat, embed = _pair(tmp_path, device, 8, 2)
        with caplog.at_level(logging.WARNING, logger=_PLANNING_LOG):
            plan = planning.plan_all_launches(
                engines={WorkerRole.CHAT: chat, WorkerRole.EMBED: embed}
            )
        assert {launch.role for launch in plan.launches} == {WorkerRole.CHAT, WorkerRole.EMBED}
        assert "does not fit" not in _warnings(caplog)
        assert "tight" not in _warnings(caplog)

    def test_a_role_that_does_not_fit_is_refused_and_names_its_engine(
        self, monkeypatch, tmp_path, caplog, unified
    ) -> None:
        device = _host(monkeypatch, tmp_path, unified=unified, total=10 * _GIB)
        chat, embed = _pair(tmp_path, device, 9, 3, other_kind=MemoryKind.FIXED_RESERVE)
        with caplog.at_level(logging.WARNING, logger=_PLANNING_LOG):
            plan = planning.plan_all_launches(
                engines={WorkerRole.CHAT: chat, WorkerRole.EMBED: embed}
            )
        served = {launch.role for launch in plan.launches}
        refused = {WorkerRole.CHAT, WorkerRole.EMBED} - served
        assert len(refused) == 1
        (role,) = refused
        engine = chat if role is WorkerRole.CHAT else embed
        ref = cfg.chat_model if role is WorkerRole.CHAT else cfg.embedding_model
        model = f"{role.value} model {ref} ({engine.name})"
        if unified:
            assert _m0(model) in _warnings(caplog)
        else:
            need, other_need = (9, 3) if role is WorkerRole.CHAT else (3, 9)
            others = "embed" if role is WorkerRole.CHAT else "chat"
            assert _m1(model, need, 9 - other_need, others) in _warnings(caplog)
        assert "tight" not in _warnings(caplog)

    def test_a_role_that_fits_alone_stays_served_beside_one_that_never_fits(
        self, monkeypatch, tmp_path, caplog, unified
    ) -> None:
        device = _host(monkeypatch, tmp_path, unified=unified, total=8 * _GIB)
        chat, embed = _pair(tmp_path, device, 4, 20)
        with caplog.at_level(logging.WARNING, logger=_PLANNING_LOG):
            plan = planning.plan_all_launches(
                engines={WorkerRole.CHAT: chat, WorkerRole.EMBED: embed}
            )
        assert {launch.role for launch in plan.launches} == {WorkerRole.CHAT}
        model = f"embed model {cfg.embedding_model} (engine-b)"
        if unified:
            assert _m0(model) in _warnings(caplog)
        else:
            assert _m2(model, 20, 7.2) in _warnings(caplog)
        assert "chat model" not in _warnings(caplog)
        assert "tight" not in _warnings(caplog)


class TestUnifiedHostRefusal:
    def test_u1_chat_too_big_is_refused_and_embed_takes_no_card(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        device = _host(monkeypatch, tmp_path, unified=True, total=16 * _GIB)
        chat, embed = _pair(tmp_path, device, 20, 4)
        with caplog.at_level(logging.WARNING, logger=_PLANNING_LOG):
            plan = planning.plan_all_launches(
                engines={WorkerRole.CHAT: chat, WorkerRole.EMBED: embed}
            )
        by_role = {launch.role: launch for launch in plan.launches}
        assert set(by_role) == {WorkerRole.EMBED}
        assert _pin_of(by_role[WorkerRole.EMBED], "")
        assert _m0(f"chat model {cfg.chat_model} (engine-a)") in _warnings(caplog)
        assert _warnings(caplog).count("will not be served") == 1


@dataclass(frozen=True)
class Row:
    """One row of the several card rule: the host, the roles and the expected plan."""

    cards: tuple[int, ...]
    roles: tuple[Role, ...]
    served: dict[tuple[WorkerRole, int], tuple[int, ...]]
    refused: frozenset[WorkerRole] = frozenset()
    group: frozenset[WorkerRole] = frozenset()
    split: tuple[int, ...] = ()
    chat_spans: bool = False

    def refusal(self, role: WorkerRole, engine: str) -> str:
        """The exact warning for *role*: no room beside others, or larger than any card."""
        model = f"{role.value} model org/repo/{model_file(role)} ({engine})"
        need = next(r.gib for r in self.roles if r.role is role)
        usable = max(self.cards) * 0.9
        spans = role is WorkerRole.CHAT and self.chat_spans and need <= sum(self.cards) * 0.9
        if need > usable and not spans:
            return _m2(model, need, usable)
        held = [0.0] * len(self.cards)
        for (other, _replica), on in self.served.items():
            charge = next(r.gib for r in self.roles if r.role is other)
            for card in on:
                if other in self.group:
                    held[card] = max(held[card], charge / len(on))
                else:
                    held[card] += charge / len(on)
        room = max(cap * 0.9 - used for cap, used in zip(self.cards, held, strict=True))
        others = ", ".join(sorted({other.value for other, _ in self.served}))
        return _m1(model, need, room, others)


C, E, R, V = WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.RERANK, WorkerRole.VISION
_T_ROWS = {
    "T1": Row((8,), (Role(C, 4), Role(E, 2)), {(E, 0): (0,), (C, 0): (0,)}),
    "T2": Row((8,), (Role(C, 4), Role(E, 20)), {(C, 0): (0,)}, frozenset({E})),
    "T3": Row((8,), (Role(C, 20), Role(E, 4)), {(E, 0): (0,)}, frozenset({C})),
    "T4": Row((8,), (Role(C, 6), Role(E, 6)), {(E, 0): (0,)}, frozenset({C})),
    "T5": Row(
        (8,), (Role(C, 6), Role(V, 6)), {(C, 0): (0,), (V, 0): (0,)}, group=frozenset({C, V})
    ),
    "T6": Row((8, 8), (Role(C, 7), Role(E, 2)), {(E, 0): (0,), (C, 0): (1,)}),
    "T7": Row((8, 8), (Role(C, 7), Role(E, 7)), {(E, 0): (0,), (C, 0): (1,)}),
    "T8": Row(
        (8, 8),
        (Role(C, 7), Role(E, 2), Role(R, 2)),
        {(C, 0): (0,), (E, 0): (1,), (R, 0): (1,)},
    ),
    "T9": Row((8, 8), (Role(C, 12), Role(E, 2)), {(E, 0): (0,)}, frozenset({C})),
    "T10": Row(
        (8, 8),
        (Role(C, 12), Role(E, 2)),
        {(E, 0): (0,), (C, 0): (1, 0)},
        split=(7, 5),
        chat_spans=True,
    ),
    "T11": Row((8, 8), (Role(C, 20), Role(E, 2)), {(E, 0): (0,)}, frozenset({C}), chat_spans=True),
    "T12": Row((8, 8), (Role(C, 4), Role(E, 20)), {(C, 0): (0,)}, frozenset({E})),
    "T13": Row(
        (8, 8),
        (Role(C, 7), Role(E, 2, replicas=2)),
        {(E, 0): (0,), (E, 1): (0,), (C, 0): (1,)},
    ),
    "T14": Row(
        (8, 8),
        (Role(C, 10), Role(E, 6, replicas=2)),
        {(E, 0): (0,), (E, 1): (1,)},
        frozenset({C}),
    ),
    "T15": Row((8, 8, 8), (Role(C, 4), Role(E, 20)), {(C, 0): (0,)}, frozenset({E})),
    "T16": Row(
        (8, 8, 8),
        (Role(C, 7), Role(E, 7), Role(V, 7)),
        {(E, 0): (0,), (V, 0): (1,), (C, 0): (2,)},
    ),
    "T17": Row(
        (8, 8, 8),
        (Role(C, 7), Role(E, 7), Role(R, 7), Role(V, 7)),
        {(E, 0): (0,), (R, 0): (1,), (V, 0): (2,), (C, 0): (2,)},
        group=frozenset({C, V}),
    ),
    "T18": Row(
        (8, 8, 8),
        (Role(C, 18), Role(E, 2)),
        {(E, 0): (0,), (C, 0): (1, 2, 0)},
        split=(7, 7, 5),
        chat_spans=True,
    ),
    "T19": Row((24, 8), (Role(C, 12), Role(E, 2)), {(E, 0): (0,), (C, 0): (0,)}),
    "T20": Row(
        (8, 8),
        (Role(C, 10), Role(E, 6), Role(R, 6)),
        {(E, 0): (0,), (R, 0): (1,)},
        frozenset({C}),
        chat_spans=True,
    ),
    "T21": Row(
        (8,),
        (Role(E, 0.3), Role(C, 7), Role(V, 7)),
        {(C, 0): (0,), (V, 0): (0,)},
        frozenset({E}),
        group=frozenset({C, V}),
    ),
}


def _mixed_engines(tmp_path: Path, row: Row, devices: list[FleetDevice]):
    """Chat on one engine, every other role on another."""
    first = CardEngine(
        "engine-a", tmp_path / "engine-a-server", row.roles, devices, spans=row.chat_spans
    )
    second = CardEngine("engine-b", tmp_path / "engine-b-server", row.roles, devices)
    return {r.role: first if r.role is C else second for r in row.roles}


def _cards_of_launch(launch: InstanceLaunch) -> tuple[int, ...]:
    """The cards a launch is pinned to, read from what reaches its process."""
    visible = launch.env_overrides.get("CUDA_VISIBLE_DEVICES", "")
    return tuple(int(part) for part in visible.split(",") if part)


def _served(plan) -> dict[tuple[WorkerRole, int], tuple[int, ...]]:
    return {(launch.role, launch.replica): _cards_of_launch(launch) for launch in plan.launches}


def _tensor_split(plan) -> tuple[int, ...]:
    for launch in plan.launches:
        if "--tensor-split" in launch.argv:
            text = launch.argv[launch.argv.index("--tensor-split") + 1]
            return tuple(int(part) for part in text.split(","))
    return ()


def _charge_on_card(plan, row: Row, card: int) -> float:
    """GiB resident on *card*: every lone role, plus a swap group's largest member."""
    sizes = {r.role: r.gib for r in row.roles}
    on_card = [launch.role for launch in plan.launches if card in _cards_of_launch(launch)]
    alone = sum(sizes[role] for role in on_card if role not in plan.co_tenants)
    return alone + max((sizes[role] for role in on_card if role in plan.co_tenants), default=0)


class TestMixedPlanAcrossCards:
    """Each role of a mixed plan gets a named card or a refusal, through plan_all_launches."""

    @pytest.mark.parametrize("row_id", sorted(_T_ROWS))
    def test_table_row(self, monkeypatch, tmp_path, caplog, row_id) -> None:
        row = _T_ROWS[row_id]
        prepare_host(monkeypatch, tmp_path, row.roles, 64 * _GIB)
        engines = _mixed_engines(tmp_path, row, cards_of(row.cards))
        with caplog.at_level(logging.WARNING, logger=_PLANNING_LOG):
            plan = planning.plan_all_launches(engines=engines)
        assert _served(plan) == row.served
        assert plan.co_tenants == row.group
        assert _tensor_split(plan) == row.split
        assert "tight" not in _warnings(caplog)
        for role in row.refused:
            assert row.refusal(role, engines[role].name) in _warnings(caplog)
        assert _warnings(caplog).count("will not be served") == len(row.refused)
        if not row.split:
            for card, size in enumerate(row.cards):
                assert _charge_on_card(plan, row, card) <= size * 0.9

    def test_the_usable_figure_of_a_refusal_counts_what_other_tenants_hold(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        roles = (Role(C, 5), Role(E, 1))
        prepare_host(monkeypatch, tmp_path, roles, 64 * _GIB)
        busy = [replace(card, free_bytes=4 * _GIB) for card in cards_of((8,))]
        chat = CardEngine("engine-a", tmp_path / "a", roles, busy)
        embed = CardEngine("engine-b", tmp_path / "b", roles, busy)
        planning.capture_plan_probe(engine=chat)
        with caplog.at_level(logging.WARNING, logger=_PLANNING_LOG):
            planning.plan_all_launches(engines={C: chat, E: embed})
        model = f"chat model org/repo/{model_file(C)} (engine-a)"
        assert _m2(model, 5, 3.6) in _warnings(caplog)

    def test_one_engine_on_the_same_cards_still_loads_tight(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        row = _T_ROWS["T9"]
        prepare_host(monkeypatch, tmp_path, row.roles, 64 * _GIB)
        monkeypatch.setattr(cfg, "num_ctx", 4096)
        engine = CardEngine(
            "engine-a", tmp_path / "engine-a-server", row.roles, cards_of(row.cards)
        )
        with caplog.at_level(logging.WARNING, logger=_PLANNING_LOG):
            plan = planning.plan_all_launches(engines=engines_for_all_roles(engine))
        assert WorkerRole.CHAT in {launch.role for launch in plan.launches}
        assert "Memory is tight for the chat model" in _warnings(caplog)
        assert "will not be served" not in _warnings(caplog)

    def test_a_role_with_no_device_charge_takes_no_card_at_any_replica(self) -> None:
        placement = plan_placement(
            [
                ModelPlacementInput(role=E, est_vram_bytes=0, replicas=2),
                ModelPlacementInput(role=C, est_vram_bytes=4 * _GIB),
            ],
            [(0, 8 * _GIB), (1, 8 * _GIB)],
            estimate_peak=lambda _role, _ratio: (),
            refuse_oversize=True,
        )
        cards = [(plan.role, plan.replica, plan.devices) for plan in placement.instances]
        assert sorted(cards, key=str) == sorted([(E, 0, ()), (E, 1, ()), (C, 0, (0,))], key=str)
        assert placement.unplaceable_roles == ()


@dataclass(frozen=True)
class PinRow:
    """A mixed plan whose engines differ in how a card reaches their processes."""

    cards: tuple[int, ...]
    backend: str
    roles: tuple[Role, ...]
    no_names: frozenset[WorkerRole]
    pinned: dict[WorkerRole, str]
    refused: frozenset[WorkerRole] = frozenset()
    other_backend: str | None = None


_ROCM_VAR = "ROCR_VISIBLE_DEVICES"
_PAIR = (Role(C, 7), Role(E, 2))
_P_ROWS = {
    "P1": PinRow(
        (8, 8),
        "CUDA",
        _PAIR,
        frozenset({E}),
        {E: "CUDA_VISIBLE_DEVICES=0", C: "CUDA_VISIBLE_DEVICES=1"},
    ),
    "P2": PinRow((8, 8), "Vulkan", _PAIR, frozenset({E}), {C: "--device"}, frozenset({E})),
    "P3": PinRow((8,), "Vulkan", (Role(C, 4), Role(E, 2)), frozenset({E}), {C: "--device", E: ""}),
    "P4": PinRow((8, 8), "ROCm", _PAIR, frozenset({E}), {E: f"{_ROCM_VAR}=0", C: f"{_ROCM_VAR}=1"}),
    "P6": PinRow(
        (8, 8),
        "CUDA",
        (Role(C, 7), Role(E, 2)),
        frozenset(),
        {C: "CUDA_VISIBLE_DEVICES=0"},
        frozenset({E}),
        other_backend="Vulkan",
    ),
    "P7": PinRow(
        (8, 8, 8),
        "Vulkan",
        (Role(C, 7), Role(E, 7), Role(V, 2)),
        frozenset({E}),
        {V: "--device", C: "--device"},
        frozenset({E}),
    ),
}


def _pin_of(launch: InstanceLaunch, want: str) -> bool:
    """Whether *launch* carries the pin the row names; an empty name means no pin at all."""
    if want == "--device":
        return "--device" in launch.argv
    if not want:
        return "--device" not in launch.argv and not any(
            key.endswith("VISIBLE_DEVICES") for key in launch.env_overrides
        )
    key, value = want.split("=")
    return launch.env_overrides.get(key) == value


def _pin_engines(tmp_path: Path, row: PinRow):
    """Chat on an engine that takes device names; the rest as the row says."""
    built = {}
    for spec in row.roles:
        name = "engine-a" if spec.role is C else "engine-b"
        backend = row.other_backend if name == "engine-b" and row.other_backend else row.backend
        built[spec.role] = CardEngine(
            f"{name}-{spec.role.value}" if spec.role is not C else name,
            tmp_path / f"{name}-{spec.role.value}-server",
            row.roles,
            cards_of(row.cards, backend),
            takes_device_names=spec.role not in row.no_names,
        )
    return built


class TestMixedPlanPins:
    """A pin that reaches its process, or a refusal, through plan_all_launches."""

    @pytest.mark.parametrize("row_id", sorted(_P_ROWS))
    def test_table_row(self, monkeypatch, tmp_path, caplog, row_id) -> None:
        row = _P_ROWS[row_id]
        prepare_host(monkeypatch, tmp_path, row.roles, 64 * _GIB)
        monkeypatch.setattr(devices_module, "amd_visible_var", lambda _env: _ROCM_VAR)
        engines = _pin_engines(tmp_path, row)
        with caplog.at_level(logging.WARNING, logger=_PLANNING_LOG):
            plan = planning.plan_all_launches(engines=engines)
        by_role = {launch.role: launch for launch in plan.launches}
        assert set(by_role) == set(row.pinned)
        for role, want in row.pinned.items():
            assert _pin_of(by_role[role], want), (role, by_role[role].env_overrides)
        for role in row.refused:
            name = engines[role].name
            model = f"{role.value} model org/repo/{model_file(role)} ({name})"
            assert _m3(model, len(row.cards), name) in _warnings(caplog)
        assert _warnings(caplog).count("will not be served") == len(row.refused)

    def test_a_role_too_big_for_any_card_that_no_pin_reaches_reads_as_the_pin(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        roles = (Role(C, 7), Role(E, 20))
        row = PinRow((8, 8), "Vulkan", roles, frozenset({E}), {C: "--device"}, frozenset({E}))
        prepare_host(monkeypatch, tmp_path, roles, 64 * _GIB)
        with caplog.at_level(logging.WARNING, logger=_PLANNING_LOG):
            plan = planning.plan_all_launches(engines=_pin_engines(tmp_path, row))
        assert {launch.role for launch in plan.launches} == {C}
        model = f"embed model org/repo/{model_file(E)} (engine-b-embed)"
        assert _m3(model, 2, "engine-b-embed") in _warnings(caplog)
        assert "more than any one GPU" not in _warnings(caplog)

    def test_p5_a_cpu_only_role_takes_no_card_and_no_pin(self, monkeypatch, tmp_path) -> None:
        roles = _PAIR
        prepare_host(monkeypatch, tmp_path, roles, 64 * _GIB)
        chat = CardEngine("engine-a", tmp_path / "a", roles, cards_of((8, 8)))
        embed = CardEngine("engine-b", tmp_path / "b", roles, [], cpu_only=True)
        plan = planning.plan_all_launches(engines={C: chat, E: embed})
        by_role = {launch.role: launch for launch in plan.launches}
        assert sorted(launch.role for launch in plan.launches) == sorted([C, E])
        assert _pin_of(by_role[E], "")
        assert _pin_of(by_role[C], "CUDA_VISIBLE_DEVICES=0")

    def test_p8_two_instances_of_one_engine_name_plan_as_the_golden(
        self, monkeypatch, tmp_path
    ) -> None:
        def two_instances(engine):
            twin = CardEngine("engine-a", engine.binary(), _PAIR, cards_of((8, 8)))
            return planning.plan_all_launches(engines={C: engine, E: twin})

        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        text = run_scenario(monkeypatch, tmp_path, SCENARIOS["two-fit"], two_instances)
        assert text == golden["two-fit"]

    def test_p9_one_engine_without_device_names_keeps_the_base_rule(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        prepare_host(monkeypatch, tmp_path, _PAIR, 64 * _GIB)
        engine = CardEngine("engine-a", tmp_path / "a", _PAIR, cards_of((8, 8), "Vulkan"))
        with caplog.at_level(logging.WARNING, logger=_PLANNING_LOG):
            plan = planning.plan_all_launches(engines=engines_for_all_roles(engine))
        assert {launch.role for launch in plan.launches} == {C, E}
        assert "will not be served" not in _warnings(caplog)

    def test_an_engine_whose_devices_cannot_be_read_is_refused(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        cards = cards_of((8, 8))
        prepare_host(monkeypatch, tmp_path, _PAIR, 64 * _GIB)
        chat = CardEngine("engine-a", tmp_path / "a", _PAIR, cards)
        embed = CardEngine("engine-b", tmp_path / "b", _PAIR, cards)
        real_get = planning._read_device_cache.get

        def reading(binary, engine):
            if engine is embed:
                raise OSError("probe failed")
            return real_get(binary, engine)

        monkeypatch.setattr(planning._read_device_cache, "get", reading)
        with caplog.at_level(logging.WARNING, logger=_PLANNING_LOG):
            plan = planning.plan_all_launches(engines={C: chat, E: embed})
        assert {launch.role for launch in plan.launches} == {C}
        model = f"embed model org/repo/{model_file(E)} (engine-b)"
        assert _m3(model, 2, "engine-b") in _warnings(caplog)

    def test_a_charged_engine_that_reports_no_device_is_refused(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        prepare_host(monkeypatch, tmp_path, _PAIR, 64 * _GIB)
        chat = CardEngine("engine-a", tmp_path / "a", _PAIR, cards_of((8, 8)))
        embed = CardEngine("engine-b", tmp_path / "b", _PAIR, [])
        with caplog.at_level(logging.WARNING, logger=_PLANNING_LOG):
            plan = planning.plan_all_launches(engines={C: chat, E: embed})
        assert {launch.role for launch in plan.launches} == {C}
        model = f"embed model org/repo/{model_file(E)} (engine-b)"
        assert _m3(model, 2, "engine-b") in _warnings(caplog)

    def test_a_saved_placement_with_a_role_the_pin_cannot_reach_falls_back_with_a_warning(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        prepare_host(monkeypatch, tmp_path, _PAIR, 64 * _GIB)
        spec = {"chat": {"devices": [0]}, "embed": {"devices": [1]}}
        monkeypatch.setattr(cfg, "placement", json.dumps(spec))
        engines = _pin_engines(tmp_path, _P_ROWS["P2"])
        with caplog.at_level(logging.WARNING, logger=_PLANNING_LOG):
            plan = planning.plan_all_launches(engines=engines)
        assert {launch.role for launch in plan.launches} == {C}
        assert "cannot be told which GPU to use" in _warnings(caplog)
        assert "using automatic placement" in _warnings(caplog)


class TestDeviceNameContract:
    def test_an_engine_that_takes_device_names_puts_them_in_argv(self, tmp_path) -> None:
        adapter = LLAMA
        assert adapter.takes_device_names is True
        spec = adapter.role_spec(C, None, None)
        request = LaunchRequest(
            binary=tmp_path / "server",
            model_path=tmp_path / "model.gguf",
            devices=(1, 0),
            n_gpu_layers=99,
            slots=1,
            ctx_per_slot=4096,
            device_names=("Vulkan1", "Vulkan0"),
            tensor_split=(7, 5),
        )
        argv = adapter.launch_argv(spec, request)
        assert argv[argv.index("--device") + 1] == "Vulkan1,Vulkan0"
        assert argv[argv.index("--tensor-split") + 1] == "7,5"

    def test_the_fake_engine_takes_no_device_names_and_never_spans(self, tmp_path) -> None:
        adapter = FakeAdapter()
        assert adapter.takes_device_names is False
        spec = adapter.role_spec(C, None, None)
        request = LaunchRequest(
            binary=tmp_path / "server",
            model_path=tmp_path / "model.gguf",
            devices=(1,),
            n_gpu_layers=99,
            slots=1,
            ctx_per_slot=4096,
            device_names=("Vulkan1",),
        )
        assert "Vulkan1" not in adapter.launch_argv(spec, request)
        estimate = adapter.estimate_memory(
            MemoryRequest(
                model_path=tmp_path / "model.gguf",
                ctx=4096,
                slots=1,
                gpu_layers=0,
                flash_attn=False,
                kv_cache_type=KvCacheType.F16,
                tensor_split=(1, 1),
            )
        )
        assert estimate.per_device == ()


def _generated_input(rng: random.Random):
    cards = [rng.choice([6, 8, 12, 24]) for _ in range(rng.choice([1, 2, 3]))]
    roles = rng.sample([C, E, R, V], rng.randint(2, 4))
    return cards, {role: rng.choice([0.5, 1, 2, 3, 4, 5, 6, 7, 9, 12, 20]) for role in roles}


def _phases_fit(assignment: dict[WorkerRole, int], sizes, cards) -> bool:
    """Whether each phase of each card holds the roles assigned to the card."""
    for card, size in enumerate(cards):
        on_card = [role for role, chosen in assignment.items() if chosen == card]
        for phase in {p for role in on_card for p in ROLE_REGISTRY[role].phases}:
            resident = sum(sizes[r] for r in on_card if phase in ROLE_REGISTRY[r].phases)
            if resident > size * 0.9 + 1e-9:
                return False
    return True


def _some_assignment_fits(sizes: dict[WorkerRole, float], cards: list[int]) -> bool:
    roles = list(sizes)
    return any(
        _phases_fit(dict(zip(roles, chosen, strict=True)), sizes, cards)
        for chosen in itertools.product(range(len(cards)), repeat=len(roles))
    )


def _overcommits(placement, sizes, cards) -> bool:
    """Whether a card holds more than its usable part, a swap group charged at its largest."""
    for card, size in enumerate(cards):
        on_card = [plan.role for plan in placement.instances if card in plan.devices]
        for phase in {p for role in on_card for p in ROLE_REGISTRY[role].phases}:
            in_phase = [role for role in on_card if phase in ROLE_REGISTRY[role].phases]
            lone = sum(sizes[r] for r in in_phase if r not in placement.co_tenants)
            group = max((sizes[r] for r in on_card if r in placement.co_tenants), default=0)
            if lone + group > size * 0.9 + 1e-6:
                return True
    return False


def generated_comparison(count: int, *, seed: int = 7, **kwargs) -> dict[str, int]:
    """Plan *count* random inputs and compare each with a brute force over assignments."""
    rng = random.Random(seed)
    result = {"overcommit": 0, "unplaceable_set": 0, "refused_inputs": 0, "served_inputs": 0}
    for _ in range(count):
        cards, sizes = _generated_input(rng)
        placement = plan_placement(
            [ModelPlacementInput(role=r, est_vram_bytes=int(s * _GIB)) for r, s in sizes.items()],
            [(i, size * _GIB) for i, size in enumerate(cards)],
            estimate_peak=lambda _role, _ratio: (),
            **kwargs,
        )
        served = {plan.role: sizes[plan.role] for plan in placement.instances}
        result["overcommit"] += _overcommits(placement, sizes, cards)
        # A swap group holds one resident at a time, which a per-phase sum cannot express.
        lone = not placement.co_tenants
        result["unplaceable_set"] += lone and not _some_assignment_fits(served, cards)
        result["refused_inputs"] += bool(placement.unplaceable_roles)
        result["served_inputs"] += bool(served)
    return result


class TestGeneratedPlansAgainstBruteForce:
    def test_the_rule_never_overcommits_a_card_and_refuses_on_some_inputs(self) -> None:
        result = generated_comparison(300, refuse_oversize=True)
        assert result["overcommit"] == 0
        assert result["unplaceable_set"] == 0
        assert result["refused_inputs"] > 0
        assert result["served_inputs"] > 0

    def test_the_oracle_goes_red_on_the_tight_placement_it_rejects(self) -> None:
        assert generated_comparison(300, refuse_oversize=False)["overcommit"] > 0


class TestEngineIdentityIsItsName:
    def test_two_adapter_instances_of_one_engine_launch_one_binary(
        self, monkeypatch, tmp_path
    ) -> None:
        prepare_host(monkeypatch, tmp_path, _PAIR, 64 * _GIB)
        busy = [replace(card, free_bytes=_GIB) for card in cards_of((8, 8))]
        chat = CardEngine("engine-a", tmp_path / "first", _PAIR, busy)
        twin = CardEngine("engine-a", tmp_path / "second", _PAIR, cards_of((8, 8)))
        plan = planning.plan_all_launches(engines={C: chat, E: twin})
        assert sorted(launch.role for launch in plan.launches) == sorted([C, E])
        assert {launch.argv[0] for launch in plan.launches} == {str(twin.binary())}

    def test_two_adapter_instances_of_one_engine_keep_the_one_engine_rule(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        device = _host(monkeypatch, tmp_path, unified=False, total=8 * _GIB)
        chat, _ = _pair(tmp_path, device, 12, 2)
        embed = _SizedEngine(
            "engine-a", tmp_path / "engine-a-server", MemoryKind.SIZED, 2 * _GIB, device
        )
        assert chat is not embed
        with caplog.at_level(logging.WARNING, logger=_PLANNING_LOG):
            plan = planning.plan_all_launches(
                engines={WorkerRole.CHAT: chat, WorkerRole.EMBED: embed}
            )
        assert {launch.role for launch in plan.launches} == {WorkerRole.EMBED}
        assert "does not fit" not in _warnings(caplog)
        assert f"Memory is tight for the chat model {cfg.chat_model}:" in _warnings(caplog)


class TestOneEngineKeepsTheBaseRule:
    def test_one_engine_on_one_gpu_loads_a_role_that_does_not_fit_tight(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        device = _host(monkeypatch, tmp_path, unified=False, total=8 * _GIB)
        chat, _ = _pair(tmp_path, device, 12, 2)
        with caplog.at_level(logging.WARNING, logger=_PLANNING_LOG):
            plan = planning.plan_all_launches(engines=engines_for_all_roles(chat))
        assert {launch.role for launch in plan.launches} == {WorkerRole.EMBED}
        assert "does not fit" not in _warnings(caplog)
        assert f"Memory is tight for the chat model {cfg.chat_model}:" in _warnings(caplog)
        assert "(engine-a)" not in _warnings(caplog)

    def test_one_engine_on_unified_memory_names_no_engine_when_a_role_does_not_fit(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        device = _host(monkeypatch, tmp_path, unified=True, total=8 * _GIB)
        chat, _ = _pair(tmp_path, device, 20, 1)
        with caplog.at_level(logging.WARNING, logger=_PLANNING_LOG):
            planning.plan_all_launches(engines=engines_for_all_roles(chat))
        assert (
            f"chat model {cfg.chat_model} does not fit available memory and will not be served"
        ) in _warnings(caplog)
        assert "(engine-a)" not in _warnings(caplog)


class TestMixedCoTenantGroup:
    """Chat on one engine and vision on another share one swap group."""

    @pytest.fixture
    def mixed_plan(self, monkeypatch, tmp_path):
        device = _host(monkeypatch, tmp_path, unified=True, total=10 * _GIB)
        monkeypatch.setattr(cfg, "embedding_model", "")
        monkeypatch.setattr(cfg, "vision_model", "org/repo/vision.gguf")
        monkeypatch.setattr(planning, "vision_role_wanted", lambda _ref: True)
        monkeypatch.setattr(planning, "_vision_mmproj", lambda _ref: tmp_path / "mmproj.gguf")
        chat, vision = _pair(tmp_path, device, 7, 6)
        engines = {WorkerRole.CHAT: chat, WorkerRole.VISION: vision}
        return planning.plan_all_launches(engines=engines), chat, vision

    def test_the_two_engines_share_one_co_tenant_group(self, mixed_plan) -> None:
        plan, _chat, _vision = mixed_plan
        assert plan.co_tenants == {WorkerRole.CHAT, WorkerRole.VISION}
        assert group_for(WorkerRole.CHAT, plan.co_tenants) is SwapGroup.CO_TENANT
        assert group_for(WorkerRole.VISION, plan.co_tenants) is SwapGroup.CO_TENANT

    def test_the_group_config_holds_the_argv_of_both_engines(self, mixed_plan) -> None:
        plan, chat, vision = mixed_plan
        members = [
            launch
            for launch in plan.launches
            if group_for(launch.role, plan.co_tenants) is SwapGroup.CO_TENANT
        ]
        ports = {launch.model_id: 5900 + i for i, launch in enumerate(members)}
        config = json.loads(build_swap_config(members, ports, swap=True))
        (group,) = config["groups"].values()
        assert group["swap"] is True
        assert sorted(group["members"]) == sorted(launch.model_id for launch in members)
        commands = {mid: entry["cmd"] for mid, entry in config["models"].items()}
        chat_id = next(launch.model_id for launch in members if launch.role is WorkerRole.CHAT)
        vision_id = next(launch.model_id for launch in members if launch.role is WorkerRole.VISION)
        assert str(chat.binary()) in commands[chat_id]
        assert str(vision.binary()) in commands[vision_id]
        assert str(vision.binary()) not in commands[chat_id]
        assert str(chat.binary()) not in commands[vision_id]
