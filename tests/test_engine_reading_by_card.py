"""A device maps to another engine's reading by the card it is, not by its index."""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.providers.fleet import planning
from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.engines import llama as llama_mod
from lilbee.providers.fleet.placement import InstancePlan
from lilbee.providers.roles import WorkerRole

_GIB = 1024**3


def _probe(*devices: FleetDevice) -> planning._PlanProbe:
    return planning._PlanProbe(devices=devices, sizing_budget=0, free_system=0, engine="engine")


def _engine_reads(monkeypatch: pytest.MonkeyPatch, *devices: FleetDevice) -> None:
    monkeypatch.setattr(planning, "_current_plan_probe", lambda _engine: _probe(*devices))


def test_a_card_the_other_engine_numbers_differently_maps_by_name_and_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    big = FleetDevice("CUDA", 0, "Big card", 24 * _GIB, 20 * _GIB)
    small = FleetDevice("CUDA", 1, "Small card", 16 * _GIB, 12 * _GIB)
    engine_small = FleetDevice("Vulkan", 0, "Small card", 16 * _GIB - (300 << 20), 10 * _GIB)
    engine_big = FleetDevice("Vulkan", 1, "Big card", 24 * _GIB - (300 << 20), 18 * _GIB)
    _engine_reads(monkeypatch, engine_small, engine_big)
    assert planning._in_engine_reading((big,), LLAMA) == (engine_big,)
    assert planning._in_engine_reading((small, big), LLAMA) == (engine_small, engine_big)


def test_a_card_with_a_uuid_maps_by_the_uuid_when_two_cards_read_alike(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = FleetDevice("CUDA", 0, "GPU", 24 * _GIB, 20 * _GIB, uuid="GPU-a")
    second = FleetDevice("CUDA", 1, "GPU", 24 * _GIB, 20 * _GIB, uuid="GPU-b")
    engine_second = FleetDevice("CUDA", 0, "GPU", 24 * _GIB, 11 * _GIB, uuid="GPU-b")
    engine_first = FleetDevice("CUDA", 1, "GPU", 24 * _GIB, 12 * _GIB, uuid="GPU-a")
    _engine_reads(monkeypatch, engine_second, engine_first)
    assert planning._in_engine_reading((first,), LLAMA) == (engine_first,)
    assert planning._in_engine_reading((second,), LLAMA) == (engine_second,)


def test_a_card_the_engine_does_not_list_keeps_its_own_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gone = FleetDevice("CUDA", 0, "Gone card", 24 * _GIB, 20 * _GIB)
    other = FleetDevice("CUDA", 0, "Other card", 8 * _GIB, 6 * _GIB)
    _engine_reads(monkeypatch, other)
    assert planning._in_engine_reading((gone,), LLAMA) == (gone,)


def test_two_cards_that_read_alike_map_to_two_engine_cards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    twins = tuple(FleetDevice("CUDA", i, "Twin", 24 * _GIB, 20 * _GIB) for i in (0, 1))
    engine_twins = tuple(
        FleetDevice("Vulkan", i, "Twin", 24 * _GIB, (10 + i) * _GIB) for i in (0, 1)
    )
    _engine_reads(monkeypatch, *reversed(engine_twins))
    assert planning._in_engine_reading(twins, LLAMA) == (engine_twins[1], engine_twins[0])


def test_an_engine_with_no_snapshot_leaves_the_devices_as_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = FleetDevice("CUDA", 0, "Card", 24 * _GIB, 20 * _GIB)
    monkeypatch.setattr(planning, "_current_plan_probe", lambda _engine: None)
    assert planning._in_engine_reading((card,), LLAMA) == (card,)


def test_a_launch_pins_the_engines_own_number_for_the_planned_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "chat.gguf"
    model.write_bytes(b"x" * 2048)
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
    monkeypatch.setattr("lilbee.providers.fleet.engines.llama.read_gguf_metadata", lambda _p: {})
    monkeypatch.setattr(planning, "_role_ctx", lambda _r, _p, _m, *_a, engine: 4096)
    monkeypatch.setattr(llama_mod, "llama_server_runtime_env", lambda: {})
    monkeypatch.setattr(
        "lilbee.providers.fleet.engines.llama.resolve_llama_server",
        lambda: Path("/bin/llama-server"),
    )
    planned = FleetDevice("CUDA", 0, "Planned card", 24 * _GIB, 20 * _GIB)
    other = FleetDevice("CUDA", 1, "Other card", 16 * _GIB, 12 * _GIB)
    engine_other = FleetDevice("CUDA", 0, "Other card", 16 * _GIB, 12 * _GIB)
    engine_planned = FleetDevice("CUDA", 1, "Planned card", 24 * _GIB, 20 * _GIB)
    _engine_reads(monkeypatch, engine_other, engine_planned)
    plan = InstancePlan(role=WorkerRole.CHAT, devices=(0,))
    launch = planning._launch_for(plan, "ref", {0: planned, 1: other}, engine=LLAMA)
    assert launch.env_overrides["CUDA_VISIBLE_DEVICES"] == "1"
