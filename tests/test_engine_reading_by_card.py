"""A device maps to another engine's reading by the card it is, not by its index."""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.core.config import cfg
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.engines import llama as llama_mod
from lilbee.providers.fleet.placement import InstancePlan
from lilbee.providers.roles import WorkerRole

_GIB = 1024**3


def _probe(*devices: FleetDevice) -> planning._PlanProbe:
    return planning._PlanProbe(devices=devices, sizing_budget=0, free_system=0, engine="engine")


def _read(devices: tuple[FleetDevice, ...]) -> tuple[FleetDevice, ...]:
    return planning._in_engine_reading(devices, LLAMA, WorkerRole.CHAT)


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
    assert _read((big,)) == (engine_big,)
    assert _read((small, big)) == (engine_small, engine_big)


def test_a_card_with_a_uuid_maps_by_the_uuid_when_two_cards_read_alike(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = FleetDevice("CUDA", 0, "GPU", 24 * _GIB, 20 * _GIB, uuid="GPU-a")
    second = FleetDevice("CUDA", 1, "GPU", 24 * _GIB, 20 * _GIB, uuid="GPU-b")
    engine_second = FleetDevice("CUDA", 0, "GPU", 24 * _GIB, 11 * _GIB, uuid="GPU-b")
    engine_first = FleetDevice("CUDA", 1, "GPU", 24 * _GIB, 12 * _GIB, uuid="GPU-a")
    _engine_reads(monkeypatch, engine_second, engine_first)
    assert _read((first,)) == (engine_first,)
    assert _read((second,)) == (engine_second,)


def test_a_card_the_engine_does_not_list_is_refused_not_pinned_by_its_fleet_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gone = FleetDevice("CUDA", 0, "Gone card", 24 * _GIB, 20 * _GIB)
    other = FleetDevice("CUDA", 0, "Other card", 8 * _GIB, 6 * _GIB)
    _engine_reads(monkeypatch, other)
    with pytest.raises(planning.CardNotListedError, match="lists a different card at device 0"):
        _read((gone,))


def test_two_cards_that_read_alike_map_to_two_engine_cards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    twins = tuple(FleetDevice("CUDA", i, "Twin", 24 * _GIB, 20 * _GIB) for i in (0, 1))
    engine_twins = tuple(
        FleetDevice("Vulkan", i, "Twin", 24 * _GIB, (10 + i) * _GIB) for i in (0, 1)
    )
    _engine_reads(monkeypatch, *reversed(engine_twins))
    assert _read(twins) == (engine_twins[1], engine_twins[0])


def test_an_engine_with_no_snapshot_leaves_the_devices_as_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = FleetDevice("CUDA", 0, "Card", 24 * _GIB, 20 * _GIB)
    monkeypatch.setattr(planning, "_current_plan_probe", lambda _engine: None)
    assert _read((card,)) == (card,)


def test_an_engine_whose_snapshot_lists_no_card_leaves_the_devices_as_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = FleetDevice("CUDA", 1, "Card", 24 * _GIB, 20 * _GIB)
    _engine_reads(monkeypatch)
    assert _read((card,)) == (card,)


def _opposite_numbering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[int, FleetDevice]:
    """The fleet's two cards, with the engine reading them in the opposite order.

    The fleet numbers the big card 0 and the small card 1; the engine numbers the small
    card 0 and the big card 1, with its own free figures.
    """
    model = tmp_path / "chat.gguf"
    model.write_bytes(b"x" * 2048)
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
    monkeypatch.setattr("lilbee.providers.fleet.engines.llama.read_gguf_metadata", lambda _p: {})
    monkeypatch.setattr(
        "lilbee.providers.fleet.engines.llama.engine_advertises", lambda _b, _flag: True
    )
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_chat_ctx", lambda *_a, **_k: 8192)
    monkeypatch.setattr(planning, "_role_ctx", lambda _r, _p, _m, *_a, engine: 4096)
    monkeypatch.setattr(llama_mod, "llama_server_runtime_env", lambda binary: {})
    monkeypatch.setattr(
        "lilbee.providers.fleet.engines.llama.resolve_llama_server",
        lambda: Path("/bin/llama-server"),
    )
    monkeypatch.setattr(cfg, "n_gpu_layers", None)
    monkeypatch.setattr(cfg, "num_ctx", None)
    big = FleetDevice("CUDA", 0, "Big card", 24 * _GIB, 20 * _GIB)
    small = FleetDevice("CUDA", 1, "Small card", 16 * _GIB, 12 * _GIB)
    engine_small = FleetDevice("CUDA", 0, "Small card", 16 * _GIB, 11 * _GIB)
    engine_big = FleetDevice("CUDA", 1, "Big card", 24 * _GIB, 19 * _GIB)
    _engine_reads(monkeypatch, engine_small, engine_big)
    return {0: big, 1: small}


def test_a_launch_pins_the_engines_own_number_for_the_planned_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    by_index = _opposite_numbering(tmp_path, monkeypatch)
    plan = InstancePlan(role=WorkerRole.CHAT, devices=(0,))
    launch = planning._launch_for(plan, "ref", by_index, engine=LLAMA)
    assert launch.env_overrides["CUDA_VISIBLE_DEVICES"] == "1"


def test_a_split_chat_subtracts_the_reservation_of_the_card_it_sits_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    by_index = _opposite_numbering(tmp_path, monkeypatch)
    seen: dict[str, list[int]] = {}

    def _fit(_path: Path, *, per_device_free_bytes: list[int], **_k: object) -> int:
        seen["free"] = per_device_free_bytes
        return 5000

    monkeypatch.setattr("lilbee.providers.fleet.ctx.fit_split_ctx", _fit)
    plan = InstancePlan(role=WorkerRole.CHAT, devices=(0, 1), tensor_split=(1, 1))
    # The fleet's card 1 is the small one; an embed server holds 3 GiB of it.
    planning._launch_for(plan, "ref", by_index, reserved_by_device={1: 3 * _GIB}, engine=LLAMA)
    # In plan order: the big card is untouched, the small card loses its 3 GiB.
    assert seen["free"] == [19 * _GIB, 8 * _GIB]


def test_a_chat_fit_target_charges_the_reservation_of_the_card_it_sits_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    by_index = _opposite_numbering(tmp_path, monkeypatch)
    plan = InstancePlan(role=WorkerRole.CHAT, devices=(1,))
    # A later role holds 3 GiB on the fleet's card 1, the small card the chat sits on.
    launch = planning._launch_for(
        plan, "ref", by_index, reserved_after_chat={1: 3 * _GIB}, engine=LLAMA
    )
    assert launch.argv[launch.argv.index("--fit-target") + 1] == str(1024 + 3 * 1024)
