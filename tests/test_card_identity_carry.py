"""A restate carries each card's clean free figure by the card's identity, never by its slot."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.engines.base import EngineAdapter
from lilbee.providers.fleet.engines.ik_llama import IK_LLAMA
from lilbee.providers.roles import EngineBackend, WorkerRole

G = 1 << 30
M = 1 << 20
_FREE_RAM = 64 * G
# The engine lists a 3090 at 24126 MiB, the driver at 24576 MiB
# (tests/conformance/fixtures/engines/llama.cpp-cuda.log).
_ENGINE_3090 = 24126 * M
_DRIVER_3090 = 24576 * M


def card(
    backend: str, index: int, name: str, total: int, free: int, uuid: str = "", loader: bool = False
) -> FleetDevice:
    """A card reading; the UUID is set only when the row names one."""
    device = FleetDevice(backend, index, name, total, free, from_loader=loader)
    return replace(device, uuid=uuid) if uuid else device


def C(index: int, name: str, total: int, free: int, uuid: str = "") -> FleetDevice:
    return card("CUDA", index, name, total, free, uuid)


def V(
    index: int, name: str, total: int, free: int, uuid: str = "", loader: bool = False
) -> FleetDevice:
    return card("Vulkan", index, name, total, free, uuid, loader)


@dataclass(frozen=True)
class Row:
    """One case: the snapshot, the new reading, the free figures expected."""

    number: int
    case: str
    snapshot: list[FleetDevice]
    reading: list[FleetDevice]
    expected: list[int]


ROWS = [
    Row(
        1, "same card, same slot", [C(0, "A", 24 * G, 23 * G)], [C(0, "A", 24 * G, 2 * G)], [23 * G]
    ),
    Row(
        2,
        "swapped card, other name",
        [C(0, "A", 24 * G, 7 * G)],
        [C(0, "B", 24 * G, 23 * G)],
        [23 * G],
    ),
    Row(
        3,
        "swapped card, same name, bigger",
        [C(0, "A", 8 * G, 7 * G)],
        [C(0, "A", 24 * G, 23 * G)],
        [23 * G],
    ),
    Row(
        4,
        "swapped card, same name, smaller",
        [C(0, "A", 24 * G, 23 * G)],
        [C(0, "A", 16 * G, 2 * G)],
        [2 * G],
    ),
    Row(
        5,
        "swapped card, same name and size, UUIDs differ",
        [C(0, "A", 24 * G, 7 * G, "GPU-old")],
        [C(0, "A", 24 * G, 23 * G, "GPU-new")],
        [23 * G],
    ),
    Row(
        6,
        "swapped card, same name and size, no UUIDs (limit)",
        [C(0, "A", 24 * G, 7 * G)],
        [C(0, "A", 24 * G, 23 * G)],
        [7 * G],
    ),
    Row(
        7,
        "moved index, no UUIDs",
        [C(0, "A", 24 * G, 23 * G), C(1, "B", 16 * G, 15 * G)],
        [C(0, "B", 16 * G, 2 * G), C(1, "A", 24 * G, 3 * G)],
        [15 * G, 23 * G],
    ),
    Row(
        8,
        "moved index, UUIDs",
        [C(0, "A", 24 * G, 23 * G, "u-a"), C(1, "A", 24 * G, 11 * G, "u-b")],
        [C(0, "A", 24 * G, 2 * G, "u-b"), C(1, "A", 24 * G, 3 * G, "u-a")],
        [11 * G, 23 * G],
    ),
    Row(
        9,
        "identical twins plus one",
        [C(0, "A", 24 * G, 23 * G), C(1, "A", 24 * G, 11 * G)],
        [C(0, "A", 24 * G, 2 * G), C(1, "A", 24 * G, 3 * G), C(2, "C", 8 * G, 8 * G)],
        [23 * G, 11 * G, 8 * G],
    ),
    Row(
        10,
        "one card becomes twins",
        [C(0, "A", 24 * G, 23 * G)],
        [C(0, "A", 24 * G, 2 * G), C(1, "A", 24 * G, 3 * G)],
        [23 * G, 3 * G],
    ),
    Row(
        11,
        "twin added in front, UUIDs",
        [C(0, "A", 24 * G, 23 * G, "u-a")],
        [C(0, "A", 24 * G, 20 * G, "u-n"), C(1, "A", 24 * G, 2 * G, "u-a")],
        [20 * G, 23 * G],
    ),
    Row(
        12,
        "twin added in front, no UUIDs (limit)",
        [C(0, "A", 24 * G, 23 * G)],
        [C(0, "A", 24 * G, 20 * G), C(1, "A", 24 * G, 2 * G)],
        [23 * G, 2 * G],
    ),
    Row(
        13,
        "one-sided UUID, same name, in the window",
        [C(0, "A", 24 * G, 23 * G, "u-a")],
        [C(0, "A", 24 * G, 2 * G)],
        [23 * G],
    ),
    Row(
        14,
        "one-sided UUID, name differs",
        [C(0, "A", 24 * G, 23 * G)],
        [C(0, "B", 24 * G, 2 * G, "u-b")],
        [2 * G],
    ),
    Row(
        15,
        "engine and driver totals 450 MiB apart",
        [C(0, "A", _ENGINE_3090, 23 * G)],
        [C(0, "A", _DRIVER_3090, 2 * G, "u-a")],
        [23 * G],
    ),
    Row(
        16,
        "window edge, exactly 1 GiB",
        [C(0, "A", 23 * G, 23 * G)],
        [C(0, "A", 24 * G, 2 * G)],
        [23 * G],
    ),
    Row(
        17,
        "window edge, 1 GiB and a byte",
        [C(0, "A", 23 * G - 1, 22 * G)],
        [C(0, "A", 24 * G, 2 * G)],
        [2 * G],
    ),
    Row(
        18,
        "cross-backend, same name, 450 MiB apart",
        [V(0, "gpu0", _ENGINE_3090, 23 * G)],
        [C(0, "gpu0", _DRIVER_3090, 2 * G, "GPU-a")],
        [23 * G],
    ),
    Row(
        19,
        "cross-backend, smaller different card",
        [C(0, "NVIDIA A", 24 * G, 23 * G)],
        [V(0, "A (vk)", 16 * G, 2 * G)],
        [2 * G],
    ),
    Row(
        20,
        "cross-backend, other name, same total, reads live",
        [C(0, "NVIDIA A", 24 * G, 23 * G)],
        [V(0, "A (vk)", 24 * G, 2 * G)],
        [2 * G],
    ),
    Row(
        21,
        "mixed listing, another card removed",
        [C(0, "A", _ENGINE_3090, 23 * G), V(0, "A", _DRIVER_3090, 22 * G), C(1, "B", 8 * G, 7 * G)],
        [C(0, "A", _ENGINE_3090, 2 * G), V(0, "A", _DRIVER_3090, 2 * G)],
        [23 * G, 22 * G],
    ),
    Row(
        22,
        "new card in a new slot",
        [C(0, "A", 24 * G, 23 * G)],
        [C(0, "A", 24 * G, 2 * G), C(1, "B", 8 * G, 7 * G)],
        [23 * G, 7 * G],
    ),
    Row(
        23,
        "card removed",
        [C(0, "A", 24 * G, 23 * G), C(1, "B", 8 * G, 7 * G)],
        [C(0, "A", 24 * G, 2 * G)],
        [23 * G],
    ),
    Row(
        24,
        "matching UUID, renamed and smaller",
        [C(0, "A", 24 * G, 23 * G, "u-a")],
        [C(0, "renamed", 20 * G, 2 * G, "u-a")],
        [20 * G],
    ),
    Row(
        25,
        "loader fallback, same host",
        [V(0, "", 8 * G, 7 * G, loader=True)],
        [V(0, "", 8 * G, 1 * G, loader=True)],
        [7 * G],
    ),
    Row(
        26,
        "driver snapshot, engine re-read, same name and size swap (limit)",
        [C(0, "A", _DRIVER_3090, 7 * G, "u-a")],
        [C(0, "A", _ENGINE_3090, 23 * G)],
        [7 * G],
    ),
    Row(
        27,
        "a slot takes a twin of its neighbour",
        [C(0, "B", 24 * G, 7 * G), C(1, "A", 24 * G, 11 * G)],
        [C(0, "A", 24 * G, 20 * G), C(1, "A", 24 * G, 3 * G)],
        [20 * G, 11 * G],
    ),
    Row(
        28,
        "mixed listing, backend order changed by a new binary",
        [V(0, "A", _DRIVER_3090, 22 * G), C(0, "A", _ENGINE_3090, 23 * G), C(1, "B", 8 * G, 7 * G)],
        [C(0, "A", _ENGINE_3090, 2 * G), V(0, "A", _DRIVER_3090, 2 * G)],
        [23 * G, 22 * G],
    ),
]


@dataclass
class Host:
    """What each engine reads now, and which binary each engine runs."""

    readings: dict[str, list[FleetDevice]] = field(default_factory=dict)
    binaries: dict[str, str] = field(default_factory=dict)


def _install(mp: pytest.MonkeyPatch, host: Host) -> None:
    def _probe(engine: EngineAdapter) -> planning_mod.DeviceReading:
        return planning_mod.DeviceReading(list(host.readings[engine.name]), EngineBackend.CUDA)

    mp.setattr(planning_mod, "_probe_engine_devices", _probe)
    mp.setattr(planning_mod, "_engine_identity", lambda engine: host.binaries[engine.name])
    mp.setattr("lilbee.providers.model_cache.free_system_memory", lambda: _FREE_RAM)


def _by_reload(host: Host, reading: list[FleetDevice]) -> planning_mod._PlanProbe | None:
    host.readings[LLAMA.name] = reading
    planning_mod.refresh_plan_devices({WorkerRole.CHAT: LLAMA})
    return planning_mod._plan_probe_store.get(LLAMA.name)


def _by_new_binary(host: Host, reading: list[FleetDevice]) -> planning_mod._PlanProbe | None:
    host.readings[LLAMA.name] = reading
    host.binaries[LLAMA.name] = "llama-v2"
    return planning_mod._current_plan_probe(LLAMA)


def _by_seed(host: Host, reading: list[FleetDevice]) -> planning_mod._PlanProbe | None:
    host.readings[IK_LLAMA.name] = reading
    return planning_mod._current_plan_probe(IK_LLAMA)


PATHS: dict[str, Callable[[Host, list[FleetDevice]], planning_mod._PlanProbe | None]] = {
    "reload": _by_reload,
    "new-binary": _by_new_binary,
    "seed-of-another-engine": _by_seed,
}


def restated_free(snapshot: list[FleetDevice], reading: list[FleetDevice], path: str) -> list[int]:
    """The free figures the plan reads after *path* restates *snapshot* against *reading*."""
    planning_mod.clear_plan_probe()
    with pytest.MonkeyPatch.context() as mp:
        host = Host({LLAMA.name: snapshot}, {LLAMA.name: "llama-v1", IK_LLAMA.name: "ik-v1"})
        _install(mp, host)
        planning_mod.capture_plan_probe({WorkerRole.CHAT: LLAMA})
        probe = PATHS[path](host, reading)
        assert probe is not None
        assert probe.free_system == _FREE_RAM
        free = [d.free_bytes for d in probe.devices]
    planning_mod.clear_plan_probe()
    return free


@pytest.mark.parametrize("path", list(PATHS))
@pytest.mark.parametrize("row", ROWS, ids=[f"row{r.number:02d}-{r.case}" for r in ROWS])
def test_each_case_plans_with_the_expected_free_figures(row: Row, path: str) -> None:
    got = restated_free(row.snapshot, row.reading, path)
    print(f"CELL row={row.number} path={path} expected={row.expected} got={got}")
    assert got == row.expected


def test_the_table_is_not_empty() -> None:
    assert len(ROWS) == 28
    assert len(PATHS) == 3


# The generator. Each physical card has a hidden id; a reading renders it with or without a
# UUID and with the engine's or the driver's total. The oracle reads the hidden id only, so it
# shares no logic with the code under test. Cases the rule cannot tell apart (two UUID-less
# cards with one name and size that change slots) are not generated.

_NAMES = ["A", "B", "C", "D", "E", "F"]
_SIZES_GIB = [8, 12, 16, 24, 48]


@dataclass(frozen=True)
class Physical:
    """A physical card: its hidden id, its name and its driver total."""

    ident: int
    name: str
    total: int


def _render(p: Physical, index: int, free: int, *, uuids: bool, engine_total: bool) -> FleetDevice:
    total = p.total - 450 * M if engine_total else p.total
    return C(index, p.name, total, min(free, total), f"GPU-{p.ident}" if uuids else "")


@st.composite
def scenarios(draw: st.DrawFn) -> tuple[list[FleetDevice], list[FleetDevice], list[int]]:
    """A snapshot, a later reading of a changed host, and the oracle's free figures."""
    uuids = draw(st.booleans())
    count = draw(st.integers(1, 5))
    names = (
        draw(st.permutations(_NAMES))
        if not uuids
        else [draw(st.sampled_from(_NAMES)) for _ in range(6)]
    )
    before = [Physical(i, names[i], draw(st.sampled_from(_SIZES_GIB)) * G) for i in range(count)]
    kept = draw(st.lists(st.sampled_from(before), unique=True, max_size=count))
    added = [
        Physical(
            100 + i,
            names[count + i] if count + i < len(names) else f"N{i}",
            draw(st.sampled_from(_SIZES_GIB)) * G,
        )
        for i in range(draw(st.integers(0, 6 - count if not uuids else 3)))
    ]
    after = draw(st.permutations(kept + added))
    # Under every rendered total, so the oracle needs no cap on the snapshot side.
    clean = {p.ident: draw(st.integers(1, p.total // G - 1)) * G for p in before}
    snapshot_engine = draw(st.booleans())
    reading_engine = draw(st.booleans())
    snapshot = [
        _render(p, i, clean[p.ident], uuids=uuids, engine_total=snapshot_engine)
        for i, p in enumerate(before)
    ]
    reading = []
    expected = []
    for i, p in enumerate(after):
        live = draw(st.integers(0, p.total // G - 1)) * G
        rendered = _render(p, i, live, uuids=uuids, engine_total=reading_engine)
        reading.append(rendered)
        expected.append(
            min(clean[p.ident], rendered.total_bytes) if p.ident in clean else rendered.free_bytes
        )
    return snapshot, reading, expected


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(scenarios(), st.sampled_from(list(PATHS)))
def test_a_generated_host_change_matches_the_oracle(
    scenario: tuple[list[FleetDevice], list[FleetDevice], list[int]], path: str
) -> None:
    snapshot, reading, expected = scenario
    assert restated_free(snapshot, reading, path) == expected
