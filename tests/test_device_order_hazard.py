"""One device number means one card: CUDA numbers follow bus order, Vulkan numbers join nothing.

The sampler stub answers by driver index with that card's true figure, so a
device that shows a figure other than its own Vulkan reading took it from a card
that no Vulkan number is known to name.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from lilbee.providers.fleet import devices, gpu_stats, host_caps, planning
from lilbee.providers.fleet.devices import FleetDevice, visible_env
from lilbee.providers.fleet.gpu_stats import GpuStat, _physical_index, probe_gpu_stats
from lilbee.providers.roles import WorkerRole

GIB = 1024**3
_VULKAN_FREE = 50 * GIB
_NVIDIA = "NVIDIA GeForce RTX 3090"
_AMD = "AMD Radeon RX 7900"
_INTEL = "Intel Arc A770"
_INSTALL_HINT = object()


def _card_free(index: int) -> int:
    return (index + 2) * GIB


def _driver_cards(count: int) -> list[host_caps.CardMemory]:
    return [host_caps.CardMemory(f"GPU-{i}", "n", 8 * GIB, 8 * GIB) for i in range(count)]


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """A host with no order or mask variable, and a sampler that answers by driver index."""

    def sample(_backend: str, wanted: frozenset[int]) -> dict[int, GpuStat]:
        return {p: GpuStat(p, 10, _card_free(p), 100 * GIB) for p in wanted}

    monkeypatch.setattr(gpu_stats, "_safe_sample", sample)
    for var in ("CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "GGML_VK_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(host_caps, "read_card_memory", lambda: [])
    return monkeypatch


def _fleet(*names: str, backend: str = "Vulkan") -> list[FleetDevice]:
    return [FleetDevice(backend, i, n, 100 * GIB, _VULKAN_FREE) for i, n in enumerate(names)]


def test_memory_vulkan_nvidia_order_1_0_keeps_each_vulkan_figure(host) -> None:
    """Vulkan 0 is driver card 1 here, so sampling index 0 would be another card's figure."""
    host.setattr(host_caps, "read_card_memory", lambda: _driver_cards(2))
    stats = probe_gpu_stats(_fleet(_NVIDIA, _NVIDIA))
    assert [s.free_bytes for s in stats.values()] == [_VULKAN_FREE, _VULKAN_FREE]
    assert [s.utilization_pct for s in stats.values()] == [None, None]


def test_memory_second_vendor_in_the_vulkan_list_keeps_each_vulkan_figure(host) -> None:
    host.setattr(host_caps, "read_card_memory", lambda: _driver_cards(1))
    stats = probe_gpu_stats(_fleet(_AMD, _NVIDIA))
    assert [s.free_bytes for s in stats.values()] == [_VULKAN_FREE, _VULKAN_FREE]


def test_memory_cuda_mask_does_not_apply_to_a_vulkan_number(host) -> None:
    host.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    host.setattr(host_caps, "read_card_memory", lambda: _driver_cards(2))
    stats = probe_gpu_stats(_fleet(_NVIDIA, _NVIDIA))
    assert [s.free_bytes for s in stats.values()] == [_VULKAN_FREE, _VULKAN_FREE]


def test_memory_lone_vulkan_device_on_a_two_card_host_keeps_its_vulkan_figure(host) -> None:
    """One Vulkan device with two cards on the driver does not say which card it is."""
    host.setattr(host_caps, "read_card_memory", lambda: _driver_cards(2))
    stats = probe_gpu_stats(_fleet(_NVIDIA))
    assert stats[0].free_bytes == _VULKAN_FREE
    assert stats[0].utilization_pct is None


@pytest.mark.parametrize("cards", [0, 1], ids=["driver-read-fails", "one-card"])
def test_memory_lone_vulkan_device_keeps_the_sample(host, cards: int) -> None:
    host.setattr(host_caps, "read_card_memory", lambda: _driver_cards(cards))
    stats = probe_gpu_stats(_fleet(_NVIDIA))
    assert stats[0].free_bytes == _card_free(0)
    assert stats[0].utilization_pct == 10


@pytest.mark.parametrize(
    ("names", "hint"),
    [((_INTEL, _INTEL), None), ((_INTEL,), _INSTALL_HINT)],
    ids=["two-vulkan-devices-take-no-sample", "lone-vulkan-device-can-take-one"],
)
def test_the_intel_hint_fires_only_when_a_tool_could_give_the_sample(host, names, hint) -> None:
    """Installing a tool does not help a device whose number no tool can name."""
    host.setattr(gpu_stats, "_detect_intel_util_hint", lambda: _INSTALL_HINT)
    fleet = _fleet(*names)
    missing = {d.index: GpuStat(d.index, None, _VULKAN_FREE, 100 * GIB) for d in fleet}
    assert gpu_stats.intel_util_hint(fleet, missing) == hint


def test_memory_cuda_devices_never_read_the_driver_count(host) -> None:
    """The join refusal is about Vulkan numbers; a CUDA tick starts no second driver read."""

    def read_cards() -> list[host_caps.CardMemory]:
        raise AssertionError("a CUDA fleet does not need the driver count")

    host.setattr(host_caps, "read_card_memory", read_cards)
    stats = probe_gpu_stats(_fleet(_NVIDIA, _NVIDIA, backend="CUDA"))
    assert [s.free_bytes for s in stats.values()] == [_card_free(0), _card_free(1)]


def test_memory_cuda_preset_order_reads_the_card_by_bus(host) -> None:
    """The child runs in bus order whatever the parent set, so the driver index is the card."""
    host.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    stats = probe_gpu_stats(_fleet(_NVIDIA, _NVIDIA, backend="CUDA"))
    assert [s.free_bytes for s in stats.values()] == [_card_free(0), _card_free(1)]


def test_a_cuda_mask_still_walks_to_the_driver_index(host) -> None:
    host.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert _physical_index("CUDA", 1) == 3


def test_another_backend_is_unaffected(host) -> None:
    host.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    for var in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
        host.delenv(var, raising=False)
    assert _physical_index("ROCm", 0) == 0


@pytest.fixture
def order_warnings(host, caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """Warnings of the device module, with the warned-once memory of earlier tests cleared."""
    devices._warn_order_replaced.cache_clear()
    host.setattr("lilbee.providers.fleet.cuda_runtime.cuda_runtime_env", lambda _binary: {})
    caplog.set_level(logging.WARNING, logger=devices.log.name)
    return caplog


_CUDA_ONE = (FleetDevice("CUDA", 1, "n", 8 * GIB, 8 * GIB),)


def _probe_order() -> str:
    return devices._probe_env(Path("/nonexistent/engine"))["CUDA_DEVICE_ORDER"]


@pytest.mark.parametrize("preset", ["FASTEST_FIRST", "fastest_first"])
def test_a_preset_order_is_replaced_with_a_warning_that_names_it(
    host, order_warnings, preset
) -> None:
    host.setenv("CUDA_DEVICE_ORDER", preset)
    assert visible_env(_CUDA_ONE)["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert _probe_order() == "PCI_BUS_ID"
    assert [r.levelno for r in order_warnings.records] == [logging.WARNING]
    assert preset in order_warnings.records[0].getMessage()


def test_the_probe_alone_warns_about_a_preset_order(host, order_warnings) -> None:
    host.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    assert _probe_order() == "PCI_BUS_ID"
    assert "FASTEST_FIRST" in order_warnings.text


@pytest.mark.parametrize("preset", [None, "PCI_BUS_ID", ""], ids=["unset", "bus", "empty"])
def test_the_bus_order_is_set_with_no_warning(host, order_warnings, preset) -> None:
    if preset is not None:
        host.setenv("CUDA_DEVICE_ORDER", preset)
    assert visible_env(_CUDA_ONE)["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert _probe_order() == "PCI_BUS_ID"
    assert order_warnings.records == []


_CHAT, _EMBED = WorkerRole.CHAT, WorkerRole.EMBED
_INPUTS = [
    SimpleNamespace(role=_CHAT, est_vram_bytes=GIB),
    SimpleNamespace(role=_EMBED, est_vram_bytes=GIB),
]


def _engine(name: str, backend: str, count: int, takes_names: bool) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        binary=lambda: Path(f"/nonexistent/{name}"),
        takes_device_names=takes_names,
        reading=[FleetDevice(backend, i, "n", 8 * GIB, 8 * GIB) for i in range(count)],
    )


def _unpinnable(host, backend: str, seen: int, cards: int) -> tuple[WorkerRole, ...]:
    """Roles refused for a plan on *backend* that sees *seen* devices on a host of *cards*."""
    first = _engine("first", backend, seen, True)
    other = _engine("other", "CUDA", seen, False)
    cache = SimpleNamespace(get=lambda _b, engine: SimpleNamespace(devices=engine.reading))
    host.setattr(planning, "_read_device_cache", cache)
    host.setattr(host_caps, "read_card_memory", lambda: _driver_cards(cards))
    engines = {_CHAT: first, _EMBED: other}
    return planning._roles_without_a_pin(_INPUTS, first.reading, engines, None)


def test_mixed_plan_with_a_mask_hiding_a_card_is_refused(host) -> None:
    host.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert _unpinnable(host, "Vulkan", seen=1, cards=2) == (_EMBED,)


def test_mixed_plan_with_gpu_devices_hiding_a_card_is_refused(host) -> None:
    host.setenv("GGML_VK_VISIBLE_DEVICES", "1")
    host.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert _unpinnable(host, "Vulkan", seen=1, cards=2) == (_EMBED,)


def test_mixed_plan_on_one_card_is_not_refused(host) -> None:
    assert _unpinnable(host, "Vulkan", seen=1, cards=1) == ()


def test_mixed_plan_with_a_failed_driver_read_uses_the_visible_count(host) -> None:
    assert _unpinnable(host, "Vulkan", seen=1, cards=0) == ()
    assert _unpinnable(host, "Vulkan", seen=2, cards=0) == (_EMBED,)


def test_cuda_plan_under_a_mask_is_not_refused(host) -> None:
    host.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert _unpinnable(host, "CUDA", seen=1, cards=2) == ()


def test_a_plan_with_no_visible_device_refuses_nothing(host) -> None:
    assert _unpinnable(host, "Vulkan", seen=0, cards=2) == ()


def test_a_plan_with_one_engine_reads_no_driver_count(host) -> None:
    def read_cards() -> list[host_caps.CardMemory]:
        raise AssertionError("one engine needs no driver count")

    host.setattr(host_caps, "read_card_memory", read_cards)
    engine = _engine("only", "Vulkan", 1, True)
    engines = {_CHAT: engine, _EMBED: engine}
    assert planning._roles_without_a_pin(_INPUTS, engine.reading, engines, None) == ()
