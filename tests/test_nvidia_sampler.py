"""The NVIDIA sampler: temperature reaches the throttle, and no driver read can stall a stream.

The stand-in is the driver's card list. The in-process driver read is made to raise, so a
sampler that reaches it fails the test instead of passing on a mock.
"""

from __future__ import annotations

import asyncio

import pytest

from lilbee.data.ingest.adaptive import CONSERVATIVE, ControllerState, Signals, decide
from lilbee.providers.fleet import gpu_stats, host_caps
from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.gpu_backends import nvidia as nvidia_mod
from lilbee.providers.fleet.gpu_stats import probe_gpu_stats, probe_throttle_signal
from lilbee.server.handlers import gpu_stats_stream

GIB = 1024**3
_NVIDIA = "NVIDIA GeForce RTX 3090"


def _smi(*rows: str):
    def run(*_a: object, **_k: object) -> str:
        return "".join(f"{row}\n" for row in rows)

    return run


@pytest.fixture(autouse=True)
def _no_driver_library(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked() -> list[host_caps.CardMemory]:
        raise AssertionError("the sampler must not read the driver library")

    monkeypatch.setattr(host_caps, "read_card_memory", blocked)
    for var in ("CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "GGML_VK_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising=False)
    nvidia_mod._cache.reset()
    monkeypatch.setattr(gpu_stats, "_shared_sample", {})


def _fleet(backend: str, count: int = 1) -> list[FleetDevice]:
    return [FleetDevice(backend, i, _NVIDIA, 24 * GIB, 20 * GIB) for i in range(count)]


def test_a_cuda_card_reports_its_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nvidia_mod, "run_smi", _smi("0, 40, 86, 1024, 24576"))
    assert probe_gpu_stats(_fleet("CUDA"))[0].temperature_c == 86


def test_the_throttle_signal_is_the_hottest_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        nvidia_mod, "run_smi", _smi("0, 40, 70, 1024, 24576", "1, 40, 86, 1024, 24576")
    )
    assert probe_throttle_signal(_fleet("CUDA", 2)).temperature_c == 86.0


def test_a_card_that_hides_its_temperature_leaves_the_signal_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nvidia_mod, "run_smi", _smi("0, 40, [N/A], 1024, 24576"))
    signal = probe_throttle_signal(_fleet("CUDA"))
    assert (signal.load_pct, signal.temperature_c) == (40.0, None)


def test_a_hot_nvidia_card_backs_the_ingest_throttle_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nvidia_mod, "run_smi", _smi("0, 40, 86, 1024, 24576"))
    gpu = probe_throttle_signal(_fleet("CUDA"))
    signals = Signals(
        throughput=10.0,
        gpu_util_pct=gpu.load_pct,
        gpu_temp_c=gpu.temperature_c,
        cpu_pct=10.0,
        ram_free_frac=0.8,
    )
    state = ControllerState(permits=8, ewma_tput=10.0, direction=1, cool_down=0)
    assert decide(CONSERVATIVE, state, signals, 1, 16).permits == 4


def test_a_lone_vulkan_nvidia_device_reports_its_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nvidia_mod, "run_smi", _smi("0, 40, 86, 1024, 24576"))
    stat = probe_gpu_stats(_fleet("Vulkan"))[0]
    assert (stat.utilization_pct, stat.temperature_c) == (40, 86)


def test_a_lone_vulkan_device_on_a_two_card_driver_takes_no_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        nvidia_mod, "run_smi", _smi("0, 40, 70, 1024, 24576", "1, 40, 86, 1024, 24576")
    )
    stat = probe_gpu_stats(_fleet("Vulkan"))[0]
    assert (stat.utilization_pct, stat.temperature_c) == (None, None)


def test_a_lone_vulkan_device_takes_no_sample_when_the_card_list_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nvidia_mod, "run_smi", _smi())
    stat = probe_gpu_stats(_fleet("Vulkan"))[0]
    assert (stat.utilization_pct, stat.temperature_c) == (None, None)


def test_a_lone_vulkan_device_takes_no_sample_when_the_count_is_unknown_on_a_cold_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter(["", "0, 40, 86, 1024, 24576\n"])
    monkeypatch.setattr(nvidia_mod, "_CACHE_TTL_S", 0.0)
    monkeypatch.setattr(nvidia_mod, "run_smi", lambda *_a, **_k: next(outputs))
    stat = probe_gpu_stats(_fleet("Vulkan"))[0]
    assert (stat.utilization_pct, stat.temperature_c) == (None, None)


def test_a_card_with_unreadable_memory_still_counts_toward_the_driver_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        nvidia_mod, "run_smi", _smi("0, 40, 70, 1024, 24576", "1, [N/A], [N/A], [N/A], [N/A]")
    )
    assert probe_gpu_stats(_fleet("Vulkan"))[0].utilization_pct is None


def test_the_stats_stream_of_a_lone_vulkan_device_never_touches_the_driver_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nvidia_mod, "run_smi", _smi("0, 40, 86, 1024, 24576"))

    async def drive() -> list[object]:
        return [c async for c in gpu_stats_stream(_fleet("Vulkan"), interval_s=0, max_ticks=1)]

    assert len(asyncio.run(drive())) >= 1
