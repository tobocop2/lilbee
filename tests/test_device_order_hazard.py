"""A reordered index space must not be merged as though it were the same one.

CUDA_DEVICE_ORDER=FASTEST_FIRST makes the engine enumerate by speed while
nvidia-smi still enumerates by bus, so fleet device 0 and sampler index 0 are
different cards. Merging them attributes one card's utilization and free VRAM
to the other, which the ingest throttle and the GPU panel then act on.
"""

from __future__ import annotations

from lilbee.providers.fleet.gpu_stats import _physical_index


def test_a_speed_ordered_cuda_host_declines_to_merge(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert _physical_index("CUDA", 0) is None


def test_the_default_bus_order_still_merges(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert _physical_index("CUDA", 0) == 0


def test_no_order_set_still_merges(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert _physical_index("CUDA", 1) == 1


def test_another_backend_is_unaffected(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("GPU_DEVICE_ORDINAL", raising=False)
    assert _physical_index("ROCm", 0) == 0
