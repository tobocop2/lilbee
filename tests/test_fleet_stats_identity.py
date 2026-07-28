"""A stat lilbee cannot attribute must not be attributed to the wrong card."""

from __future__ import annotations

import pytest

from lilbee.providers.fleet.gpu_stats import _physical_index

_MASKS = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for var in _MASKS:
        monkeypatch.delenv(var, raising=False)


class TestAnUntranslatableMaskYieldsNoIndex:
    """Returning the fleet index unchanged was a guess, and on a masked host it
    is the wrong card: its utilization, temperature and free VRAM then drive both
    the GPU panel and the adaptive ingest throttle."""

    @pytest.mark.parametrize(
        "mask",
        ["GPU-abcdef12", "MIG-GPU-abcdef12/1/0", "0,GPU-abcdef12"],
        ids=["uuid", "mig", "mixed"],
    )
    def test_a_mask_naming_devices_by_identity_gives_up(self, monkeypatch, mask: str) -> None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
        assert _physical_index("CUDA", 0) is None

    def test_an_index_past_the_end_of_the_mask_gives_up(self, monkeypatch) -> None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
        assert _physical_index("CUDA", 1) is None

    def test_a_numeric_mask_is_still_undone(self, monkeypatch) -> None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
        assert _physical_index("CUDA", 1) == 3

    def test_no_mask_leaves_the_index_alone(self) -> None:
        assert _physical_index("CUDA", 1) == 1


class TestTheOrdinalMaskIsUndoneToo:
    """GPU_DEVICE_ORDINAL reindexes exactly as the other two do and was invisible."""

    def test_an_ordinal_mask_is_translated(self, monkeypatch) -> None:
        monkeypatch.setenv("GPU_DEVICE_ORDINAL", "4,5")
        assert _physical_index("ROCm", 1) == 5

    def test_an_ordinal_mask_by_identity_gives_up(self, monkeypatch) -> None:
        monkeypatch.setenv("GPU_DEVICE_ORDINAL", "GPU-abcdef12")
        assert _physical_index("ROCm", 0) is None
