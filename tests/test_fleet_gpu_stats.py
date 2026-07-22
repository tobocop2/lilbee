"""Tests for merging vendor SMI samples into the engine device index space."""

from __future__ import annotations

from types import SimpleNamespace


class TestMaskedHostsAttributeStatsToTheRightCards:
    """The engine numbers devices after a visible-devices mask, SMI tools before it.

    Under CUDA_VISIBLE_DEVICES=2,3 the fleet holds devices 0 and 1 while
    nvidia-smi keeps reporting 0..3, so an ordinal merge showed another tenant's
    utilization, temperature and free memory as this fleet's, and paced adaptive
    ingest off a neighbour's workload.
    """

    def test_a_cuda_mask_shifts_the_query_and_the_merge(self, monkeypatch) -> None:
        from lilbee.providers.fleet import gpu_stats as mod
        from lilbee.providers.fleet.gpu_backends import UtilSample

        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
        asked: list[frozenset[int]] = []

        def _sample(_backend, indices):
            asked.append(indices)
            return {
                2: UtilSample(
                    index=2, utilization_pct=90, free_bytes=1, total_bytes=8, temperature_c=None
                ),
                3: UtilSample(
                    index=3, utilization_pct=10, free_bytes=2, total_bytes=8, temperature_c=None
                ),
            }

        monkeypatch.setattr(mod, "_safe_sample", _sample)
        devices = [
            SimpleNamespace(index=0, backend="CUDA", name="A40", free_bytes=0, total_bytes=8),
            SimpleNamespace(index=1, backend="CUDA", name="A40", free_bytes=0, total_bytes=8),
        ]

        stats = mod.probe_gpu_stats(devices)

        assert asked == [frozenset({2, 3})]
        assert stats[0].utilization_pct == 90
        assert stats[1].utilization_pct == 10

    def test_an_unmasked_host_is_unchanged(self, monkeypatch) -> None:
        from lilbee.providers.fleet import gpu_stats as mod
        from lilbee.providers.fleet.gpu_backends import UtilSample

        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setattr(
            mod,
            "_safe_sample",
            lambda _b, _i: {
                0: UtilSample(
                    index=0, utilization_pct=42, free_bytes=1, total_bytes=8, temperature_c=None
                )
            },
        )
        devices = [
            SimpleNamespace(index=0, backend="CUDA", name="A40", free_bytes=0, total_bytes=8)
        ]

        assert mod.probe_gpu_stats(devices)[0].utilization_pct == 42

    def test_the_amd_pair_composes_in_the_order_the_runtime_applies_it(self, monkeypatch) -> None:
        """ROCr filters first; HIP then re-indexes within the survivors."""
        from lilbee.providers.fleet.gpu_stats import _physical_index

        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1,2,3")
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,2")

        assert _physical_index("ROCm", 0) == 1  # ROCr 1,2,3 -> HIP picks its 0 -> 1
        assert _physical_index("ROCm", 1) == 3  # HIP picks its 2 -> ROCr's third -> 3

    def test_a_uuid_mask_is_left_alone_rather_than_guessed_at(self, monkeypatch) -> None:
        from lilbee.providers.fleet.gpu_stats import _physical_index

        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abcdef")

        assert _physical_index("CUDA", 0) == 0

    def test_intel_is_deliberately_not_translated(self, monkeypatch) -> None:
        """ONEAPI_DEVICE_SELECTOR is a selector grammar, not an index list."""
        from lilbee.providers.fleet.gpu_stats import _physical_index

        monkeypatch.setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1,2")

        assert _physical_index("SYCL", 0) == 0
