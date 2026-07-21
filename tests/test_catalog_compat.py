"""Unit tests for catalog.compat: classify(), SUPPORTED_ARCHS, UnsupportedArchError."""

from __future__ import annotations

import pytest

from lilbee.catalog.compat import (
    SUPPORTED_ARCHS,
    UnsupportedArchError,
    classify,
)
from lilbee.catalog.types import ModelCompat


def test_supported_archs_includes_llama() -> None:
    assert "llama" in SUPPORTED_ARCHS


def test_supported_archs_is_frozenset() -> None:
    assert isinstance(SUPPORTED_ARCHS, frozenset)


def test_supported_archs_nonempty() -> None:
    assert len(SUPPORTED_ARCHS) > 50


def test_classify_known_supported() -> None:
    assert classify("llama") is ModelCompat.SUPPORTED


def test_classify_unknown_string_is_unsupported() -> None:
    assert classify("this-arch-will-never-exist") is ModelCompat.UNSUPPORTED


def test_classify_empty_is_unknown() -> None:
    assert classify("") is ModelCompat.UNKNOWN


def test_unsupported_arch_error_carries_fields() -> None:
    err = UnsupportedArchError("acme/foo-GGUF", "kimi_k2")
    assert err.ref == "acme/foo-GGUF"
    assert err.architecture == "kimi_k2"
    assert "kimi_k2" in str(err)
    assert "acme/foo-GGUF" in str(err)


def test_unsupported_arch_error_is_exception() -> None:
    with pytest.raises(UnsupportedArchError):
        raise UnsupportedArchError("a", "b")


class TestExpertOffloadFitBudget:
    """Offload puts a sparse model's experts in system RAM, so the fit budget follows.

    The budget may only borrow RAM where it was device memory to begin with;
    every other platform path already reports system RAM.
    """

    @staticmethod
    def _ram(monkeypatch, *, gib: int) -> None:
        # Installed RAM, which is what the headroom scales from: it is added to a
        # capacity-based VRAM budget, so mixing in a live figure made a catalog
        # entry fit or not depending on what else the machine was doing. Patching
        # the free reading instead left these tests reading the host's own RAM,
        # so they passed or failed by the size of the machine running them.
        monkeypatch.setattr(
            "lilbee.providers.model_cache.total_system_memory", lambda: gib * 1024**3
        )

    @staticmethod
    def _nvidia(monkeypatch, *, present: bool) -> None:
        monkeypatch.setattr("lilbee.providers.model_cache.has_nvidia_gpu", lambda: present)

    def test_no_headroom_when_offload_is_off(self, monkeypatch) -> None:
        from lilbee.core.config import cfg
        from lilbee.runtime import hardware

        monkeypatch.setattr(cfg, "cpu_moe", False)
        monkeypatch.setattr(cfg, "n_cpu_moe", None)
        self._nvidia(monkeypatch, present=True)
        assert hardware._expert_offload_headroom() == 0

    def test_headroom_borrows_system_ram_on_a_discrete_gpu_host(self, monkeypatch) -> None:
        from lilbee.core.config import cfg
        from lilbee.runtime import hardware

        monkeypatch.setattr(cfg, "cpu_moe", True)
        monkeypatch.setattr(cfg, "gpu_memory_fraction", 0.5)
        self._nvidia(monkeypatch, present=True)
        self._ram(monkeypatch, gib=64)
        assert hardware._expert_offload_headroom() == 32 * 1024**3

    def test_layer_count_alone_enables_the_headroom(self, monkeypatch) -> None:
        from lilbee.core.config import cfg
        from lilbee.runtime import hardware

        monkeypatch.setattr(cfg, "cpu_moe", False)
        monkeypatch.setattr(cfg, "n_cpu_moe", 8)
        monkeypatch.setattr(cfg, "gpu_memory_fraction", 1.0)
        self._nvidia(monkeypatch, present=True)
        self._ram(monkeypatch, gib=32)
        assert hardware._expert_offload_headroom() == 32 * 1024**3

    def test_no_double_count_without_a_discrete_gpu(self, monkeypatch) -> None:
        # Apple unified memory, a non-NVIDIA host, and a CPU-only host all report
        # system RAM as the budget already; borrowing it again invents capacity.
        from lilbee.core.config import cfg
        from lilbee.runtime import hardware

        monkeypatch.setattr(cfg, "cpu_moe", True)
        self._nvidia(monkeypatch, present=False)
        self._ram(monkeypatch, gib=64)
        assert hardware._expert_offload_headroom() == 0

    def test_a_model_past_vram_can_still_run_with_offload(self, monkeypatch) -> None:
        # The case that matters: a 49GB sparse coder on a 24GB card.
        from lilbee.core.config import cfg
        from lilbee.runtime import hardware
        from lilbee.runtime.hardware import FitLevel, compute_fit

        monkeypatch.setattr(cfg, "cpu_moe", True)
        monkeypatch.setattr(cfg, "gpu_memory_fraction", 1.0)
        self._nvidia(monkeypatch, present=True)
        self._ram(monkeypatch, gib=64)
        monkeypatch.setattr(
            "lilbee.providers.model_cache.get_available_memory", lambda *_a, **_k: 24 * 1024**3
        )
        budget = hardware.available_memory_for_fit()
        assert budget is not None
        assert compute_fit(49 * 1024**3, budget).level is FitLevel.FITS

    def test_headroom_survives_a_probe_failure(self, monkeypatch) -> None:
        from lilbee.core.config import cfg
        from lilbee.runtime import hardware

        def _boom() -> bool:
            raise RuntimeError("probe exploded")

        monkeypatch.setattr(cfg, "cpu_moe", True)
        monkeypatch.setattr("lilbee.providers.model_cache.has_nvidia_gpu", _boom)
        assert hardware._expert_offload_headroom() == 0
