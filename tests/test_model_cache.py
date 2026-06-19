"""Tests for the loader-mode helpers (kv-size, dynamic ctx, available memory)."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from unittest import mock

import pytest

from lilbee.providers.model_cache import (
    _BUFFER_OVERHEAD_FRACTION,
    _DYNAMIC_CTX_FLOOR,
    _KV_BYTES_PER_CTX_TOKEN,
    _try_nvidia_memory,
    compute_dynamic_ctx,
    estimate_model_memory,
    free_system_memory,
    get_available_memory,
    has_nvidia_gpu,
    kv_bytes_per_token,
    total_system_memory,
)


class TestKvBytesPerToken:
    def test_returns_default_when_meta_is_empty(self) -> None:
        assert kv_bytes_per_token({}) == _KV_BYTES_PER_CTX_TOKEN

    def test_returns_default_when_meta_is_none(self) -> None:
        assert kv_bytes_per_token(None) == _KV_BYTES_PER_CTX_TOKEN

    def test_uses_key_value_lengths_when_present(self) -> None:
        meta = {
            "block_count": "32",
            "head_count": "32",
            "head_count_kv": "8",
            "key_length": "128",
            "value_length": "128",
        }
        # 32 layers * 8 heads * (128 + 128) = 65536 bytes per token at f16=2
        assert kv_bytes_per_token(meta) == 32 * 8 * 256 * 2

    def test_falls_back_to_embedding_length_split(self) -> None:
        """Without key_length/value_length, derives head_dim from embedding_length."""
        meta = {
            "block_count": "32",
            "head_count": "32",
            "head_count_kv": "8",
            "embedding_length": "4096",
        }
        # head_dim = 4096 / 32 = 128 -> kv_dim = 256 -> 32*8*256*2
        assert kv_bytes_per_token(meta) == 32 * 8 * 256 * 2

    def test_returns_default_on_missing_keys(self) -> None:
        assert kv_bytes_per_token({"block_count": "abc"}) == _KV_BYTES_PER_CTX_TOKEN


class TestEstimateModelMemory:
    def test_zero_bytes_when_path_missing(self, tmp_path: Path) -> None:
        # Missing file: estimate is just the KV cache (no weight, no overhead).
        absent = tmp_path / "nope.gguf"
        bytes_ = estimate_model_memory(absent, n_ctx=128, kv_bytes_per_tok=4)
        assert bytes_ == 128 * 4

    def test_includes_file_weights_kv_and_overhead(self, tmp_path: Path) -> None:
        model = tmp_path / "m.gguf"
        weight = 10_000
        model.write_bytes(b"x" * weight)
        bytes_ = estimate_model_memory(model, n_ctx=100, kv_bytes_per_tok=8)
        expected = weight + (100 * 8) + int(weight * _BUFFER_OVERHEAD_FRACTION)
        assert bytes_ == expected


class TestComputeDynamicCtx:
    def test_returns_clamped_when_kv_per_token_is_zero(self) -> None:
        ctx = compute_dynamic_ctx(
            model_bytes=1_000_000,
            available_bytes=10_000_000,
            training_ctx=8192,
            kv_bytes_per_tok=0,
            ceiling=4096,
        )
        assert ctx == 4096

    def test_returns_target_when_kv_per_token_is_zero_and_target_set(self) -> None:
        """The zero-kv fast-path honors the target instead of maxing to ceiling."""
        ctx = compute_dynamic_ctx(
            model_bytes=1_000_000,
            available_bytes=10_000_000,
            training_ctx=40_960,
            kv_bytes_per_tok=0,
            ceiling=40_960,
            target=8192,
        )
        assert ctx == 8192

    def test_returns_floor_when_budget_zero_or_negative(self) -> None:
        ctx = compute_dynamic_ctx(
            model_bytes=10_000_000,
            available_bytes=1_000_000,  # less than model
            training_ctx=8192,
            kv_bytes_per_tok=2048,
            ceiling=4096,
        )
        assert ctx == _DYNAMIC_CTX_FLOOR

    def test_quantizes_to_multiple(self) -> None:
        ctx = compute_dynamic_ctx(
            model_bytes=1_000_000,
            available_bytes=20_000_000,
            training_ctx=100_000,
            kv_bytes_per_tok=2048,
            ceiling=8192,
            quantum=256,
        )
        assert ctx % 256 == 0
        assert ctx <= 8192

    def test_target_caps_below_ceiling_when_ram_allows_more(self) -> None:
        """With plenty of RAM, n_ctx aims for target instead of maxing to ceiling."""
        ctx = compute_dynamic_ctx(
            model_bytes=1_000_000,
            available_bytes=10_000_000_000,
            training_ctx=40_960,
            kv_bytes_per_tok=2048,
            ceiling=40_960,
            target=8192,
        )
        assert ctx == 8192

    def test_target_clamped_down_by_host_ram(self) -> None:
        """When RAM cannot back target, the picker scales target down to fit."""
        ctx = compute_dynamic_ctx(
            model_bytes=1_000_000,
            available_bytes=10_000_000,  # ~9 MB budget after weights
            training_ctx=40_960,
            kv_bytes_per_tok=2048,
            ceiling=40_960,
            target=8192,
            quantum=256,
        )
        # raw_ctx = ~4395; target=8192 too big; picker uses raw_ctx, quantized.
        assert ctx < 8192
        assert ctx % 256 == 0

    def test_target_never_exceeds_training_ctx(self) -> None:
        """A 32K-training model with target=64K stays at 32K."""
        ctx = compute_dynamic_ctx(
            model_bytes=1_000_000,
            available_bytes=10_000_000_000,
            training_ctx=32_768,
            kv_bytes_per_tok=2048,
            ceiling=131_072,
            target=65_536,
        )
        assert ctx == 32_768

    def test_scaled_target_still_clamped_by_available_ram(self) -> None:
        # Simulate a 32GiB host scaling to 16384, but only 2GiB of headroom
        # at chat-worker spawn. Picker must drop below the scaled target.
        model_bytes = 3 * 1024**3
        available_bytes = model_bytes + 2 * 1024**3  # 2 GiB free for KV
        kv_bytes_per_tok = 256 * 1024  # exaggerated to force a tight clamp
        training_ctx = 128_000
        ceiling = training_ctx  # no artificial upper bound; training_ctx is the only ceiling

        picked = compute_dynamic_ctx(
            model_bytes=model_bytes,
            available_bytes=available_bytes,
            training_ctx=training_ctx,
            kv_bytes_per_tok=kv_bytes_per_tok,
            ceiling=ceiling,
            target=16384,
        )

        assert picked < 16384, (
            "scaled target must be clamped by available_bytes when KV cost exceeds budget"
        )


class TestGetAvailableMemory:
    def test_macos_uses_psutil_total(self, monkeypatch) -> None:
        fake_psutil = mock.MagicMock()
        fake_psutil.virtual_memory.return_value.total = 1_000_000_000
        monkeypatch.setitem(__import__("sys").modules, "psutil", fake_psutil)
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        assert get_available_memory(0.5) == 500_000_000

    def test_linux_falls_back_to_psutil_when_no_nvidia(self, monkeypatch) -> None:
        fake_psutil = mock.MagicMock()
        fake_psutil.virtual_memory.return_value.total = 800_000_000
        monkeypatch.setitem(__import__("sys").modules, "psutil", fake_psutil)
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setattr("lilbee.providers.model_cache._try_nvidia_memory", lambda: None)
        assert get_available_memory(0.25) == 200_000_000

    def test_linux_with_nvidia_uses_gpu_total(self, monkeypatch) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setattr(
            "lilbee.providers.model_cache._try_nvidia_memory", lambda: 4_000_000_000
        )
        assert get_available_memory(0.5) == 2_000_000_000


class TestFreeSystemMemory:
    def test_returns_live_psutil_available(self, monkeypatch) -> None:
        # Unlike get_available_memory (total capacity), this is what's free right
        # now -- the number that decides whether a model load would swap-thrash.
        fake_psutil = mock.MagicMock()
        fake_psutil.virtual_memory.return_value.available = 7_000_000_000
        monkeypatch.setitem(__import__("sys").modules, "psutil", fake_psutil)
        assert free_system_memory() == 7_000_000_000


class TestTotalSystemMemory:
    def test_returns_psutil_total(self, monkeypatch) -> None:
        fake_psutil = mock.MagicMock()
        fake_psutil.virtual_memory.return_value.total = 8_000_000_000
        monkeypatch.setitem(__import__("sys").modules, "psutil", fake_psutil)
        assert total_system_memory() == 8_000_000_000


def _fake_nvidia_run(smi=None, pynvml_out=""):
    """subprocess.run fake: nvidia-smi returns *smi* (stdout str, or an Exception to
    raise, or None for a nonzero exit); the isolated python+pynvml subprocess returns
    *pynvml_out* on stdout."""

    def run(cmd, *_a, **_k):
        if cmd and cmd[0] == "nvidia-smi":
            if isinstance(smi, Exception):
                raise smi
            return mock.MagicMock(returncode=0 if smi is not None else 1, stdout=smi or "")
        return mock.MagicMock(returncode=0, stdout=pynvml_out)

    return run


class TestTryNvidiaMemory:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        _try_nvidia_memory.cache_clear()
        yield
        _try_nvidia_memory.cache_clear()

    def test_nvidia_smi_is_primary(self, monkeypatch) -> None:
        monkeypatch.setattr("subprocess.run", _fake_nvidia_run(smi="8192\n"))
        assert _try_nvidia_memory() == 8192 * 1024 * 1024

    def test_nvidia_smi_takes_minimum_across_lines(self, monkeypatch) -> None:
        monkeypatch.setattr("subprocess.run", _fake_nvidia_run(smi="24576\n8192\n"))
        assert _try_nvidia_memory() == 8192 * 1024 * 1024

    def test_falls_back_to_isolated_pynvml_subprocess(self, monkeypatch) -> None:
        # nvidia-smi missing -> the total comes from the isolated python+pynvml subprocess.
        monkeypatch.setattr(
            "subprocess.run",
            _fake_nvidia_run(smi=FileNotFoundError(), pynvml_out=str(8 * 1024**3) + "\n"),
        )
        assert _try_nvidia_memory() == 8 * 1024**3

    def test_never_initializes_nvml_in_process(self, monkeypatch) -> None:
        # The fix: NVML must never be imported/initialized in THIS process -- the pynvml
        # fallback runs as a fresh subprocess, so it cannot poison a later CUDA probe.
        calls: list[list[str]] = []

        def run(cmd, *_a, **_k):
            calls.append(cmd)
            if cmd[0] == "nvidia-smi":
                raise FileNotFoundError
            return mock.MagicMock(returncode=0, stdout="123\n")

        monkeypatch.setattr("subprocess.run", run)
        real_import = __import__("builtins").__import__

        def _no_inprocess_pynvml(name, *a, **k):
            if name == "pynvml":
                raise AssertionError("pynvml must not be imported in this process")
            return real_import(name, *a, **k)

        monkeypatch.setattr("builtins.__import__", _no_inprocess_pynvml)
        assert _try_nvidia_memory() == 123
        assert any(c[0] == sys.executable for c in calls)  # used the isolated subprocess

    def test_returns_none_when_both_fail(self, monkeypatch) -> None:
        monkeypatch.setattr("subprocess.run", mock.MagicMock(side_effect=FileNotFoundError))
        assert _try_nvidia_memory() is None

    @pytest.mark.parametrize("returncode", [1, 2])
    def test_returns_none_when_nvidia_smi_nonzero_and_no_pynvml(
        self, monkeypatch, returncode: int
    ) -> None:
        def run(cmd, *_a, **_k):
            if cmd[0] == "nvidia-smi":
                return mock.MagicMock(returncode=returncode, stdout="")
            return mock.MagicMock(returncode=0, stdout="")  # pynvml subprocess: no output

        monkeypatch.setattr("subprocess.run", run)
        assert _try_nvidia_memory() is None


class TestHasNvidiaGpu:
    def test_true_when_memory_detected(self, monkeypatch) -> None:
        monkeypatch.setattr("lilbee.providers.model_cache._try_nvidia_memory", lambda: 8 * 1024**3)
        assert has_nvidia_gpu() is True

    def test_false_when_no_gpu(self, monkeypatch) -> None:
        monkeypatch.setattr("lilbee.providers.model_cache._try_nvidia_memory", lambda: None)
        assert has_nvidia_gpu() is False
