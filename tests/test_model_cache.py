"""Tests for the llama-cpp loader-mode helpers (kv-size, dynamic ctx, available memory)."""

from __future__ import annotations

import platform
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
    get_available_memory,
    kv_bytes_per_token,
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


class TestTryNvidiaMemory:
    def test_returns_total_from_pynvml_when_available(self, monkeypatch) -> None:
        """pynvml success path returns the GPU total in bytes."""
        fake_pynvml = mock.MagicMock()
        fake_info = mock.MagicMock()
        fake_info.total = 16 * 1024 * 1024 * 1024
        fake_pynvml.nvmlDeviceGetMemoryInfo.return_value = fake_info
        monkeypatch.setitem(__import__("sys").modules, "pynvml", fake_pynvml)
        assert _try_nvidia_memory() == 16 * 1024 * 1024 * 1024

    def test_returns_none_when_pynvml_and_nvidia_smi_fail(self, monkeypatch) -> None:
        # Force pynvml import to fail (no module installed by default in CI).
        original_import = __import__("builtins").__import__

        def _no_pynvml(name, *args, **kwargs):
            if name == "pynvml":
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _no_pynvml)
        monkeypatch.setattr("subprocess.run", mock.MagicMock(side_effect=FileNotFoundError))
        assert _try_nvidia_memory() is None

    def test_returns_total_from_nvidia_smi_when_pynvml_unavailable(self, monkeypatch) -> None:
        original_import = __import__("builtins").__import__

        def _no_pynvml(name, *args, **kwargs):
            if name == "pynvml":
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _no_pynvml)
        result = mock.MagicMock(returncode=0, stdout="8192\n")
        monkeypatch.setattr("subprocess.run", mock.MagicMock(return_value=result))
        assert _try_nvidia_memory() == 8192 * 1024 * 1024

    @pytest.mark.parametrize("returncode", [1, 2])
    def test_returns_none_when_nvidia_smi_nonzero_exit(self, monkeypatch, returncode: int) -> None:
        original_import = __import__("builtins").__import__

        def _no_pynvml(name, *args, **kwargs):
            if name == "pynvml":
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _no_pynvml)
        result = mock.MagicMock(returncode=returncode, stdout="")
        monkeypatch.setattr("subprocess.run", mock.MagicMock(return_value=result))
        assert _try_nvidia_memory() is None
