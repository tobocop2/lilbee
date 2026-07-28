"""Tests for the loader-mode helpers (kv-size, dynamic ctx, available memory)."""

from __future__ import annotations

import platform
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from lilbee.core import system as system_mod
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
        monkeypatch.setattr(
            "lilbee.providers.model_cache._try_nvidia_memory", lambda reducer=min: None
        )
        assert get_available_memory(0.25) == 200_000_000

    def test_linux_with_nvidia_uses_gpu_total(self, monkeypatch) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setattr(
            "lilbee.providers.model_cache._try_nvidia_memory",
            lambda reducer=min: 4_000_000_000,
        )
        assert get_available_memory(0.5) == 2_000_000_000

    def test_total_uses_sum_reducer_default_uses_min(self, monkeypatch) -> None:
        """total=True sums every card; the default sizes against the smallest."""
        captured: dict[str, object] = {}

        def fake(reducer=min):  # per-card totals for a 3-GPU host
            captured["reducer"] = reducer
            return reducer([10, 20, 30])

        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setattr("lilbee.providers.model_cache._try_nvidia_memory", fake)
        assert get_available_memory(1.0) == 10
        assert captured["reducer"] is min
        assert get_available_memory(1.0, total=True) == 60
        assert captured["reducer"] is sum


class TestFreeSystemMemory:
    def test_returns_live_psutil_available(self, monkeypatch, tmp_path) -> None:
        # Unlike get_available_memory (total capacity), this is what's free right
        # now -- the number that decides whether a model load would swap-thrash.
        # No cgroup, so the host figure stands; CI itself runs in a capped one.
        monkeypatch.setattr(system_mod, "_CGROUP_ROOT", tmp_path / "absent")
        fake_psutil = mock.MagicMock()
        fake_psutil.virtual_memory.return_value.available = 7_000_000_000
        monkeypatch.setitem(__import__("sys").modules, "psutil", fake_psutil)
        assert free_system_memory() == 7_000_000_000


class TestTotalSystemMemory:
    def test_returns_psutil_total(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(system_mod, "_CGROUP_ROOT", tmp_path / "absent")
        fake_psutil = mock.MagicMock()
        fake_psutil.virtual_memory.return_value.total = 8_000_000_000
        monkeypatch.setitem(__import__("sys").modules, "psutil", fake_psutil)
        assert total_system_memory() == 8_000_000_000


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


class TestHeterogeneousGpuSizing:
    def test_pynvml_takes_minimum_total_across_devices(self, monkeypatch) -> None:
        # Heterogeneous GPUs: sizing against the smallest card is conservative.
        fake_pynvml = mock.MagicMock()
        fake_pynvml.nvmlDeviceGetCount.return_value = 2
        infos = {0: mock.MagicMock(total=24 * 1024**3), 1: mock.MagicMock(total=8 * 1024**3)}
        fake_pynvml.nvmlDeviceGetHandleByIndex.side_effect = lambda i: i
        fake_pynvml.nvmlDeviceGetMemoryInfo.side_effect = lambda handle: infos[handle]
        monkeypatch.setitem(__import__("sys").modules, "pynvml", fake_pynvml)
        assert _try_nvidia_memory() == 8 * 1024**3

    def test_pynvml_sum_reducer_totals_every_device(self, monkeypatch) -> None:
        # The fit chip sums all cards: a model can tensor-split across the fleet.
        fake_pynvml = mock.MagicMock()
        fake_pynvml.nvmlDeviceGetCount.return_value = 2
        infos = {0: mock.MagicMock(total=24 * 1024**3), 1: mock.MagicMock(total=8 * 1024**3)}
        fake_pynvml.nvmlDeviceGetHandleByIndex.side_effect = lambda i: i
        fake_pynvml.nvmlDeviceGetMemoryInfo.side_effect = lambda handle: infos[handle]
        monkeypatch.setitem(__import__("sys").modules, "pynvml", fake_pynvml)
        assert _try_nvidia_memory(sum) == 32 * 1024**3

    def test_pynvml_zero_devices_falls_through_to_nvidia_smi(self, monkeypatch) -> None:
        fake_pynvml = mock.MagicMock()
        fake_pynvml.nvmlDeviceGetCount.return_value = 0
        monkeypatch.setitem(__import__("sys").modules, "pynvml", fake_pynvml)
        result = mock.MagicMock(returncode=0, stdout="4096\n")
        monkeypatch.setattr("subprocess.run", mock.MagicMock(return_value=result))
        assert _try_nvidia_memory() == 4096 * 1024 * 1024

    def test_nvidia_smi_takes_minimum_total_across_lines(self, monkeypatch) -> None:
        original_import = __import__("builtins").__import__

        def _no_pynvml(name, *args, **kwargs):
            if name == "pynvml":
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _no_pynvml)
        result = mock.MagicMock(returncode=0, stdout="24576\n8192\n")
        monkeypatch.setattr("subprocess.run", mock.MagicMock(return_value=result))
        assert _try_nvidia_memory() == 8192 * 1024 * 1024

    def test_nvidia_smi_sum_reducer_totals_every_line(self, monkeypatch) -> None:
        original_import = __import__("builtins").__import__

        def _no_pynvml(name, *args, **kwargs):
            if name == "pynvml":
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _no_pynvml)
        result = mock.MagicMock(returncode=0, stdout="24576\n8192\n")
        monkeypatch.setattr("subprocess.run", mock.MagicMock(return_value=result))
        assert _try_nvidia_memory(sum) == (24576 + 8192) * 1024 * 1024


class TestHasNvidiaGpu:
    def test_true_when_a_device_is_detected(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "lilbee.providers.model_cache._nvidia_device_totals",
            lambda: [("GPU-aaa", 8 * 1024**3)],
        )
        assert has_nvidia_gpu() is True

    def test_false_when_no_gpu(self, monkeypatch) -> None:
        monkeypatch.setattr("lilbee.providers.model_cache._nvidia_device_totals", lambda: None)
        assert has_nvidia_gpu() is False

    def test_an_empty_mask_does_not_hide_the_card(self, monkeypatch) -> None:
        """This answers "does the host have one", not "may this process use it".

        An orchestrator exporting an empty CUDA_VISIBLE_DEVICES is the case the
        env cleanup exists for, and it asks this first: if the empty mask hid
        the card here, the cleanup could never run.
        """
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        monkeypatch.setattr(
            "lilbee.providers.model_cache._nvidia_device_totals",
            lambda: [("GPU-aaa", 8 * 1024**3)],
        )
        assert has_nvidia_gpu() is True


class TestCudaVisibleDevicesMask:
    """NVML and nvidia-smi report every card the driver knows about.

    CUDA_VISIBLE_DEVICES is read by the CUDA runtime, not by either tool, so
    budgets taken straight off them describe a machine the engine does not have.
    """

    _FLEET = (("GPU-aaa", 24 * 1024**3), ("GPU-bbb", 8 * 1024**3), ("GPU-ccc", 16 * 1024**3))

    def _totals(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "lilbee.providers.model_cache._nvidia_device_totals", lambda: list(self._FLEET)
        )

    def test_unmasked_sums_the_whole_fleet(self, monkeypatch) -> None:
        self._totals(monkeypatch)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        assert _try_nvidia_memory(sum) == 48 * 1024**3

    def test_a_masked_container_sums_only_what_it_was_given(self, monkeypatch) -> None:
        """One card of an eight-card host summed all eight and approved models
        eight times too large for the card the container actually had."""
        self._totals(monkeypatch)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        assert _try_nvidia_memory(sum) == 24 * 1024**3

    def test_masking_out_the_smallest_card_raises_the_min_budget(self, monkeypatch) -> None:
        """The default reducer sized every budget against a card the engine cannot see."""
        self._totals(monkeypatch)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2")
        assert _try_nvidia_memory() == 16 * 1024**3

    def test_uuid_entries_select_the_same_way(self, monkeypatch) -> None:
        self._totals(monkeypatch)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-ccc")
        assert _try_nvidia_memory() == 16 * 1024**3

    def test_enumeration_stops_at_the_first_entry_naming_no_device(self, monkeypatch) -> None:
        """CUDA stops there, so 0,9,1 means one card and not two."""
        self._totals(monkeypatch)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,9,1")
        assert _try_nvidia_memory(sum) == 24 * 1024**3

    def test_an_empty_mask_leaves_no_budget_at_all(self, monkeypatch) -> None:
        self._totals(monkeypatch)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        assert _try_nvidia_memory() is None


class TestNvidiaSmiRowsThatDoNotParse:
    """nvidia-smi output is text; a row that is not a memory figure is skipped
    rather than allowed to poison the budget."""

    def test_a_blank_row_is_dropped(self) -> None:
        from lilbee.providers.model_cache import _parse_smi_row

        assert _parse_smi_row("") is None
        assert _parse_smi_row("  , GPU-aaa") is None

    def test_a_non_numeric_memory_figure_is_dropped(self) -> None:
        from lilbee.providers.model_cache import _parse_smi_row

        assert _parse_smi_row("[N/A], GPU-aaa") is None

    def test_an_older_smi_without_the_uuid_column_still_yields_a_device(self) -> None:
        from lilbee.providers.model_cache import _parse_smi_row

        assert _parse_smi_row("8192") == ("", 8192 * 1024 * 1024)


class TestSystemMemoryUnderACgroupCap:
    """Both readers answer for this process, not for the machine it runs on."""

    @staticmethod
    def _capped(monkeypatch, tmp_path, *, limit: int, used: int | None = None) -> None:
        (tmp_path / "memory.max").write_text(f"{limit}\n")
        if used is not None:
            (tmp_path / "memory.current").write_text(f"{used}\n")
        monkeypatch.setattr(system_mod, "_CGROUP_ROOT", tmp_path)
        monkeypatch.setattr(
            "psutil.virtual_memory",
            lambda: SimpleNamespace(total=64 * 1024**3, available=60 * 1024**3),
        )

    def test_total_is_capped_by_the_cgroup_limit(self, monkeypatch, tmp_path) -> None:
        self._capped(monkeypatch, tmp_path, limit=8 * 1024**3)
        assert total_system_memory() == 8 * 1024**3

    def test_free_is_the_cap_minus_what_the_cgroup_holds(self, monkeypatch, tmp_path) -> None:
        self._capped(monkeypatch, tmp_path, limit=8 * 1024**3, used=3 * 1024**3)
        assert free_system_memory() == 5 * 1024**3

    def test_a_cap_with_no_usage_file_bounds_free_at_the_cap(self, monkeypatch, tmp_path) -> None:
        self._capped(monkeypatch, tmp_path, limit=8 * 1024**3)
        assert free_system_memory() == 8 * 1024**3

    def test_an_uncapped_cgroup_leaves_the_host_figures_alone(self, monkeypatch, tmp_path) -> None:
        (tmp_path / "memory.max").write_text("max\n")
        monkeypatch.setattr(system_mod, "_CGROUP_ROOT", tmp_path)
        monkeypatch.setattr(
            "psutil.virtual_memory",
            lambda: SimpleNamespace(total=64 * 1024**3, available=60 * 1024**3),
        )
        assert total_system_memory() == 64 * 1024**3
        assert free_system_memory() == 60 * 1024**3

    def test_an_unreadable_host_raises_rather_than_answering_zero(self, monkeypatch) -> None:
        # A zero budget refuses every model with no reason given; the fleet's
        # sizing paths want the failure surfaced instead.
        monkeypatch.setattr("psutil.virtual_memory", mock.Mock(side_effect=RuntimeError("boom")))
        with pytest.raises(RuntimeError):
            total_system_memory()

    def test_the_coarse_budget_is_capped_too(self, monkeypatch, tmp_path) -> None:
        # The catalog fit chip reads this on a host with no device list, and a
        # capped container must not be told a model fits the machine's RAM.
        monkeypatch.setattr(system_mod, "_CGROUP_ROOT", tmp_path)
        (tmp_path / "memory.max").write_text(f"{8 * 1024**3}\n")
        monkeypatch.setattr(
            "psutil.virtual_memory",
            lambda: SimpleNamespace(total=64 * 1024**3, available=60 * 1024**3),
        )
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        assert get_available_memory(0.5) == 4 * 1024**3
