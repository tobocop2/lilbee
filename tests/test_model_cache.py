"""Tests for MemoryAwareModelCache."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest import mock

import pytest

from lilbee.providers.model_cache import (
    MemoryAwareModelCache,
    _CacheEntry,
    _try_nvidia_memory,
    compute_dynamic_ctx,
    estimate_model_memory,
    get_available_memory,
    kv_bytes_per_token,
)


@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    """Create a temp directory with fake GGUF files."""
    d = tmp_path / "models"
    d.mkdir()
    # 1 MB fake model
    (d / "chat.gguf").write_bytes(b"\x00" * 1_000_000)
    (d / "embed.gguf").write_bytes(b"\x00" * 500_000)
    return d


def _fake_loader(path: Path, *, mode: str = "chat") -> mock.MagicMock:
    """Simulate loading a Llama model: returns a unique mock per call."""
    m = mock.MagicMock()
    m._model_path = str(path)
    m._mode = mode
    return m


@mock.patch("lilbee.providers.model_cache.get_available_memory", return_value=10 * 1024**3)
def test_cache_hit_returns_same_instance(_mem: object, model_dir: Path) -> None:
    cache = MemoryAwareModelCache(loader=_fake_loader)
    path = model_dir / "chat.gguf"

    first = cache.load_model(path, mode="chat")
    second = cache.load_model(path, mode="chat")

    assert first is second


@mock.patch("lilbee.providers.model_cache.get_available_memory", return_value=10 * 1024**3)
def test_different_paths_return_different_instances(_mem: object, model_dir: Path) -> None:
    cache = MemoryAwareModelCache(loader=_fake_loader)

    chat = cache.load_model(model_dir / "chat.gguf", mode="chat")
    embed = cache.load_model(model_dir / "embed.gguf", mode="embed")

    assert chat is not embed
    stats = cache.get_stats()
    assert stats["loaded_models"] == 2


@mock.patch("lilbee.providers.model_cache.get_available_memory")
def test_lru_eviction_when_memory_tight(mock_mem: mock.MagicMock, model_dir: Path) -> None:
    """When a new model won't fit, the oldest (LRU) is evicted."""
    # Available memory barely fits one model at a time
    # Each model is ~1MB file + overhead ~= 5MB estimated
    mock_mem.return_value = 6_000_000

    cache = MemoryAwareModelCache(loader=_fake_loader)
    chat_path = model_dir / "chat.gguf"
    embed_path = model_dir / "embed.gguf"

    first = cache.load_model(chat_path, mode="chat")
    assert cache.get_stats()["loaded_models"] == 1

    # Loading second model should evict first
    _second = cache.load_model(embed_path, mode="embed")
    stats = cache.get_stats()
    assert stats["loaded_models"] == 1
    # First model should have been evicted (close called)
    first.close.assert_called_once()


def test_keep_alive_expiry(model_dir: Path) -> None:
    """Models past keep_alive TTL are evicted on next load."""
    with mock.patch("lilbee.providers.model_cache.get_available_memory", return_value=10 * 1024**3):
        cache = MemoryAwareModelCache(keep_alive_seconds=1, loader=_fake_loader)
        path = model_dir / "chat.gguf"

        model = cache.load_model(path, mode="chat")
        assert cache.get_stats()["loaded_models"] == 1

    # Simulate time passing by manipulating the entry's last_used
    with cache._lock:
        entry = cache._cache[f"{path}:chat"]
        entry.last_used = time.monotonic() - 10  # 10 seconds ago

    evicted = cache.evict_stale()
    assert evicted == 1
    assert cache.get_stats()["loaded_models"] == 0
    model.close.assert_called_once()


def test_memory_estimation(model_dir: Path) -> None:
    """File size + KV cache + overhead."""
    path = model_dir / "chat.gguf"
    file_size = path.stat().st_size  # 1_000_000
    n_ctx = 2048

    estimated = estimate_model_memory(path, n_ctx=n_ctx)

    kv = n_ctx * 2048
    overhead = int(file_size * 0.10)
    expected = file_size + kv + overhead
    assert estimated == expected


def test_memory_estimation_missing_file(tmp_path: Path) -> None:
    """Non-existent file gives 0 for file_size component."""
    path = tmp_path / "missing.gguf"
    estimated = estimate_model_memory(path, n_ctx=512)
    # Just KV cache + 0 overhead
    assert estimated == 512 * 2048


@mock.patch("lilbee.providers.model_cache.get_available_memory", return_value=10 * 1024**3)
def test_thread_safety(_mem: object, model_dir: Path) -> None:
    """Concurrent loads don't crash or corrupt state."""
    cache = MemoryAwareModelCache(loader=_fake_loader)
    path = model_dir / "chat.gguf"
    results: list[object] = []
    errors: list[Exception] = []

    def load() -> None:
        try:
            m = cache.load_model(path, mode="chat")
            results.append(m)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=load) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not errors
    assert len(results) == 10
    # All threads should get the same cached instance
    assert all(r is results[0] for r in results)


@mock.patch("lilbee.providers.model_cache.get_available_memory", return_value=10 * 1024**3)
def test_unload_all(_mem: object, model_dir: Path) -> None:
    cache = MemoryAwareModelCache(loader=_fake_loader)
    m1 = cache.load_model(model_dir / "chat.gguf", mode="chat")
    m2 = cache.load_model(model_dir / "embed.gguf", mode="embed")

    cache.unload_all()

    assert cache.get_stats()["loaded_models"] == 0
    m1.close.assert_called_once()
    m2.close.assert_called_once()


@mock.patch("lilbee.providers.model_cache.get_available_memory", return_value=10 * 1024**3)
def test_unload_path_evicts_only_matching_path(_mem: object, model_dir: Path) -> None:
    cache = MemoryAwareModelCache(loader=_fake_loader)
    chat_path = model_dir / "chat.gguf"
    embed_path = model_dir / "embed.gguf"
    chat = cache.load_model(chat_path, mode="chat")
    embed = cache.load_model(embed_path, mode="embed")

    evicted = cache.unload_path(chat_path)

    assert evicted == 1
    chat.close.assert_called_once()
    embed.close.assert_not_called()
    stats = cache.get_stats()
    assert stats["loaded_models"] == 1
    assert stats["models"][0]["path"] == str(embed_path)


@mock.patch("lilbee.providers.model_cache.get_available_memory", return_value=10 * 1024**3)
def test_unload_path_evicts_all_modes_for_same_path(_mem: object, model_dir: Path) -> None:
    cache = MemoryAwareModelCache(loader=_fake_loader)
    path = model_dir / "embed.gguf"
    embed = cache.load_model(path, mode="embed")
    rerank = cache.load_model(path, mode="rerank")

    evicted = cache.unload_path(path)

    assert evicted == 2
    embed.close.assert_called_once()
    rerank.close.assert_called_once()
    assert cache.get_stats()["loaded_models"] == 0


@mock.patch("lilbee.providers.model_cache.get_available_memory", return_value=10 * 1024**3)
def test_unload_path_unknown_path_is_noop(_mem: object, model_dir: Path) -> None:
    cache = MemoryAwareModelCache(loader=_fake_loader)
    cache.load_model(model_dir / "chat.gguf", mode="chat")

    evicted = cache.unload_path(model_dir / "missing.gguf")

    assert evicted == 0
    assert cache.get_stats()["loaded_models"] == 1


@mock.patch("lilbee.providers.model_cache.get_available_memory", return_value=10 * 1024**3)
def test_get_stats(_mem: object, model_dir: Path) -> None:
    cache = MemoryAwareModelCache(
        max_memory_fraction=0.8, keep_alive_seconds=120, loader=_fake_loader
    )
    cache.load_model(model_dir / "chat.gguf", mode="chat")

    stats = cache.get_stats()

    assert stats["loaded_models"] == 1
    assert stats["keep_alive_seconds"] == 120
    assert stats["memory_fraction"] == 0.8
    assert len(stats["models"]) == 1
    assert stats["models"][0]["embedding"] is False
    assert stats["models"][0]["estimated_mb"] >= 0


@mock.patch("psutil.virtual_memory")
def test_get_available_memory_darwin(mock_vm: mock.MagicMock) -> None:
    """macOS path uses psutil total memory."""
    mock_vm.return_value = mock.MagicMock(total=16 * 1024**3)
    with mock.patch("lilbee.providers.model_cache.platform.system", return_value="Darwin"):
        result = get_available_memory(0.75)
    assert result == int(16 * 1024**3 * 0.75)


@mock.patch("psutil.virtual_memory")
def test_get_available_memory_linux_fallback(mock_vm: mock.MagicMock) -> None:
    """Linux without NVIDIA falls back to psutil."""
    mock_vm.return_value = mock.MagicMock(total=32 * 1024**3)
    with (
        mock.patch("lilbee.providers.model_cache.platform.system", return_value="Linux"),
        mock.patch("lilbee.providers.model_cache._try_nvidia_memory", return_value=None),
    ):
        result = get_available_memory(0.5)
    assert result == int(32 * 1024**3 * 0.5)


@mock.patch("psutil.virtual_memory")
def test_get_available_memory_linux_nvidia(mock_vm: mock.MagicMock) -> None:
    """Linux with NVIDIA uses GPU memory."""
    mock_vm.return_value = mock.MagicMock(total=32 * 1024**3)
    gpu_total = 8 * 1024**3
    with (
        mock.patch("lilbee.providers.model_cache.platform.system", return_value="Linux"),
        mock.patch("lilbee.providers.model_cache._try_nvidia_memory", return_value=gpu_total),
    ):
        result = get_available_memory(0.75)
    assert result == int(gpu_total * 0.75)


def test_cache_entry_touch() -> None:
    """Touch updates last_used timestamp."""
    entry = _CacheEntry(
        model=mock.MagicMock(),
        path=Path("/fake"),
        mode="chat",
        estimated_bytes=100,
    )
    old = entry.last_used
    time.sleep(0.05)  # Windows timer resolution is ~15.6ms
    entry.touch()
    assert entry.last_used > old


@mock.patch("lilbee.providers.model_cache.get_available_memory", return_value=10 * 1024**3)
def test_evict_stale_with_zero_keep_alive(_mem: object, model_dir: Path) -> None:
    """keep_alive=0 means evict_stale is a no-op (immediate unload is handled elsewhere)."""
    cache = MemoryAwareModelCache(keep_alive_seconds=0, loader=_fake_loader)
    cache.load_model(model_dir / "chat.gguf", mode="chat")
    evicted = cache.evict_stale()
    assert evicted == 0


@mock.patch("lilbee.providers.model_cache.get_available_memory", return_value=10 * 1024**3)
def test_unload_entry_no_close(_mem: object, model_dir: Path) -> None:
    """Models without a close method don't error on eviction."""

    def loader_no_close(path: Path, *, mode: str = "chat") -> str:
        return "plain-string-model"

    cache = MemoryAwareModelCache(loader=loader_no_close)
    cache.load_model(model_dir / "chat.gguf", mode="chat")
    cache.unload_all()  # Should not raise
    assert cache.get_stats()["loaded_models"] == 0


@mock.patch("lilbee.providers.model_cache.get_available_memory", return_value=10 * 1024**3)
def test_unload_entry_close_error(_mem: object, model_dir: Path) -> None:
    """Models whose close() raises are handled gracefully."""

    def loader_bad_close(path: Path, *, mode: str = "chat") -> mock.MagicMock:
        m = mock.MagicMock()
        m.close.side_effect = RuntimeError("GPU error")
        return m

    cache = MemoryAwareModelCache(loader=loader_bad_close)
    cache.load_model(model_dir / "chat.gguf", mode="chat")
    cache.unload_all()  # Should not raise despite close() error
    assert cache.get_stats()["loaded_models"] == 0


def test_try_nvidia_memory_pynvml_success() -> None:
    """pynvml path returns GPU memory."""
    mock_pynvml = mock.MagicMock()
    mock_info = mock.MagicMock()
    mock_info.total = 8 * 1024**3
    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_info

    with mock.patch.dict("sys.modules", {"pynvml": mock_pynvml}):
        result = _try_nvidia_memory()

    assert result == 8 * 1024**3


def test_try_nvidia_memory_pynvml_fails_nvidia_smi_success() -> None:
    """Falls back to nvidia-smi when pynvml fails."""
    with (
        mock.patch.dict("sys.modules", {"pynvml": None}),
        mock.patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = mock.MagicMock(returncode=0, stdout="8192\n")
        result = _try_nvidia_memory()

    assert result == 8192 * 1024 * 1024


def test_try_nvidia_memory_all_fail() -> None:
    """Returns None when no NVIDIA detection works."""
    with (
        mock.patch.dict("sys.modules", {"pynvml": None}),
        mock.patch("subprocess.run", side_effect=FileNotFoundError),
    ):
        result = _try_nvidia_memory()

    assert result is None


def test_try_nvidia_memory_nvidia_smi_nonzero_exit() -> None:
    """nvidia-smi with non-zero exit returns None."""
    with (
        mock.patch.dict("sys.modules", {"pynvml": None}),
        mock.patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = mock.MagicMock(returncode=1, stdout="")
        result = _try_nvidia_memory()

    assert result is None


@mock.patch("psutil.virtual_memory")
def test_get_available_memory_windows_nvidia(mock_vm: mock.MagicMock) -> None:
    """Windows + NVIDIA picks pynvml VRAM, not system RAM."""
    mock_vm.return_value = mock.MagicMock(total=16 * 1024**3)
    gpu_total = 8 * 1024**3
    with (
        mock.patch("lilbee.providers.model_cache.platform.system", return_value="Windows"),
        mock.patch("lilbee.providers.model_cache._try_nvidia_memory", return_value=gpu_total),
    ):
        result = get_available_memory(0.75)
    assert result == int(gpu_total * 0.75)


@mock.patch("psutil.virtual_memory")
def test_get_available_memory_windows_no_nvidia(mock_vm: mock.MagicMock) -> None:
    """Windows without NVIDIA falls back to system RAM."""
    mock_vm.return_value = mock.MagicMock(total=16 * 1024**3)
    with (
        mock.patch("lilbee.providers.model_cache.platform.system", return_value="Windows"),
        mock.patch("lilbee.providers.model_cache._try_nvidia_memory", return_value=None),
    ):
        result = get_available_memory(0.5)
    assert result == int(16 * 1024**3 * 0.5)


def test_kv_bytes_per_token_uses_metadata() -> None:
    """KV bytes/token = 2 * n_layers * n_kv_heads * (key_length + value_length) * elem_bytes."""
    meta = {
        "block_count": "32",
        "head_count_kv": "8",
        "key_length": "128",
        "value_length": "128",
    }
    # f16: 2 bytes/elem, no leading factor of 2 here because key+value already sum
    expected = 32 * 8 * (128 + 128) * 2
    assert kv_bytes_per_token(meta, kv_elem_bytes=2) == expected


def test_kv_bytes_per_token_quantized_kv() -> None:
    """q8_0 cuts per-token bytes in half vs f16."""
    meta = {
        "block_count": "32",
        "head_count_kv": "8",
        "key_length": "128",
        "value_length": "128",
    }
    f16 = kv_bytes_per_token(meta, kv_elem_bytes=2)
    q8 = kv_bytes_per_token(meta, kv_elem_bytes=1)
    assert q8 == f16 // 2


def test_kv_bytes_per_token_fallback_when_meta_missing() -> None:
    """Missing metadata returns the conservative flat constant."""
    assert kv_bytes_per_token(None) == 2048
    assert kv_bytes_per_token({}) == 2048


def test_kv_bytes_per_token_derives_head_dim_from_embedding() -> None:
    """Without explicit key_length, fall back to embedding_length / head_count."""
    meta = {
        "block_count": "16",
        "head_count_kv": "4",
        "head_count": "8",
        "embedding_length": "1024",
    }
    head_dim = 1024 // 8
    expected = 16 * 4 * (2 * head_dim) * 2
    assert kv_bytes_per_token(meta, kv_elem_bytes=2) == expected


def test_compute_dynamic_ctx_typical_case() -> None:
    """Picks the largest 256-multiple ctx that fits the budget."""
    # 4 GB model, 8 GB available, 256 KB/token KV.
    model_bytes = 4 * 1024**3
    available = 8 * 1024**3
    kv_per_tok = 256 * 1024
    ctx = compute_dynamic_ctx(
        model_bytes=model_bytes,
        available_bytes=available,
        training_ctx=131072,
        kv_bytes_per_tok=kv_per_tok,
        ceiling=16384,
    )
    overhead = int(model_bytes * 0.10)
    budget = available - model_bytes - overhead
    raw = budget // kv_per_tok
    expected = (min(raw, 16384) // 256) * 256
    assert ctx == expected
    assert ctx % 256 == 0


def test_compute_dynamic_ctx_returns_floor_when_oversubscribed() -> None:
    """When the model alone exceeds available memory, return the floor."""
    ctx = compute_dynamic_ctx(
        model_bytes=10 * 1024**3,
        available_bytes=2 * 1024**3,
        training_ctx=8192,
        kv_bytes_per_tok=128 * 1024,
        ceiling=16384,
    )
    assert ctx == 512


def test_compute_dynamic_ctx_honors_training_ctx_upper_bound() -> None:
    """Even with infinite memory, never exceed the model's training window."""
    ctx = compute_dynamic_ctx(
        model_bytes=100,
        available_bytes=10**12,
        training_ctx=4096,
        kv_bytes_per_tok=1,
        ceiling=131072,
    )
    assert ctx == 4096


def test_compute_dynamic_ctx_honors_ceiling() -> None:
    """The ceiling caps a generous training window on a generous host."""
    ctx = compute_dynamic_ctx(
        model_bytes=100,
        available_bytes=10**12,
        training_ctx=131072,
        kv_bytes_per_tok=1,
        ceiling=8192,
    )
    assert ctx == 8192


def test_compute_dynamic_ctx_zero_kv_falls_back_to_training_or_ceiling() -> None:
    """Zero/negative kv_bytes_per_tok (degenerate metadata) returns min(training, ceiling)."""
    ctx = compute_dynamic_ctx(
        model_bytes=1_000_000,
        available_bytes=10**12,
        training_ctx=4096,
        kv_bytes_per_tok=0,
        ceiling=8192,
    )
    assert ctx == 4096
