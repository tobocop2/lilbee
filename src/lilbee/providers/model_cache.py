"""Memory-aware LRU model cache for llama-cpp-python instances.

Tracks loaded Llama models in an OrderedDict, evicting least-recently-used
entries when memory is tight or keep-alive TTL expires. Thread-safe via Lock.
"""

from __future__ import annotations

import logging
import platform
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)

LoaderMode = Literal["chat", "embed", "rerank"]
MODE_CHAT: LoaderMode = "chat"
MODE_EMBED: LoaderMode = "embed"
MODE_RERANK: LoaderMode = "rerank"

# Fallback KV cache estimate when GGUF metadata can't be read.
# 2048 bytes/token undershoots real KV size for modern models (Gemma3-4B is
# ~640 KB/token f16) but is fine as a coarse pre-load eviction signal.
_KV_BYTES_PER_CTX_TOKEN = 2048

# Metal/CUDA buffer overhead as fraction of model weight memory
_BUFFER_OVERHEAD_FRACTION = 0.10

# Default context length for estimation when metadata unavailable
_DEFAULT_CTX_LEN = 2048

# Floor for the dynamic n_ctx computation (smaller is unusable for chat)
_DYNAMIC_CTX_FLOOR = 512

# Default ceiling for dynamic n_ctx (overridable via cfg.num_ctx_max)
_DYNAMIC_CTX_DEFAULT_CEILING = 16384

# Round dynamic n_ctx down to a multiple of this (clean batch sizes)
_DYNAMIC_CTX_QUANTUM = 256

# KV cache element size for f16 (bytes). Quantized KV reduces this.
_KV_ELEM_BYTES_F16 = 2


@dataclass
class _CacheEntry:
    """A loaded model with metadata for eviction decisions."""

    model: Any
    path: Path
    mode: LoaderMode
    estimated_bytes: int
    loaded_at: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        """Update last-used timestamp."""
        self.last_used = time.monotonic()

    @property
    def embedding(self) -> bool:
        """True if the underlying Llama was opened with embedding=True."""
        return self.mode in (MODE_EMBED, MODE_RERANK)


def kv_bytes_per_token(meta: dict[str, str] | None, kv_elem_bytes: int = _KV_ELEM_BYTES_F16) -> int:
    """Estimate per-token KV cache size in bytes from GGUF metadata.

    Formula: 2 (K + V) * n_layers * n_kv_heads * head_dim * elem_bytes.
    Falls back to ``_KV_BYTES_PER_CTX_TOKEN`` when metadata is missing.
    """
    if not meta:
        return _KV_BYTES_PER_CTX_TOKEN
    try:
        n_layers = int(meta["block_count"])
        head_count_kv = int(meta.get("head_count_kv") or meta["head_count"])
        if "key_length" in meta and "value_length" in meta:
            kv_dim = int(meta["key_length"]) + int(meta["value_length"])
        else:
            embed = int(meta["embedding_length"])
            head_count = int(meta.get("head_count") or head_count_kv)
            head_dim = embed // head_count
            kv_dim = 2 * head_dim
    except (KeyError, ValueError, ZeroDivisionError):
        return _KV_BYTES_PER_CTX_TOKEN
    return n_layers * head_count_kv * kv_dim * kv_elem_bytes


def estimate_model_memory(
    model_path: Path,
    n_ctx: int = _DEFAULT_CTX_LEN,
    kv_bytes_per_tok: int = _KV_BYTES_PER_CTX_TOKEN,
) -> int:
    """Estimate memory consumption for a GGUF model.
    Approximation: file_size (weights) + KV cache + 10% buffer overhead.
    """
    file_bytes = model_path.stat().st_size if model_path.exists() else 0
    kv_bytes = n_ctx * kv_bytes_per_tok
    overhead = int(file_bytes * _BUFFER_OVERHEAD_FRACTION)
    return file_bytes + kv_bytes + overhead


def compute_dynamic_ctx(
    *,
    model_bytes: int,
    available_bytes: int,
    training_ctx: int,
    kv_bytes_per_tok: int,
    ceiling: int = _DYNAMIC_CTX_DEFAULT_CEILING,
    floor: int = _DYNAMIC_CTX_FLOOR,
    quantum: int = _DYNAMIC_CTX_QUANTUM,
) -> int:
    """Pick the largest n_ctx that fits in available memory.

    Subtracts model weights and a 10% buffer overhead from ``available_bytes``,
    then divides the remainder by ``kv_bytes_per_tok``. Clamps to
    ``[floor, min(training_ctx, ceiling)]`` and rounds down to ``quantum``.
    """
    if kv_bytes_per_tok <= 0:
        return min(training_ctx, ceiling)
    overhead = int(model_bytes * _BUFFER_OVERHEAD_FRACTION)
    budget = available_bytes - model_bytes - overhead
    if budget <= 0:
        return floor
    raw_ctx = budget // kv_bytes_per_tok
    upper = min(training_ctx, ceiling)
    bounded = max(floor, min(raw_ctx, upper))
    quantized = (bounded // quantum) * quantum
    return max(floor, quantized)


def get_available_memory(fraction: float) -> int:
    """Return usable GPU/unified memory in bytes, scaled by *fraction*.
    - macOS (Apple Silicon): unified memory via psutil
    - Linux with NVIDIA GPU: pynvml -> nvidia-smi -> psutil fallback
    - Other: psutil system memory
    """
    import psutil

    system = platform.system()

    if system == "Darwin":
        total = psutil.virtual_memory().total
        return int(total * fraction)

    if system in ("Linux", "Windows"):
        nvidia_mem = _try_nvidia_memory()
        if nvidia_mem is not None:
            return int(nvidia_mem * fraction)

    total = psutil.virtual_memory().total
    return int(total * fraction)


def _try_nvidia_memory() -> int | None:
    """Try to get NVIDIA GPU total memory via pynvml, then nvidia-smi."""
    try:
        import pynvml  # type: ignore[import-untyped]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return int(info.total)
    except Exception:  # noqa: S110 -- optional GPU detect; absence is expected on non-NVIDIA hosts
        pass

    try:
        import subprocess

        # nvidia-smi ships with the NVIDIA driver and is always on PATH when
        # present; fully-qualifying it would break on every install layout.
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            mib = int(result.stdout.strip().split("\n")[0])
            return mib * 1024 * 1024
    except Exception:  # noqa: S110 -- optional GPU detect; same rationale as above
        pass

    return None


class MemoryAwareModelCache:
    """LRU cache for Llama model instances with memory-aware eviction.
    Models are evicted when:
    - A new model won't fit in the memory budget (LRU evicted first)
    - A model's keep-alive TTL has expired (checked on load and via evict_stale)
    """

    def __init__(
        self,
        max_memory_fraction: float = 0.75,
        keep_alive_seconds: int = 300,
        loader: Any = None,
    ) -> None:
        self._fraction = max_memory_fraction
        self._keep_alive = keep_alive_seconds
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._loader = loader

    def load_model(self, model_path: Path, mode: LoaderMode) -> Any:
        """Load or return a cached Llama model instance.

        Evicts stale entries first, then evicts LRU if memory is tight.
        The cache key includes the mode so chat, embed, and rerank loaders
        stay isolated (they construct Llama with different flags and must
        not be aliased).
        """
        key = f"{model_path}:{mode}"

        with self._lock:
            self._evict_stale_locked()

            if key in self._cache:
                entry = self._cache[key]
                entry.touch()
                self._cache.move_to_end(key)
                log.debug("Cache hit: %s (%s)", model_path.name, mode)
                return entry.model

            estimated = estimate_model_memory(model_path)
            available = get_available_memory(self._fraction)
            self._evict_for_space_locked(estimated, available)

            log.info(
                "Loading model %s in %s mode (est. %d MB, available %d MB)",
                model_path.name,
                mode,
                estimated // (1024 * 1024),
                available // (1024 * 1024),
            )
            model = self._loader(model_path, mode=mode)

            self._cache[key] = _CacheEntry(
                model=model,
                path=model_path,
                mode=mode,
                estimated_bytes=estimated,
            )
            return model

    def _evict_stale_locked(self) -> int:
        """Remove models past keep_alive TTL. Must hold self._lock."""
        if self._keep_alive <= 0:
            return 0
        now = time.monotonic()
        stale_keys = [
            k for k, entry in self._cache.items() if (now - entry.last_used) > self._keep_alive
        ]
        for k in stale_keys:
            self._unload_entry(k)
        return len(stale_keys)

    def _evict_for_space_locked(self, needed: int, available: int) -> None:
        """Evict LRU entries until *needed* bytes fit within *available*."""
        current_usage = sum(e.estimated_bytes for e in self._cache.values())
        while self._cache and (current_usage + needed) > available:
            oldest_key = next(iter(self._cache))
            oldest = self._cache[oldest_key]
            current_usage -= oldest.estimated_bytes
            log.info("Evicting LRU model %s to free memory", oldest.path.name)
            self._unload_entry(oldest_key)

    def _unload_entry(self, key: str) -> None:
        """Remove and close a single cache entry. Must hold self._lock."""
        entry = self._cache.pop(key, None)
        if entry is not None:
            try:
                entry.model.close()
            except AttributeError:
                pass
            except Exception:
                log.debug("Error closing model %s", entry.path.name, exc_info=True)

    def evict_stale(self) -> int:
        """Remove models past keep_alive TTL. Returns count evicted."""
        with self._lock:
            return self._evict_stale_locked()

    def unload_all(self) -> None:
        """Clear entire cache, closing all models."""
        with self._lock:
            keys = list(self._cache.keys())
            for k in keys:
                self._unload_entry(k)

    def unload_path(self, model_path: Path) -> int:
        """Evict every cache entry for *model_path* (across all modes). Returns count."""
        with self._lock:
            keys = [k for k, entry in self._cache.items() if entry.path == model_path]
            for k in keys:
                self._unload_entry(k)
            return len(keys)

    def get_stats(self) -> dict[str, Any]:
        """Return cache statistics for monitoring."""
        with self._lock:
            entries = []
            for _key, entry in self._cache.items():
                entries.append(
                    {
                        "path": str(entry.path),
                        "mode": entry.mode,
                        "embedding": entry.embedding,
                        "estimated_mb": entry.estimated_bytes // (1024 * 1024),
                        "age_seconds": int(time.monotonic() - entry.loaded_at),
                        "idle_seconds": int(time.monotonic() - entry.last_used),
                    }
                )
            return {
                "loaded_models": len(self._cache),
                "total_estimated_mb": sum(e.estimated_bytes for e in self._cache.values())
                // (1024 * 1024),
                "keep_alive_seconds": self._keep_alive,
                "memory_fraction": self._fraction,
                "models": entries,
            }
