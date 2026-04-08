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
from typing import Any

log = logging.getLogger(__name__)

# KV cache memory estimate: 2048 bytes per context token (conservative)
_KV_BYTES_PER_CTX_TOKEN = 2048

# Metal/CUDA buffer overhead as fraction of model weight memory
_BUFFER_OVERHEAD_FRACTION = 0.10

# Default context length for estimation when metadata unavailable
_DEFAULT_CTX_LEN = 2048


@dataclass
class _CacheEntry:
    """A loaded model with metadata for eviction decisions."""

    model: Any
    path: Path
    embedding: bool
    estimated_bytes: int
    loaded_at: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        """Update last-used timestamp."""
        self.last_used = time.monotonic()


def estimate_model_memory(model_path: Path, n_ctx: int = _DEFAULT_CTX_LEN) -> int:
    """Estimate memory consumption for a GGUF model.

    Approximation: file_size (weights) + KV cache + 10% buffer overhead.
    """
    file_bytes = model_path.stat().st_size if model_path.exists() else 0
    kv_bytes = n_ctx * _KV_BYTES_PER_CTX_TOKEN
    overhead = int(file_bytes * _BUFFER_OVERHEAD_FRACTION)
    return file_bytes + kv_bytes + overhead


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

    if system == "Linux":
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
    except Exception:
        pass

    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            mib = int(result.stdout.strip().split("\n")[0])
            return mib * 1024 * 1024
    except Exception:
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

    def load_model(self, model_path: Path, embedding: bool) -> Any:
        """Load or return a cached Llama model instance.

        Evicts stale entries first, then evicts LRU if memory is tight.
        """
        key = str(model_path)

        with self._lock:
            self._evict_stale_locked()

            if key in self._cache:
                entry = self._cache[key]
                entry.touch()
                self._cache.move_to_end(key)
                log.debug("Cache hit: %s", model_path.name)
                return entry.model

            estimated = estimate_model_memory(model_path)
            available = get_available_memory(self._fraction)
            self._evict_for_space_locked(estimated, available)

            log.info(
                "Loading model %s (est. %d MB, available %d MB)",
                model_path.name,
                estimated // (1024 * 1024),
                available // (1024 * 1024),
            )
            model = self._loader(model_path, embedding=embedding)

            self._cache[key] = _CacheEntry(
                model=model,
                path=model_path,
                embedding=embedding,
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

    def get_stats(self) -> dict[str, Any]:
        """Return cache statistics for monitoring."""
        with self._lock:
            entries = []
            for _key, entry in self._cache.items():
                entries.append(
                    {
                        "path": str(entry.path),
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
