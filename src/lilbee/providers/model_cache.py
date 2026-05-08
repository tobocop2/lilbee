"""Llama-cpp loader-mode constants and dynamic-context / GPU-memory helpers."""

from __future__ import annotations

import logging
import platform
from enum import StrEnum
from pathlib import Path

log = logging.getLogger(__name__)


class LoaderMode(StrEnum):
    """Which task to configure llama.cpp for at load time."""

    CHAT = "chat"
    EMBED = "embed"
    RERANK = "rerank"

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

# Round dynamic n_ctx down to a multiple of this (clean batch sizes)
_DYNAMIC_CTX_QUANTUM = 256

# KV cache element size for f16 (bytes). Quantized KV reduces this.
_KV_ELEM_BYTES_F16 = 2


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
    ceiling: int,
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
