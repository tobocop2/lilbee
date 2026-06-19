"""Loader-mode constants and dynamic-context / GPU-memory helpers for llama-server."""

from __future__ import annotations

import functools
import logging
import platform
import subprocess
import sys
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
    target: int | None = None,
    floor: int = _DYNAMIC_CTX_FLOOR,
    quantum: int = _DYNAMIC_CTX_QUANTUM,
) -> int:
    """Pick the n_ctx that best fits target, ceiling, and host RAM.

    Selection rule, in order:

    1. ``upper = min(training_ctx, ceiling)`` is the hard upper bound; the
       model cannot exceed its training window and the caller may cap below it.
    2. If ``target`` is provided, prefer it (clamped to ``[floor, upper]``)
       so a 40K-context model still loads at 8K when chat doesn't need more,
       rather than maximising n_ctx just because RAM allows it.
    3. ``raw_ctx = budget // kv_bytes_per_tok`` is the largest n_ctx the host
       can physically back. The result is clamped to ``raw_ctx`` so we never
       over-allocate on memory-constrained boxes.
    4. Result is quantized down to ``quantum`` and floored at ``floor``.
    """
    upper = min(training_ctx, ceiling)
    if kv_bytes_per_tok <= 0:
        if target is not None:
            return max(floor, min(target, upper))
        return upper
    overhead = int(model_bytes * _BUFFER_OVERHEAD_FRACTION)
    budget = available_bytes - model_bytes - overhead
    if budget <= 0:
        return floor
    raw_ctx = budget // kv_bytes_per_tok
    # Aim for target when set, but never above what host RAM or model training_ctx permit.
    desired = min(target, raw_ctx, upper) if target is not None else min(raw_ctx, upper)
    bounded = max(floor, desired)
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


def free_system_memory() -> int:
    """Live allocatable system RAM in bytes (free + reclaimable), right now.

    The load-time counterpart to :func:`get_available_memory`, which scales total
    capacity for sizing rather than reporting what is free this instant.
    """
    import psutil

    return int(psutil.virtual_memory().available)


def total_system_memory() -> int:
    """Total installed system RAM in bytes."""
    import psutil

    return int(psutil.virtual_memory().total)


def has_nvidia_gpu() -> bool:
    """Whether an NVIDIA GPU is detectable on this host (NVML or nvidia-smi)."""
    return _try_nvidia_memory() is not None


# A pynvml probe that runs in its OWN process: prints the minimum device total in
# bytes, or nothing. Run as a subprocess so NVML init never touches THIS process.
_PYNVML_MIN_TOTAL_SNIPPET = (
    "import pynvml; pynvml.nvmlInit();"
    "t=[pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).total"
    " for i in range(pynvml.nvmlDeviceGetCount())];"
    "pynvml.nvmlShutdown();"
    "print(min(t)) if t else None"
)
_NVIDIA_PROBE_TIMEOUT_S = 10


@functools.cache
def _try_nvidia_memory() -> int | None:
    """Minimum NVIDIA GPU total memory (bytes), detected WITHOUT touching NVML/CUDA here.

    Initializing NVML in this process (``pynvml.nvmlInit``) leaves NVIDIA driver state
    that breaks a later ``llama-server --list-devices`` CUDA probe on newer drivers, so
    detection runs only as subprocesses: ``nvidia-smi`` (ships with the driver), then an
    isolated python+pynvml process. The minimum across visible devices is conservative on
    a heterogeneous-GPU host. Cached: the subprocess runs once per process.
    """
    return _nvidia_smi_min_total() or _pynvml_min_total_isolated()


def _nvidia_smi_min_total() -> int | None:
    """Minimum GPU total memory (bytes) via ``nvidia-smi``, or None."""
    try:
        # nvidia-smi ships with the NVIDIA driver and is always on PATH when present;
        # fully-qualifying it would break on every install layout.
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    mibs = [int(line) for line in result.stdout.strip().splitlines() if line.strip()]
    return min(mibs) * 1024 * 1024 if mibs else None


def _pynvml_min_total_isolated() -> int | None:
    """Minimum GPU total memory (bytes) via pynvml run in an isolated subprocess, or None."""
    try:
        proc = subprocess.run(  # noqa: S603 - this interpreter + a fixed in-repo snippet
            [sys.executable, "-c", _PYNVML_MIN_TOTAL_SNIPPET],
            capture_output=True,
            text=True,
            timeout=_NVIDIA_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    out = proc.stdout.strip()
    return int(out) if out.isdigit() else None
