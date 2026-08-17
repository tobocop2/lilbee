"""Loader-mode constants and dynamic-context / GPU-memory helpers for llama-server."""

from __future__ import annotations

import logging
import os
import platform
from collections.abc import Callable
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


def kv_bytes_per_token(
    meta: dict[str, str] | None,
    k_elem_bytes: float = _KV_ELEM_BYTES_F16,
    v_elem_bytes: float | None = None,
) -> int:
    """Estimate per-token KV cache size in bytes from GGUF metadata.

    Formula: n_layers * n_kv_heads * (k_dim * k_elem_bytes + v_dim * v_elem_bytes).
    K and V are charged separately because a launch can quantize one and not the
    other: a quantized V cache needs flash attention, so the engine keeps V at
    f16 when flash attention is not certain. ``v_elem_bytes`` defaults to the K
    cost for callers whose caches match. Falls back to
    ``_KV_BYTES_PER_CTX_TOKEN`` when metadata is missing.
    """
    if v_elem_bytes is None:
        v_elem_bytes = k_elem_bytes
    if not meta:
        return _KV_BYTES_PER_CTX_TOKEN
    try:
        n_layers = int(meta["block_count"])
        head_count_kv = int(meta.get("head_count_kv") or meta["head_count"])
        if "key_length" in meta and "value_length" in meta:
            k_dim = int(meta["key_length"])
            v_dim = int(meta["value_length"])
        else:
            embed = int(meta["embedding_length"])
            head_count = int(meta.get("head_count") or head_count_kv)
            k_dim = v_dim = embed // head_count
    except (KeyError, ValueError, ZeroDivisionError):
        return _KV_BYTES_PER_CTX_TOKEN
    return int(n_layers * head_count_kv * (k_dim * k_elem_bytes + v_dim * v_elem_bytes))


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
    """Pick the n_ctx that best fits target, ceiling, and ``available_bytes``.

    Selection rule, in order:

    1. ``upper = min(training_ctx, ceiling)`` is the hard upper bound; the
       model cannot exceed its training window and the caller may cap below it.
    2. If ``target`` is provided, prefer it (clamped to ``[floor, upper]``)
       so a 40K-context model still loads at 8K when chat doesn't need more,
       rather than maximising n_ctx just because the memory allows it.
    3. ``raw_ctx = budget // kv_bytes_per_tok`` is the largest n_ctx the
       available memory can physically back. The result is clamped to
       ``raw_ctx`` so we never over-allocate on memory-constrained boxes.
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
    # Aim for target when set, but never above what the memory or training_ctx permit.
    desired = min(target, raw_ctx, upper) if target is not None else min(raw_ctx, upper)
    bounded = max(floor, desired)
    quantized = (bounded // quantum) * quantum
    return max(floor, quantized)


def get_available_memory(fraction: float, *, total: bool = False) -> int:
    """Return usable GPU/unified memory in bytes, scaled by *fraction*.
    - macOS (Apple Silicon): unified memory via psutil
    - Linux with NVIDIA GPU: pynvml -> nvidia-smi -> psutil fallback
    - Other: psutil system memory

    With multiple NVIDIA GPUs, *total* sums every card's memory (whole-fleet
    capacity, for deciding whether a model can run tensor-split across all of
    them); the default sizes against the smallest single card.

    A coarse figure for callers with no device list to hand. The fleet has one
    and sizes against it instead
    (:func:`lilbee.providers.fleet.planning.plan_sizing_budget`), because this
    answers with system RAM on every host without an NVIDIA card. That system
    figure is the process's, cgroup cap included, not the machine's.
    """
    system = platform.system()

    if system == "Darwin":
        return int(total_system_memory() * fraction)

    if system in ("Linux", "Windows"):
        nvidia_mem = _try_nvidia_memory(sum if total else min)
        if nvidia_mem is not None:
            return int(nvidia_mem * fraction)

    return int(total_system_memory() * fraction)


def free_system_memory() -> int:
    """Live allocatable system RAM in bytes (free + reclaimable), right now.

    The load-time counterpart to :func:`get_available_memory`, which scales total
    capacity for sizing rather than reporting what is free this instant.

    Bounded by what this process's cgroup still has, for the reason in
    :func:`lilbee.core.system.cgroup_memory_limit`.
    """
    import psutil

    from lilbee.core.system import cgroup_memory_limit, cgroup_memory_used

    host_free = int(psutil.virtual_memory().available)
    limit = cgroup_memory_limit()
    if limit is None:
        return host_free
    used = cgroup_memory_used()
    return min(host_free, limit if used is None else max(0, limit - used))


def total_system_memory() -> int:
    """Total system RAM in bytes this process may use, cgroup cap included.

    Raises rather than answering zero when the host cannot be read: every caller
    here is sizing a real placement, and a budget computed from zero refuses
    every model without saying why.
    """
    from lilbee.core.system import capped_total_memory

    return capped_total_memory()


def has_nvidia_gpu() -> bool:
    """Whether an NVIDIA GPU is physically present on this host (NVML or nvidia-smi).

    Deliberately unmasked. ``CUDA_VISIBLE_DEVICES`` says what a CUDA process may
    use, not what the machine has, and the callers of this ask the second
    question: one of them exists to delete an empty mask that an orchestrator
    left behind, which it could never do if the empty mask hid the card first.
    """
    return _nvidia_device_totals() is not None


def _try_nvidia_memory(reducer: Callable[[list[int]], int] = min) -> int | None:
    """NVIDIA GPU total memory the CUDA runtime can actually reach, or ``None``.

    *reducer* combines the per-device totals. ``min`` (the default) sizes against
    the smallest card, the safe budget for a single server that has not been told
    which card it will run on. ``sum`` gives whole-fleet capacity, used only by
    the catalog fit chip to decide whether a model can run split across every card.

    Restricted to the devices ``CUDA_VISIBLE_DEVICES`` exposes. Neither NVML nor
    nvidia-smi applies that mask on its own: it is read by the CUDA runtime, and
    both tools report every card the driver knows about. Unmasked, a container
    given one card of an eight-card host summed all eight and approved models
    eight times too large for the card it had, and a fleet whose smallest card
    was masked out sized every budget against a card the engine cannot see.
    """
    totals = _nvidia_device_totals()
    if not totals:
        return None
    visible = _apply_cuda_visible_mask(totals)
    return reducer([total for _uuid, total in visible]) if visible else None


def _nvidia_device_totals() -> list[tuple[str, int]] | None:
    """``[(uuid, total_bytes), ...]`` in driver enumeration order, or ``None``.

    ``None`` means no NVIDIA GPU was detectable at all, which is the expected
    outcome on every non-NVIDIA host.
    """
    try:
        import pynvml  # type: ignore[import-untyped]

        pynvml.nvmlInit()
        totals = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            totals.append(
                (
                    _decoded(pynvml.nvmlDeviceGetUUID(handle)),
                    int(pynvml.nvmlDeviceGetMemoryInfo(handle).total),
                )
            )
        pynvml.nvmlShutdown()
        if totals:
            return totals
    except Exception:  # noqa: S110 -- optional GPU detect; absence is expected on non-NVIDIA hosts
        pass

    try:
        import subprocess

        # nvidia-smi ships with the NVIDIA driver and is always on PATH when
        # present; fully-qualifying it would break on every install layout.
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,uuid", "--format=csv,noheader,nounits"],  # noqa: S607
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
        if result.returncode == 0:
            rows = [_parse_smi_row(line) for line in result.stdout.strip().splitlines()]
            parsed = [row for row in rows if row is not None]
            if parsed:
                return parsed
    except Exception:  # noqa: S110 -- optional GPU detect; same rationale as above
        pass

    return None


def _decoded(value: str | bytes) -> str:
    """pynvml returns ``str`` on recent versions and ``bytes`` on older ones."""
    return value.decode() if isinstance(value, bytes) else value


def _parse_smi_row(line: str) -> tuple[str, int] | None:
    """One ``memory.total,uuid`` CSV row as ``(uuid, total_bytes)``.

    The UUID column is optional so an older nvidia-smi that only echoes the
    memory still yields a device; only a UUID-keyed mask needs it.
    """
    fields = [field.strip() for field in line.split(",")]
    if not fields or not fields[0]:
        return None
    try:
        mib = int(fields[0])
    except ValueError:
        return None
    return (fields[1] if len(fields) > 1 else "", mib * 1024 * 1024)


def _apply_cuda_visible_mask(devices: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """The subset of *devices* ``CUDA_VISIBLE_DEVICES`` exposes, in its order.

    Entries are driver indexes or ``GPU-``/``MIG-`` UUIDs. An unset variable
    masks nothing; an empty one exposes nothing. CUDA stops enumerating at the
    first entry that names no device, and so does this, which is what makes
    ``0,9,1`` on a two-card host mean one card rather than two.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return devices
    visible: list[tuple[str, int]] = []
    for entry in (part.strip() for part in raw.split(",")):
        matched = _resolve_cuda_entry(entry, devices)
        if matched is None:
            break
        visible.append(matched)
    return visible


def _resolve_cuda_entry(entry: str, devices: list[tuple[str, int]]) -> tuple[str, int] | None:
    """The device an entry of ``CUDA_VISIBLE_DEVICES`` names, ``None`` if it names none."""
    if entry.isdigit():
        index = int(entry)
        return devices[index] if index < len(devices) else None
    # UUIDs may be abbreviated to any unique prefix.
    if entry.startswith(("GPU-", "MIG-")):
        matches = [device for device in devices if device[0].startswith(entry)]
        return matches[0] if len(matches) == 1 else None
    return None
