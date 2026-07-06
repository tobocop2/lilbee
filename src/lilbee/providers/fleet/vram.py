"""gguf-parser-backed, UMA-aware memory estimation for one llama-server instance.

See docs/architecture.md (VRAM estimation) for why this replaced the hand-rolled
weights + KV-cache math.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from lilbee.core.config.enums import KvCacheType
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet.adapters import FLAG_BATCH_SIZE, FLAG_UBATCH_SIZE
from lilbee.providers.fleet.binary import resolve_gguf_parser

# gguf-parser CLI flags (the batch flags are shared with the llama-server argv builder).
_FLAG_PATH = "--path"
_FLAG_CTX = "--ctx-size"
_FLAG_PARALLEL = "--parallel"
_FLAG_GPU_LAYERS = "--gpu-layers"
_FLAG_CACHE_K = "--cache-type-k"
_FLAG_CACHE_V = "--cache-type-v"
_FLAG_MMPROJ = "--mmproj-path"
_FLAG_FLASH = "--flash-attention"
_FLAG_NO_FLASH = "--no-flash-attention"
_FLAG_TENSOR_SPLIT = "--tensor-split"
_FLAG_SPLIT_MODE = "--split-mode"
_SPLIT_MODE_LAYER = "layer"
_FLAG_JSON = "--json"

# gguf-parser JSON keys: the per-instance footprint lives under estimate.items
# (v0.24.x) or estimate.memory (upstream's post-v0.24.1 rename of the same list).
_KEY_ESTIMATE = "estimate"
_KEY_ITEMS = "items"
_KEY_MEMORY = "memory"
_KEY_RAM = "ram"
_KEY_VRAMS = "vrams"
_KEY_UMA = "uma"
_KEY_NONUMA = "nonuma"

_PROVIDER = "llama-server"
_PARSE_TIMEOUT_S = 60
_CACHE_SIZE = 64

# Mirrors vLLM's gpu_memory_utilization default: never charge a GPU past 90% of
# its free VRAM, leaving headroom for allocator fragmentation and driver overhead.
USABLE_VRAM_FRACTION = 0.9


@dataclass(frozen=True)
class GgufVramEstimate:
    """One instance's footprint from gguf-parser, for both memory models.

    ``per_device_*`` carry the per-GPU breakdown gguf-parser returns once a
    ``tensor_split`` is supplied. A tensor-split instance OOMs on its busiest card,
    so the planner fits/charges ``peak_footprint`` (the max device), never the sum.
    """

    vram_bytes: int
    """Discrete-GPU model: bytes resident in device VRAM (summed over devices)."""
    ram_bytes: int
    """Discrete-GPU model: host RAM bytes (mmap pages, compute buffers)."""
    unified_bytes: int
    """Unified-memory model: total resident footprint (RAM + would-be VRAM)."""
    per_device_vram: tuple[int, ...] = ()
    """Discrete-GPU VRAM per device (gguf-parser ``vrams[].nonuma``)."""
    per_device_unified: tuple[int, ...] = ()
    """Unified-memory footprint per device (``vrams[].uma``)."""

    def footprint(self, *, unified: bool) -> int:
        """Total bytes to charge against a shared budget for this memory model."""
        return self.unified_bytes if unified else self.vram_bytes

    def peak_footprint(self, *, unified: bool) -> int:
        """The busiest single device's bytes -- what must fit on one GPU.

        Falls back to the total when there is no per-device breakdown (a
        single-device estimate, or an estimate run without a tensor split).
        """
        per_device = self.per_device_unified if unified else self.per_device_vram
        return max(per_device) if per_device else self.footprint(unified=unified)


def estimate_instance_footprint(
    model_path: Path,
    *,
    ctx: int,
    slots: int,
    gpu_layers: int,
    flash_attn: bool,
    kv_cache_type: KvCacheType,
    mmproj_path: Path | None = None,
    tensor_split: tuple[int, ...] = (),
    batch_size: int | None = None,
) -> GgufVramEstimate:
    """gguf-parser's UMA-aware footprint for one llama-server instance.

    Pass *tensor_split* (the per-device proportions a multi-GPU instance launches
    with) so gguf-parser reports the real per-device breakdown; without it the
    estimate is single-device and the per-GPU peak that actually OOMs is invisible.
    Pass *batch_size* when the launch raises ``--batch-size``/``--ubatch-size``
    (pooled embed/rerank), so the compute-buffer estimate matches the launch.
    """
    mmproj = str(mmproj_path) if mmproj_path is not None else None
    mmproj_mtime = mmproj_path.stat().st_mtime_ns if mmproj_path is not None else 0
    return _cached_footprint(
        str(model_path),
        model_path.stat().st_mtime_ns,
        ctx,
        slots,
        gpu_layers,
        flash_attn,
        kv_cache_type.value,
        mmproj,
        mmproj_mtime,
        tensor_split,
        batch_size,
    )


@lru_cache(maxsize=_CACHE_SIZE)
def _cached_footprint(
    path_str: str,
    _mtime_ns: int,
    ctx: int,
    slots: int,
    gpu_layers: int,
    flash_attn: bool,
    kv_cache_type: str,
    mmproj: str | None,
    _mmproj_mtime_ns: int,
    tensor_split: tuple[int, ...],
    batch_size: int | None,
) -> GgufVramEstimate:
    """Memoised gguf-parser run keyed on path + mtime + sizing.

    The mtime args participate in the cache key only; a re-pulled file at the same
    path invalidates automatically because its mtime changes.
    """
    argv = estimator_argv(
        path_str,
        ctx=ctx,
        slots=slots,
        gpu_layers=gpu_layers,
        flash_attn=flash_attn,
        kv_cache_type=kv_cache_type,
        mmproj=mmproj,
        tensor_split=tensor_split,
        batch_size=batch_size,
    )
    return _parse_estimate(_run_parser(argv, path_str), path_str)


def estimator_argv(
    path_str: str,
    *,
    ctx: int,
    slots: int,
    gpu_layers: int,
    flash_attn: bool,
    kv_cache_type: str,
    mmproj: str | None,
    tensor_split: tuple[int, ...],
    batch_size: int | None,
) -> list[str]:
    """The gguf-parser command line for one instance's sizing parameters."""
    argv = [
        str(resolve_gguf_parser()),
        _FLAG_PATH,
        path_str,
        _FLAG_CTX,
        str(ctx),
        _FLAG_PARALLEL,
        str(slots),
        _FLAG_GPU_LAYERS,
        str(gpu_layers),
        _FLAG_CACHE_K,
        kv_cache_type,
        _FLAG_CACHE_V,
        kv_cache_type,
        _FLAG_FLASH if flash_attn else _FLAG_NO_FLASH,
        _FLAG_JSON,
    ]
    if batch_size is not None:
        # Pooled embed/rerank launch with --batch-size/--ubatch-size raised to the
        # context; the default ubatch (512) would under-estimate their compute buffer.
        argv += [FLAG_BATCH_SIZE, str(batch_size), FLAG_UBATCH_SIZE, str(batch_size)]
    if tensor_split:
        # The split proportions are gguf-parser's only signal for the device count,
        # so it returns one ``vrams[]`` entry per GPU instead of a single total.
        argv += [
            _FLAG_TENSOR_SPLIT,
            ",".join(str(p) for p in tensor_split),
            _FLAG_SPLIT_MODE,
            _SPLIT_MODE_LAYER,
        ]
    if mmproj is not None:
        argv += [_FLAG_MMPROJ, mmproj]
    return argv


def _run_parser(argv: list[str], path_str: str) -> str:
    """Run gguf-parser, returning its JSON stdout or a user-facing error."""
    try:
        proc = subprocess.run(  # noqa: S603 - argv[0] is the resolved gguf-parser
            argv, capture_output=True, text=True, timeout=_PARSE_TIMEOUT_S, check=True
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProviderError(
            f"Could not size the model {path_str!r}: the memory estimator failed to run.",
            provider=_PROVIDER,
            kind=ProviderErrorKind.SERVER,
        ) from exc
    return proc.stdout


def _parse_estimate(stdout: str, path_str: str) -> GgufVramEstimate:
    """Parse gguf-parser JSON into a UMA-aware footprint.

    Accepts both estimate payload keys: ``items`` (the pinned v0.24.x releases)
    and ``memory`` (upstream renamed the key after v0.24.1), so an engine built
    past the pin still sizes instead of failing every launch plan.
    """
    try:
        estimate = json.loads(stdout)[_KEY_ESTIMATE]
        item = (estimate.get(_KEY_ITEMS) or estimate[_KEY_MEMORY])[0]
        ram = item[_KEY_RAM]
        vrams = item[_KEY_VRAMS]
        per_device_vram = tuple(int(v[_KEY_NONUMA]) for v in vrams)
        per_device_unified = tuple(int(v[_KEY_UMA]) for v in vrams)
        return GgufVramEstimate(
            vram_bytes=sum(per_device_vram),
            ram_bytes=int(ram[_KEY_NONUMA]),
            unified_bytes=int(ram[_KEY_UMA]) + sum(per_device_unified),
            per_device_vram=per_device_vram,
            per_device_unified=per_device_unified,
        )
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ProviderError(
            f"Could not size the model {path_str!r}: unexpected estimator output "
            f"({type(exc).__name__}: {exc}).",
            provider=_PROVIDER,
            kind=ProviderErrorKind.SERVER,
        ) from exc
