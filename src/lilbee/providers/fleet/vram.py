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
from lilbee.providers.fleet.binary import resolve_gguf_parser

# gguf-parser CLI flags.
_FLAG_PATH = "--path"
_FLAG_CTX = "--ctx-size"
_FLAG_PARALLEL = "--parallel"
_FLAG_GPU_LAYERS = "--gpu-layers"
_FLAG_CACHE_K = "--cache-type-k"
_FLAG_CACHE_V = "--cache-type-v"
_FLAG_MMPROJ = "--mmproj-path"
_FLAG_FLASH = "--flash-attention"
_FLAG_NO_FLASH = "--no-flash-attention"
_FLAG_JSON = "--json"

# gguf-parser JSON keys (estimate.items[0] carries the per-instance footprint).
_KEY_ESTIMATE = "estimate"
_KEY_ITEMS = "items"
_KEY_RAM = "ram"
_KEY_VRAMS = "vrams"
_KEY_UMA = "uma"
_KEY_NONUMA = "nonuma"

_PROVIDER = "llama-server"
_PARSE_TIMEOUT_S = 60
_CACHE_SIZE = 64


@dataclass(frozen=True)
class GgufVramEstimate:
    """One instance's footprint from gguf-parser, for both memory models."""

    vram_bytes: int
    """Discrete-GPU model: bytes resident in device VRAM (summed over devices)."""
    ram_bytes: int
    """Discrete-GPU model: host RAM bytes (mmap pages, compute buffers)."""
    unified_bytes: int
    """Unified-memory model: total resident footprint (RAM + would-be VRAM)."""

    def footprint(self, *, unified: bool) -> int:
        """Bytes to charge against the budget for this host's memory model."""
        return self.unified_bytes if unified else self.vram_bytes


def estimate_instance_footprint(
    model_path: Path,
    *,
    ctx: int,
    slots: int,
    gpu_layers: int,
    flash_attn: bool,
    kv_cache_type: KvCacheType,
    mmproj_path: Path | None = None,
) -> GgufVramEstimate:
    """gguf-parser's UMA-aware footprint for one llama-server instance."""
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
) -> GgufVramEstimate:
    """Memoised gguf-parser run keyed on path + mtime + sizing.

    The mtime args participate in the cache key only; a re-pulled file at the same
    path invalidates automatically because its mtime changes.
    """
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
    if mmproj is not None:
        argv += [_FLAG_MMPROJ, mmproj]
    return _parse_estimate(_run_parser(argv, path_str), path_str)


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
    """Parse gguf-parser JSON into a UMA-aware footprint."""
    try:
        item = json.loads(stdout)[_KEY_ESTIMATE][_KEY_ITEMS][0]
        ram = item[_KEY_RAM]
        vrams = item[_KEY_VRAMS]
        vram_nonuma = sum(int(v[_KEY_NONUMA]) for v in vrams)
        vram_uma = sum(int(v[_KEY_UMA]) for v in vrams)
        return GgufVramEstimate(
            vram_bytes=vram_nonuma,
            ram_bytes=int(ram[_KEY_NONUMA]),
            unified_bytes=int(ram[_KEY_UMA]) + vram_uma,
        )
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ProviderError(
            f"Could not size the model {path_str!r}: estimator returned no usable result.",
            provider=_PROVIDER,
            kind=ProviderErrorKind.SERVER,
        ) from exc
