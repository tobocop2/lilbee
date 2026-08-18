"""gguf-parser-backed, UMA-aware memory estimation for one llama-server instance.

See docs/architecture.md (VRAM estimation).
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
from lilbee.providers.fleet.proc import run_bounded

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
_FLAG_OVERRIDE_TENSOR = "--override-tensor"
_BUFFER_TYPE_CPU = "CPU"

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
# Bounded wait for a timed-out parser to die before abandoning it (matches the
# device probe): a parser wedged in driver I/O must not hang the warm-up thread.
_PARSE_KILL_WAIT_S = 5.0
# Sized for a whole plan on a wide box, not for one sweep. The ratio ladder and
# the context bisection key separately (the key carries ctx), and slot fitting
# adds more, so an eight-GPU chat split touches on the order of a hundred keys.
# Too small and the winning candidate's keys are evicted before the launch reads
# them back, which spawns gguf-parser again to recompute what was just measured.
_CACHE_SIZE = 256

# Mirrors vLLM's gpu_memory_utilization default: never charge a GPU past 90% of
# its free VRAM, leaving headroom for allocator fragmentation and driver overhead.
# The default for cfg.usable_vram_fraction, which is what callers should read.
USABLE_VRAM_FRACTION = 0.9


def usable_vram_fraction() -> float:
    """Share of a card placement may charge.

    Configurable because it decides admission rather than merely tuning it: at
    the default, a host whose chat model lands just over the line is refused chat
    with no way for its owner to say the card has the room.
    """
    from lilbee.core.config import cfg

    return cfg.usable_vram_fraction


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
    kv_cache_type_v: KvCacheType | None = None,
    mmproj_path: Path | None = None,
    tensor_split: tuple[int, ...] = (),
    batch_size: int | None = None,
    expert_offload: tuple[str, ...] = (),
) -> GgufVramEstimate:
    """gguf-parser's UMA-aware footprint for one llama-server instance.

    Pass *tensor_split* (the per-device proportions a multi-GPU instance launches
    with) so gguf-parser reports the real per-device breakdown; without it the
    estimate is single-device and the per-GPU peak that actually OOMs is invisible.
    Pass *batch_size* when the launch raises ``--batch-size``/``--ubatch-size``
    (pooled embed/rerank), so the compute-buffer estimate matches the launch.

    With an *mmproj_path* the discrete-GPU number is corrected: gguf-parser's
    nonuma merge overcharges a multimodal projector by roughly 10 GiB of compute
    buffer (v0.24.x and current main), while its unified-memory accounting stays
    accurate, so the projector is re-charged from that side
    (:func:`_corrected_projector_estimate`).
    """

    def run(mmproj: Path | None) -> GgufVramEstimate:
        return _cached_footprint(
            engine_build_identity(),
            str(model_path),
            model_path.stat().st_mtime_ns,
            ctx,
            slots,
            gpu_layers,
            flash_attn,
            kv_cache_type.value,
            (kv_cache_type_v or kv_cache_type).value,
            str(mmproj) if mmproj is not None else None,
            mmproj.stat().st_mtime_ns if mmproj is not None else 0,
            tensor_split,
            batch_size,
            expert_offload,
        )

    if mmproj_path is None:
        return run(None)
    with_projector = run(mmproj_path)
    return _corrected_projector_estimate(run(None), with_projector, mmproj_path.stat().st_size)


def _corrected_projector_estimate(
    base: GgufVramEstimate, with_projector: GgufVramEstimate, mmproj_bytes: int
) -> GgufVramEstimate:
    """Charge the projector at its unified-memory delta, floored at its weights.

    The floor covers a mmap-shared projector whose uma delta hides weights that
    still occupy VRAM once offloaded. The charge lands on the first device:
    llama.cpp loads the projector on the main GPU, not across a split.
    """
    projector = max(with_projector.unified_bytes - base.unified_bytes, mmproj_bytes)
    per_device_vram = tuple(
        vram + (projector if i == 0 else 0) for i, vram in enumerate(base.per_device_vram)
    )
    return GgufVramEstimate(
        vram_bytes=base.vram_bytes + projector,
        ram_bytes=with_projector.ram_bytes,
        unified_bytes=with_projector.unified_bytes,
        per_device_vram=per_device_vram,
        per_device_unified=with_projector.per_device_unified,
    )


def engine_build_identity() -> str:
    """Which engine build these numbers describe.

    Part of the memo key. The estimate prices what one particular llama-server
    will allocate, and the key held the model, the sizing and the parser's own
    arguments without a trace of that, so swapping the engine kept the previous
    engine's answers.
    """
    from lilbee.providers.fleet.binary import _engine_build_id

    return _engine_build_id()


@lru_cache(maxsize=_CACHE_SIZE)
def _cached_footprint(
    _engine_id: str,
    path_str: str,
    _mtime_ns: int,
    ctx: int,
    slots: int,
    gpu_layers: int,
    flash_attn: bool,
    kv_cache_type: str,
    kv_cache_type_v: str,
    mmproj: str | None,
    _mmproj_mtime_ns: int,
    tensor_split: tuple[int, ...],
    batch_size: int | None,
    expert_offload: tuple[str, ...],
) -> GgufVramEstimate:
    """Memoised gguf-parser run keyed on engine + path + mtime + sizing.

    The mtime and engine args participate in the cache key only; a re-pulled file
    at the same path invalidates automatically because its mtime changes, and a
    swapped engine invalidates because its build identity does.
    """
    argv = estimator_argv(
        path_str,
        ctx=ctx,
        slots=slots,
        gpu_layers=gpu_layers,
        flash_attn=flash_attn,
        kv_cache_type=kv_cache_type,
        kv_cache_type_v=kv_cache_type_v,
        mmproj=mmproj,
        tensor_split=tensor_split,
        batch_size=batch_size,
        expert_offload=expert_offload,
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
    kv_cache_type_v: str,
    mmproj: str | None,
    tensor_split: tuple[int, ...],
    batch_size: int | None,
    expert_offload: tuple[str, ...] = (),
) -> list[str]:
    """The gguf-parser command line for one instance's sizing parameters.

    ``ctx`` is the per-slot context, as it is for the launch argv, and
    ``--ctx-size`` carries the total across slots. The parser's ``--parallel``
    does not divide the context the way llama-server's does, so the multiply has
    to reach the parser through ``--ctx-size`` or the attention cache is
    under-reserved by the slot count.

    ``--parallel`` is still passed, and it is not inert. A hybrid or recurrent
    model holds a per-sequence recurrent state that scales with the sequence
    count and not with the context, exactly as llama.cpp sizes it, so the parser
    reads the slot count from this flag. Both flags are load-bearing, for
    different halves of the memory.
    """
    argv = [
        str(resolve_gguf_parser()),
        _FLAG_PATH,
        path_str,
        _FLAG_CTX,
        str(ctx * slots),
        _FLAG_PARALLEL,
        str(slots),
        _FLAG_GPU_LAYERS,
        str(gpu_layers),
        _FLAG_CACHE_K,
        kv_cache_type,
        _FLAG_CACHE_V,
        kv_cache_type_v,
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
    if expert_offload:
        # Without these the estimate charges the GPU for experts the launch keeps
        # in system memory, and the planner sizes slots against a footprint that
        # never materializes.
        argv += [
            _FLAG_OVERRIDE_TENSOR,
            ",".join(f"{pattern}={_BUFFER_TYPE_CPU}" for pattern in expert_offload),
        ]
    if mmproj is not None:
        argv += [_FLAG_MMPROJ, mmproj]
    return argv


def _run_parser(argv: list[str], path_str: str) -> str:
    """Run gguf-parser, returning its JSON stdout or a user-facing error."""
    failed = ProviderError(
        f"Could not size the model {path_str!r}: the memory estimator failed to run.",
        provider=_PROVIDER,
        kind=ProviderErrorKind.SERVER,
    )
    try:
        stdout, returncode = run_bounded(
            argv, timeout_s=_PARSE_TIMEOUT_S, kill_wait_s=_PARSE_KILL_WAIT_S, label="gguf-parser"
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise failed from exc
    if returncode != 0:
        raise failed
    return stdout


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
