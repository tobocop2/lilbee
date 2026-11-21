"""The request and result types of a memory estimate, shared by the estimators and the engines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from lilbee.core.config.enums import KvCacheType


class MemoryKind(StrEnum):
    """How much an engine's memory figures can be trusted."""

    SIZED = "sized"
    FIXED_RESERVE = "fixed_reserve"


@dataclass(frozen=True)
class MemoryRequest:
    """What the memory estimate needs to know about one instance."""

    model_path: Path
    ctx: int
    slots: int
    gpu_layers: int
    flash_attn: bool
    kv_cache_type: KvCacheType
    mmproj_path: Path | None = None
    tensor_split: tuple[int, ...] = ()
    batch_size: int | None = None
    expert_offload: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoryEstimate:
    """One instance's memory footprint under both memory models.

    A tensor-split instance fails on its busiest device, so the planner fits
    ``peak_footprint`` (the largest device), never the sum.
    """

    kind: MemoryKind
    device_bytes: int
    """Discrete-GPU model: bytes resident in device memory, summed over devices."""
    host_bytes: int
    """Discrete-GPU model: bytes held in host memory."""
    shared_bytes: int
    """Unified-memory model: the whole resident footprint."""
    per_device: tuple[int, ...] = ()
    """Discrete-GPU model: bytes on each device."""
    per_device_shared: tuple[int, ...] = ()
    """Unified-memory model: bytes on each device."""

    def footprint(self, *, unified: bool) -> int:
        """Total bytes to charge against a shared budget for this memory model."""
        return self.shared_bytes if unified else self.device_bytes

    def peak_footprint(self, *, unified: bool) -> int:
        """The busiest single device's bytes, or the total without a per-device breakdown."""
        per_device = self.per_device_shared if unified else self.per_device
        return max(per_device) if per_device else self.footprint(unified=unified)
