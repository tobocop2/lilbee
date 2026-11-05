"""Shared 0-as-auto replica-count resolution used by planning, provider, and pipeline.

The ``embed_replicas`` / ``vision_replicas`` knobs default to 0, meaning "auto:
one replica per GPU". Resolving that consistently in one place keeps the planning
hot path, the vision OCR gate, and the ingest fan-out from disagreeing on the
effective replica count.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING

from lilbee.providers.roles import ROLE_REGISTRY, WorkerRole

if TYPE_CHECKING:
    from pathlib import Path

    from lilbee.providers.fleet.engines.base import EngineAdapter

log = logging.getLogger(__name__)

# Auto resolves to at least one replica even when no GPU is enumerated.
_MIN_REPLICAS = 1


def resolve_replica_count(role: WorkerRole, device_count: int) -> int:
    """Requested data-parallel instances for *role* (0 = auto = one per GPU).

    Embed and vision honor their ``*_replicas`` knob; an explicit value wins,
    0 means one replica per GPU (falling to one when GPU-less). Other roles run
    one instance. Capping to residual VRAM happens in placement.
    """
    from lilbee.core.config import cfg

    knob = ROLE_REGISTRY[role].replica_knob
    if knob is None:
        return _MIN_REPLICAS
    return getattr(cfg, knob) or max(_MIN_REPLICAS, device_count)


def gpu_device_count(engine: EngineAdapter) -> int:
    """Effective GPU count *engine* will use, floored at one.

    Counts the devices auto placement packs, so a shared-memory GPU beside a dedicated
    card adds no replica. An engine with no binary, or a binary that cannot load, sizes
    as GPU-less: the paths that start the engine raise on their own. Neither count is
    kept, so a process that outlives the fix counts its cards again.
    """
    from lilbee.providers.base import ProviderError
    from lilbee.providers.fleet.devices import EngineDllMissingError, EngineIllegalInstructionError

    try:
        binary = engine.binary()
    except ProviderError:
        log.debug("engine binary not found; treating host as GPU-less for replica sizing")
        return _MIN_REPLICAS
    try:
        return _probed_gpu_count(engine, binary)
    except (EngineDllMissingError, EngineIllegalInstructionError):
        log.debug("engine cannot load; sizing for a GPU-less host")
        return _MIN_REPLICAS


@functools.cache
def _probed_gpu_count(engine: EngineAdapter, binary: Path) -> int:
    """Count *engine*'s packable devices once per process (cached); a missing DLL raises uncached.

    Resolved the same way planning does (binary ``--list-devices`` view).
    """
    from lilbee.providers.fleet.planning import packable_devices, resolve_devices

    return max(_MIN_REPLICAS, len(packable_devices(resolve_devices(binary, engine))))
