"""Shared 0-as-auto replica-count resolution used by planning, provider, and pipeline.

The ``embed_replicas`` / ``vision_replicas`` knobs default to 0, meaning "auto:
one replica per GPU". Resolving that consistently in one place keeps the planning
hot path, the vision OCR gate, and the ingest fan-out from disagreeing on the
effective replica count.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from lilbee.providers.roles import ROLE_REGISTRY, WorkerRole

if TYPE_CHECKING:
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


def sizing_device_count(engine: EngineAdapter) -> int:
    """GPU count for sizing, floored at one, and one when the probe fails."""
    from lilbee.providers.base import ProviderError

    try:
        return gpu_device_count(engine)
    except ProviderError as exc:
        log.debug("device probe failed (%s); sizing for a GPU-less host", exc)
        return _MIN_REPLICAS


def gpu_device_count(engine: EngineAdapter) -> int:
    """Effective GPU count *engine* will use, floored at one.

    Counts the devices auto placement packs, so a shared-memory GPU beside a dedicated
    card adds no replica. An engine with no binary, or a binary that cannot load, sizes
    as GPU-less: the paths that start the engine raise on their own. The probe reads the
    device read cache, so a failed probe is remembered for its failure window.
    """
    from lilbee.providers.base import ProviderError
    from lilbee.providers.fleet.devices import EngineDllMissingError, EngineIllegalInstructionError
    from lilbee.providers.fleet.planning import packable_devices, resolve_devices

    try:
        binary = engine.binary()
    except ProviderError:
        log.debug("engine binary not found; treating host as GPU-less for replica sizing")
        return _MIN_REPLICAS
    try:
        return max(_MIN_REPLICAS, len(packable_devices(resolve_devices(binary, engine))))
    except (EngineDllMissingError, EngineIllegalInstructionError):
        log.debug("engine cannot load; sizing for a GPU-less host")
        return _MIN_REPLICAS
