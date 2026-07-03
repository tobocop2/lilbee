"""Shared 0-as-auto replica-count resolution used by planning, provider, and pipeline.

The ``embed_replicas`` / ``vision_replicas`` knobs default to 0, meaning "auto:
one replica per GPU". Resolving that consistently in one place keeps the planning
hot path, the vision OCR gate, and the ingest fan-out from disagreeing on the
effective replica count.
"""

from __future__ import annotations

import functools
import logging

from lilbee.core.config import cfg
from lilbee.providers.roles import ROLE_REGISTRY, WorkerRole

log = logging.getLogger(__name__)

# Auto resolves to at least one replica even when no GPU is enumerated.
_MIN_REPLICAS = 1


def resolve_replica_count(role: WorkerRole, device_count: int) -> int:
    """Requested data-parallel instances for *role* (0 = auto = one per GPU).

    Embed and vision honor their ``*_replicas`` knob; an explicit value wins,
    0 means one replica per GPU (falling to one when GPU-less). Other roles run
    one instance. Capping to residual VRAM happens in placement.
    """

    knob = ROLE_REGISTRY[role].replica_knob
    if knob is None:
        return _MIN_REPLICAS
    return getattr(cfg, knob) or max(_MIN_REPLICAS, device_count)


@functools.cache
def gpu_device_count() -> int:
    """Effective GPU count lilbee will use; fixed for the process lifetime (cached).

    Resolved the same way planning does (binary ``--list-devices`` view), and
    floored at one so auto means "one replica" on a GPU-less host. Returns one
    when the engine binary is absent so callers that size concurrency (e.g.
    ingest) degrade gracefully instead of raising.
    """
    from lilbee.providers.base import ProviderError
    from lilbee.providers.fleet.binary import resolve_llama_server
    from lilbee.providers.fleet.planning import resolve_devices

    try:
        binary = resolve_llama_server()
    except ProviderError:
        log.debug("llama-server not found; treating host as GPU-less for replica sizing")
        return _MIN_REPLICAS
    devices = resolve_devices(binary)
    return max(_MIN_REPLICAS, len(devices))
