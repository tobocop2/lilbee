"""Shared 0-as-auto replica-count resolution used by planning, provider, and pipeline.

The ``embed_replicas`` / ``vision_replicas`` knobs default to 0, meaning "auto:
one replica per GPU". Resolving that consistently in one place keeps the planning
hot path, the vision OCR gate, and the ingest fan-out from disagreeing on the
effective replica count.
"""

from __future__ import annotations

import functools

from lilbee.providers.roles import WorkerRole

# Roles whose ``*_replicas`` knob scales data-parallel instances; others run one.
# Single source of truth for the elastic (embed/vision) roles, reused by the
# fleet provider to mark which placed instances belong to the ingest pool.
REPLICATED_ROLES = (WorkerRole.EMBED, WorkerRole.VISION)
# Auto resolves to at least one replica even when no GPU is enumerated.
_MIN_REPLICAS = 1


def resolve_replica_count(role: WorkerRole, device_count: int) -> int:
    """Requested data-parallel instances for *role* (0 = auto = one per GPU).

    Embed and vision honor their ``*_replicas`` knob; an explicit value wins,
    0 means one replica per GPU (falling to one when GPU-less). Other roles run
    one instance. Capping to residual VRAM happens in placement.
    """
    from lilbee.core.config import cfg

    if role not in REPLICATED_ROLES:
        return _MIN_REPLICAS
    if role is WorkerRole.EMBED:
        return cfg.embed_replicas or max(_MIN_REPLICAS, device_count)
    return cfg.vision_replicas or max(_MIN_REPLICAS, device_count)


@functools.cache
def gpu_device_count() -> int:
    """Effective GPU count lilbee will use; fixed for the process lifetime (cached).

    Resolved the same way planning does (binary ``--list-devices`` view), and
    floored at one so auto means "one replica" on a GPU-less host.
    """
    from lilbee.providers.fleet.binary import resolve_llama_server
    from lilbee.providers.fleet.planning import resolve_devices

    devices = resolve_devices(resolve_llama_server())
    return max(_MIN_REPLICAS, len(devices))
