"""Per-role llama-server specs and the argv builder for the fleet.

A data table (not per-role functions) keyed by ``WorkerRole``: each spec carries
the OpenAI endpoint path, the role-specific server flags, and whether the role is
viable on a server today. ``build_server_argv`` reads a spec plus placement data
to assemble one llama-server command line.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lilbee.providers.worker.transport import WorkerRole

_HOST = "127.0.0.1"


@dataclass(frozen=True)
class RoleServerSpec:
    """How one role maps onto a llama-server instance."""

    role: WorkerRole
    endpoint_path: str
    extra_args: tuple[str, ...]
    server_capable: bool


# chat/embed have stable OpenAI surfaces; rerank (/v1/rerank) and vision
# (mtmd-over-HTTP) are experimental in llama-server, so they stay in-process
# until validated. server_capable gates the planner's fallback.
ROLE_SPECS: dict[WorkerRole, RoleServerSpec] = {
    WorkerRole.CHAT: RoleServerSpec(
        role=WorkerRole.CHAT,
        endpoint_path="/v1/chat/completions",
        extra_args=(),
        server_capable=True,
    ),
    WorkerRole.EMBED: RoleServerSpec(
        role=WorkerRole.EMBED,
        endpoint_path="/v1/embeddings",
        extra_args=("--embeddings",),
        server_capable=True,
    ),
    WorkerRole.RERANK: RoleServerSpec(
        role=WorkerRole.RERANK,
        endpoint_path="/v1/rerank",
        extra_args=("--reranking", "--pooling", "rank"),
        server_capable=False,
    ),
    WorkerRole.VISION: RoleServerSpec(
        role=WorkerRole.VISION,
        endpoint_path="/v1/chat/completions",
        extra_args=(),
        server_capable=False,
    ),
}


def build_server_argv(
    *,
    binary: Path,
    spec: RoleServerSpec,
    model_path: Path,
    devices: tuple[int, ...],
    n_gpu_layers: int,
    slots: int,
    ctx_per_slot: int,
    tensor_split: tuple[int, ...] = (),
) -> list[str]:
    """Assemble the llama-server command line for one instance, minus ``--port``.

    The port is claimed and appended at spawn time (avoiding a batch-allocation
    race). Single-device instances are pinned via the visible-device env in the
    child (so no split flag); multi-device instances split across the placement's
    GPUs by ``tensor_split`` (per-device proportion), so unequal cards split by
    capacity rather than evenly. ``--ctx-size`` is the per-slot context times the
    slot count, since llama-server divides total context across parallel slots.
    """
    argv = [
        str(binary),
        "--model",
        str(model_path),
        "--host",
        _HOST,
        "--n-gpu-layers",
        str(n_gpu_layers),
        "--parallel",
        str(slots),
        "--cont-batching",
        "--ctx-size",
        str(ctx_per_slot * slots),
    ]
    if len(devices) > 1:
        ratio = tensor_split or tuple(1 for _ in devices)
        argv += ["--tensor-split", ",".join(str(r) for r in ratio)]
    argv += list(spec.extra_args)
    return argv
