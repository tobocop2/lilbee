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


# Every role runs on the fleet by mirroring the in-process primitive over HTTP:
# rerank uses rank-pooling embeddings (--pooling rank -> /v1/embeddings, with the
# same query</s></s>candidate pairing as in-process), NOT the template-dependent
# /v1/rerank; vision uses the chat endpoint with an --mmproj projector. This keeps
# the in-process robustness without depending on a model's embedded rerank template.
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
        endpoint_path="/v1/embeddings",
        extra_args=("--embeddings", "--pooling", "rank"),
        server_capable=True,
    ),
    WorkerRole.VISION: RoleServerSpec(
        role=WorkerRole.VISION,
        endpoint_path="/v1/chat/completions",
        extra_args=(),
        server_capable=True,
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
    mmproj: Path | None = None,
    flash_attn: str | None = None,
    cache_type: str | None = None,
    batch_size: int | None = None,
    threads: int | None = None,
) -> list[str]:
    """Assemble the llama-server command line for one instance, minus ``--port``.

    The port is claimed and appended at spawn time (avoiding a batch-allocation
    race). Single-device instances are pinned via the visible-device env in the
    child (so no split flag); multi-device instances split across the placement's
    GPUs by ``tensor_split`` (per-device proportion), so unequal cards split by
    capacity rather than evenly. ``--ctx-size`` is the per-slot context times the
    slot count, since llama-server divides total context across parallel slots.

    The optional flags mirror the in-process loader for the same role+config:
    ``flash_attn`` (``on``/``off``) and ``cache_type`` apply to chat;
    ``batch_size`` sets ``--batch-size``/``--ubatch-size`` for embed/rerank (the
    server caps embeddings at ``n_ubatch``, default 512, so a full-context embed
    needs both raised); ``threads`` matches the vision loader's full-core setting.
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
    if flash_attn is not None:
        argv += ["--flash-attn", flash_attn]
    if cache_type is not None:
        argv += ["--cache-type-k", cache_type, "--cache-type-v", cache_type]
    if batch_size is not None:
        argv += ["--batch-size", str(batch_size), "--ubatch-size", str(batch_size)]
    if threads is not None:
        argv += ["--threads", str(threads), "--threads-batch", str(threads)]
    if mmproj is not None:  # vision: the CLIP/mtmd projector sidecar
        argv += ["--mmproj", str(mmproj)]
    if len(devices) > 1:
        ratio = tensor_split or tuple(1 for _ in devices)
        argv += ["--tensor-split", ",".join(str(r) for r in ratio)]
    argv += list(spec.extra_args)
    return argv
