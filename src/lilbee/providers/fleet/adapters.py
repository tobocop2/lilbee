"""Per-role llama-server specs and the argv builder for the fleet.

A data table (not per-role functions) keyed by ``WorkerRole``: each spec carries
the OpenAI endpoint path, the role-specific server flags, and whether the role is
viable on a server today. ``build_server_argv`` reads a spec plus placement data
to assemble one llama-server command line.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

from lilbee.core.config.enums import RerankerType
from lilbee.providers.roles import RerankMode, WorkerRole

_HOST = "127.0.0.1"
# llama-server batch flags; gguf-parser accepts the same names, so vram.py shares these.
FLAG_BATCH_SIZE = "--batch-size"
FLAG_UBATCH_SIZE = "--ubatch-size"


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
        # --jinja renders the model's own chat template and parses native
        # tool-call syntax into structured message.tool_calls.
        # --reasoning-format none keeps <think>...</think> inline in content;
        # without it, recent llama-server extracts reasoning into a separate
        # reasoning_content field and lilbee's <think>-based parser sees none.
        extra_args=("--jinja", "--reasoning-format", "none"),
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

# Decoder-only architectures. Served generatively as rerankers (Qwen3-Reranker,
# mxbai-rerank-v2), and their embeddings use last-token (EOS) pooling, not the
# encoder default of mean/cls.
_DECODER_ARCHS: frozenset[str] = frozenset(
    {"qwen2", "qwen3", "llama", "mistral", "gemma", "gemma2", "gemma3", "phi3"}
)

LLM_RERANK_SPEC = RoleServerSpec(
    role=WorkerRole.RERANK,
    endpoint_path="/v1/chat/completions",
    extra_args=("--jinja",),
    server_capable=True,
)

_RERANK_MODE_SPECS: dict[RerankMode, RoleServerSpec] = {
    RerankMode.CROSS_ENCODER: ROLE_SPECS[WorkerRole.RERANK],
    RerankMode.LLM: LLM_RERANK_SPEC,
}

# An LLM reranker scores one chat request per candidate; this is both the client's
# per-rerank request fan-out and the server's --parallel slot ceiling, so the
# server can decode concurrently instead of serializing the fan-out.
LLM_RERANK_CONCURRENCY = 8


def resolve_rerank_mode(reranker_type: RerankerType, arch: str | None) -> RerankMode:
    """Pick the reranker serving mode from the config setting and GGUF arch.

    ``auto`` serves a known decoder arch generatively; encoder/unknown archs stay
    cross-encoder. Explicit settings override the arch.
    """
    if reranker_type is RerankerType.LLM:
        return RerankMode.LLM
    if reranker_type is RerankerType.CROSS_ENCODER:
        return RerankMode.CROSS_ENCODER
    if arch in _DECODER_ARCHS:
        return RerankMode.LLM
    return RerankMode.CROSS_ENCODER


def rerank_spec(mode: RerankMode) -> RoleServerSpec:
    """The server spec for a RERANK launch given its resolved mode."""
    return _RERANK_MODE_SPECS[mode]


class PoolingType(StrEnum):
    """A llama-server ``--pooling`` value (also the GGUF pooling_type enum names)."""

    NONE = "none"
    MEAN = "mean"
    CLS = "cls"
    LAST = "last"
    RANK = "rank"


# GGUF <arch>.pooling_type integer (as read_gguf_metadata returns it, a string) ->
# the --pooling value. NONE (0) is omitted: a 0 on an embedder is the unset default,
# so it falls through to the arch-based choice rather than per-token (non-)pooling.
_GGUF_POOLING: dict[str, PoolingType] = {
    "1": PoolingType.MEAN,
    "2": PoolingType.CLS,
    "3": PoolingType.LAST,
    "4": PoolingType.RANK,
}


def embed_spec(meta: dict[str, str] | None) -> RoleServerSpec:
    """The EMBED server spec with the model's pooling resolved for llama-server."""
    # The GGUF's declared pooling wins; else a decoder-only embedder pools on its
    # last/EOS token (llama-server would otherwise default to mean, which is wrong).
    arch = meta.get("architecture") if meta else None
    pooling_type = meta.get("pooling_type") if meta else None
    pooling = _GGUF_POOLING.get(pooling_type or "")
    if pooling is None and arch in _DECODER_ARCHS:
        pooling = PoolingType.LAST
    base = ROLE_SPECS[WorkerRole.EMBED]
    if pooling is None:
        return base
    return replace(base, extra_args=(*base.extra_args, "--pooling", pooling.value))


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

    ``--ctx-size`` is the per-slot context times the slot count, since
    llama-server divides total context across parallel slots.
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
        argv += [FLAG_BATCH_SIZE, str(batch_size), FLAG_UBATCH_SIZE, str(batch_size)]
    if threads is not None:
        argv += ["--threads", str(threads), "--threads-batch", str(threads)]
    if mmproj is not None:  # vision: the CLIP/mtmd projector sidecar
        argv += ["--mmproj", str(mmproj)]
    if len(devices) > 1:
        ratio = tensor_split or tuple(1 for _ in devices)
        argv += ["--tensor-split", ",".join(str(r) for r in ratio)]
    argv += list(spec.extra_args)
    return argv
