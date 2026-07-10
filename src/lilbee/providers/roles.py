"""Engine-neutral role identifiers shared across the provider stack.

``WorkerRole`` names the four inference roles a local engine serves; the fleet
maps each to one llama-server instance. ``OcrBackend`` names the PDF-OCR paths.
These outlive any particular engine, so they live here rather than inside an
engine-specific module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class WorkerRole(StrEnum):
    """Inference role identifier; addresses one llama-server in the fleet."""

    EMBED = "embed"
    RERANK = "rerank"
    CHAT = "chat"
    VISION = "vision"


class RerankMode(StrEnum):
    """Resolved reranker serving mode for one RERANK server.

    ``CROSS_ENCODER`` serves an encoder GGUF with rank-pooling embeddings;
    ``LLM`` serves a decoder GGUF generatively and scores yes/no logprobs.
    """

    CROSS_ENCODER = "cross_encoder"
    LLM = "llm"


@dataclass(frozen=True)
class RoleInfo:
    """The per-role knowledge the fleet needs to configure one llama-server.

    One row per ``WorkerRole``, so adding a role is a single registry entry and the
    scattered planning/placement/replica tuples all derive from here.
    """

    role: WorkerRole
    config_field: str  # the cfg ``*_model`` field whose value this role serves
    replicated: bool  # runs N data-parallel replicas (embed/vision)
    replica_knob: str | None  # cfg int field scaling replicas, None when not replicated
    offload_all_layers: bool  # loader offloads every layer, ignoring cfg.n_gpu_layers
    flash_attn: bool  # runs with flash attention (chat/vision)
    pooled: bool  # pooled single-slot search role (embed/cross-encoder rerank)
    placement_rank: int  # placement order; the elastic chat model is charged last


ROLE_REGISTRY: dict[WorkerRole, RoleInfo] = {
    WorkerRole.CHAT: RoleInfo(
        role=WorkerRole.CHAT,
        config_field="chat_model",
        replicated=False,
        replica_knob=None,
        offload_all_layers=False,
        flash_attn=True,
        pooled=False,
        placement_rank=2,
    ),
    WorkerRole.EMBED: RoleInfo(
        role=WorkerRole.EMBED,
        config_field="embedding_model",
        replicated=True,
        replica_knob="embed_replicas",
        offload_all_layers=True,
        flash_attn=False,
        pooled=True,
        placement_rank=0,
    ),
    WorkerRole.RERANK: RoleInfo(
        role=WorkerRole.RERANK,
        config_field="reranker_model",
        replicated=False,
        replica_knob=None,
        offload_all_layers=True,
        flash_attn=False,
        pooled=True,
        placement_rank=0,
    ),
    WorkerRole.VISION: RoleInfo(
        role=WorkerRole.VISION,
        config_field="vision_model",
        replicated=True,
        replica_knob="vision_replicas",
        offload_all_layers=True,
        flash_attn=True,
        pooled=False,
        placement_rank=1,
    ),
}
"""Single source of truth for per-role fleet configuration, ordered chat/embed/rerank/vision."""


MODEL_FIELD_TO_ROLE: dict[str, WorkerRole] = {
    info.config_field: role for role, info in ROLE_REGISTRY.items()
}
"""Config model-role field name -> the worker whose server serves it.

A model-role setting change reloads just that role's server (off-thread) rather
than dropping the whole fleet, so unrelated roles keep serving uninterrupted.
"""


MODEL_ROLE_FIELDS: frozenset[str] = frozenset(MODEL_FIELD_TO_ROLE)
"""The cfg ``*_model`` field names, as a set (settings overlay + reload routing)."""


REPLICATED_ROLES: tuple[WorkerRole, ...] = tuple(
    role for role, info in ROLE_REGISTRY.items() if info.replicated
)
"""Roles whose ``*_replicas`` knob scales data-parallel instances; others run one."""


OcrBackend = Literal["vision"]
"""PDF-OCR backends routed to the engine. Tesseract runs inline, not on a server."""


def configured_model_message(role: WorkerRole, configured: str, requested: str) -> str:
    """User-facing rejection for a per-call model that differs from the configured one."""
    return (
        f"This engine serves the configured {role} model ({configured}). "
        f"To use {requested!r}, set it as the {role} model in lilbee settings "
        f"(TUI /settings), then retry; the engine reloads automatically."
    )
