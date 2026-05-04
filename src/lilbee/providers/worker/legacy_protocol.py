"""Per-call worker subprocess protocol used by ``WorkerManager``.

Reached for vision OCR when ``cfg.worker_pool_enabled = False``.
The pool path uses :mod:`lilbee.providers.worker.transport_pipe`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lilbee.core.config import cfg


@dataclass(frozen=True)
class EmbedRequest:
    """Request to embed a list of texts."""

    texts: list[str]
    model: str
    request_id: int = 0


@dataclass(frozen=True)
class EmbedResponse:
    """Response with embedding vectors or an error."""

    vectors: list[list[float]] = field(default_factory=list)
    error: str = ""
    request_id: int = 0


@dataclass(frozen=True)
class VisionRequest:
    """Request to run vision OCR on a PNG image."""

    png_bytes: bytes
    model: str
    prompt: str = ""
    request_id: int = 0


@dataclass(frozen=True)
class VisionResponse:
    """Response with extracted text or an error."""

    text: str = ""
    error: str = ""
    request_id: int = 0


@dataclass(frozen=True)
class LoadModelRequest:
    """Request to (re)load a specific model in the worker."""

    model: str
    model_type: str = "embed"  # "embed" or "vision"


@dataclass(frozen=True)
class ShutdownRequest:
    """Sentinel to tell the worker to exit."""


@dataclass(frozen=True)
class ConfigSnapshot:
    """Minimal config fields needed by the child process."""

    models_dir: str
    embedding_model: str
    embedding_dim: int
    num_ctx: int | None
    gpu_memory_fraction: float
    vision_model: str


WorkerRequest = EmbedRequest | VisionRequest | LoadModelRequest | ShutdownRequest
WorkerResponse = EmbedResponse | VisionResponse


def config_snapshot_from_cfg() -> ConfigSnapshot:
    """Build a ConfigSnapshot from the current cfg singleton."""
    return ConfigSnapshot(
        models_dir=str(cfg.models_dir),
        embedding_model=cfg.embedding_model,
        embedding_dim=cfg.embedding_dim,
        num_ctx=cfg.num_ctx,
        gpu_memory_fraction=cfg.gpu_memory_fraction,
        vision_model=cfg.vision_model,
    )
