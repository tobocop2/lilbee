"""Engine-neutral role identifiers shared across the provider stack.

``WorkerRole`` names the four inference roles a local engine serves; the fleet
maps each to one llama-server instance. ``OcrBackend`` names the PDF-OCR paths.
These outlive any particular engine, so they live here rather than inside an
engine-specific module.
"""

from __future__ import annotations

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


MODEL_FIELD_TO_ROLE: dict[str, WorkerRole] = {
    "chat_model": WorkerRole.CHAT,
    "embedding_model": WorkerRole.EMBED,
    "reranker_model": WorkerRole.RERANK,
    "vision_model": WorkerRole.VISION,
}
"""Config model-role field name -> the worker whose server serves it.

A model-role setting change reloads just that role's server (off-thread) rather
than dropping the whole fleet, so unrelated roles keep serving uninterrupted.
"""


OcrBackend = Literal["vision"]
"""PDF-OCR backends routed to the engine. Tesseract runs inline, not on a server."""
