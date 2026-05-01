"""Status snapshot of the local knowledge base."""

from __future__ import annotations

from pydantic import BaseModel

from lilbee.core.config import cfg
from lilbee.core.services import get_services


class StatusConfig(BaseModel):
    """Configuration section of a status response.

    Exposes all four role-bound model fields (chat, embedding, vision,
    reranker) so the TUI status screen and plugin callers can show
    what's active per role.
    """

    documents_dir: str
    data_dir: str
    chat_model: str
    embedding_model: str
    vision_model: str = ""
    reranker_model: str = ""
    enable_ocr: bool | None = None


class SourceInfo(BaseModel):
    """A single indexed source in a status response."""

    filename: str
    file_hash: str
    chunk_count: int
    ingested_at: str


class StatusResult(BaseModel):
    """Full status response for the knowledge base."""

    command: str = "status"
    config: StatusConfig
    sources: list[SourceInfo]
    total_chunks: int


def gather_status() -> StatusResult:
    """Collect status data as a typed model (shared by human + JSON output)."""
    sources = get_services().store.get_sources()
    sorted_sources = sorted(sources, key=lambda x: x["filename"])
    total_chunks = sum(s["chunk_count"] for s in sources)
    return StatusResult(
        config=StatusConfig(
            documents_dir=str(cfg.documents_dir),
            data_dir=str(cfg.data_dir),
            chat_model=cfg.chat_model,
            embedding_model=cfg.embedding_model,
            vision_model=cfg.vision_model,
            reranker_model=cfg.reranker_model,
            enable_ocr=cfg.enable_ocr,
        ),
        sources=[
            SourceInfo(
                filename=s["filename"],
                file_hash=s["file_hash"][:12],
                chunk_count=s["chunk_count"],
                ingested_at=s["ingested_at"][:19],
            )
            for s in sorted_sources
        ],
        total_chunks=total_chunks,
    )
