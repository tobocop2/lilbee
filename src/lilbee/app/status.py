"""Status snapshot of the local knowledge base."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from lilbee.app.services import get_services
from lilbee.core.config import cfg
from lilbee.core.system import LOCAL_ROOT_DIRNAME, default_data_dir

LILBEE_LABEL_MAX_LEN = 40
"""Soft cap on the compact-label width to keep the status pill inside the bar."""


def _project_root() -> Path:
    """Walk past a trailing ``.lilbee`` marker to the project dir that owns it."""
    root = cfg.data_root
    if root.name == LOCAL_ROOT_DIRNAME:
        return root.parent
    return root


def lilbee_label() -> str:
    """Status-bar pill text for the active lilbee.

    User-set ``lilbee_name`` always wins. Global data dir always renders
    "global". Otherwise the project path: compact (``~``-substituted,
    truncated from the left to ``LILBEE_LABEL_MAX_LEN``) by default, or
    full absolute when ``show_lilbee_path`` is on (toggle: Ctrl+L).
    """
    if cfg.lilbee_name:
        return cfg.lilbee_name
    if cfg.data_root.expanduser().resolve() == default_data_dir().resolve():
        return "global"
    full = str(_project_root().expanduser().resolve())
    if cfg.show_lilbee_path:
        return full
    home = str(Path.home())
    compact = f"~{full[len(home) :]}" if full.startswith(home) else full
    if len(compact) <= LILBEE_LABEL_MAX_LEN:
        return compact
    leaf = _project_root().name or compact
    head_budget = LILBEE_LABEL_MAX_LEN - len(leaf) - 2
    if head_budget <= 0:
        return f"…/{leaf}"
    return f"…{compact[-(LILBEE_LABEL_MAX_LEN - 1) :]}"


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
