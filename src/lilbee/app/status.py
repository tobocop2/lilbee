"""Status snapshot of the local knowledge base."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel

from lilbee.app.services import get_services
from lilbee.core.config import cfg
from lilbee.core.system import LOCAL_ROOT_DIRNAME, default_data_dir
from lilbee.data.ingest.skip_marker import describe_skips, load_skip_markers
from lilbee.data.types import SkippedSource

LILBEE_LABEL_MAX_LEN = 40
"""Hard cap on the compact-label width. The leaf gets its own internal
ellipsis when even the leaf alone would breach the cap."""

STATUS_SKIPPED_LIMIT = 50
"""Cap on held-out files in one status response; ``skipped_total`` carries the real count."""

_ELLIPSIS = "…"


def _project_root() -> Path:
    """Walk past a trailing ``.lilbee`` marker to the project dir that owns it."""
    root = cfg.data_root
    if root.name == LOCAL_ROOT_DIRNAME:
        return root.parent
    return root


def _truncate_leaf(leaf: str, max_len: int) -> str:
    """Shrink an over-long leaf to fit a budget, with an internal ellipsis."""
    if len(leaf) <= max_len:
        return leaf
    if max_len <= 1:
        return _ELLIPSIS
    keep = max_len - 1
    head = keep // 2
    tail = keep - head
    return f"{leaf[:head]}{_ELLIPSIS}{leaf[-tail:] if tail else ''}"


def _compact_path(full: str) -> str:
    """Render *full* with ``~`` substituted for ``$HOME`` when it leads."""
    home = str(Path.home())
    if full == home:
        return "~"
    home_prefix = f"{home}{os.sep}"
    return f"~{os.sep}{full[len(home_prefix) :]}" if full.startswith(home_prefix) else full


def lilbee_label() -> str:
    """Status-bar pill text for the active lilbee.

    Precedence: ``lilbee_name`` override > ``"global"`` (when data_root
    is the platform default) > project path. ``show_lilbee_path``
    (toggled by F4) returns the full absolute path instead of the
    compact / "global" form.
    """
    if cfg.lilbee_name:
        return cfg.lilbee_name
    is_global = cfg.data_root.expanduser().resolve() == default_data_dir().resolve()
    if cfg.show_lilbee_path:
        return str(default_data_dir() if is_global else _project_root().expanduser().resolve())
    if is_global:
        return "global"
    full = str(_project_root().expanduser().resolve())
    compact = _compact_path(full)
    if len(compact) <= LILBEE_LABEL_MAX_LEN:
        return compact
    leaf = _project_root().name or compact
    leaf_budget = LILBEE_LABEL_MAX_LEN - 1 - len(os.sep)
    return f"{_ELLIPSIS}{os.sep}{_truncate_leaf(leaf, leaf_budget)}"


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


class EntityStatus(BaseModel):
    """Entity-extraction section of a status response (present when enabled)."""

    types: list[str]
    rows: int


class StatusResult(BaseModel):
    """Full status response for the knowledge base."""

    command: str = "status"
    config: StatusConfig
    sources: list[SourceInfo]
    document_count: int
    total_chunks: int
    entities: EntityStatus | None = None
    skipped: list[SkippedSource] = []
    """Files a skip marker holds out of the index, capped at ``STATUS_SKIPPED_LIMIT``."""
    skipped_total: int = 0


def gather_status() -> StatusResult:
    """Collect status data as a typed model (shared by human + JSON output)."""
    sources = get_services().store.get_sources()
    sorted_sources = sorted(sources, key=lambda x: x["filename"])
    total_chunks = sum(s["chunk_count"] for s in sources)
    skipped, skipped_total = held_out_sources()
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
        document_count=len(sources),
        total_chunks=total_chunks,
        entities=entity_status(),
        skipped=skipped,
        skipped_total=skipped_total,
    )


def held_out_sources() -> tuple[list[SkippedSource], int]:
    """Held-out files with reasons, capped at ``STATUS_SKIPPED_LIMIT``, and the real count."""
    held_out = sorted(load_skip_markers(cfg.data_root))
    return describe_skips(cfg.data_root, held_out[:STATUS_SKIPPED_LIMIT]), len(held_out)


def entity_status() -> EntityStatus | None:
    """Entity types + extracted row count; None while the feature is off."""
    if not cfg.entity_extraction:
        return None
    from lilbee.core.config import ENTITIES_TABLE
    from lilbee.retrieval.entities import load_schema

    store = get_services().store
    schema = load_schema(store)
    table = store.open_table(ENTITIES_TABLE)
    rows = table.count_rows() if table is not None else 0
    return EntityStatus(types=[t.name for t in schema.types] if schema else [], rows=rows)
