"""Knowledge base reset (delete all documents and data)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from pydantic import BaseModel

from lilbee.core.config import cfg
from lilbee.core.security import validate_path_within


class ResetResult(BaseModel):
    """Result of a full knowledge base reset."""

    command: str = "reset"
    deleted_docs: int
    deleted_data: int
    skipped: list[str] = []
    documents_dir: str
    data_dir: str


def _clear_dir(base_dir: Path, skipped: list[str]) -> int:
    """Delete all items in *base_dir*, appending undeletable paths to *skipped*."""
    log = logging.getLogger(__name__)
    deleted = 0
    if not base_dir.exists():
        return deleted
    for item in list(base_dir.iterdir()):
        validate_path_within(item, base_dir)
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except OSError as exc:
            log.warning("Could not delete %s: %s", item, exc)
            skipped.append(str(item))
            continue
        deleted += 1
    return deleted


def perform_reset() -> ResetResult:
    """Delete all documents and data. Returns summary of what was deleted."""
    skipped: list[str] = []
    deleted_docs = _clear_dir(cfg.documents_dir, skipped)
    deleted_data = _clear_dir(cfg.data_dir, skipped)

    return ResetResult(
        deleted_docs=deleted_docs,
        deleted_data=deleted_data,
        skipped=skipped,
        documents_dir=str(cfg.documents_dir),
        data_dir=str(cfg.data_dir),
    )
