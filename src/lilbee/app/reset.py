"""Knowledge base reset (delete all documents and data)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from pydantic import BaseModel

from lilbee.core import settings
from lilbee.core.config import cfg
from lilbee.data.store.held_import import remove_legacy_files
from lilbee.runtime.lock import library_held_for_reset


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
        try:
            # iterdir yields direct children only, so the entry is within
            # base_dir by construction. Registered source roots live outside this
            # dir (only their config entry is here), so reset never reaches a
            # user's corpus: it deletes owned files, and perform_reset
            # un-registers the roots from config.toml separately.
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except OSError as exc:
            # best-effort: reset is "delete as much as you can", and the
            # caller surfaces the skipped list to the user verbatim.
            log.warning("Could not delete %s: %s", item, exc)
            skipped.append(str(item))
            continue
        deleted += 1
    return deleted


def perform_reset() -> ResetResult:
    """Delete all documents and data and un-register every linked source.

    Raises ``ResetRefusedError`` while a sync, an import, a wiki build or an add that
    moves source keys, in this process or another, runs against the same data root,
    or when the lock that shows one cannot be taken: any of them would write back
    what the reset removed.
    """
    skipped: list[str] = []
    with library_held_for_reset(cfg):
        # config.toml lives at data_root, next to (not inside) the two cleared
        # dirs. Without un-registering the roots here, in the file AND in this
        # process, the next sync walks every linked root and re-indexes it, so
        # content reappears after a "factory reset". Other settings survive:
        # reset deletes data, not configuration. The file is written first: a
        # config lock that cannot be taken then refuses the reset before it
        # deletes anything.
        settings.delete_value(cfg.data_root, "linked_roots")
        cfg.linked_roots = {}
        deleted_docs = _clear_dir(cfg.documents_dir, skipped)
        deleted_data = _clear_dir(cfg.data_dir, skipped)
        # The held-out record files of an older lilbee sit at data_root too; left
        # there, the next open would import them into the emptied index.
        remove_legacy_files(cfg.data_root)

    return ResetResult(
        deleted_docs=deleted_docs,
        deleted_data=deleted_data,
        skipped=skipped,
        documents_dir=str(cfg.documents_dir),
        data_dir=str(cfg.data_dir),
    )
