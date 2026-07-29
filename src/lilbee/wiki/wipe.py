"""Remove a generated wiki: its pages on disk and its rows in the store."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from lilbee.core.config import cfg

from .shared import WIKI_BUILD_LOCK, WIKI_CONTENT_SUBDIRS, WikiSubdir

if TYPE_CHECKING:
    from lilbee.core.config import Config
    from lilbee.data.store import Store

log = logging.getLogger(__name__)

# Every subdir holding pages a wipe removes: published content plus the
# quarantined drafts and the archive prune leaves behind.
_PAGE_SUBDIRS: tuple[WikiSubdir, ...] = (
    *WIKI_CONTENT_SUBDIRS,
    WikiSubdir.DRAFTS,
    WikiSubdir.ARCHIVE,
)


@dataclass(frozen=True)
class WipeReport:
    """What a wipe removed, and whether the store delete actually landed."""

    pages_removed: int
    sources_cleared: int
    rows_deleted: bool

    def summary(self) -> str:
        """One line for the CLI, MCP, and HTTP responses."""
        pages = f"{self.pages_removed} page{'s' if self.pages_removed != 1 else ''}"
        if not self.rows_deleted:
            return f"Removed {pages}, but deleting the store rows failed; run the wipe again"
        return f"Removed {pages} and the store rows for {self.sources_cleared} of them"


def _count_pages(wiki_root: Path) -> int:
    """Count the ``.md`` pages a wipe is about to remove."""
    total = 0
    for subdir in _PAGE_SUBDIRS:
        directory = wiki_root / subdir
        if directory.is_dir():
            total += sum(1 for _ in directory.rglob("*.md"))
    return total


def wipe_wiki(store: Store, config: Config | None = None) -> WipeReport:
    """Delete every generated wiki page and the store rows behind it.

    Pages go first. A crash in between then leaves rows whose page is gone,
    which the next prune reconciles away; the reverse order would leave pages
    on disk that no check ever retries, because an uncited page reads as
    nothing to do.
    """
    if config is None:
        config = cfg
    wiki_root = config.data_root / config.wiki_dir
    with WIKI_BUILD_LOCK:
        sources = store.wiki_chunk_sources() | store.wiki_citation_sources()
        pages_removed = _count_pages(wiki_root)
        if wiki_root.is_dir():
            shutil.rmtree(wiki_root)
        rows_deleted = store.delete_all_wiki_rows()
    if not rows_deleted:
        log.warning("Wiki wipe removed the pages but failed to delete the store rows")
    log.info("Wiki wipe: %d pages, %d indexed sources", pages_removed, len(sources))
    return WipeReport(
        pages_removed=pages_removed,
        sources_cleared=len(sources),
        rows_deleted=rows_deleted,
    )
