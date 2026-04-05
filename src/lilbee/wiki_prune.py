"""Prune stale and orphaned wiki pages.

Pruning rules:
1. All cited sources deleted -> archive the page
2. Concept cluster shrinks below 3 sources -> archive synthesis page
3. >50% of citations are stale (stale_hash or excerpt_missing) -> flag for regeneration

Archived pages are moved to wiki/archive/ and removed from the vector store.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from lilbee.config import Config, cfg
from lilbee.store import Store
from lilbee.wiki_lint import IssueSeverity, lint_wiki_page

log = logging.getLogger(__name__)

STALE_CITATION_THRESHOLD = 0.5


class PruneAction(Enum):
    """What happened to a wiki page during pruning."""

    ARCHIVED = "archived"
    FLAGGED = "flagged"


@dataclass(frozen=True)
class PruneRecord:
    """A single pruning action taken on a wiki page."""

    wiki_source: str
    action: PruneAction
    reason: str


@dataclass
class PruneReport:
    """Aggregated results from pruning wiki pages."""

    records: list[PruneRecord] = field(default_factory=list)

    @property
    def archived_count(self) -> int:
        return sum(1 for r in self.records if r.action == PruneAction.ARCHIVED)

    @property
    def flagged_count(self) -> int:
        return sum(1 for r in self.records if r.action == PruneAction.FLAGGED)


def _archive_page(
    wiki_source: str,
    wiki_root: Path,
    store: Store,
    config: Config,
) -> None:
    """Move a wiki page to wiki/archive/ and clean up store data."""
    relative = wiki_source.removeprefix(config.wiki_dir + "/")
    source_path = wiki_root / relative

    archive_dir = wiki_root / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / source_path.name

    if source_path.exists():
        shutil.move(str(source_path), str(archive_path))
        log.info("Archived wiki page %s -> %s", source_path, archive_path)
    else:
        log.warning("Wiki page file not found for archival: %s", source_path)

    store.delete_by_source(wiki_source)
    store.delete_citations_for_wiki(wiki_source)


def _check_all_sources_deleted(
    wiki_source: str,
    store: Store,
    documents_dir: Path,
) -> bool:
    """Return True if every cited source file has been deleted from disk."""
    citations = store.get_citations_for_wiki(wiki_source)
    if not citations:
        return False
    source_files = {c["source_filename"] for c in citations}
    return all(not (documents_dir / f).exists() for f in source_files)


def _check_cluster_below_threshold(
    wiki_source: str,
    store: Store,
    documents_dir: Path,
    min_sources: int = 3,
) -> bool:
    """Return True if a synthesis page's live source count dropped below min_sources."""
    if "/concepts/" not in wiki_source:
        return False
    citations = store.get_citations_for_wiki(wiki_source)
    if not citations:
        return False
    source_files = {c["source_filename"] for c in citations}
    live_count = sum(1 for f in source_files if (documents_dir / f).exists())
    return live_count < min_sources


def _check_stale_majority(
    wiki_source: str,
    store: Store,
    config: Config,
) -> bool:
    """Return True if >50% of citations are stale (stale_hash or excerpt_missing)."""
    issues = lint_wiki_page(wiki_source, store, config)
    if not issues:
        return False
    citations = store.get_citations_for_wiki(wiki_source)
    if not citations:
        return False
    stale_count = sum(
        1
        for i in issues
        if i.severity == IssueSeverity.WARNING
        and ("stale hash" in i.message.lower() or "excerpt not found" in i.message.lower())
    )
    return stale_count / len(citations) > STALE_CITATION_THRESHOLD


def prune_wiki(
    store: Store,
    config: Config | None = None,
) -> PruneReport:
    """Scan all wiki pages and prune stale/orphaned ones.

    Returns a PruneReport with all actions taken.
    """
    if config is None:
        config = cfg

    wiki_root = config.data_root / config.wiki_dir
    report = PruneReport()

    if not wiki_root.exists():
        return report

    for subdir in ("summaries", "concepts"):
        subdir_path = wiki_root / subdir
        if not subdir_path.exists():
            continue
        for md_path in sorted(subdir_path.rglob("*.md")):
            relative = md_path.relative_to(wiki_root)
            wiki_source = f"{config.wiki_dir}/{relative}"

            if _check_all_sources_deleted(wiki_source, store, config.documents_dir):
                _archive_page(wiki_source, wiki_root, store, config)
                report.records.append(
                    PruneRecord(
                        wiki_source=wiki_source,
                        action=PruneAction.ARCHIVED,
                        reason="all cited sources deleted",
                    )
                )
                continue

            if _check_cluster_below_threshold(wiki_source, store, config.documents_dir):
                _archive_page(wiki_source, wiki_root, store, config)
                report.records.append(
                    PruneRecord(
                        wiki_source=wiki_source,
                        action=PruneAction.ARCHIVED,
                        reason="concept cluster below 3 live sources",
                    )
                )
                continue

            if _check_stale_majority(wiki_source, store, config):
                report.records.append(
                    PruneRecord(
                        wiki_source=wiki_source,
                        action=PruneAction.FLAGGED,
                        reason="majority of citations stale",
                    )
                )

    if report.records:
        log.info(
            "Wiki prune: %d archived, %d flagged for regeneration",
            report.archived_count,
            report.flagged_count,
        )
    return report
