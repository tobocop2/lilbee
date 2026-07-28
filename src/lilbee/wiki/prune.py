"""Prune stale and orphaned wiki pages.

Pruning rules:
1. All cited sources deleted -> archive the page
2. Synthesis cluster shrinks below MIN_CLUSTER_SOURCES live sources -> archive the page
3. Stale citations (stale_hash or excerpt_missing) exceed
   ``wiki_stale_citation_threshold`` -> flag the page in the prune report

Archived pages are moved to wiki/archive/ and removed from the vector store.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from lilbee.core.config import Config, cfg
from lilbee.data.store import Store
from lilbee.wiki.index import append_wiki_log, update_wiki_index
from lilbee.wiki.lint import IssueType, lint_wiki_page
from lilbee.wiki.persistence import subdir_from_wiki_source
from lilbee.wiki.shared import (
    MIN_CLUSTER_SOURCES,
    WIKI_BUILD_LOCK,
    WIKI_CONTENT_SUBDIRS,
    WikiLogAction,
    WikiSubdir,
)

log = logging.getLogger(__name__)

_STALE_TYPES = {IssueType.STALE_HASH, IssueType.EXCERPT_MISSING}


class PruneAction(Enum):
    """What happened to a wiki page during pruning."""

    ARCHIVED = "archived"
    FLAGGED = "flagged"
    RECONCILED = "reconciled"


@dataclass(frozen=True)
class PruneRecord:
    """A single pruning action taken on a wiki page."""

    wiki_source: str
    action: PruneAction
    reason: str

    def to_dict(self) -> dict[str, str]:
        """Serialize to a plain dict suitable for JSON output."""
        return {
            "wiki_source": self.wiki_source,
            "action": self.action.value,
            "reason": self.reason,
        }


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

    @property
    def reconciled_count(self) -> int:
        return sum(1 for r in self.records if r.action == PruneAction.RECONCILED)


def _delete_wiki_rows(wiki_source: str, store: Store) -> bool:
    """Delete a wiki page's chunk and citation rows. Returns whether it succeeded."""
    try:
        store.delete_by_source(wiki_source)
        store.delete_citations_for_wiki(wiki_source)
    except Exception:
        log.warning(
            "Failed to delete store rows for %s; the next prune pass retries them",
            wiki_source,
            exc_info=True,
        )
        return False
    return True


def _archive_page(
    wiki_source: str,
    wiki_root: Path,
    store: Store,
    config: Config,
) -> None:
    """Move a wiki page to wiki/archive/, then delete its store rows.

    The file moves first so a crash or a failed delete leaves rows without a
    page, which :func:`_reconcile_orphan_rows` deletes on the next pass.
    Deleting rows first would leave a page on disk with no citations, and every
    archival check reads an uncited page as "nothing to do", so nothing would
    ever retry it.
    """
    relative = wiki_source.removeprefix(config.wiki_dir + "/")
    source_path = wiki_root / relative

    # Mirror the source subdir under archive/ (archive/concepts/foo.md), not a flat
    # archive/foo.md: same-slug pages from different subdirs would overwrite there.
    archive_path = wiki_root / WikiSubdir.ARCHIVE / relative
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    if source_path.exists():
        shutil.move(source_path, archive_path)
        log.info("Archived wiki page %s -> %s", source_path, archive_path)
    else:
        log.warning("Wiki page file not found for archival: %s", source_path)
    _delete_wiki_rows(wiki_source, store)


def _check_all_sources_deleted(
    wiki_source: str,
    store: Store,
) -> bool:
    """Return True if every cited source file has been deleted from disk."""
    from lilbee.data.ingest.discovery import resolve_source_path

    citations = store.get_citations_for_wiki(wiki_source)
    if not citations:
        return False
    source_files = {c["source_filename"] for c in citations}
    return all(not resolve_source_path(f).exists() for f in source_files)


def _check_cluster_below_threshold(
    wiki_source: str,
    store: Store,
    min_sources: int = MIN_CLUSTER_SOURCES,
) -> bool:
    """Return True if a synthesis page's live source count dropped below min_sources."""
    from lilbee.data.ingest.discovery import resolve_source_path

    if f"/{WikiSubdir.SYNTHESIS}/" not in wiki_source:
        return False
    citations = store.get_citations_for_wiki(wiki_source)
    if not citations:
        return False
    source_files = {c["source_filename"] for c in citations}
    live_count = sum(1 for f in source_files if resolve_source_path(f).exists())
    return live_count < min_sources


def _check_stale_majority(
    wiki_source: str,
    store: Store,
    config: Config,
) -> bool:
    """Return True if the stale citation fraction exceeds ``wiki_stale_citation_threshold``."""
    issues = lint_wiki_page(wiki_source, store, config)
    if not issues:
        return False
    citations = store.get_citations_for_wiki(wiki_source)
    if not citations:
        return False
    stale_count = sum(1 for i in issues if i.issue_type in _STALE_TYPES)
    return stale_count / len(citations) > config.wiki_stale_citation_threshold


def _archive_and_record(
    wiki_source: str,
    wiki_root: Path,
    store: Store,
    config: Config,
    reason: str,
) -> PruneRecord:
    """Archive a wiki page and return its PruneRecord."""
    _archive_page(wiki_source, wiki_root, store, config)
    return PruneRecord(wiki_source=wiki_source, action=PruneAction.ARCHIVED, reason=reason)


def _evaluate_page(
    wiki_source: str, wiki_root: Path, store: Store, config: Config
) -> PruneRecord | None:
    """Check a single wiki page against pruning rules. Returns a record or None."""
    if _check_all_sources_deleted(wiki_source, store):
        return _archive_and_record(
            wiki_source, wiki_root, store, config, "all cited sources deleted"
        )
    if _check_cluster_below_threshold(wiki_source, store):
        return _archive_and_record(
            wiki_source,
            wiki_root,
            store,
            config,
            f"synthesis cluster below {MIN_CLUSTER_SOURCES} live sources",
        )
    if _check_stale_majority(wiki_source, store, config):
        return PruneRecord(
            wiki_source=wiki_source,
            action=PruneAction.FLAGGED,
            reason="majority of citations stale",
        )
    return None


def _reconcile_orphan_rows(store: Store, wiki_root: Path, config: Config) -> list[PruneRecord]:
    """Delete wiki rows whose page is no longer a file under a content subdir.

    The page scan only ever revisits pages still on disk, so rows left behind by
    an interrupted archive, a manual delete, or a migration would otherwise keep
    serving retired content in search forever.
    """
    prefix = config.wiki_dir + "/"
    records: list[PruneRecord] = []
    for wiki_source in sorted(store.wiki_chunk_sources() | store.wiki_citation_sources()):
        subdir = subdir_from_wiki_source(wiki_source, config.wiki_dir)
        page_path = wiki_root / wiki_source.removeprefix(prefix)
        if subdir in WIKI_CONTENT_SUBDIRS and page_path.is_file():
            continue
        if not _delete_wiki_rows(wiki_source, store):
            continue
        log.info("Reconciled orphaned wiki rows for %s (no page on disk)", wiki_source)
        records.append(
            PruneRecord(
                wiki_source=wiki_source,
                action=PruneAction.RECONCILED,
                reason="indexed rows without a page on disk",
            )
        )
    return records


def _finalize_prune(report: PruneReport, wiki_root: Path, config: Config) -> None:
    """Update wiki index and log after pruning.

    A pass that reconciled rows for a wiki directory the user deleted writes
    nothing back: index.md and log.md would recreate the tree it removed.
    """
    if not report.records:
        return
    log.info(
        "Wiki prune: %d archived, %d flagged, %d reconciled",
        report.archived_count,
        report.flagged_count,
        report.reconciled_count,
    )
    if not wiki_root.exists():
        return
    update_wiki_index(config)
    for rec in report.records:
        append_wiki_log(
            WikiLogAction.PRUNE,
            f"{rec.action.value} {rec.wiki_source}: {rec.reason}",
            config,
        )


def prune_wiki(store: Store, config: Config | None = None) -> PruneReport:
    """Scan all wiki pages and prune stale/orphaned ones.

    The page scan covers pages still on disk; reconciliation then covers rows
    whose page is not, including the case of a wiki directory removed wholesale.
    Archiving and the index rewrite make this a writer, so it holds the wiki
    build mutex for the whole pass.
    """
    if config is None:
        config = cfg
    wiki_root = config.data_root / config.wiki_dir
    report = PruneReport()
    with WIKI_BUILD_LOCK:
        for subdir in WIKI_CONTENT_SUBDIRS:
            subdir_path = wiki_root / subdir
            if not subdir_path.is_dir():
                continue
            for md_path in sorted(subdir_path.rglob("*.md")):
                relative = md_path.relative_to(wiki_root)
                wiki_source = f"{config.wiki_dir}/{relative.as_posix()}"
                record = _evaluate_page(wiki_source, wiki_root, store, config)
                if record:
                    report.records.append(record)
        report.records.extend(_reconcile_orphan_rows(store, wiki_root, config))
        _finalize_prune(report, wiki_root, config)
    return report
