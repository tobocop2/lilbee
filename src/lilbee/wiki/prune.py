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
from lilbee.wiki.lint import IssueType, lint_wiki_page
from lilbee.wiki.shared import MIN_CLUSTER_SOURCES

log = logging.getLogger(__name__)


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
    min_sources: int = MIN_CLUSTER_SOURCES,
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
    _STALE_TYPES = {IssueType.STALE_HASH, IssueType.EXCERPT_MISSING}
    stale_count = sum(1 for i in issues if i.issue_type in _STALE_TYPES)
    return stale_count / len(citations) > config.wiki_stale_citation_threshold


def _archive_and_record(
    wiki_source: str,
    wiki_root: Path,
    store: Store,
    config: Config,
    reason: str,
) -> PruneRecord:
    """Archive a wiki page and return a PruneRecord for the action."""
    _archive_page(wiki_source, wiki_root, store, config)
    return PruneRecord(wiki_source=wiki_source, action=PruneAction.ARCHIVED, reason=reason)


def _evaluate_page(
    wiki_source: str, wiki_root: Path, store: Store, config: Config
) -> PruneRecord | None:
    """Check a single wiki page against pruning rules. Returns a record or None."""
    if _check_all_sources_deleted(wiki_source, store, config.documents_dir):
        return _archive_and_record(
            wiki_source, wiki_root, store, config, "all cited sources deleted"
        )
    if _check_cluster_below_threshold(wiki_source, store, config.documents_dir):
        return _archive_and_record(
            wiki_source,
            wiki_root,
            store,
            config,
            f"concept cluster below {MIN_CLUSTER_SOURCES} live sources",
        )
    if _check_stale_majority(wiki_source, store, config):
        return PruneRecord(
            wiki_source=wiki_source,
            action=PruneAction.FLAGGED,
            reason="majority of citations stale",
        )
    return None


def _finalize_prune(report: PruneReport, config: Config) -> None:
    """Update wiki index and log after pruning."""
    if not report.records:
        return
    log.info(
        "Wiki prune: %d archived, %d flagged",
        report.archived_count,
        report.flagged_count,
    )
    from lilbee.wiki.index import append_wiki_log, update_wiki_index

    update_wiki_index(config)
    for rec in report.records:
        append_wiki_log(f"pruned ({rec.action.value})", f"{rec.wiki_source}: {rec.reason}", config)


def prune_wiki(store: Store, config: Config | None = None) -> PruneReport:
    """Scan all wiki pages and prune stale/orphaned ones."""
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
            wiki_source = f"{config.wiki_dir}/{relative.as_posix()}"
            record = _evaluate_page(wiki_source, wiki_root, store, config)
            if record:
                report.records.append(record)
    _finalize_prune(report, config)
    return report
