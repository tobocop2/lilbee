"""Per-source batched-generation helpers and legacy concept-page archival.

The batched build (one LLM call per source that emits sections for
every pre-extracted entity plus 3-5 LLM-curated concepts) lives here:
section-finalization, label matching, source hashing, and the page
splitter that turns the model's response into per-section bodies.
Also owns the one-time migration that archives legacy concept pages
(written before per-source batched generation) and unwraps stale
``[[archived-slug]]`` links.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from lilbee.core.config import Config
from lilbee.core.text import make_slug
from lilbee.data.ingest import file_hash
from lilbee.data.store import CitationRecord, SearchChunk, Store
from lilbee.wiki.citations import (
    ParsedCitation,
    footnote_marker_keys,
    render_citation_block,
    scrub_unverified_markers,
    strip_citation_block,
    verify_citations,
    wiki_sourced_count,
)
from lilbee.wiki.entity_extractor import EntityKind
from lilbee.wiki.page import assemble_content, build_frontmatter
from lilbee.wiki.persistence import (
    delete_pending_marker_if_present,
    divert_concept_collision,
    persist_and_finalize,
)
from lilbee.wiki.quality import check_faithfulness
from lilbee.wiki.shared import (
    WIKI_CONTENT_SUBDIRS,
    PageTarget,
    WikiSubdir,
    atomic_write_text,
)
from lilbee.wiki.stats import BuildStats

log = logging.getLogger(__name__)

# Sentinel file for the one-time legacy-concepts archival. Lives under
# data_dir (NOT inside wiki/) so Obsidian sync and wiki tree-walkers
# never surface it. The on-disk filename is preserved across renames so
# upgrading installs do not re-run the migration.
_LEGACY_CONCEPTS_MIGRATED_SENTINEL = ".phase-d-migrated"

# Legacy wiki concepts that we move to archive/ as part of the one-time
# migration. Matches wiki/<WikiSubdir.CONCEPTS>/*.md recursively.
_ARCHIVE_CONCEPTS_SUBPATH = Path(WikiSubdir.ARCHIVE) / WikiSubdir.CONCEPTS


def hash_existing_sources(source_names: list[str]) -> dict[str, str]:
    """Hash each source file that still exists on disk (used for citation staleness)."""
    from lilbee.data.ingest.discovery import resolve_source_path

    out: dict[str, str] = {}
    for name in source_names:
        source_path = resolve_source_path(name)
        if source_path.exists():
            out[name] = file_hash(source_path)
    return out


def _first_label_where(
    candidates: list[tuple[list[str], EntityKind]],
    predicate: Callable[[str], bool],
) -> tuple[EntityKind, str] | None:
    for labels, kind in candidates:
        for label in labels:
            if predicate(label.lower()):
                return (kind, label)
    return None


def match_label(
    lowered_name: str,
    candidates: Sequence[tuple[set[str], EntityKind]],
) -> tuple[EntityKind, str] | None:
    """Case-insensitive match of *lowered_name* against ordered *candidates*.

    Each candidate is an ``(expected labels, kind)`` pair. Returns
    ``(kind, original_label)`` on hit, ``None`` otherwise. Every candidate set
    is tried for an exact match before any is tried for a substring match, so a
    header naming one set's label exactly is not taken by another set's label
    that merely contains it. Substring overlap in either direction accommodates
    the LLM adding qualifiers ("Brake System (hydraulic)" vs "brake system").
    Labels are ordered by length then alphabetically, so overlapping labels
    ("Ford" and "Henry Ford") bind the same way on every run.
    """
    ordered = [
        (sorted(expected, key=lambda label: (-len(label), label)), kind)
        for expected, kind in candidates
    ]
    return _first_label_where(ordered, lambda low: low == lowered_name) or _first_label_where(
        ordered, lambda low: bool(low) and (low in lowered_name or lowered_name in low)
    )


def short_source_hash(source: str) -> str:
    """8-char sha256 digest of *source* (stable collision-marker suffix)."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:8]


def archive_legacy_concept_pages(
    wiki_root: Path, data_dir: Path, store: Store, config: Config
) -> None:
    """One-time migration: archive legacy concept pages.

    Runs idempotently, gated by ``{data_dir}/.phase-d-migrated``:

    1. Delete each ``wiki/concepts/*.md`` page's chunk and citation
       rows, then move the file to ``wiki/archive/concepts/``
       preserving relative subpaths. Store cleanup comes first so an
       interrupted migration leaves the page on disk rather than rows
       serving a page nothing scans. Older concept pages stay readable
       but drop out of retrieval and the active browse surface.
    2. Unwrap stale ``[[archived-slug]]`` references across the
       remaining pages so a reader clicking a link does not hit a
       404. Archived slugs become plain text.
    3. Write the sentinel so future builds skip this path.

    Freshly LLM-curated concept pages written AFTER the sentinel exists
    are never touched.
    """
    sentinel = data_dir / _LEGACY_CONCEPTS_MIGRATED_SENTINEL
    if sentinel.exists():
        return
    concepts_dir = wiki_root / WikiSubdir.CONCEPTS
    archive_dir = wiki_root / _ARCHIVE_CONCEPTS_SUBPATH
    archived_slugs: list[str] = []
    if concepts_dir.is_dir():
        for src in sorted(concepts_dir.rglob("*.md")):
            rel = src.relative_to(concepts_dir)
            dest = archive_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            slug = str(rel.with_suffix("")).replace("\\", "/")
            wiki_source = f"{config.wiki_dir}/{WikiSubdir.CONCEPTS}/{slug}.md"
            store.delete_by_source(wiki_source)
            if not store.delete_citations_for_wiki(wiki_source):
                # Sentinel unwritten: the next build retries the migration. Pages
                # already moved this pass still need their inbound links unwrapped,
                # since the retry only sees what is left in concepts/ and would
                # never revisit them, leaving those links pointing at a 404.
                log.warning("Citation delete failed for %s; migration will retry", wiki_source)
                _unwrap_archived_links(wiki_root, archived_slugs)
                return
            src.replace(dest)
            archived_slugs.append(slug)

    if archived_slugs:
        _unwrap_archived_links(wiki_root, archived_slugs)

    data_dir.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(datetime.now(UTC).isoformat(), encoding="utf-8")
    if archived_slugs:
        log.info(
            "Legacy-concepts migration: archived %d concept pages, sentinel written at %s",
            len(archived_slugs),
            sentinel,
        )


def _unwrap_archived_links(wiki_root: Path, archived_slugs: list[str]) -> None:
    """Rewrite ``[[slug]]`` → ``slug`` (plain text) across remaining wiki pages.

    The existing ``rewrite_links_across_wiki`` path is the wrong
    tool here: it compiles an *additive* surface map, not a
    removal pass. Walk the active wiki content subdirs once per
    archived slug is acceptable because the archive count is
    bounded (concepts that existed pre-migration). Pages whose body
    did not change are not rewritten.
    """
    if not archived_slugs:
        return
    patterns = [(re.compile(r"\[\[" + re.escape(slug) + r"\]\]"), slug) for slug in archived_slugs]
    for subdir in WIKI_CONTENT_SUBDIRS:
        subdir_path = wiki_root / subdir
        if not subdir_path.is_dir():
            continue
        for md_path in subdir_path.rglob("*.md"):
            original = md_path.read_text(encoding="utf-8")
            rewritten = original
            for pattern, replacement in patterns:
                rewritten = pattern.sub(replacement, rewritten)
            if rewritten != original:
                atomic_write_text(md_path, rewritten)


def finalize_section(
    *,
    header_label: str,
    kind: EntityKind,
    body: str,
    chunks: list[SearchChunk],
    citation_resolver: Callable[[list[ParsedCitation]], list[CitationRecord]],
    source_names: list[str],
    store: Store,
    config: Config,
    source: str,
    written_concept_slugs: dict[str, str],
    drafts_dir: Path,
    shared_parsed_citations: list[ParsedCitation],
    scoring_chunks_by_label: dict[str, list[SearchChunk]],
    stats: BuildStats | None = None,
) -> Path | None:
    """Citation-check, faithfulness-check, write one batched section.

    Shared by entity and concept sections from the per-source batched
    call. Returns the written page path, or ``None`` if the section
    failed any gate (no citations, empty body, slug collision marker
    handled via side channel). ``shared_parsed_citations`` is the
    definition list parsed once over the whole response: every
    section replays it so pages other than the last one still have
    their footnotes resolved.

    ``scoring_chunks_by_label`` maps an entity label to the chunks its
    extraction refs named. Faithfulness scores against those rather
    than the whole-source mean, so a section about one entity in a
    multi-topic source is not compared to the document-wide centroid.
    A label with no entry (concepts, or refs that fell outside the
    budgeted chunks) scores against the full pool.

    Citation counts and the section's outcome are recorded on *stats*.
    """
    stats = BuildStats.ensure(stats)
    slug = make_slug(header_label)
    if not slug:
        log.info("Empty slug for batched section %r; skipping", header_label)
        return None

    # Only replay citation keys the section's prose references. Keys are read
    # from the citation-stripped body: the response's trailing block lands in
    # the last section, whose definitions would otherwise make it claim every
    # citation in the response.
    section_keys = footnote_marker_keys(strip_citation_block(body))
    relevant = [c for c in shared_parsed_citations if c.citation_key in section_keys]
    resolved = citation_resolver(relevant)
    verified = verify_citations(resolved, chunks, header_label, config)
    dropped = len(relevant) - wiki_sourced_count(resolved, config) - len(verified)
    if not verified:
        stats.record_citations(0, dropped)
        log.info("No valid citations for batched section %s, skipping", header_label)
        return None

    score = check_faithfulness(
        scoring_chunks_by_label.get(header_label, chunks), body, header_label, config
    )
    threshold = config.wiki_embedding_faithfulness_threshold
    page_type = WikiSubdir.CONCEPTS if kind is EntityKind.CONCEPT else WikiSubdir.ENTITIES
    subdir = page_type if score >= threshold else WikiSubdir.DRAFTS
    if subdir == WikiSubdir.DRAFTS:
        log.info(
            "Batched section %s scored %.2f (< %.2f), sending to drafts",
            header_label,
            score,
            threshold,
        )

    clean_body = scrub_unverified_markers(strip_citation_block(body), verified)
    frontmatter = build_frontmatter(config, source_names, score, chunks=chunks)
    citation_block = render_citation_block(verified)
    full_content = assemble_content(frontmatter, clean_body, citation_block)

    # Recorded before the collision return: the section's footnotes were parsed and
    # verified either way, and the verify rate counts every outcome that got that far.
    stats.record_citations(len(verified), dropped)

    # Concept collision: the second source proposing a slug loses and writes to a
    # drafts collision marker; the winning source's page stays untouched. This
    # applies whether the section publishes or is routed to drafts -- a below-
    # threshold concept still claims the slug, and two such drafts would otherwise
    # overwrite each other at drafts/<slug>.md.
    if kind is EntityKind.CONCEPT:
        first_source = written_concept_slugs.get(slug)
        if first_source is not None and first_source != source:
            stats.record_pending_marker()
            divert_concept_collision(
                slug=slug,
                source=source,
                first_source=first_source,
                content=full_content,
                drafts_dir=drafts_dir,
                origin_subdir=page_type,
            )
            return None
        written_concept_slugs.setdefault(slug, source)

    # Successful regen of a previously-PENDING slug: remove the old
    # marker so the drafts surface no longer lists it.
    delete_pending_marker_if_present(drafts_dir, slug)

    wiki_root = config.data_root / config.wiki_dir
    target = PageTarget(
        wiki_root=wiki_root,
        subdir=subdir,
        slug=slug,
        wiki_source=f"{config.wiki_dir}/{subdir}/{slug}.md",
        page_type=page_type,
        label=header_label,
    )
    page_path = persist_and_finalize(
        full_content, target, verified, source_names, store, config, stats=stats
    )
    log.info(
        "Generated batched page for %s -> %s (score=%.2f, citations=%d)",
        header_label,
        target.subdir,
        score,
        len(verified),
    )
    return page_path
