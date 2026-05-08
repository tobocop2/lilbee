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
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from lilbee.core.config import Config
from lilbee.data.ingest import file_hash
from lilbee.data.store import CitationRecord, SearchChunk, Store
from lilbee.wiki.citation import (
    ParsedCitation,
    parse_wiki_citations,
    render_citation_block,
    strip_citation_block,
)
from lilbee.wiki.citations import verify_citations
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
    make_slug,
)

log = logging.getLogger(__name__)

# In-body ``[^keyN]`` footnote-marker pattern. Module-scope so the
# batched-generation hot path (`finalize_section`) does not recompile
# it on every recovered section.
_FOOTNOTE_MARKER_RE = re.compile(r"\[\^([a-zA-Z0-9_\-]+)\]")

# Sentinel file for the one-time legacy-concepts archival. Lives under
# data_dir (NOT inside wiki/) so Obsidian sync and wiki tree-walkers
# never surface it. The on-disk filename is preserved across renames so
# upgrading installs do not re-run the migration.
_LEGACY_CONCEPTS_MIGRATED_SENTINEL = ".phase-d-migrated"

# Legacy wiki concepts that we move to archive/ as part of the one-time
# migration. Matches wiki/<WikiSubdir.CONCEPTS>/*.md recursively.
_ARCHIVE_CONCEPTS_SUBPATH = Path(WikiSubdir.ARCHIVE) / WikiSubdir.CONCEPTS


def hash_existing_sources(source_names: list[str], documents_dir: Path) -> dict[str, str]:
    """Hash each source file that still exists on disk (used for citation staleness)."""
    out: dict[str, str] = {}
    for name in source_names:
        source_path = documents_dir / name
        if source_path.exists():
            out[name] = file_hash(source_path)
    return out


def match_label(
    lowered_name: str,
    expected: set[str],
    kind: EntityKind,
) -> tuple[EntityKind, str] | None:
    """Case-insensitive substring match of *lowered_name* against *expected*.

    Returns ``(kind, original_label)`` on hit, ``None`` otherwise.
    A substring match (not equality) accommodates the LLM adding
    qualifiers ("Brake System (hydraulic)" vs "brake system").
    """
    for label in expected:
        low = label.lower()
        if low and (low in lowered_name or lowered_name in low):
            return (kind, label)
    return None


def chunks_for_source(chunks: list[SearchChunk], source: str) -> list[SearchChunk]:
    """Return the subset of *chunks* whose ``source`` matches, preserving order."""
    return [c for c in chunks if c.source == source]


def short_source_hash(source: str) -> str:
    """8-char sha256 digest of *source* (stable collision-marker suffix)."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:8]


def _group_chunks_by_page(
    chunks: list[SearchChunk],
) -> list[tuple[int, list[SearchChunk]]]:
    """Group chunks by ``page_start``, preserving in-document order within a page.

    Returns ``(page_start, chunks)`` tuples sorted ascending by page number.
    Chunks with ``page_start=0`` (non-paginated sources) collapse to a single
    entry keyed at 0, so a markdown or code source still emits exactly one
    summary file until structure detection arrives in a later stage.
    """
    grouped: dict[int, list[SearchChunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.page_start, []).append(chunk)
    return sorted(grouped.items())


def archive_legacy_concept_pages(wiki_root: Path, data_dir: Path) -> None:
    """One-time migration: archive legacy concept pages.

    Runs idempotently, gated by ``{data_dir}/.phase-d-migrated``:

    1. Move every ``wiki/concepts/*.md`` to ``wiki/archive/concepts/``
       preserving relative subpaths. Older concept pages stay
       readable but drop out of the active wiki browse surface.
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
            src.replace(dest)
            archived_slugs.append(str(rel.with_suffix("")).replace("\\", "/"))

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

    The existing ``_rewrite_links_across_wiki`` path is the wrong
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
                md_path.write_text(rewritten, encoding="utf-8")


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
) -> Path | None:
    """Citation-check, faithfulness-check, write one batched section.

    Shared by entity and concept sections from the per-source batched
    call. Returns the written page path, or ``None`` if the section
    failed any gate (no citations, empty body, slug collision marker
    handled via side channel). ``shared_parsed_citations`` is the
    definition list parsed once over the whole response: every
    section replays it so pages other than the last one still have
    their footnotes resolved.
    """
    slug = make_slug(header_label)
    if not slug:
        log.info("Empty slug for batched section %r; skipping", header_label)
        return None

    # Only replay citation keys that this section actually references
    # in the body; otherwise every section would claim every citation.
    section_keys = {ref.citation_key for ref in parse_wiki_citations(body)}
    # Fall back to in-body ``[^keyN]`` references when no definitions
    # live inside the section: count occurrences of the footnote
    # marker against the shared definition set.
    section_keys.update(_FOOTNOTE_MARKER_RE.findall(body))
    relevant = [c for c in shared_parsed_citations if c.citation_key in section_keys]
    verified = verify_citations(citation_resolver(relevant), chunks, header_label, config)
    if not verified:
        log.info("No valid citations for batched section %s, skipping", header_label)
        return None

    score = check_faithfulness(chunks, body, header_label, config)
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

    clean_body = strip_citation_block(body)
    frontmatter = build_frontmatter(config, source_names, score, chunks=chunks)
    citation_block = render_citation_block(verified)
    full_content = assemble_content(frontmatter, clean_body, citation_block)

    # Concept collision: the second source proposing a slug loses
    # and writes to a drafts collision marker; the winning source's
    # page stays untouched.
    if kind is EntityKind.CONCEPT and subdir == WikiSubdir.CONCEPTS:
        first_source = written_concept_slugs.get(slug)
        if first_source is not None and first_source != source:
            return divert_concept_collision(
                slug=slug,
                source=source,
                first_source=first_source,
                content=full_content,
                drafts_dir=drafts_dir,
            )
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
    page_path = persist_and_finalize(full_content, target, verified, source_names, store, config)
    log.info(
        "Generated batched page for %s -> %s (score=%.2f, citations=%d)",
        header_label,
        target.subdir,
        score,
        len(verified),
    )
    return page_path
