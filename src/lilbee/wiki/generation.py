"""Top-level wiki build orchestrators.

Two public entry points live here:

- :func:`build_wiki` produces entity and LLM-curated concept pages
  per source, runs the one-time legacy-concept-page archival first,
  then rewrites ``[[link]]`` slugs across all wiki content subdirs.
- :func:`generate_synthesis_pages` produces cross-source synthesis
  pages from concept clusters spanning 3+ documents.

Both reuse the per-source batch path and the single-page pipeline
from :mod:`lilbee.wiki.synthesis` and :mod:`lilbee.wiki.page`.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TypedDict

from lilbee.app.services import get_services
from lilbee.core.config import Config, cfg
from lilbee.data.store import SearchChunk, Store
from lilbee.providers.base import LLMProvider
from lilbee.retrieval.clustering import SourceClusterer
from lilbee.runtime.progress import (
    DetailedProgressCallback,
    EventType,
    WikiPageEvent,
    WikiPhase,
    WikiPhaseEvent,
    noop_callback,
)
from lilbee.wiki.batch import archive_legacy_concept_pages
from lilbee.wiki.entity_extractor import EntityKind, ExtractedEntity, get_entity_extractor
from lilbee.wiki.index import append_wiki_log, update_wiki_index
from lilbee.wiki.links import apply_rewriter, compile_rewriter
from lilbee.wiki.shared import (
    MIN_CLUSTER_SOURCES,
    WIKI_BUILD_LOCK,
    WIKI_CONTENT_SUBDIRS,
    WikiLogAction,
    WikiSubdir,
    atomic_write_text,
)
from lilbee.wiki.stats import BuildStats, BuildStatsDict
from lilbee.wiki.synthesis import (
    generate_source_batch,
    generate_synthesis_page,
    group_entities_by_primary_source,
)

log = logging.getLogger(__name__)

_ENTITY_LIKE_SUBDIRS: tuple[str, ...] = (WikiSubdir.CONCEPTS, WikiSubdir.ENTITIES)


def _generate_for_cluster(
    label: str,
    sources: frozenset[str],
    provider: LLMProvider,
    store: Store,
    config: Config,
    stats: BuildStats,
) -> Path | None:
    """Gather chunks for a cluster and generate a synthesis page."""
    source_names = sorted(sources)
    chunks_by_source: dict[str, list] = {}
    for name in source_names:
        chunks = store.get_chunks_by_source(name)
        if chunks:
            chunks_by_source[name] = chunks

    if len(chunks_by_source) < MIN_CLUSTER_SOURCES:
        return None

    return generate_synthesis_page(
        label, source_names, chunks_by_source, provider, store, config, stats
    )


def generate_synthesis_pages(
    provider: LLMProvider,
    store: Store,
    clusterer: SourceClusterer,
    config: Config | None = None,
    on_progress: DetailedProgressCallback = noop_callback,
    stats: BuildStats | None = None,
    cancel: threading.Event | None = None,
) -> list[Path]:
    """Generate synthesis pages for source clusters spanning 3+ documents.

    Setting *cancel* stops at the next cluster boundary so a disconnected
    client does not leave the run holding the wiki build mutex.
    """
    if config is None:
        config = cfg
    stats = BuildStats.ensure(stats)

    clusters = clusterer.get_clusters(min_sources=MIN_CLUSTER_SOURCES)
    if not clusters:
        log.info("No source clusters span %d+ sources, skipping synthesis", MIN_CLUSTER_SOURCES)
        return []

    on_progress(EventType.WIKI_PHASE, WikiPhaseEvent(phase=WikiPhase.GENERATE, total=len(clusters)))
    pages: list[Path] = []
    for index, cluster in enumerate(clusters, start=1):
        if cancel is not None and cancel.is_set():
            log.info("Wiki synthesis cancelled after %d of %d clusters", index - 1, len(clusters))
            break
        page = _generate_for_cluster(cluster.label, cluster.sources, provider, store, config, stats)
        if page is not None:
            pages.append(page)
        on_progress(
            EventType.WIKI_PAGE,
            WikiPageEvent(
                label=cluster.label,
                pages=0 if page is None else 1,
                current=index,
                total=len(clusters),
            ),
        )

    log.info("Generated %d synthesis pages", len(pages))
    return pages


def _all_sources_in_scope(
    grouped: dict[str, list[ExtractedEntity]],
    store: Store,
    config: Config,
    extract_concepts: bool,
) -> set[str]:
    """Union of sources with entities and (when enabled) eligible for concept curation.

    Seed the union with every entity's primary source (the keys of
    ``grouped``). When ``extract_concepts`` is True AND
    ``wiki_batch_min_chunks`` is satisfied, add any source in the store
    that passes the floor. This gives concept-only sources (no extracted
    entities) their chance at curation while keeping zero-entity short
    sources skipped entirely.
    """
    sources: set[str] = set(grouped)
    if not extract_concepts:
        return sources
    try:
        records = store.get_sources()
    except Exception as exc:
        log.warning("get_sources failed; sticking to entity-grouped sources: %s", exc)
        return sources
    for record in records:
        name = record.get("filename", "") if isinstance(record, dict) else ""
        if not name:
            continue
        if name in sources:
            continue
        chunk_count = record.get("chunk_count", 0) if isinstance(record, dict) else 0
        if chunk_count >= config.wiki_batch_min_chunks:
            sources.add(name)
    return sources


def _entity_surface_map(entities: list[ExtractedEntity]) -> dict[str, str]:
    """Build the surface-form -> slug map for the ``[[link]]`` rewriter.

    Includes both the entity's human label (e.g. *"Henry Ford"*) and
    the slug-with-hyphens-as-spaces variant (*"henry ford"*) so the
    rewriter catches either form in body text.
    """
    mapping: dict[str, str] = {}
    for entity in entities:
        mapping[entity.label] = entity.slug
        spaced = entity.slug.replace("-", " ")
        if spaced and spaced != entity.label:
            mapping[spaced] = entity.slug
    return mapping


def _augment_surface_map_with_existing_pages(
    surface_to_slug: dict[str, str], wiki_root: Path
) -> None:
    """Add slugs for pages already on disk so an incremental rebuild of
    one concept still links to its unchanged neighbors. **Mutates
    surface_to_slug in place.** Only enriches the map with the
    hyphen-to-space surface form because frontmatter labels aren't
    read here; body prose typically uses the spaced form so this
    covers the common case.
    """
    for subdir in _ENTITY_LIKE_SUBDIRS:
        subdir_path = wiki_root / subdir
        if not subdir_path.is_dir():
            continue
        for md_path in subdir_path.rglob("*.md"):
            slug = md_path.stem
            spaced = slug.replace("-", " ")
            surface_to_slug.setdefault(spaced, slug)


def rewrite_links_across_wiki(entities: list[ExtractedEntity], config: Config) -> None:
    """Rewrite ``[[slug]]`` links on every page under ``wiki/`` content subdirs.

    A page never receives a link to itself: the rewriter takes the
    owning slug and drops it inside its match callback, so the
    surface map is shared unmodified across every page in the walk
    (no O(M) dict rebuild per file). The map is augmented with
    slugs from the existing on-disk corpus so a touched page still
    links to untouched neighbors. The alternation regex + lookup are
    compiled once per build and reused across pages.
    """
    surface_to_slug = _entity_surface_map(entities)
    wiki_root = config.data_root / config.wiki_dir
    _augment_surface_map_with_existing_pages(surface_to_slug, wiki_root)
    rewriter = compile_rewriter(surface_to_slug)
    if rewriter is None:
        return

    for subdir in WIKI_CONTENT_SUBDIRS:
        subdir_path = wiki_root / subdir
        if not subdir_path.is_dir():
            continue
        is_entity_subdir = subdir in _ENTITY_LIKE_SUBDIRS
        for md_path in subdir_path.rglob("*.md"):
            owning_slug = md_path.stem if is_entity_subdir else None
            original = md_path.read_text(encoding="utf-8")
            rewritten = apply_rewriter(original, rewriter, skip_slug=owning_slug)
            if rewritten != original:
                atomic_write_text(md_path, rewritten)


def build_wiki(
    entities: list[ExtractedEntity],
    provider: LLMProvider,
    store: Store,
    config: Config | None = None,
    *,
    extract_concepts: bool = True,
    on_progress: DetailedProgressCallback = noop_callback,
    stats: BuildStats | None = None,
    cancel: threading.Event | None = None,
) -> list[Path]:
    """Produce entity and LLM-curated concept pages per source.

    Per-entity / per-concept fan-out is collapsed into a per-source
    batched call: for each source in ``entities``' chunk refs, one LLM
    call identifies 3-5 concepts AND writes a wiki section for every
    pre-extracted entity belonging to that source. Output sections are
    split, citation-verified, embedding-scored, and landed under
    ``wiki/entities/`` or ``wiki/concepts/`` depending on kind.

    ``extract_concepts=False`` (used by the incremental-ingest hook)
    drops the concept-curation paragraph from the prompt so a
    touched source does not churn concept slugs.

    A one-time archive migration runs first (idempotently, gated by
    ``{data_dir}/.phase-d-migrated``), moving legacy concept pages
    under ``wiki/archive/concepts/`` and unwrapping stale
    ``[[archived-slug]]`` links across the remaining pages.
    """
    if config is None:
        config = cfg
    stats = BuildStats.ensure(stats)
    wiki_root = config.data_root / config.wiki_dir
    archive_legacy_concept_pages(wiki_root, config.data_dir, store, config)

    grouped = group_entities_by_primary_source(entities)
    all_sources = _all_sources_in_scope(grouped, store, config, extract_concepts)
    written_concept_slugs: dict[str, str] = {}
    pages: list[Path] = []

    on_progress(
        EventType.WIKI_PHASE, WikiPhaseEvent(phase=WikiPhase.GENERATE, total=len(all_sources))
    )
    for index, source in enumerate(sorted(all_sources), start=1):
        if cancel is not None and cancel.is_set():
            log.info("Wiki build cancelled after %d of %d sources", index - 1, len(all_sources))
            break
        source_entities = grouped.get(source, [])
        chunks = store.get_chunks_by_source(source)
        chunk_count = len(chunks)
        source_extract = extract_concepts and chunk_count >= config.wiki_batch_min_chunks
        source_pages: list[Path] = []
        if source_entities or source_extract:
            source_pages = generate_source_batch(
                source=source,
                entities=source_entities,
                chunks=chunks,
                provider=provider,
                store=store,
                config=config,
                extract_concepts=source_extract,
                written_concept_slugs=written_concept_slugs,
                stats=stats,
            )
            pages.extend(source_pages)
        else:
            log.info(
                "Skipping source %s: %d entities, %d chunks, min=%d, extract=%s",
                source,
                len(source_entities),
                chunk_count,
                config.wiki_batch_min_chunks,
                source_extract,
            )
        on_progress(
            EventType.WIKI_PAGE,
            WikiPageEvent(
                label=source,
                pages=len(source_pages),
                current=index,
                total=len(all_sources),
            ),
        )

    rewrite_links_across_wiki(entities, config)
    log.info("Generated %d batched wiki pages", len(pages))
    return pages


class WikiBuildSummary(TypedDict):
    """Result of a full wiki build/update."""

    paths: list[str]
    entities: int
    count: int
    stats: BuildStatsDict


class WikiEntityCandidate(TypedDict):
    """One NER entity a build would write a page for."""

    slug: str
    label: str
    kind: EntityKind
    type_hint: str
    mentions: int
    sources: list[str]


DRY_RUN_CONCEPT_NOTE = (
    "LLM-curated concepts are not part of a dry run. "
    "Run the build to see which concepts the LLM proposes."
)


def _corpus_chunks(store: Store) -> list[SearchChunk]:
    """Every chunk of every ingested source, in source order."""
    chunks: list[SearchChunk] = []
    for record in store.get_sources():
        chunks.extend(store.get_chunks_by_source(record["filename"]))
    return chunks


def preview_build_entities(config: Config) -> list[WikiEntityCandidate]:
    """Entity candidates a build would cover, extracted with no LLM call.

    The dry run every surface shares. Concepts come from the per-source
    batched call, so they are absent here: see :data:`DRY_RUN_CONCEPT_NOTE`.
    """
    svc = get_services()
    extractor = get_entity_extractor(config.wiki_entity_mode, svc.provider, config)
    return [
        WikiEntityCandidate(
            slug=entity.slug,
            label=entity.label,
            kind=entity.kind,
            type_hint=entity.type_hint,
            mentions=len(entity.chunk_refs),
            sources=sorted({ref.source for ref in entity.chunk_refs}),
        )
        for entity in extractor.extract(_corpus_chunks(svc.store))
    ]


def run_full_build(
    config: Config | None = None,
    on_progress: DetailedProgressCallback = noop_callback,
    cancel: threading.Event | None = None,
) -> WikiBuildSummary:
    """Extract entities and build wiki pages for every ingested source.

    Holds the wiki build mutex for the whole run, so a build started from any
    surface waits for one already in flight instead of interleaving writes.
    Setting *cancel* stops the run at the next source boundary and releases the
    mutex; without it a client that disconnects mid-stream leaves a build
    holding the mutex until it finishes the whole corpus.
    """
    if config is None:
        config = cfg
    stats = BuildStats()
    with WIKI_BUILD_LOCK:
        svc = get_services()
        on_progress(EventType.WIKI_PHASE, WikiPhaseEvent(phase=WikiPhase.EXTRACT))
        extractor = get_entity_extractor(config.wiki_entity_mode, svc.provider, config)
        entities = extractor.extract(_corpus_chunks(svc.store))
        pages = build_wiki(
            entities,
            svc.provider,
            svc.store,
            config,
            extract_concepts=config.wiki_extract_concepts,
            on_progress=on_progress,
            stats=stats,
            cancel=cancel,
        )
        on_progress(EventType.WIKI_PHASE, WikiPhaseEvent(phase=WikiPhase.INDEX))
        update_wiki_index(config)
        append_wiki_log(
            WikiLogAction.BUILD,
            f"{len(pages)} pages from {len(entities)} records; {stats.summary_line()}",
            config,
        )
    return {
        "paths": [str(p) for p in pages],
        "entities": len(entities),
        "count": len(pages),
        "stats": stats.as_dict(),
    }


class WikiSynthesizeSummary(TypedDict):
    """Result of running synthesis-page generation."""

    paths: list[str]
    count: int
    stats: BuildStatsDict


def run_full_synthesize(
    config: Config | None = None,
    on_progress: DetailedProgressCallback = noop_callback,
    cancel: threading.Event | None = None,
) -> WikiSynthesizeSummary:
    """Generate synthesis pages for cross-source clusters.

    Shares the wiki build mutex with :func:`run_full_build` so synthesis and a
    build never write the same tree at once.
    """
    if config is None:
        config = cfg
    stats = BuildStats()
    svc = get_services()
    with WIKI_BUILD_LOCK:
        paths = generate_synthesis_pages(
            svc.provider, svc.store, svc.clusterer, config, on_progress, stats, cancel
        )
        append_wiki_log(
            WikiLogAction.SYNTHESIZE,
            f"{len(paths)} synthesis pages; {stats.summary_line()}",
            config,
        )
    return {
        "paths": [str(p) for p in paths],
        "count": len(paths),
        "stats": stats.as_dict(),
    }
