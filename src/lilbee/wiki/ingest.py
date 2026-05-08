"""Wiki post-ingest hook: regenerate pages touched by a recent sync.

This module is the wiki layer's response to a completed
``lilbee.data.ingest.pipeline.sync()`` run. The pipeline doesn't know
the wiki rules; it just calls :func:`incremental_update` once the
ingest has settled. The function bails out early when the wiki feature
is off or the touched-page set exceeds ``cfg.wiki_ingest_update_cap``.
"""

from __future__ import annotations

import asyncio
import logging

from lilbee.app.services import get_services
from lilbee.core.config import cfg

log = logging.getLogger(__name__)


async def incremental_update(changed_sources: set[str]) -> None:
    """Regenerate only the wiki pages touched by *changed_sources*.

    Builds a fresh ``ExtractedEntity`` set from the current corpus,
    keeps the records that either have no page on disk yet or whose
    chunk trail includes one of the changed sources, and regenerates
    just those. Above ``cfg.wiki_ingest_update_cap`` touched pages the
    auto-update bails out and logs a manual-update hint instead.
    """
    if not cfg.wiki or not changed_sources:
        return
    from lilbee.data.store import SearchChunk
    from lilbee.wiki import append_wiki_log, build_wiki, update_wiki_index
    from lilbee.wiki.entity_extractor import EntityKind, get_entity_extractor
    from lilbee.wiki.shared import WikiLogAction, WikiSubdir

    svc = get_services()
    extractor = get_entity_extractor(cfg.wiki_entity_mode, svc.provider, cfg)

    chunks: list[SearchChunk] = []
    for record in svc.store.get_sources():
        chunks.extend(svc.store.get_chunks_by_source(record["filename"]))
    entities = await asyncio.to_thread(extractor.extract, chunks)

    wiki_root = cfg.data_root / cfg.wiki_dir
    touched = []
    for entity in entities:
        # The extractor emits only ENTITY kind; CONCEPT is reserved for
        # LLM-curated pages produced inside the batched call. Keeping
        # the dispatch neutral guards against a future extractor that
        # re-introduces CONCEPT.
        subdir = WikiSubdir.CONCEPTS if entity.kind is EntityKind.CONCEPT else WikiSubdir.ENTITIES
        page_path = wiki_root / subdir / f"{entity.slug}.md"
        if not page_path.exists():
            touched.append(entity)
            continue
        if any(ref.source in changed_sources for ref in entity.chunk_refs):
            touched.append(entity)

    if not touched:
        return

    if len(touched) > cfg.wiki_ingest_update_cap:
        # warning, not info: the default LILBEE_LOG_LEVEL is WARNING, so
        # log.info would silently drop the manual-update hint and the user
        # would see no signal at all during `lilbee sync` when the cap trips.
        log.warning(
            "Wiki auto-update skipped: %d pages touched (cap %d). "
            "Run 'lilbee wiki update' to refresh.",
            len(touched),
            cfg.wiki_ingest_update_cap,
        )
        append_wiki_log(
            WikiLogAction.INGEST,
            f"skipped: {len(touched)} pages exceeds cap {cfg.wiki_ingest_update_cap}",
        )
        return

    # extract_concepts=False so an incremental sync does not churn
    # concept slugs. Concept curation is a deliberate, user-invoked
    # refresh (full `lilbee wiki build`).
    pages = await asyncio.to_thread(
        build_wiki, touched, svc.provider, svc.store, cfg, extract_concepts=False
    )
    update_wiki_index()
    append_wiki_log(
        WikiLogAction.INGEST,
        f"{len(pages)} pages regenerated for {', '.join(sorted(changed_sources))}",
    )
