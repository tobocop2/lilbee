"""Wiki post-ingest hook: regenerate pages touched by a recent sync."""

from __future__ import annotations

import asyncio
import logging

from lilbee.app.services import get_services
from lilbee.core.config import cfg

log = logging.getLogger(__name__)


async def incremental_update(changed_sources: set[str]) -> None:
    """Regenerate only the wiki pages touched by *changed_sources*.

    Builds a fresh ``ExtractedEntity`` set from the current corpus and
    keeps the records whose chunk trail includes one of the changed
    sources. An entity with no page on disk is not by itself a reason to
    regenerate: its page may be a draft or a marker held for review, and
    re-queueing those every sync burns LLM calls and overwrites pending
    review content. Above ``cfg.wiki_ingest_update_cap`` touched pages
    the auto-update bails out and logs a manual-update hint instead.
    """
    if not cfg.wiki or not changed_sources:
        return
    from lilbee.data.store import SearchChunk
    from lilbee.wiki import append_wiki_log, build_wiki, update_wiki_index
    from lilbee.wiki.entity_extractor import get_entity_extractor
    from lilbee.wiki.shared import WIKI_BUILD_LOCK, WikiLogAction
    from lilbee.wiki.stats import BuildStats

    svc = get_services()
    extractor = get_entity_extractor(cfg.wiki_entity_mode, svc.provider, cfg)

    chunks: list[SearchChunk] = []
    for record in svc.store.get_sources():
        chunks.extend(svc.store.get_chunks_by_source(record["filename"]))
    entities = await asyncio.to_thread(extractor.extract, chunks)

    touched = [
        entity
        for entity in entities
        if any(ref.source in changed_sources for ref in entity.chunk_refs)
    ]

    if not touched:
        return

    if len(touched) > cfg.wiki_ingest_update_cap:
        # warning, not info: the default LILBEE_LOG_LEVEL is WARNING, so
        # log.info would silently drop the manual-update hint and the user
        # would see no signal at all during `lilbee sync` when the cap trips.
        log.warning(
            "Wiki auto-update skipped: %d pages touched (cap %d). "
            "Run 'lilbee wiki update' for a full rebuild.",
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
    def _regenerate() -> None:
        stats = BuildStats()
        with WIKI_BUILD_LOCK:
            pages = build_wiki(
                touched, svc.provider, svc.store, cfg, extract_concepts=False, stats=stats
            )
            update_wiki_index()
            append_wiki_log(
                WikiLogAction.INGEST,
                f"{len(pages)} pages regenerated for {', '.join(sorted(changed_sources))}; "
                f"{stats.summary_line()}",
            )

    # The mutex is taken in the worker thread: acquiring it here would block the
    # event loop for the length of another surface's build.
    await asyncio.to_thread(_regenerate)
