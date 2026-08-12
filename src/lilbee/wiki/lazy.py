"""Generate one wiki page on demand, from the index rather than from a build.

A build walks every source document and spends a call on each. This path walks
one subject: it takes that subject's chunks across every source naming it and
writes a single page. The cost is one call for the page someone actually asked
for, and the evidence is wider than a build's, which assigns each entity to the
one source mentioning it most and never sees the rest.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING

from lilbee.app.services import get_services
from lilbee.core.config import cfg
from lilbee.core.text import clean_label_for_display
from lilbee.runtime.progress import (
    DetailedProgressCallback,
    EventType,
    WikiPageEvent,
    WikiPhase,
    WikiPhaseEvent,
    noop_callback,
)

from .batch import hash_existing_sources
from .citations import resolve_multi_source_citations
from .generation import rewrite_links_across_wiki
from .page import (
    chunks_to_text,
    generate_page,
    truncate_chunks_to_budget,
)
from .shared import WIKI_BUILD_LOCK
from .stubs import WikiStub, load_stub_index

if TYPE_CHECKING:
    import threading
    from pathlib import Path

    from lilbee.core.config import Config
    from lilbee.data.store import CitationRecord, SearchChunk, Store

    from .citations import ParsedCitation
    from .stats import BuildStats

log = logging.getLogger(__name__)


class UnknownStubError(LookupError):
    """Raised when a slug names no entry in the wiki index."""


def _chunks_for_stub(stub: WikiStub, store: Store) -> tuple[dict[str, list[SearchChunk]], int]:
    """The stub's chunks grouped by source, plus how many refs went unresolved.

    Refs are looked up per source and filtered by index, so a source that was
    re-ingested with fewer chunks contributes what it still has instead of
    failing the page.
    """
    wanted: dict[str, set[int]] = {}
    for source, index in stub.chunk_refs:
        wanted.setdefault(source, set()).add(index)
    if not wanted:
        # Subtracting a re-indexed source can empty the refs while leaving real
        # evidence: the cap may have given every ref to the source that went.
        # The recorded sources are the truth, so fall back to reading them.
        wanted = {source: set() for source in stub.sources}

    # Most-mentioning source first, so when the context budget truncates it
    # drops the documents with least to say rather than the alphabetically
    # unlucky ones.
    counts = dict(stub.source_mentions)
    order = sorted(wanted, key=lambda name: (-counts.get(name, 0), name))

    by_source: dict[str, list[SearchChunk]] = {}
    resolved = 0
    for source in order:
        available = {c.chunk_index: c for c in store.get_chunks_by_source(source)}
        indexes = wanted[source] or set(available)
        kept = [available[i] for i in sorted(indexes) if i in available]
        resolved += len(kept)
        if kept:
            by_source[source] = kept
    return by_source, max(0, len(stub.chunk_refs) - resolved)


def _resolve(slug: str, stubs: dict[str, WikiStub]) -> WikiStub | None:
    """Find a stub by bare slug or by the subdir-qualified form.

    Every surface shows pages as ``entities/ford``, so that is what a user
    types, while the index is keyed by the bare slug. Both resolve.
    """
    direct = stubs.get(slug)
    if direct is not None:
        return direct
    return next((stub for stub in stubs.values() if stub.wiki_slug == slug), None)


def resolve_stub(slug: str, config: Config | None = None) -> WikiStub | None:
    """The index entry for *slug*, by bare or subdir-qualified form."""
    return _resolve(slug, load_stub_index(config or cfg))


def generate_stub_page(
    slug: str,
    store: Store,
    config: Config | None = None,
    *,
    stats: BuildStats | None = None,
    on_progress: DetailedProgressCallback = noop_callback,
    cancel: threading.Event | None = None,
) -> Path | None:
    """Write the page for one indexed subject. Returns its path, or None.

    Runs the same citation verification, faithfulness gate, and drafts
    quarantine a build does; nothing here bypasses them. Holds the wiki mutex,
    so a page generated from the browse tree cannot interleave with a build.
    Emits the same wiki_phase/wiki_page events a build does; a *cancel* set
    before the model call skips it.
    """
    if config is None:
        config = cfg
    with WIKI_BUILD_LOCK:
        stubs = load_stub_index(config)
        stub = _resolve(slug, stubs)
        if stub is None:
            raise UnknownStubError(f"no indexed page named {slug!r}")

        chunks_by_source, unresolved = _chunks_for_stub(stub, store)
        if not chunks_by_source:
            log.warning("No chunks remain for %s; the index is stale", slug)
            return None
        if unresolved:
            log.info("%d of %s's indexed chunks are gone", unresolved, slug)

        on_progress(EventType.WIKI_PHASE, WikiPhaseEvent(phase=WikiPhase.GENERATE, total=1))
        if cancel is not None and cancel.is_set():
            log.info("Generation of %s cancelled before the model call", slug)
            return None

        source_names = sorted(chunks_by_source)
        all_chunks = [c for chunks in chunks_by_source.values() for c in chunks]
        source_list = "\n".join(f"- {name}" for name in source_names)
        display = clean_label_for_display(stub.label)
        render = functools.partial(
            config.wiki_entity_page_prompt.format, topic=display, source_list=source_list
        )
        all_chunks = truncate_chunks_to_budget(all_chunks, config, len(render(chunks_text="")))
        prompt = render(chunks_text=chunks_to_text(all_chunks))
        source_hashes = hash_existing_sources(source_names)

        def resolver(parsed: list[ParsedCitation]) -> list[CitationRecord]:
            return resolve_multi_source_citations(
                parsed, source_names, source_hashes, chunks_by_source
            )

        path = generate_page(
            label=stub.label,
            prompt=prompt,
            chunks=all_chunks,
            citation_resolver=resolver,
            page_type=stub.subdir,
            slug=stub.slug,
            source_names=source_names,
            provider=get_services().provider,
            store=store,
            config=config,
            stats=stats,
            # These documents mention the subject; the page does not replace
            # them. Pruning here would delete every document that named it.
            supersedes_sources=False,
        )
        if path is not None:
            # The link pass a build runs; without it the page has no [[links]]
            # and sits alone in the graph. No entities: the surface map is then
            # built from the pages on disk, this one included, so links go both
            # ways.
            rewrite_links_across_wiki([], config)
            on_progress(
                EventType.WIKI_PAGE,
                WikiPageEvent(label=stub.label, pages=1, current=1, total=1),
            )
        return path
