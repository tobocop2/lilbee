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

from .batch import hash_existing_sources
from .citations import resolve_multi_source_citations
from .page import (
    chunks_to_text,
    generate_page,
    truncate_chunks_to_budget,
)
from .shared import WIKI_BUILD_LOCK
from .stubs import WikiStub, load_stub_index

if TYPE_CHECKING:
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

    by_source: dict[str, list[SearchChunk]] = {}
    resolved = 0
    for source in sorted(wanted):
        available = {c.chunk_index: c for c in store.get_chunks_by_source(source)}
        kept = [available[i] for i in sorted(wanted[source]) if i in available]
        resolved += len(kept)
        if kept:
            by_source[source] = kept
    return by_source, len(stub.chunk_refs) - resolved


def generate_stub_page(
    slug: str,
    store: Store,
    config: Config | None = None,
    *,
    stats: BuildStats | None = None,
) -> Path | None:
    """Write the page for one indexed subject. Returns its path, or None.

    Runs the same citation verification, faithfulness gate, and drafts
    quarantine a build does; nothing here bypasses them. Holds the wiki mutex,
    so a page generated from the browse tree cannot interleave with a build.
    """
    if config is None:
        config = cfg
    with WIKI_BUILD_LOCK:
        stub = load_stub_index(config).get(slug)
        if stub is None:
            raise UnknownStubError(f"no indexed page named {slug!r}")

        chunks_by_source, unresolved = _chunks_for_stub(stub, store)
        if not chunks_by_source:
            log.warning("No chunks remain for %s; the index is stale", slug)
            return None
        if unresolved:
            log.info("%d of %s's indexed chunks are gone", unresolved, slug)

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

        return generate_page(
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
        )
