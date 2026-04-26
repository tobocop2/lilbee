"""Wiki generation entry points and back-compat surface.

This module is now a thin orchestration layer:

- :func:`run_full_build` and :func:`run_full_synthesize` are the
  single entry points used by the CLI, the HTTP server, and the MCP
  surface. They wire the stored services container into the per-source
  batched build (:mod:`lilbee.wiki.generation`) and the cross-source
  synthesizer.
- The single-page pipeline, citation/quality/persistence helpers, the
  batched-output parser, and the per-source builder live in
  :mod:`lilbee.wiki.page`, :mod:`lilbee.wiki.citations`,
  :mod:`lilbee.wiki.quality`, :mod:`lilbee.wiki.persistence`,
  :mod:`lilbee.wiki.batch`, :mod:`lilbee.wiki.synthesis`,
  :mod:`lilbee.wiki.cache`, and :mod:`lilbee.wiki.generation`.

External callers and tests reach a stable name surface through the
re-exports below (kept so ``monkeypatch.setattr("lilbee.wiki.gen.X")``
in cross-cutting tests continues to flip the bindings the orchestrators
resolve).
"""

from __future__ import annotations

import logging
from typing import TypedDict

from lilbee.chunk import chunk_text  # noqa: F401  (re-export)
from lilbee.config import Config, cfg
from lilbee.services import get_services
from lilbee.store import SearchChunk

# Re-exports kept on this module's namespace because external test
# suites (``tests/test_cli.py``, ``tests/test_mcp.py``) and downstream
# imports (``lilbee.cli.commands``) reach into ``lilbee.wiki.gen.X``
# directly. Listing them via ``from ... import X`` makes the binding
# late-bound: monkeypatching ``lilbee.wiki.gen.<name>`` rebinds the
# attribute the orchestrators below resolve.
from lilbee.wiki.batch import (  # noqa: F401  (re-exports)
    _chunks_for_source,
    _finalize_section,
    _group_chunks_by_page,
    _hash_existing_sources,
    _match_label,
    _maybe_run_phase_d_migration,
    _short_source_hash,
    _unwrap_archived_links,
)
from lilbee.wiki.cache import (  # noqa: F401  (re-exports)
    _find_cached_leaf,
    _leaf_hash,
    _normalize_whitespace,
)
from lilbee.wiki.citation import strip_citation_block  # noqa: F401  (re-export)
from lilbee.wiki.citations import (  # noqa: F401  (re-exports)
    _build_citation_record,
    _decode_excerpt_escapes,
    _extract_excerpt,
    _find_excerpt_location,
    _find_excerpt_source,
    _match_citation_source,
    _render_provenance,
    _resolve_citations,
    _resolve_multi_source_citations,
    _verify_citations,
)
from lilbee.wiki.entity_extractor import get_entity_extractor
from lilbee.wiki.generation import (  # noqa: F401  (re-exports)
    _all_sources_in_scope,
    _augment_surface_map_with_existing_pages,
    _entity_surface_map,
    _generate_for_cluster,
    _rewrite_links_across_wiki,
    build_wiki,
    generate_synthesis_pages,
)
from lilbee.wiki.index import append_wiki_log, update_wiki_index
from lilbee.wiki.page import (  # noqa: F401  (re-exports)
    WikiProgressCallback,
    _assemble_content,
    _build_frontmatter,
    _build_wiki_messages,
    _chunks_to_text,
    _generate_page,
    _truncate_chunks_to_budget,
    _write_page,
    index_wiki_page,
)
from lilbee.wiki.persistence import (  # noqa: F401  (re-exports)
    _delete_pending_marker_if_present,
    _divert_concept_collision,
    _divert_to_drafts,
    _persist_and_finalize,
    _subdir_from_wiki_source,
    _write_pending_marker,
)
from lilbee.wiki.quality import (  # noqa: F401  (re-exports)
    _check_faithfulness,
    _content_change_ratio,
    _diff_summary,
    _embedding_faithfulness_score,
    _mean_vector,
    _title_content_coherence,
)
from lilbee.wiki.shared import WIKI_LOG_ACTION_BUILD
from lilbee.wiki.synthesis import (  # noqa: F401  (re-exports)
    _build_batch_prompt,
    _generate_source_batch,
    _generate_synthesis_page,
    _group_entities_by_primary_source,
    _prefix_heading,
    _split_batched_output,
)

log = logging.getLogger(__name__)

__all__ = [
    "WikiBuildSummary",
    "WikiProgressCallback",
    "WikiSynthesizeSummary",
    "build_wiki",
    "generate_synthesis_pages",
    "index_wiki_page",
    "run_full_build",
    "run_full_synthesize",
]


class WikiBuildSummary(TypedDict):
    """Result of a full wiki build/update."""

    paths: list[str]
    entities: int
    count: int


def run_full_build(config: Config | None = None) -> WikiBuildSummary:
    """Extract entities and build wiki pages for every ingested source."""
    if config is None:
        config = cfg
    svc = get_services()
    chunks: list[SearchChunk] = []
    for record in svc.store.get_sources():
        chunks.extend(svc.store.get_chunks_by_source(record["filename"]))

    extractor = get_entity_extractor(config.wiki_entity_mode, svc.provider, config)
    entities = extractor.extract(chunks)
    pages = build_wiki(
        entities,
        svc.provider,
        svc.store,
        config,
        extract_concepts=config.wiki_extract_concepts,
    )
    update_wiki_index()
    append_wiki_log(WIKI_LOG_ACTION_BUILD, f"{len(pages)} pages from {len(entities)} records")
    return {
        "paths": [str(p) for p in pages],
        "entities": len(entities),
        "count": len(pages),
    }


class WikiSynthesizeSummary(TypedDict):
    """Result of running synthesis-page generation."""

    paths: list[str]
    count: int


def run_full_synthesize(config: Config | None = None) -> WikiSynthesizeSummary:
    """Generate synthesis pages for cross-source clusters of 3+ documents.

    Shared entry point for MCP ``wiki_synthesize`` and ``POST
    /api/wiki/synthesize``. Concurrency contract matches
    :func:`run_full_build`: not safe to run in parallel with itself or
    with other wiki write paths; callers serialize via an external lock
    on shared event loops.
    """
    if config is None:
        config = cfg
    svc = get_services()
    paths = generate_synthesis_pages(svc.provider, svc.store, svc.clusterer, config)
    return {
        "paths": [str(p) for p in paths],
        "count": len(paths),
    }
