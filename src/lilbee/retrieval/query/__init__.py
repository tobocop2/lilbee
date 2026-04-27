"""RAG query pipeline -- embed question, search, generate answer with citations."""

from __future__ import annotations

from lilbee.retrieval.query.dedup import (
    _relevance_weight,
    deduplicate_sources,
    diversify_sources,
    filter_results,
    prepare_results,
    sort_by_relevance,
)
from lilbee.retrieval.query.formatting import (
    CONTEXT_TEMPLATE,
    _extract_cited_indices,
    _format_citation,
    build_context,
    display_source_path,
    format_source,
    strip_llm_citations,
)
from lilbee.retrieval.query.searcher import AskResult, ChatMessage, Searcher

__all__ = [
    "CONTEXT_TEMPLATE",
    "AskResult",
    "ChatMessage",
    "Searcher",
    "_extract_cited_indices",
    "_format_citation",
    "_relevance_weight",
    "build_context",
    "deduplicate_sources",
    "display_source_path",
    "diversify_sources",
    "filter_results",
    "format_source",
    "prepare_results",
    "sort_by_relevance",
    "strip_llm_citations",
]
