"""RAG query pipeline. Embed question, search, generate answer with citations."""

from __future__ import annotations

from lilbee.retrieval.query.dedup import (
    diversify_sources,
    filter_results,
    prepare_results,
    sort_by_relevance,
)
from lilbee.retrieval.query.formatting import (
    SOURCES_BLOCK_MARKER,
    build_context,
    display_source_path,
    format_source,
    format_sources_block,
    strip_llm_citations,
)
from lilbee.retrieval.query.searcher import AskResult, ChatMessage, Searcher

__all__ = [
    "SOURCES_BLOCK_MARKER",
    "AskResult",
    "ChatMessage",
    "Searcher",
    "build_context",
    "display_source_path",
    "diversify_sources",
    "filter_results",
    "format_source",
    "format_sources_block",
    "prepare_results",
    "sort_by_relevance",
    "strip_llm_citations",
]
