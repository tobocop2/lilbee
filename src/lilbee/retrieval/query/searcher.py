"""RAG search pipeline -- embed, search, expand, rerank, generate."""

from __future__ import annotations

import logging
import re
from collections.abc import Generator
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from lilbee.core.config import Config
from lilbee.core.config.enums import ChatMode
from lilbee.data.store import (
    ChunkType,
    MemoryKind,
    MemoryRow,
    SearchChunk,
    Store,
    cosine_sim,
    human_recall_predicate,
)
from lilbee.providers.base import LLMProvider
from lilbee.retrieval.embedder import Embedder
from lilbee.retrieval.query.dedup import (
    _greedy_cover,
    _relevance_weight,
    deduplicate_sources,
    filter_results,
    order_by_fusion,
    prepare_results,
)
from lilbee.retrieval.query.expansion import (
    CONDENSE_HISTORY_TURNS,
    CONDENSE_MAX_TOKENS,
    CONDENSE_PROMPT,
    EXPANSION_MAX_TOKENS,
    EXPANSION_PROMPT,
)
from lilbee.retrieval.query.formatting import (
    CONTEXT_TEMPLATE,
    build_context,
    cited_subset,
    strip_llm_citations,
)
from lilbee.retrieval.query.history_window import estimate_text_tokens
from lilbee.retrieval.query.intent import (
    AggregateKind,
    AggregateQuery,
    document_references,
    matches_reference,
    parse_aggregate,
)
from lilbee.retrieval.query.memory import format_memory_block
from lilbee.retrieval.query.tokenize import _idf_weights, _tokenize
from lilbee.retrieval.reasoning import (
    StreamToken,
    cap_events_as_stream_tokens,
    effective_reasoning_cap,
    stream_chat_with_cap,
    strip_reasoning,
)

if TYPE_CHECKING:
    from lilbee.retrieval.concepts import ConceptGraph
    from lilbee.retrieval.reranker import Reranker

log = logging.getLogger(__name__)

# BM25 probe needs at least this many hits to compare top vs. runner-up
# scores for the expansion-skip heuristic.
_MIN_BM25_PROBE_RESULTS = 2

# Substring candidates fetched per document reference before token-exact
# disambiguation picks the unique winner.
_KNOWN_ITEM_CANDIDATES = 50

# Content-based known-item resolution: BM25 hits probed for a reference, and
# the fraction of them one source must own to count as that document. A
# docket-style number lives in a document's text, not its filename, so
# filename matching alone can never resolve it; concentration keeps the
# fallback conservative, since a number cited across many filings spreads.
_KNOWN_ITEM_PROBE_K = 6
_KNOWN_ITEM_PROBE_MAJORITY = 0.75

# Structured-query mode names (the ``mode:`` prefix shortcut). Single source for
# both the prefix parser and the dispatch in ``_search_structured``. "term"/"vec"/
# "hyde" pick a retrieval strategy; "wiki"/"raw" are ChunkType scope shortcuts.
_STRUCTURED_QUERY_MODES = ("term", "vec", "hyde", ChunkType.WIKI.value, ChunkType.RAW.value)

# Leading list markers models prepend to expansion output despite the prompt:
# "1.", "2)", "-", "*", "•".
_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+[.)]\s*|[-*•]\s+)")


def _strip_list_marker(line: str) -> str:
    """Drop a leading list marker from an expansion variant line."""
    return _LIST_MARKER_RE.sub("", line).strip()


# Half-saturation constant for BM25 confidence: a raw score of 5 reads as 0.5,
# 20 as 0.8, 45 as 0.9. A plain sigmoid saturated at ~0.99 for any raw score
# above 5, which made the top-vs-runner-up gap condition unsatisfiable exactly
# when BM25 was most certain, so the skip never fired.
_BM25_HALF_SATURATION = 5.0


def _bm25_confidence(score: float | None) -> float:
    """Squash a raw, unbounded BM25 score into (0, 1) without saturating.

    ``s / (s + k)`` keeps strong scores distinguishable (unlike a sigmoid,
    which flattens everything past ~5 to within 0.01 of 1.0). Absent or
    non-positive scores read as 0, so a missing FTS signal never trips the
    expansion skip.
    """
    if score is None or score <= 0.0:
        return 0.0
    return score / (score + _BM25_HALF_SATURATION)


# RAG mode answer when retrieval finds no usable sources: a grounded refusal
# instead of free-wheeling on the model's parametric knowledge. Users who want
# off-corpus answers can switch to chat mode.
_GROUNDED_REFUSAL = "I couldn't find anything in the indexed documents that answers that."

# Ask/search needs an embedder to ground an answer. When none is loaded, refuse
# with an actionable message rather than hard-failing or silently answering
# ungrounded; chat mode stays available for an off-corpus reply.
SEARCH_NEEDS_EMBEDDER = (
    "Search needs an embedding model to ground answers in your documents. "
    "Add one, or switch to chat mode for an ungrounded reply."
)

# Token reserve for the generated answer when budgeting RAG context to num_ctx.
_ANSWER_RESERVE_TOKENS = 1024
# Approximate token cost of the Context/Question template wrapper.
_CONTEXT_TEMPLATE_TOKENS = 16
# Approximate per-source overhead: the "[i] " marker, the provenance header
# (source path plus page/line span), and the blank-line separator.
_PER_SOURCE_TOKENS = 24


class ChatMessage(TypedDict):
    """A single chat message with role and content."""

    role: str
    content: str


class AskResult(BaseModel):
    """Structured result from ask_raw: answer text, retrieved sources, and the cited subset.

    ``sources`` is the full retrieved/reranked set; ``cited_sources`` is the subset the
    answer actually referenced via [n] markers (empty when it cited nothing), so a JSON
    consumer can tell whether the answer was grounded without re-parsing the text.
    """

    answer: str
    sources: list[SearchChunk]
    cited_sources: list[SearchChunk] = Field(default_factory=list)


class Searcher:
    """RAG search pipeline -- embed, search, expand, rerank, generate.
    All search and answer operations go through this class.
    Constructed with injected dependencies via the Services container.
    """

    def __init__(
        self,
        config: Config,
        provider: LLMProvider,
        store: Store,
        embedder: Embedder,
        reranker: Reranker,
        concepts: ConceptGraph,
    ) -> None:
        self._config = config
        self._provider = provider
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._concepts = concepts

    def _apply_temporal_filter(
        self, results: list[SearchChunk], question: str
    ) -> list[SearchChunk]:
        if not self._config.temporal_filtering:
            return results
        from lilbee.runtime.temporal import detect_temporal, resolve_date_range

        keyword = detect_temporal(question)
        if keyword is None:
            return results
        date_range = resolve_date_range(keyword)
        source_dates = self._store.source_ingested_at_map()
        filtered: list[SearchChunk] = []
        for r in results:
            ingested_at = source_dates.get(r.source, "")
            if not ingested_at:
                filtered.append(r)
                continue
            try:
                doc_date = datetime.fromisoformat(ingested_at)
                if date_range.start <= doc_date <= date_range.end:
                    filtered.append(r)
            except (ValueError, TypeError):
                filtered.append(r)
        return filtered if filtered else results

    def _apply_guardrails(
        self,
        variants: list[tuple[str, list[float]]],
        question_vec: list[float],
    ) -> list[tuple[str, list[float]]]:
        """Drop expansion variants whose embedding drifts too far from the question."""
        if not self._config.expansion_guardrails:
            return variants
        threshold = self._config.expansion_similarity_threshold
        return [(text, vec) for text, vec in variants if cosine_sim(question_vec, vec) >= threshold]

    def _concept_query_expansion(self, question: str) -> list[str]:
        if not self._config.concept_graph:
            return []
        try:
            if not self._concepts.get_graph():
                return []
            return self._concepts.expand_query(question)
        except Exception:
            log.debug("Concept query expansion failed", exc_info=True)
            return []

    def _llm_expand(self, question: str, count: int) -> list[str]:
        """Call the LLM to produce ``count`` alternative phrasings.

        Reasoning is stripped before the line split: a reasoning chat model
        otherwise contributes its deliberation as "variants" that get embedded
        and searched. List numbering is stripped per line since models add it
        despite the prompt, and "1." pollutes the BM25 arm of every variant
        search.
        """
        prompt = EXPANSION_PROMPT.format(count=count, question=question)
        messages = [{"role": "user", "content": prompt}]
        response = self._provider.chat(
            messages, stream=False, options={"num_predict": EXPANSION_MAX_TOKENS}
        )
        text = strip_reasoning(response.text).strip()
        variants = [_strip_list_marker(line.strip()) for line in text.split("\n") if line.strip()]
        return [v for v in variants if v][:count]

    def _expand_query(
        self, question: str, question_vec: list[float]
    ) -> list[tuple[str, list[float]]]:
        """Return ``(variant, variant_vec)`` pairs for downstream search.

        LLM variants run through ``_apply_guardrails``; concept-graph
        variants bypass it since they come from deterministic traversal.
        Embeddings batch per source: one provider round-trip per source.
        """
        count = self._config.query_expansion_count
        if count <= 0 and not self._config.concept_graph:
            return []
        # Short queries skip LLM expansion: BM25/vector signal is already strong
        # and the LLM round-trip dominates latency on small local models.
        # Concept-graph expansion still runs.
        short_threshold = self._config.expansion_short_query_tokens
        skip_llm = short_threshold > 0 and len(_tokenize(question)) <= short_threshold
        try:
            llm_variants: list[tuple[str, list[float]]] = []
            if count > 0 and not skip_llm:
                llm_texts = list(self._llm_expand(question, count))
                if llm_texts:
                    llm_vectors = self._embedder.embed_query_batch(llm_texts)
                    llm_variants = list(zip(llm_texts, llm_vectors, strict=True))
            llm_variants = self._apply_guardrails(llm_variants, question_vec)

            concept_texts = list(self._concept_query_expansion(question))
            if concept_texts:
                concept_vectors = self._embedder.embed_query_batch(concept_texts)
                llm_variants.extend(zip(concept_texts, concept_vectors, strict=True))

            return llm_variants
        except Exception as exc:
            log.warning("Query expansion disabled for this call: %s", exc)
            log.debug("Query expansion exception", exc_info=True)
            return []

    def _should_skip_expansion(self, question: str, chunk_type: ChunkType | None = None) -> bool:
        if self._config.expansion_skip_threshold <= 0:
            return False
        # Probe the same pool the scoped search returns, else a confident hit in
        # the wrong sub-pool could skip expansion the scoped result actually needs.
        results = self._store.bm25_probe(
            question, top_k=_MIN_BM25_PROBE_RESULTS, chunk_type=chunk_type
        )
        if not results:
            return False
        top_raw = results[0].bm25_score or 0.0
        if _bm25_confidence(top_raw) < self._config.expansion_skip_threshold:
            return False
        if len(results) < _MIN_BM25_PROBE_RESULTS:
            return True
        # Relative gap in raw score space: any squash compresses the spread
        # between two strong scores toward zero, so a squashed-space gap test
        # can never fire exactly when the lexical arm is most certain.
        second_raw = results[1].bm25_score or 0.0
        relative_gap = (top_raw - second_raw) / top_raw if top_raw > 0 else 0.0
        return relative_gap >= self._config.expansion_skip_gap

    def _apply_concept_boost(self, results: list[SearchChunk], question: str) -> list[SearchChunk]:
        if not self._config.concept_graph or not results:
            return results
        try:
            if not self._concepts.get_graph():
                return results
            query_concepts = self._concepts.extract_concepts(question)
            if not query_concepts:
                return results
            boosted = self._concepts.boost_results(results, query_concepts)
            # boost_results returns copies with adjusted relevance_score/distance in
            # input order; re-sort so the boost actually re-ranks for callers that
            # consume search() order directly (CLI search, MCP lilbee_search).
            # order_by_fusion normalizes per scoring family so HyDE (distance) hits
            # can't outrank a strong hybrid (RRF) hit purely on scale.
            return order_by_fusion(boosted)
        except Exception:
            log.debug("Concept boost failed", exc_info=True)
            return results

    def _hyde_search(
        self, question: str, top_k: int, chunk_type: ChunkType | None = None
    ) -> list[SearchChunk]:
        """Hypothetical Document Embedding search.
        Gao et al. 2022, "Precise Zero-Shot Dense Retrieval without
        Relevance Labels" -- generates a hypothetical answer passage,
        embeds it, and uses the embedding to search for real documents.

        The passage is deliberately embedded with ``embed_query`` (the query
        instruction), not the document prefix: it stands in for the user's
        query against the doc-prefixed index, staying in the same vector
        space as every other query this searcher issues. Changing that is a
        retrieval-quality experiment for the embedding bench, not a refactor.
        """
        try:
            response = self._provider.chat(
                [{"role": "user", "content": self._config.hyde_prompt.format(question=question)}],
                stream=False,
                options={"num_predict": EXPANSION_MAX_TOKENS},
            )
            # Reasoning models front-load deliberation; embedding it instead
            # of the passage would search for the model's thought process.
            text = strip_reasoning(response.text).strip()
            if not text:
                return []
            hyde_vec = self._embedder.embed_query(text)
            return self._store.search(hyde_vec, top_k=top_k, query_text=None, chunk_type=chunk_type)
        except Exception:
            log.debug("HyDE search failed", exc_info=True)
            return []

    def _normalize_chunk_type(self, chunk_type: ChunkType | None) -> ChunkType | None:
        """Drop ``chunk_type=ChunkType.WIKI`` when wiki generation is disabled.

        With wiki off the chunks table contains only raw rows, so the
        filter would return empty. Logging once keeps the surprise out
        of the user's way while surfacing the misuse in logs.
        """
        if chunk_type == ChunkType.WIKI and not self._config.wiki:
            log.warning(
                "wiki scope requested but wiki is disabled; searching the full pool instead"
            )
            return None
        return chunk_type

    def _parse_structured_query(self, question: str) -> tuple[str | None, str]:
        stripped = question.strip()
        for mode in _STRUCTURED_QUERY_MODES:
            prefix = f"{mode}:"
            if stripped.lower().startswith(prefix):
                return mode, stripped[len(prefix) :].strip()
        return None, question

    def _search_structured(
        self,
        mode: str,
        query: str,
        top_k: int,
        chunk_type: ChunkType | None = None,
    ) -> list[SearchChunk]:
        if mode == "term":
            return self._store.bm25_probe(query, top_k=top_k, chunk_type=chunk_type)
        if mode == "vec":
            query_vec = self._embedder.embed_query(query)
            return self._store.search(
                query_vec, top_k=top_k, query_text=None, chunk_type=chunk_type
            )
        if mode == "hyde":
            return self._hyde_search(query, top_k, chunk_type=chunk_type)
        if mode in (ChunkType.WIKI, ChunkType.RAW):
            # Explicit ``chunk_type`` arg beats the ``wiki:``/``raw:`` prefix shortcut.
            # Route the prefix-derived type through the same wiki-disabled guard the
            # explicit arg gets, so ``wiki:`` doesn't bypass it and search an empty pool.
            requested = chunk_type if chunk_type is not None else ChunkType(mode)
            effective = self._normalize_chunk_type(requested)
            query_vec = self._embedder.embed_query(query)
            return self._store.search(
                query_vec, top_k=top_k, query_text=query, chunk_type=effective
            )
        return []

    def select_context(
        self, results: list[SearchChunk], question: str, max_sources: int | None = None
    ) -> list[SearchChunk]:
        """Pick ``max_sources`` chunks.

        Results carrying ``rerank_score`` keep the cross-encoder order (top
        ``max_sources``); otherwise greedy IDF-weighted set cover.
        """
        if max_sources is None:
            max_sources = self._config.max_context_sources
        if len(results) <= max_sources:
            return results
        if any(r.rerank_score is not None for r in results):
            return results[:max_sources]

        question_terms = set(_tokenize(question))
        if not question_terms:
            return results[:max_sources]

        chunk_tokens = [set(_tokenize(r.chunk)) for r in results]
        term_weights = _idf_weights(question_terms, chunk_tokens)
        if not any(term_weights.values()):
            return results[:max_sources]

        weights = [_relevance_weight(r) for r in results]
        selected = _greedy_cover(chunk_tokens, question_terms, term_weights, max_sources, weights)
        selected.sort()
        return [results[i] for i in selected]

    def _merge_variant_results(
        self,
        question: str,
        query_vec: list[float],
        results: list[SearchChunk],
        seen: set[tuple[str, int]],
        top_k: int,
        chunk_type: ChunkType | None,
    ) -> None:
        """Append unseen variant-search hits to ``results`` (in place)."""
        for variant, variant_vec in self._expand_query(question, query_vec):
            variant_results = self._store.search(
                variant_vec,
                top_k=top_k,
                query_text=variant,
                chunk_type=chunk_type,
            )
            for r in variant_results:
                key = (r.source, r.chunk_index)
                if key not in seen:
                    results.append(r)
                    seen.add(key)

    def _merge_hyde_results(
        self,
        question: str,
        results: list[SearchChunk],
        seen: set[tuple[str, int]],
        top_k: int,
        chunk_type: ChunkType | None = None,
    ) -> None:
        """Append unseen HyDE hits to ``results`` (in place), down-weighted by
        ``hyde_weight`` in canonical score space (a weight of 1.0 trusts HyDE
        hits as much as direct hits; lower discounts them proportionally)."""
        for r in self._hyde_search(question, top_k, chunk_type=chunk_type):
            key = (r.source, r.chunk_index)
            if key in seen:
                continue
            if r.score is not None:
                r = r.model_copy(update={"score": r.score * self._config.hyde_weight})
            results.append(r)
            seen.add(key)

    def search(
        self,
        question: str,
        top_k: int = 0,
        *,
        chunk_type: ChunkType | None = None,
    ) -> list[SearchChunk]:
        """Embed question and search with expansion, HyDE, and concept boost.
        Returns up to top_k*2 candidates for downstream filtering.

        When *chunk_type* is set (``"raw"`` or ``"wiki"``), only chunks of
        that type are returned. An explicit ``chunk_type`` always wins
        over the ``wiki:``/``raw:`` prefix shortcut in *question* so the
        user-facing scope choice has the final say.

        When ``chunk_type="wiki"`` but wiki generation is disabled on the
        config, the filter is normalized to ``None`` (mixed pool) and a
        warning is logged: with wiki off the chunks table has no wiki rows,
        so honouring the filter would silently return zero results.

        A ``mode:`` prefix (``term:``/``vec:``/``hyde:``/``wiki:``/``raw:``)
        forces a single explicit retrieval strategy and so skips expansion and
        concept boost, but the temporal date-range filter still applies -- it is
        a filter, not a re-ranking, and a "recent" query must be honored in any
        mode.
        """
        if top_k == 0:
            top_k = self._config.top_k
        chunk_type = self._normalize_chunk_type(chunk_type)
        mode, clean_query = self._parse_structured_query(question)
        if mode is not None:
            structured = self._search_structured(mode, clean_query, top_k, chunk_type=chunk_type)
            return self._apply_temporal_filter(structured, clean_query)
        query_vec = self._embedder.embed_query(question)
        results = self._store.search(
            query_vec,
            top_k=top_k,
            query_text=question,
            chunk_type=chunk_type,
        )
        # Query expansion (variant + HyDE searches) is skipped for short/term
        # queries, but concept boost is a separate graph re-rank that should still
        # apply -- the early return used to drop it on the skip path.
        if not self._should_skip_expansion(question, chunk_type):
            seen = {(r.source, r.chunk_index) for r in results}
            self._merge_variant_results(question, query_vec, results, seen, top_k, chunk_type)
            if self._config.hyde:
                self._merge_hyde_results(question, results, seen, top_k, chunk_type)
        results = self._apply_concept_boost(results, question)
        # Merged variant/HyDE hits arrive appended, not ranked; every consumer
        # of this method (bare search surfaces included) gets one global order
        # over the canonical score rather than insertion order.
        results = order_by_fusion(results)
        # Apply the date-range filter here so the bare search() path (e.g. /api/search)
        # honors a "recent"/"today" query, matching the chat/ask path.
        results = self._apply_temporal_filter(results, question)
        return results[: top_k * 2]

    def _condense_question(self, question: str, history: list[ChatMessage]) -> str:
        """Rewrite a follow-up into a standalone retrieval query.

        Retrieval sees only the query text; without this, "what about his
        brother?" is embedded and BM25-matched with its pronouns. The
        rewritten form drives retrieval only; the user's original wording
        still reaches the answering prompt. Falls back to the original
        question on any failure or empty rewrite.
        """
        recent = history[-CONDENSE_HISTORY_TURNS:]
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
        prompt = CONDENSE_PROMPT.format(history=transcript, question=question)
        try:
            response = self._provider.chat(
                [{"role": "user", "content": prompt}],
                stream=False,
                options={"num_predict": CONDENSE_MAX_TOKENS},
            )
            rewritten = strip_reasoning(response.text).strip().splitlines()
            first_line = rewritten[0].strip() if rewritten else ""
            if first_line:
                log.debug("Condensed follow-up %r -> %r", question, first_line)
                return first_line
        except Exception:
            log.debug("History condensation failed; using the raw question", exc_info=True)
        return question

    def _known_item_results(self, question: str) -> list[SearchChunk]:
        """Resolve a document named in *question* to its own chunks.

        A question that names a document wants that document, not a ranking:
        similarity search retrieves neighbors of the question's wording,
        which for "summarize survey_214.pdf" is mostly noise. Resolution is
        conservative: only a reference matching exactly one source routes;
        anything ambiguous falls back to topical retrieval. Chunks come back
        in document order with full canonical confidence, since their
        relevance is established by the name match, not by similarity.
        """
        for ref in document_references(question):
            filename = self._resolve_reference_filename(ref)
            if filename is None:
                continue
            chunks = self._store.get_chunks_by_source(filename)
            if not chunks:
                continue
            chunks.sort(key=lambda c: c.chunk_index)
            log.info("Known-item route: %r resolved to %s", ref, filename)
            return [c.model_copy(update={"score": 1.0}) for c in chunks]
        return []

    def _resolve_reference_filename(self, ref: str) -> str | None:
        """The one source *ref* names, or ``None`` when nothing resolves uniquely.

        Filename resolution first: substring search over-matches (a bare
        "482" hits every zero-padded id containing it), so token-exact
        matching disambiguates and only a unique winner routes. A unique
        substring hit still routes for word refs (quoted titles never
        token-match hyphenated filenames) but not numeric ones: "12" inside
        "notes-2012" is the false match token comparison exists to reject.

        When no filename knows the reference, it may be a docket-style number
        living in the document's own text; a BM25 probe resolves it when the
        hits concentrate in a single source.
        """
        candidates = self._store.get_sources(search=ref, limit=_KNOWN_ITEM_CANDIDATES)
        matches = [s for s in candidates if matches_reference(ref, s["filename"])]
        if len(matches) == 1:
            return str(matches[0]["filename"])
        if not matches and len(candidates) == 1 and not ref.strip().isdigit():
            return str(candidates[0]["filename"])
        if matches:
            return None  # several sources genuinely carry the reference
        return self._resolve_reference_by_content(ref)

    def _resolve_reference_by_content(self, ref: str) -> str | None:
        """Resolve *ref* to the single source whose text owns it, if any."""
        hits = self._store.bm25_probe(ref, top_k=_KNOWN_ITEM_PROBE_K)
        if len(hits) < _KNOWN_ITEM_PROBE_K:
            return None
        counts: dict[str, int] = {}
        for hit in hits:
            counts[hit.source] = counts.get(hit.source, 0) + 1
        top_source, owned = max(counts.items(), key=lambda kv: kv[1])
        if owned / len(hits) >= _KNOWN_ITEM_PROBE_MAJORITY:
            return top_source
        return None

    def build_rag_context(
        self,
        question: str,
        top_k: int = 0,
        history: list[ChatMessage] | None = None,
        *,
        chunk_type: ChunkType | None = None,
    ) -> tuple[list[SearchChunk], list[ChatMessage]] | None:
        """Build RAG context from search results.

        ``chunk_type`` restricts the pool to ``"raw"`` or ``"wiki"`` rows;
        ``None`` (default) searches the mixed pool.
        """
        retrieval_query = question
        if history and self._config.history_rewrite:
            retrieval_query = self._condense_question(question, history)
        known_item = (
            self._known_item_results(retrieval_query) if self._config.intent_routing else []
        )
        if known_item:
            # The named document IS the context; ranking and reranking would
            # only reorder or drop parts of it. The budget fit below still
            # trims to the context window, keeping the document's head.
            results = known_item
        else:
            results = self.search(retrieval_query, top_k=top_k, chunk_type=chunk_type)
            results = filter_results(
                results, self._config.max_distance, self._config.min_relevance_score
            )
            if not results:
                return None
            results = prepare_results(results)
            if self._config.reranker_model:
                results = self._reranker.rerank(retrieval_query, results)
            # Temporal filtering already ran inside search(); no need to repeat it here.
            results = self.select_context(results, retrieval_query)
        system = self._system_with_memory(self._config.rag_system_prompt, question)
        results = self._fit_context_budget(results, system, question, history)
        context = build_context(results)
        prompt = CONTEXT_TEMPLATE.format(context=context, question=question)
        messages: list[ChatMessage] = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return results, messages

    def _fit_context_budget(
        self,
        results: list[SearchChunk],
        system: str,
        question: str,
        history: list[ChatMessage] | None,
    ) -> list[SearchChunk]:
        """Drop the lowest-ranked sources until the assembled prompt fits num_ctx.

        ``max_context_sources`` caps by count; this caps by tokens so a
        retrieval-heavy query degrades gracefully instead of erroring with
        CONTEXT_OVERFLOW. The top-ranked source is always kept.
        """
        ctx = self._config.num_ctx or self._config.chat_n_ctx_target
        reserve = (
            estimate_text_tokens(system)
            + estimate_text_tokens(question)
            + sum(estimate_text_tokens(m["content"]) for m in history or [])
            + _ANSWER_RESERVE_TOKENS
            + _CONTEXT_TEMPLATE_TOKENS
        )
        budget = ctx - reserve
        kept: list[SearchChunk] = []
        used = 0
        for r in results:
            cost = estimate_text_tokens(r.chunk) + _PER_SOURCE_TOKENS
            if kept and used + cost > budget:
                break
            kept.append(r)
            used += cost
        if len(kept) < len(results):
            log.info(
                "Kept %d of %d sources to fit the model context window.",
                len(kept),
                len(results),
            )
        return kept

    def _system_with_memory(self, base_prompt: str, question: str) -> str:
        """Append the local-owner memory block to *base_prompt* when memory is enabled."""
        block = self._memory_block(question)
        return f"{base_prompt}\n\n{block}" if block else base_prompt

    def _memory_block(self, question: str) -> str:
        """Recall the local human's preferences and relevant facts as a system block.

        Preferences are always included; facts are recalled by similarity. Empty
        when memory is disabled or nothing matches. MCP agents never reach this path
        (their tools recall explicitly under their own owner).
        """
        if not self._config.memory_enabled:
            return ""
        # The human's answers see their own memories plus any an agent shared.
        owner_predicate = human_recall_predicate()
        preferences = self._store.get_memories(
            owner_predicate=owner_predicate,
            kind=MemoryKind.PREFERENCE,
        )
        facts: list[MemoryRow] = []
        if self._config.memory_top_k > 0 and self._embedder.embedding_available():
            vector = self._embedder.embed_query(question)
            facts = self._store.search_memories(
                vector,
                owner_predicate=owner_predicate,
                top_k=self._config.memory_top_k,
                max_distance=self._config.memory_max_distance,
            )
        return format_memory_block(preferences, facts, self._config.memory_token_budget)

    def _answer_aggregate(self, aggregate: AggregateQuery) -> str:
        """Answer a count-shaped question with an exact full-corpus scan.

        Top-k retrieval sees a handful of chunks out of the whole corpus, so
        it structurally cannot count; the faithful-but-useless outcome is a
        model hedging that "the context does not provide precise counts".
        Counting is a scan, and a scan needs no language model: the numbers
        below are exact, not generated.
        """
        if aggregate.kind is AggregateKind.TOTAL_SOURCES:
            sources = self._store.count_sources()
            chunks = self._store.count_chunks()
            return f"The index holds {sources} documents split into {chunks} searchable passages."
        if aggregate.kind is AggregateKind.TERM_MENTIONS:
            chunk_hits, source_hits = self._store.count_term_mentions(aggregate.term)
            return (
                f"Exact scan of the whole index: {source_hits} documents mention "
                f"{aggregate.term!r}, across {chunk_hits} passages. This counts literal "
                f"mentions of the phrase, not paraphrases."
            )
        return (
            "Answering that count needs structured records (dates, identifiers, or "
            "entities) that aren't extracted from this corpus yet. I can count "
            "documents or passages that mention a specific term, or you can ask for "
            "the passages themselves and count from those."
        )

    def route_direct_answer(self, question: str) -> str | None:
        """The exact-scan answer for a count-shaped question, else ``None``.

        Every retrieval entry point must consult this before building RAG
        context: ask_raw/ask_stream do (covering CLI and TUI), and the HTTP
        handlers call it themselves because they assemble their own prompts
        from build_rag_context. An entry point that skips it hedges at the
        count questions every other surface answers exactly.
        """
        if not self._config.intent_routing:
            return None
        aggregate = parse_aggregate(question)
        if aggregate is None:
            return None
        log.info("Aggregate route: %s for %r", aggregate.kind.value, question)
        return self._answer_aggregate(aggregate)

    def skip_retrieval(self) -> bool:
        """Whether this turn should bypass RAG: chat-only mode or no embedder."""
        return (
            self._config.chat_mode == ChatMode.CHAT.value
            or not self._embedder.embedding_available()
        )

    def search_unavailable(self) -> bool:
        """Search mode is active but retrieval can't run because no embedder is loaded.

        Ask refuses cleanly in this state (best UX: tell the user search needs an
        embedder) rather than silently answering ungrounded. Chat mode is exempt --
        it intentionally answers off-corpus, so it falls back instead of refusing.
        """
        return (
            self._config.chat_mode != ChatMode.CHAT.value
            and not self._embedder.embedding_available()
        )

    def direct_messages(
        self, question: str, history: list[ChatMessage] | None = None
    ) -> list[ChatMessage]:
        """Build messages for direct LLM chat (no RAG context)."""
        messages: list[ChatMessage] = [
            {
                "role": "system",
                "content": self._system_with_memory(self._config.general_system_prompt, question),
            }
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})
        return messages

    def _messages_for_provider(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        """Convert ChatMessage list to provider-expected format."""
        return [{"role": m["role"], "content": m["content"]} for m in messages]

    def _direct_chat(
        self,
        question: str,
        history: list[ChatMessage] | None,
        options: dict[str, Any] | None,
    ) -> str:
        """Run a no-RAG chat turn and return the cleaned response."""
        messages = self.direct_messages(question, history)
        provider_messages = self._messages_for_provider(messages)
        opts = options if options is not None else self._config.generation_options()
        result = self._provider.chat(provider_messages, options=opts or None)
        raw = result.text
        return raw if self._config.show_reasoning else strip_reasoning(raw)

    def ask_raw(
        self,
        question: str,
        top_k: int = 0,
        history: list[ChatMessage] | None = None,
        options: dict[str, Any] | None = None,
        *,
        chunk_type: ChunkType | None = None,
    ) -> AskResult:
        """Ask a question. Refuses cleanly without an embedder (search can't
        ground); falls back to direct chat only when chat_mode is 'chat'."""
        if self.search_unavailable():
            return AskResult(answer=SEARCH_NEEDS_EMBEDDER, sources=[])
        if self.skip_retrieval():
            return AskResult(answer=self._direct_chat(question, history, options), sources=[])
        aggregate_answer = self.route_direct_answer(question)
        if aggregate_answer is not None:
            return AskResult(answer=aggregate_answer, sources=[])
        rag = self.build_rag_context(question, top_k=top_k, history=history, chunk_type=chunk_type)
        if rag is None:
            return AskResult(answer=_GROUNDED_REFUSAL, sources=[])
        results, messages = rag
        provider_messages = self._messages_for_provider(messages)
        opts = options if options is not None else self._config.generation_options()
        result = self._provider.chat(provider_messages, options=opts or None)
        raw = result.text
        clean = raw if self._config.show_reasoning else strip_reasoning(raw)
        return AskResult(answer=clean, sources=results, cited_sources=cited_subset(clean, results))

    def ask(
        self,
        question: str,
        top_k: int = 0,
        history: list[ChatMessage] | None = None,
        options: dict[str, Any] | None = None,
        *,
        chunk_type: ChunkType | None = None,
    ) -> str:
        """Ask a question and get a formatted answer with citations."""
        result = self.ask_raw(
            question, top_k=top_k, history=history, options=options, chunk_type=chunk_type
        )
        if not result.sources:
            return result.answer
        answer = strip_llm_citations(result.answer)
        source_list = result.cited_sources if result.cited_sources else result.sources
        citations = deduplicate_sources(source_list)
        return f"{answer}\n\nSources:\n" + "\n".join(citations)

    def _stream_direct(
        self,
        question: str,
        history: list[ChatMessage] | None,
        options: dict[str, Any] | None,
    ) -> Generator[StreamToken, None, None]:
        """Streaming branch with the general system prompt (no RAG context)."""
        messages = self.direct_messages(question, history)
        provider_messages = self._messages_for_provider(messages)
        opts = options if options is not None else self._config.generation_options()
        events = stream_chat_with_cap(
            self._provider,
            cast("list[dict[str, Any]]", provider_messages),
            options=opts,
            model=self._config.chat_model,
            show_reasoning=self._config.show_reasoning,
            cap_chars=effective_reasoning_cap(),
        )
        try:
            yield from cap_events_as_stream_tokens(events)
        except (ConnectionError, OSError) as exc:
            yield StreamToken(content=f"\n\n[Connection lost: {exc}]", is_reasoning=False)

    def ask_stream(
        self,
        question: str,
        top_k: int = 0,
        history: list[ChatMessage] | None = None,
        options: dict[str, Any] | None = None,
        *,
        chunk_type: ChunkType | None = None,
    ) -> Generator[StreamToken, None, None]:
        """Stream answer tokens with citations appended at the end."""
        if self.search_unavailable():
            yield StreamToken(content=SEARCH_NEEDS_EMBEDDER, is_reasoning=False)
            return
        if self.skip_retrieval():
            yield from self._stream_direct(question, history, options)
            return
        aggregate_answer = self.route_direct_answer(question)
        if aggregate_answer is not None:
            yield StreamToken(content=aggregate_answer, is_reasoning=False)
            return

        rag = self.build_rag_context(question, top_k=top_k, history=history, chunk_type=chunk_type)
        if rag is None:
            yield StreamToken(content=_GROUNDED_REFUSAL, is_reasoning=False)
            return
        results, messages = rag
        provider_messages = self._messages_for_provider(messages)
        opts = options if options is not None else self._config.generation_options()
        answer_parts: list[str] = []
        events = stream_chat_with_cap(
            self._provider,
            cast("list[dict[str, Any]]", provider_messages),
            options=opts,
            model=self._config.chat_model,
            show_reasoning=self._config.show_reasoning,
            cap_chars=effective_reasoning_cap(),
        )
        try:
            for token in cap_events_as_stream_tokens(events):
                if not token.is_reasoning:
                    answer_parts.append(token.content)
                yield token
        except (ConnectionError, OSError) as exc:
            yield StreamToken(content=f"\n\n[Connection lost: {exc}]", is_reasoning=False)
        # LLM-generated citation blocks in streamed tokens cannot be
        # retroactively stripped. The system prompt discourages them; this
        # only filters the code-appended Sources block to cited chunks.
        full_answer = "".join(answer_parts)
        used = cited_subset(full_answer, results)
        source_list = used if used else results
        citations = deduplicate_sources(source_list)
        yield StreamToken(content="\n\nSources:\n" + "\n".join(citations), is_reasoning=False)
