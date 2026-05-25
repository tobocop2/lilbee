"""RAG search pipeline -- embed, search, expand, rerank, generate."""

from __future__ import annotations

import logging
from collections.abc import Generator
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel
from typing_extensions import TypedDict

from lilbee.core.config import Config
from lilbee.core.config.enums import ChatMode
from lilbee.data.store import (
    ChunkType,
    SearchChunk,
    Store,
    cosine_sim,
)
from lilbee.providers.base import LLMProvider
from lilbee.retrieval.embedder import Embedder
from lilbee.retrieval.query.dedup import (
    _greedy_cover,
    _relevance_weight,
    deduplicate_sources,
    filter_results,
    prepare_results,
)
from lilbee.retrieval.query.expansion import EXPANSION_MAX_TOKENS, EXPANSION_PROMPT
from lilbee.retrieval.query.formatting import (
    CONTEXT_TEMPLATE,
    _extract_cited_indices,
    build_context,
    strip_llm_citations,
)
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


class ChatMessage(TypedDict):
    """A single chat message with role and content."""

    role: str
    content: str


class AskResult(BaseModel):
    """Structured result from ask_raw -- answer text + raw search results."""

    answer: str
    sources: list[SearchChunk]


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
        """Call the LLM to produce ``count`` alternative phrasings."""
        prompt = EXPANSION_PROMPT.format(count=count, question=question)
        messages = [{"role": "user", "content": prompt}]
        response = self._provider.chat(
            messages, stream=False, options={"num_predict": EXPANSION_MAX_TOKENS}
        )
        if not isinstance(response, str):
            return []
        variants = [line.strip() for line in response.strip().split("\n") if line.strip()]
        return variants[:count]

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
                    llm_vectors = self._embedder.embed_batch(llm_texts)
                    llm_variants = list(zip(llm_texts, llm_vectors, strict=True))
            llm_variants = self._apply_guardrails(llm_variants, question_vec)

            concept_texts = list(self._concept_query_expansion(question))
            if concept_texts:
                concept_vectors = self._embedder.embed_batch(concept_texts)
                llm_variants.extend(zip(concept_texts, concept_vectors, strict=True))

            return llm_variants
        except Exception as exc:
            log.warning("Query expansion disabled for this call: %s", exc)
            log.debug("Query expansion exception", exc_info=True)
            return []

    def _should_skip_expansion(self, question: str) -> bool:
        if self._config.expansion_skip_threshold <= 0:
            return False
        results = self._store.bm25_probe(question, top_k=2)
        if not results:
            return False
        top_score = results[0].relevance_score or 0
        if top_score < self._config.expansion_skip_threshold:
            return False
        if len(results) < _MIN_BM25_PROBE_RESULTS:
            return True
        second_score = results[1].relevance_score or 0
        return (top_score - second_score) >= self._config.expansion_skip_gap

    def _apply_concept_boost(self, results: list[SearchChunk], question: str) -> list[SearchChunk]:
        if not self._config.concept_graph or not results:
            return results
        try:
            if not self._concepts.get_graph():
                return results
            query_concepts = self._concepts.extract_concepts(question)
            if not query_concepts:
                return results
            return self._concepts.boost_results(results, query_concepts)
        except Exception:
            log.debug("Concept boost failed", exc_info=True)
            return results

    def _hyde_search(self, question: str, top_k: int) -> list[SearchChunk]:
        """Hypothetical Document Embedding search.
        Gao et al. 2022, "Precise Zero-Shot Dense Retrieval without
        Relevance Labels" -- generates a hypothetical answer passage,
        embeds it, and uses the embedding to search for real documents.
        """
        try:
            response = self._provider.chat(
                [{"role": "user", "content": self._config.hyde_prompt.format(question=question)}],
                stream=False,
                options={"num_predict": EXPANSION_MAX_TOKENS},
            )
            if not isinstance(response, str) or not response.strip():
                return []
            hyde_vec = self._embedder.embed(response.strip())
            return self._store.search(hyde_vec, top_k=top_k, query_text=None)
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
        for prefix in ("term:", "vec:", "hyde:", "wiki:", "raw:"):
            if question.strip().lower().startswith(prefix):
                return prefix[:-1], question.strip()[len(prefix) :].strip()
        return None, question

    def _search_structured(
        self,
        mode: str,
        query: str,
        top_k: int,
        chunk_type: ChunkType | None = None,
    ) -> list[SearchChunk]:
        if mode == "term":
            return self._store.bm25_probe(query, top_k=top_k)
        if mode == "vec":
            query_vec = self._embedder.embed(query)
            return self._store.search(query_vec, top_k=top_k, query_text=None)
        if mode == "hyde":
            return self._hyde_search(query, top_k)
        if mode in (ChunkType.WIKI, ChunkType.RAW):
            # Explicit ``chunk_type`` arg beats the ``wiki:``/``raw:`` prefix shortcut.
            effective = chunk_type if chunk_type is not None else ChunkType(mode)
            query_vec = self._embedder.embed(query)
            return self._store.search(
                query_vec, top_k=top_k, query_text=query, chunk_type=effective
            )
        return []

    def select_context(
        self, results: list[SearchChunk], question: str, max_sources: int | None = None
    ) -> list[SearchChunk]:
        """Pick ``max_sources`` chunks by greedy IDF-weighted set cover."""
        if max_sources is None:
            max_sources = self._config.max_context_sources
        if len(results) <= max_sources:
            return results

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
    ) -> None:
        """Append unseen HyDE hits to ``results`` (in place), reweighted by ``hyde_weight``."""
        for r in self._hyde_search(question, top_k):
            key = (r.source, r.chunk_index)
            if key in seen:
                continue
            if r.distance is not None and self._config.hyde_weight > 0:
                r = r.model_copy(update={"distance": r.distance / self._config.hyde_weight})
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
        """
        if top_k == 0:
            top_k = self._config.top_k
        chunk_type = self._normalize_chunk_type(chunk_type)
        mode, clean_query = self._parse_structured_query(question)
        if mode is not None:
            return self._search_structured(mode, clean_query, top_k, chunk_type=chunk_type)
        query_vec = self._embedder.embed(question)
        results = self._store.search(
            query_vec,
            top_k=top_k,
            query_text=question,
            chunk_type=chunk_type,
        )
        if self._should_skip_expansion(question):
            return results[: top_k * 2]
        seen = {(r.source, r.chunk_index) for r in results}
        self._merge_variant_results(question, query_vec, results, seen, top_k, chunk_type)
        if self._config.hyde:
            self._merge_hyde_results(question, results, seen, top_k)
        results = self._apply_concept_boost(results, question)
        return results[: top_k * 2]

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
        results = self.search(question, top_k=top_k, chunk_type=chunk_type)
        results = filter_results(
            results, self._config.max_distance, self._config.min_relevance_score
        )
        if not results:
            return None
        results = prepare_results(results)
        if self._config.reranker_model:
            results = self._reranker.rerank(question, results)
        results = self._apply_temporal_filter(results, question)
        results = self.select_context(results, question)
        context = build_context(results)
        prompt = CONTEXT_TEMPLATE.format(context=context, question=question)
        messages: list[ChatMessage] = [
            {"role": "system", "content": self._config.rag_system_prompt}
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return results, messages

    def _direct_messages(
        self, question: str, history: list[ChatMessage] | None = None
    ) -> list[ChatMessage]:
        """Build messages for direct LLM chat (no RAG context)."""
        messages: list[ChatMessage] = [
            {"role": "system", "content": self._config.general_system_prompt}
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
        messages = self._direct_messages(question, history)
        provider_messages = self._messages_for_provider(messages)
        opts = options if options is not None else self._config.generation_options()
        raw = str(self._provider.chat(provider_messages, options=opts or None) or "")
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
        """Ask a question. Skips retrieval when chat_mode is 'chat' or
        when no embedding model is configured."""
        if (
            self._config.chat_mode == ChatMode.CHAT.value
            or not self._embedder.embedding_available()
        ):
            return AskResult(answer=self._direct_chat(question, history, options), sources=[])
        rag = self.build_rag_context(question, top_k=top_k, history=history, chunk_type=chunk_type)
        if rag is None:
            return AskResult(answer=self._direct_chat(question, history, options), sources=[])
        results, messages = rag
        provider_messages = self._messages_for_provider(messages)
        opts = options if options is not None else self._config.generation_options()
        raw = str(self._provider.chat(provider_messages, options=opts or None) or "")
        clean = raw if self._config.show_reasoning else strip_reasoning(raw)
        return AskResult(answer=clean, sources=results)

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
        cited = _extract_cited_indices(result.answer)
        used = [result.sources[i - 1] for i in sorted(cited) if 1 <= i <= len(result.sources)]
        answer = strip_llm_citations(result.answer)
        source_list = used if used else result.sources
        citations = deduplicate_sources(source_list)
        return f"{answer}\n\nSources:\n" + "\n".join(citations)

    def _stream_direct(
        self,
        question: str,
        history: list[ChatMessage] | None,
        options: dict[str, Any] | None,
    ) -> Generator[StreamToken, None, None]:
        """Streaming branch with the general system prompt (no RAG context)."""
        messages = self._direct_messages(question, history)
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
        if (
            self._config.chat_mode == ChatMode.CHAT.value
            or not self._embedder.embedding_available()
        ):
            yield from self._stream_direct(question, history, options)
            return

        rag = self.build_rag_context(question, top_k=top_k, history=history, chunk_type=chunk_type)
        if rag is None:
            yield from self._stream_direct(question, history, options)
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
        cited = _extract_cited_indices(full_answer)
        used = [results[i - 1] for i in sorted(cited) if 1 <= i <= len(results)]
        source_list = used if used else results
        citations = deduplicate_sources(source_list)
        yield StreamToken(content="\n\nSources:\n" + "\n".join(citations), is_reasoning=False)
