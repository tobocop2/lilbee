"""RAG query pipeline -- embed question, search, generate answer with citations."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from lilbee.reasoning import StreamToken

from pydantic import BaseModel
from typing_extensions import TypedDict

from lilbee.config import Config, cfg
from lilbee.embedder import Embedder
from lilbee.providers.base import LLMProvider
from lilbee.store import SearchChunk, Store


class ChatMessage(TypedDict):
    """A single chat message with role and content."""

    role: str
    content: str


CONTEXT_TEMPLATE = """Context:
{context}

Question: {question}"""


def format_source(result: SearchChunk) -> str:
    """Format a search result as a source citation line."""
    if result.content_type == "pdf":
        ps, pe = result.page_start, result.page_end
        pages = f"page {ps}" if ps == pe else f"pages {ps}-{pe}"
        return f"  → {result.source}, {pages}"

    if result.content_type == "code":
        ls, le = result.line_start, result.line_end
        lines = f"line {ls}" if ls == le else f"lines {ls}-{le}"
        return f"  → {result.source}, {lines}"

    return f"  → {result.source}"


def deduplicate_sources(results: list[SearchChunk], max_citations: int = 5) -> list[str]:
    """Merge results from same source into deduplicated citation lines."""
    seen: set[str] = set()
    citations: list[str] = []
    for r in results:
        line = format_source(r)
        if line not in seen:
            seen.add(line)
            citations.append(line)
            if len(citations) >= max_citations:
                break
    return citations


def _sort_key(r: SearchChunk) -> float:
    """Sort key: lower = more relevant."""
    if r.relevance_score is not None:
        return -r.relevance_score
    if r.distance is not None:
        return r.distance
    return float("inf")


def sort_by_relevance(results: list[SearchChunk]) -> list[SearchChunk]:
    """Sort search results by relevance (works for both hybrid and vector results)."""
    return sorted(results, key=_sort_key)


def diversify_sources(
    results: list[SearchChunk], max_per_source: int | None = None
) -> list[SearchChunk]:
    """Cap results per source document to ensure diversity.

    Source diversity filtering: Zhai 2008, "Statistical Language Models for
    Information Retrieval" -- caps per-source representation to prevent
    any single document from dominating results.
    """
    if max_per_source is None:
        max_per_source = cfg.diversity_max_per_source
    counts: dict[str, int] = {}
    diverse: list[SearchChunk] = []
    for r in results:
        count = counts.get(r.source, 0)
        if count < max_per_source:
            diverse.append(r)
            counts[r.source] = count + 1
    return diverse


def prepare_results(results: list[SearchChunk]) -> list[SearchChunk]:
    """Sort by relevance and apply source diversity cap."""
    return diversify_sources(sort_by_relevance(results))


def build_context(results: list[SearchChunk]) -> str:
    """Build context block from search results."""
    parts: list[str] = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r.chunk}")
    return "\n\n".join(parts)


_EXPANSION_PROMPT = (
    "Generate {count} alternative search queries for the following question. "
    "Return ONLY the queries, one per line, no numbering or explanation.\n\n"
    "Question: {question}"
)

_EXPANSION_MAX_TOKENS = 200


_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "about",
        "between",
        "through",
        "after",
        "before",
        "above",
        "below",
        "and",
        "or",
        "but",
        "not",
        "no",
        "if",
        "then",
        "than",
        "that",
        "this",
        "it",
        "its",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "when",
        "where",
        "why",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "they",
    }
)

_MIN_OVERLAP_RATIO = 0.3


def _tokenize_query(text: str) -> set[str]:
    """Tokenize into lowercase words, removing stop words."""
    return {w for w in text.lower().split() if w not in _STOP_WORDS and len(w) > 1}


_DEFAULT_HYDE_PROMPT = (
    "Write a 50-100 word passage that directly answers this question as if "
    "it were an excerpt from a real document. Do not include any preamble, "
    "just write the passage.\n\nQuestion: {question}"
)


class AskResult(BaseModel):
    """Structured result from ask_raw -- answer text + raw search results."""

    answer: str
    sources: list[SearchChunk]


class Searcher:
    """RAG search pipeline -- owns config, provider, store, and embedder.

    All query operations go through this class.
    """

    def __init__(
        self,
        config: Config,
        provider: LLMProvider,
        store: Store,
        embedder: Embedder,
    ) -> None:
        self._config = config
        self._provider = provider
        self._store = store
        self._embedder = embedder

    def _apply_temporal_filter(
        self, results: list[SearchChunk], question: str
    ) -> list[SearchChunk]:
        if not self._config.temporal_filtering:
            return results
        from lilbee.temporal import detect_temporal, resolve_date_range

        keyword = detect_temporal(question)
        if keyword is None:
            return results
        date_range = resolve_date_range(keyword)
        source_dates = {s["filename"]: s.get("ingested_at", "") for s in self._store.get_sources()}
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

    def _apply_guardrails(self, variants: list[str], question: str) -> list[str]:
        if not self._config.expansion_guardrails:
            return variants
        original_tokens = _tokenize_query(question)
        if not original_tokens:
            return variants
        validated: list[str] = []
        for variant in variants:
            variant_tokens = _tokenize_query(variant)
            if not variant_tokens:
                continue
            overlap = len(original_tokens & variant_tokens) / len(original_tokens)
            if overlap >= _MIN_OVERLAP_RATIO:
                validated.append(variant)
        return validated

    def _concept_query_expansion(self, question: str) -> list[str]:
        if not self._config.concept_graph:
            return []
        try:
            from lilbee.concepts import ConceptGraph

            cg = ConceptGraph(self._config, self._store)
            if not cg.get_graph():
                return []
            return cg.expand_query(question)
        except Exception:
            return []

    def _expand_query(self, question: str) -> list[str]:
        count = self._config.query_expansion_count
        if count == 0:
            return []
        try:
            prompt = _EXPANSION_PROMPT.format(count=count, question=question)
            messages = [{"role": "user", "content": prompt}]
            response = self._provider.chat(
                messages, stream=False, options={"num_predict": _EXPANSION_MAX_TOKENS}
            )
            if not isinstance(response, str):
                return []
            variants = [line.strip() for line in response.strip().split("\n") if line.strip()]
            variants = variants[:count]
            variants = self._apply_guardrails(variants, question)
            variants.extend(self._concept_query_expansion(question))
            return variants
        except Exception:
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
        if len(results) < 2:
            return True
        second_score = results[1].relevance_score or 0
        return (top_score - second_score) >= self._config.expansion_skip_gap

    def _apply_concept_boost(self, results: list[SearchChunk], question: str) -> list[SearchChunk]:
        if not self._config.concept_graph:
            return results
        try:
            from lilbee.concepts import ConceptGraph

            cg = ConceptGraph(self._config, self._store)
            if not cg.get_graph():
                return results
            query_concepts = cg.extract_concepts(question)
            return cg.boost_results(results, query_concepts)
        except Exception:
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
                options={"num_predict": _EXPANSION_MAX_TOKENS},
            )
            if not isinstance(response, str) or not response.strip():
                return []
            hyde_vec = self._embedder.embed(response.strip())
            return self._store.search(hyde_vec, top_k=top_k, query_text=None)
        except Exception:
            return []

    def _parse_structured_query(self, question: str) -> tuple[str | None, str]:
        for prefix in ("term:", "vec:", "hyde:"):
            if question.strip().lower().startswith(prefix):
                return prefix[:-1], question.strip()[len(prefix) :].strip()
        return None, question

    def _search_structured(self, mode: str, query: str, top_k: int) -> list[SearchChunk]:
        if mode == "term":
            return self._store.bm25_probe(query, top_k=top_k)
        if mode == "vec":
            query_vec = self._embedder.embed(query)
            return self._store.search(query_vec, top_k=top_k, query_text=None)
        if mode == "hyde":
            return self._hyde_search(query, top_k)
        return []

    def select_context(
        self, results: list[SearchChunk], question: str, max_sources: int | None = None
    ) -> list[SearchChunk]:
        if max_sources is None:
            max_sources = self._config.max_context_sources
        if len(results) <= max_sources:
            return results
        query_terms = _tokenize_query(question)
        if not query_terms:
            return results[:max_sources]
        selected: list[SearchChunk] = []
        covered: set[str] = set()
        remaining = list(results)
        for _ in range(max_sources):
            if not remaining or covered == query_terms:
                break
            best_idx = 0
            best_gain = -1
            for i, chunk in enumerate(remaining):
                chunk_terms = _tokenize_query(chunk.chunk)
                gain = len((chunk_terms & query_terms) - covered)
                if gain > best_gain or (gain == best_gain and i < best_idx):
                    best_gain = gain
                    best_idx = i
            if best_gain <= 0 and selected:
                break
            chosen = remaining.pop(best_idx)
            selected.append(chosen)
            covered |= _tokenize_query(chosen.chunk) & query_terms
        return selected

    def search(self, question: str, top_k: int = 0) -> list[SearchChunk]:
        """Search with expansion and reranking.

        Returns up to top_k*2 candidates for downstream filtering.
        """
        if top_k == 0:
            top_k = self._config.top_k
        mode, clean_query = self._parse_structured_query(question)
        if mode is not None:
            return self._search_structured(mode, clean_query, top_k)
        query_vec = self._embedder.embed(question)
        results = self._store.search(query_vec, top_k=top_k, query_text=question)
        if self._should_skip_expansion(question):
            return results[: top_k * 2]
        seen = {(r.source, r.chunk_index) for r in results}
        variants = self._expand_query(question)
        if variants:
            for variant in variants:
                variant_vec = self._embedder.embed(variant)
                variant_results = self._store.search(variant_vec, top_k=top_k, query_text=variant)
                for r in variant_results:
                    key = (r.source, r.chunk_index)
                    if key not in seen:
                        results.append(r)
                        seen.add(key)
        if self._config.hyde:
            hyde_results = self._hyde_search(question, top_k)
            for r in hyde_results:
                key = (r.source, r.chunk_index)
                if key not in seen:
                    if r.distance is not None and self._config.hyde_weight > 0:
                        r = r.model_copy(update={"distance": r.distance / self._config.hyde_weight})
                    results.append(r)
                    seen.add(key)
        results = self._apply_concept_boost(results, question)
        return results[: top_k * 2]

    def build_rag_context(
        self,
        question: str,
        top_k: int = 0,
        history: list[ChatMessage] | None = None,
    ) -> tuple[list[SearchChunk], list[ChatMessage]] | None:
        results = self.search(question, top_k=top_k)
        if not results:
            return None
        results = prepare_results(results)
        if self._config.reranker_model:
            from lilbee.reranker import Reranker

            reranker = Reranker(self._config)
            results = reranker.rerank(question, results)
        results = self._apply_temporal_filter(results, question)
        results = self.select_context(results, question)
        context = build_context(results)
        prompt = CONTEXT_TEMPLATE.format(context=context, question=question)
        messages: list[ChatMessage] = [{"role": "system", "content": self._config.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return results, messages

    def ask_raw(
        self,
        question: str,
        top_k: int = 0,
        history: list[ChatMessage] | None = None,
        options: dict[str, Any] | None = None,
    ) -> AskResult:
        rag = self.build_rag_context(question, top_k=top_k, history=history)
        if rag is None:
            return AskResult(
                answer="No relevant documents found. Try ingesting some documents first.",
                sources=[],
            )
        results, messages = rag
        opts = options if options is not None else self._config.generation_options()
        answer = self._provider.chat(cast(list[dict[str, Any]], messages), options=opts or None)
        return AskResult(answer=str(answer) or "", sources=results)

    def ask(
        self,
        question: str,
        top_k: int = 0,
        history: list[ChatMessage] | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        result = self.ask_raw(question, top_k=top_k, history=history, options=options)
        if not result.sources:
            return result.answer
        citations = deduplicate_sources(result.sources)
        return f"{result.answer}\n\nSources:\n" + "\n".join(citations)

    def ask_stream(
        self,
        question: str,
        top_k: int = 0,
        history: list[ChatMessage] | None = None,
        options: dict[str, Any] | None = None,
    ) -> Generator[StreamToken, None, None]:
        from lilbee.reasoning import StreamToken, filter_reasoning

        rag = self.build_rag_context(question, top_k=top_k, history=history)
        if rag is None:
            yield StreamToken(
                content="No relevant documents found. Try ingesting some documents first.",
                is_reasoning=False,
            )
            return
        results, messages = rag
        opts = options if options is not None else self._config.generation_options()
        raw_stream = self._provider.chat(
            cast(list[dict[str, Any]], messages), stream=True, options=opts or None
        )
        try:
            for st in filter_reasoning(
                cast(Iterator[str], raw_stream), show=self._config.show_reasoning
            ):
                if st.content:
                    yield st
        except (ConnectionError, OSError) as exc:
            yield StreamToken(content=f"\n\n[Connection lost: {exc}]", is_reasoning=False)
        citations = deduplicate_sources(results)
        yield StreamToken(content="\n\nSources:\n" + "\n".join(citations), is_reasoning=False)


# ---------------------------------------------------------------------------
# Module-level convenience API -- delegates through runtime singleton
# ---------------------------------------------------------------------------


def get_provider() -> LLMProvider:
    """Return the runtime LLM provider singleton."""
    from lilbee.runtime import get_provider as _get_provider

    return _get_provider()


def search_context(question: str, top_k: int = 0) -> list[SearchChunk]:
    """Search with expansion and reranking (convenience wrapper)."""
    from lilbee.runtime import get_searcher

    return get_searcher().search(question, top_k=top_k)


def build_rag_context(
    question: str,
    top_k: int = 0,
    history: list[ChatMessage] | None = None,
) -> tuple[list[SearchChunk], list[ChatMessage]] | None:
    from lilbee.runtime import get_searcher

    return get_searcher().build_rag_context(question, top_k=top_k, history=history)


def ask_raw(
    question: str,
    top_k: int = 0,
    history: list[ChatMessage] | None = None,
    options: dict[str, Any] | None = None,
) -> AskResult:
    from lilbee.runtime import get_searcher

    return get_searcher().ask_raw(question, top_k=top_k, history=history, options=options)


def ask(
    question: str,
    top_k: int = 0,
    history: list[ChatMessage] | None = None,
    options: dict[str, Any] | None = None,
) -> str:
    from lilbee.runtime import get_searcher

    return get_searcher().ask(question, top_k=top_k, history=history, options=options)


def ask_stream(
    question: str,
    top_k: int = 0,
    history: list[ChatMessage] | None = None,
    options: dict[str, Any] | None = None,
) -> Generator[StreamToken, None, None]:
    from lilbee.runtime import get_searcher

    return get_searcher().ask_stream(question, top_k=top_k, history=history, options=options)


def select_context(
    results: list[SearchChunk], question: str, max_sources: int | None = None
) -> list[SearchChunk]:
    from lilbee.runtime import get_searcher

    return get_searcher().select_context(results, question, max_sources=max_sources)


def _expand_query(question: str) -> list[str]:
    from lilbee.runtime import get_searcher

    return get_searcher()._expand_query(question)


def _should_skip_expansion(question: str) -> bool:
    from lilbee.runtime import get_searcher

    return get_searcher()._should_skip_expansion(question)


def _apply_guardrails(variants: list[str], question: str) -> list[str]:
    from lilbee.runtime import get_searcher

    return get_searcher()._apply_guardrails(variants, question)


def _hyde_search(question: str, top_k: int) -> list[SearchChunk]:
    from lilbee.runtime import get_searcher

    return get_searcher()._hyde_search(question, top_k)


def _apply_temporal_filter(results: list[SearchChunk], question: str) -> list[SearchChunk]:
    from lilbee.runtime import get_searcher

    return get_searcher()._apply_temporal_filter(results, question)
