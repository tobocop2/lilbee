"""RAG query pipeline -- embed question, search, generate answer with citations."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Generator, Iterator
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from lilbee.concepts import ConceptGraph
    from lilbee.reasoning import StreamToken
    from lilbee.reranker import Reranker

from pydantic import BaseModel
from typing_extensions import TypedDict

from lilbee.config import Config, cfg
from lilbee.embedder import Embedder
from lilbee.providers.base import LLMProvider
from lilbee.store import CitationRecord, SearchChunk, Store, cosine_sim

log = logging.getLogger(__name__)

# Minimum token length — single characters carry no selection signal.
_MIN_TOKEN_LEN = 2

# Any run of non-alphanumeric characters (whitespace, punctuation,
# hyphens) is treated as a word boundary so ``"state-of-the-art"``
# becomes four tokens rather than collapsing into one.
_TOKEN_SPLIT_RE = re.compile(r"\W+")


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens for IDF-weighted context selection."""
    return [word for word in _TOKEN_SPLIT_RE.split(text.lower()) if len(word) >= _MIN_TOKEN_LEN]


def _idf_weights(
    question_terms: set[str],
    chunk_tokens: list[set[str]],
) -> dict[str, float]:
    """Return IDF weight per question term over the candidate corpus.

    ``log(N / (1 + df))`` is clamped to zero for terms that appear in
    (almost) every candidate chunk — the corpus-specific stopwords that
    a hand-curated English list would otherwise approximate. The only
    caller is ``select_context``, which short-circuits on empty
    ``results``, so ``chunk_tokens`` is guaranteed non-empty here.
    """
    n = len(chunk_tokens)
    df: dict[str, int] = {}
    for tokens in chunk_tokens:
        for term in tokens & question_terms:
            df[term] = df.get(term, 0) + 1
    return {t: max(0.0, math.log(n / (1 + df.get(t, 0)))) for t in question_terms}


def _greedy_cover(
    chunk_tokens: list[set[str]],
    question_terms: set[str],
    term_weights: dict[str, float],
    budget: int,
) -> list[int]:
    """Greedy IDF-weighted set cover over candidate chunks.

    Picks chunks whose distinctive (high-IDF) query terms contribute
    the most uncovered weight until either the budget is exhausted or
    no chunk can add anything new. Any remaining budget is filled in
    retrieval order so callers always receive ``budget`` chunks when
    that many are available.
    """
    selected: list[int] = []
    covered: set[str] = set()
    remaining = list(range(len(chunk_tokens)))
    while remaining and len(selected) < budget:
        best_pos = -1
        best_gain = 0.0
        for pos, idx in enumerate(remaining):
            new_terms = (chunk_tokens[idx] & question_terms) - covered
            gain = sum(term_weights[t] for t in new_terms)
            if gain > best_gain:
                best_gain = gain
                best_pos = pos
        if best_pos < 0:
            break
        chosen = remaining.pop(best_pos)
        selected.append(chosen)
        covered |= chunk_tokens[chosen] & question_terms

    for idx in remaining:
        if len(selected) >= budget:
            break
        selected.append(idx)
    return selected


class ChatMessage(TypedDict):
    """A single chat message with role and content."""

    role: str
    content: str


CONTEXT_TEMPLATE = """Context:
{context}

Question: {question}"""


def _format_citation(citation: CitationRecord) -> str:
    """Format a single citation record as an indented attribution line."""
    if citation["page_start"] or citation["page_end"]:
        ps, pe = citation["page_start"], citation["page_end"]
        pages = f"page {ps}" if ps == pe else f"pages {ps}-{pe}"
        return f"    → {citation['source_filename']}, {pages}"
    if citation["line_start"] or citation["line_end"]:
        ls, le = citation["line_start"], citation["line_end"]
        lines = f"line {ls}" if ls == le else f"lines {ls}-{le}"
        return f"    → {citation['source_filename']}, {lines}"
    return f"    → {citation['source_filename']}"


def format_source(result: SearchChunk, citations: list[CitationRecord] | None = None) -> str:
    """Format a search result as a source citation line.
    For wiki chunks, shows the wiki page path followed by indented transitive citations.
    """
    if result.chunk_type == "wiki" and citations:
        parts = [f"  → {result.source}"]
        for cit in citations:
            parts.append(_format_citation(cit))
        return "\n".join(parts)

    if result.content_type == "pdf":
        ps, pe = result.page_start, result.page_end
        pages = f"page {ps}" if ps == pe else f"pages {ps}-{pe}"
        return f"  → {result.source}, {pages}"

    if result.content_type == "code":
        ls, le = result.line_start, result.line_end
        lines = f"line {ls}" if ls == le else f"lines {ls}-{le}"
        return f"  → {result.source}, {lines}"

    return f"  → {result.source}"


def _source_slug(source_name: str) -> str:
    """Derive the wiki filename stem from a raw source name.
    Mirrors the slug logic in gen.py: "subdir/doc.md" -> "subdir--doc".
    """
    return source_name.replace("/", "--").rsplit(".", 1)[0]


def _wiki_covered_raw_sources(results: list[SearchChunk]) -> set[str]:
    """Build a set of raw source names that have wiki coverage.
    Wiki chunks have sources like "wiki/summaries/subdir--doc.md" while raw
    chunks have sources like "subdir/doc.md". Match by comparing the wiki
    file stem against the slug derived from the raw source name.
    """
    wiki_stems: set[str] = set()
    for r in results:
        if r.chunk_type == "wiki":
            # "wiki/summaries/subdir--doc.md" -> "subdir--doc"
            filename = r.source.rsplit("/", 1)[-1]
            wiki_stems.add(filename.rsplit(".", 1)[0])
    if not wiki_stems:
        return set()
    raw_covered: set[str] = set()
    for r in results:
        if r.chunk_type != "wiki" and _source_slug(r.source) in wiki_stems:
            raw_covered.add(r.source)
    return raw_covered


def prefer_wiki(results: list[SearchChunk]) -> list[SearchChunk]:
    """When both wiki and raw chunks exist for the same source, prefer wiki."""
    covered = _wiki_covered_raw_sources(results)
    if not covered:
        return results
    return [r for r in results if r.chunk_type == "wiki" or r.source not in covered]


def deduplicate_sources(
    results: list[SearchChunk],
    max_citations: int = 5,
    citations_map: dict[str, list[CitationRecord]] | None = None,
) -> list[str]:
    """Merge results from same source into deduplicated citation lines."""
    seen: set[str] = set()
    citation_lines: list[str] = []
    for r in results:
        cits = (citations_map or {}).get(r.source)
        line = format_source(r, citations=cits)
        if line not in seen:
            seen.add(line)
            citation_lines.append(line)
            if len(citation_lines) >= max_citations:
                break
    return citation_lines


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
    return "\n\n".join(f"[{i}] {r.chunk}" for i, r in enumerate(results, 1))


_EXPANSION_PROMPT = (
    "Generate {count} alternative search queries for the following question. "
    "Return ONLY the queries, one per line, no numbering or explanation.\n\n"
    "Question: {question}"
)

_EXPANSION_MAX_TOKENS = 200


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

    def _apply_guardrails(
        self,
        variants: list[tuple[str, list[float]]],
        question_vec: list[float],
    ) -> list[tuple[str, list[float]]]:
        """Drop expansion variants that drift too far from the question.

        Compares sentence-level cosine similarity between the question
        embedding and each variant embedding. Language-agnostic and uses
        embeddings already computed by the caller.
        """
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
        prompt = _EXPANSION_PROMPT.format(count=count, question=question)
        messages = [{"role": "user", "content": prompt}]
        response = self._provider.chat(
            messages, stream=False, options={"num_predict": _EXPANSION_MAX_TOKENS}
        )
        if not isinstance(response, str):
            return []
        variants = [line.strip() for line in response.strip().split("\n") if line.strip()]
        return variants[:count]

    def _expand_query(
        self, question: str, question_vec: list[float]
    ) -> list[tuple[str, list[float]]]:
        """Return ``(variant, variant_vec)`` pairs for downstream search.

        Expansion is disabled entirely when both ``query_expansion_count``
        is zero and concept-graph expansion is off — matching the old
        early-return so ``LILBEE_QUERY_EXPANSION_COUNT=0`` remains an
        off-switch for users who have ``cfg.concept_graph`` enabled.

        LLM expansions run through the similarity guardrail; concept-graph
        expansions bypass it because they come from deterministic graph
        traversal and are expected to be partial phrases with lower
        similarity to the full question.
        """
        count = self._config.query_expansion_count
        if count <= 0 and not self._config.concept_graph:
            return []
        try:
            llm_variants: list[tuple[str, list[float]]] = []
            if count > 0:
                for text in self._llm_expand(question, count):
                    llm_variants.append((text, self._embedder.embed(text)))
            llm_variants = self._apply_guardrails(llm_variants, question_vec)
            for concept in self._concept_query_expansion(question):
                llm_variants.append((concept, self._embedder.embed(concept)))
            return llm_variants
        except (ConnectionError, OSError, RuntimeError) as exc:
            log.warning("Query expansion disabled for this call: %s", exc, exc_info=True)
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
            if not self._concepts.get_graph():
                return results
            query_concepts = self._concepts.extract_concepts(question)
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
                options={"num_predict": _EXPANSION_MAX_TOKENS},
            )
            if not isinstance(response, str) or not response.strip():
                return []
            hyde_vec = self._embedder.embed(response.strip())
            return self._store.search(hyde_vec, top_k=top_k, query_text=None)
        except Exception:
            log.debug("HyDE search failed", exc_info=True)
            return []

    def _parse_structured_query(self, question: str) -> tuple[str | None, str]:
        for prefix in ("term:", "vec:", "hyde:", "wiki:", "raw:"):
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
        if mode in ("wiki", "raw"):
            query_vec = self._embedder.embed(query)
            return self._store.search(query_vec, top_k=top_k, query_text=query, chunk_type=mode)
        return []

    def select_context(
        self, results: list[SearchChunk], question: str, max_sources: int | None = None
    ) -> list[SearchChunk]:
        """Greedy IDF-weighted cover: pick chunks whose distinctive query
        terms add the most information, then fill any remaining budget
        in retrieval order.

        IDF is computed over the candidate ``results`` list itself, so
        terms that appear in every candidate (the corpus-specific
        stopwords) contribute zero weight without needing a hand-curated
        stoplist.
        """
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

        selected = _greedy_cover(chunk_tokens, question_terms, term_weights, max_sources)
        selected.sort()
        return [results[i] for i in selected]

    def search(self, question: str, top_k: int = 0) -> list[SearchChunk]:
        """Embed question and search with expansion, HyDE, and concept boost.
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
        for variant, variant_vec in self._expand_query(question, query_vec):
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
        """Build RAG context from search results."""
        results = self.search(question, top_k=top_k)
        mode, _ = self._parse_structured_query(question)
        if mode is None:
            results = prefer_wiki(results)
        if not results:
            return None
        results = prepare_results(results)
        if self._config.reranker_model:
            results = self._reranker.rerank(question, results)
        results = self._apply_temporal_filter(results, question)
        results = self.select_context(results, question)
        context = build_context(results)
        prompt = CONTEXT_TEMPLATE.format(context=context, question=question)
        messages: list[ChatMessage] = [{"role": "system", "content": self._config.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return results, messages

    _NO_EMBED_WARNING = (
        "Chat only — no document search configured. "
        "Install an embedding model: lilbee models install nomic-embed-text\n\n"
    )

    def _direct_messages(
        self, question: str, history: list[ChatMessage] | None = None
    ) -> list[ChatMessage]:
        """Build messages for direct LLM chat (no RAG context)."""
        messages: list[ChatMessage] = [{"role": "system", "content": self._config.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})
        return messages

    def _messages_for_provider(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        """Convert ChatMessage list to provider-expected format."""
        return [{"role": m["role"], "content": m["content"]} for m in messages]

    def ask_raw(
        self,
        question: str,
        top_k: int = 0,
        history: list[ChatMessage] | None = None,
        options: dict[str, Any] | None = None,
    ) -> AskResult:
        """Ask a question and get a structured result."""
        if not self._embedder.embedding_available():
            messages = self._direct_messages(question, history)
            provider_messages = self._messages_for_provider(messages)
            opts = options if options is not None else self._config.generation_options()
            answer = self._provider.chat(provider_messages, options=opts or None)
            return AskResult(answer=self._NO_EMBED_WARNING + str(answer or ""), sources=[])
        rag = self.build_rag_context(question, top_k=top_k, history=history)
        if rag is None:
            return AskResult(
                answer="No relevant documents found. Try ingesting some documents first.",
                sources=[],
            )
        results, messages = rag
        provider_messages = self._messages_for_provider(messages)
        opts = options if options is not None else self._config.generation_options()
        answer = self._provider.chat(provider_messages, options=opts or None)
        return AskResult(answer=str(answer) or "", sources=results)

    def ask(
        self,
        question: str,
        top_k: int = 0,
        history: list[ChatMessage] | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Ask a question and get a formatted answer with citations."""
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
        """Stream answer tokens with citations appended at the end."""
        from lilbee.reasoning import StreamToken, filter_reasoning

        if not self._embedder.embedding_available():
            yield StreamToken(content=self._NO_EMBED_WARNING, is_reasoning=False)
            messages = self._direct_messages(question, history)
            provider_messages = self._messages_for_provider(messages)
            opts = options if options is not None else self._config.generation_options()
            raw = self._provider.chat(provider_messages, stream=True, options=opts or None)
            try:
                for st in filter_reasoning(
                    cast(Iterator[str], raw), show=self._config.show_reasoning
                ):
                    if st.content:
                        yield st
            except (ConnectionError, OSError) as exc:
                yield StreamToken(content=f"\n\n[Connection lost: {exc}]", is_reasoning=False)
            return

        rag = self.build_rag_context(question, top_k=top_k, history=history)
        if rag is None:
            yield StreamToken(
                content="No relevant documents found. Try ingesting some documents first.",
                is_reasoning=False,
            )
            return
        results, messages = rag
        provider_messages = self._messages_for_provider(messages)
        opts = options if options is not None else self._config.generation_options()
        raw_stream = self._provider.chat(provider_messages, stream=True, options=opts or None)
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
