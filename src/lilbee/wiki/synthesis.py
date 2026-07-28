"""Cross-source synthesis pages and per-source batched generation.

Two related orchestrators live here:

- ``generate_synthesis_page`` and friends produce a single
  cross-source page from a concept cluster spanning 3+ documents.
- ``generate_source_batch`` issues one LLM call per source that
  emits sections for every pre-extracted entity plus 3-5 LLM-curated
  concepts; the response is split into per-section bodies and each
  section is finalized via :func:`finalize_section`.

The shared output-parsing helpers (``_split_batched_output``,
``_prefix_heading``, ``match_label``) cover both paths.
"""

from __future__ import annotations

import functools
import logging
import re
from collections.abc import Callable
from pathlib import Path

import yaml

from lilbee.core.config import Config
from lilbee.core.text import clean_label_for_display, make_slug
from lilbee.data.store import CitationRecord, SearchChunk, Store
from lilbee.providers.base import LLMProvider
from lilbee.retrieval.reasoning import strip_reasoning
from lilbee.wiki.batch import (
    finalize_section,
    hash_existing_sources,
    match_label,
)
from lilbee.wiki.citation import ParsedCitation, parse_wiki_citations
from lilbee.wiki.citations import resolve_multi_source_citations
from lilbee.wiki.entity_extractor import EntityKind, ExtractedEntity
from lilbee.wiki.page import (
    build_wiki_messages,
    chunks_to_text,
    generate_page,
    truncate_chunks_to_budget,
    wiki_generation_options,
)
from lilbee.wiki.persistence import write_pending_marker
from lilbee.wiki.shared import (
    PENDING_MARKER_KEYWORD_PARSE,
    PendingKind,
    WikiSubdir,
)

log = logging.getLogger(__name__)

# Section headers the batch parser recognizes: H1 (``# Name``) or H2
# (``## Name``). The name capture is anchored to the rest of the line so
# labels like ``## Brake System (hydraulic)`` still parse. Bold lines are
# not headers: a mid-body ``**emphasis**`` would otherwise truncate its
# section and open a bogus one.
_SECTION_HEADER_RE = re.compile(r"^##?\s+(?P<name>[^\n]+)\s*$", re.MULTILINE)

# Machine-readable concept declaration the batched prompt requires when
# concept curation is on. Only a declared name may open a concept section.
_CONCEPT_DECLARATION_PREFIX = "CONCEPTS:"
_CONCEPT_DECLARATION_SEPARATOR = ";"
_CONCEPT_DECLARATION_RE = re.compile(
    rf"^\s*{_CONCEPT_DECLARATION_PREFIX}\s*(?P<labels>[^\n]+)$",
    re.MULTILINE | re.IGNORECASE,
)

_PENDING_PARSE_MARKER_PREFIX = f"<!-- {PENDING_MARKER_KEYWORD_PARSE}"


def generate_synthesis_page(
    topic: str,
    source_names: list[str],
    chunks_by_source: dict[str, list[SearchChunk]],
    provider: LLMProvider,
    store: Store,
    config: Config,
) -> Path | None:
    """Generate a single synthesis page for a concept cluster.
    Returns the path to the generated page, or None on failure.
    """
    all_chunks = [c for cs in chunks_by_source.values() for c in cs]
    if not all_chunks:
        log.warning("No chunks for synthesis topic %r, skipping", topic)
        return None

    all_chunks = truncate_chunks_to_budget(all_chunks, config)
    chunks_text = chunks_to_text(all_chunks)
    source_list = "\n".join(f"- {name}" for name in sorted(source_names))
    template = config.wiki_synthesis_prompt
    display_topic = clean_label_for_display(topic)
    prompt = template.format(topic=display_topic, source_list=source_list, chunks_text=chunks_text)
    slug = make_slug(topic)

    source_hashes = hash_existing_sources(source_names)

    def resolver(parsed: list[ParsedCitation]) -> list[CitationRecord]:
        return resolve_multi_source_citations(parsed, source_names, source_hashes, chunks_by_source)

    return generate_page(
        label=topic,
        prompt=prompt,
        chunks=all_chunks,
        citation_resolver=resolver,
        page_type=WikiSubdir.SYNTHESIS,
        slug=slug,
        source_names=source_names,
        provider=provider,
        store=store,
        config=config,
    )


def _split_batched_output(
    text: str,
    expected_entity_labels: set[str],
    expected_concept_labels: set[str] | None = None,
) -> dict[str, tuple[EntityKind, str]]:
    """Parse the batched LLM response into per-label bodies.

    Splits on H1/H2 headers, then binds each header to an expected entity
    label or to a concept the response declared, via :func:`match_label`.
    A header matching neither is dropped as noise. Labels whose section
    could not be recovered are simply absent from the result; the caller
    loops over the expected sets to write their PENDING markers.
    """
    concepts = expected_concept_labels or set()
    recovered: dict[str, tuple[EntityKind, str]] = {}
    matches = list(_SECTION_HEADER_RE.finditer(text))
    for i, match in enumerate(matches):
        name = match.group("name").strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        lowered = name.lower()
        kind_label = match_label(lowered, expected_entity_labels, EntityKind.ENTITY) or match_label(
            lowered, concepts, EntityKind.CONCEPT
        )
        if kind_label is None:
            log.info("Dropping section %r: matches no expected entity or declared concept", name)
            continue
        kind, label = kind_label
        recovered[label] = (kind, _prefix_heading(label, body))
    return recovered


def _parse_declared_concepts(text: str) -> set[str]:
    """Read the ``CONCEPTS: a; b; c`` line the batched prompt requires.

    An absent declaration means the response curated no concepts, so every
    non-entity header in it is noise.
    """
    match = _CONCEPT_DECLARATION_RE.search(text)
    if match is None:
        return set()
    parts = match.group("labels").split(_CONCEPT_DECLARATION_SEPARATOR)
    return {part.strip() for part in parts if part.strip()}


def _prefix_heading(label: str, body: str) -> str:
    """Rebuild the section's ``# Label`` H1 from the label it was bound to.

    Splitting consumes the model's own header line. Rebuilding the heading
    from the matched label rather than the model's wording keeps the
    title/body coherence gate reading the label the page is filed under,
    including when the header matched only as a substring of it.
    """
    return f"# {clean_label_for_display(label)}\n\n{body}"


def _existing_concept_labels(wiki_root: Path) -> list[str]:
    """Published concept slugs as spaced names, so rebuilds reuse established names."""
    concepts_dir = wiki_root / WikiSubdir.CONCEPTS
    if not concepts_dir.is_dir():
        return []
    return sorted({path.stem.replace("-", " ") for path in concepts_dir.rglob("*.md")})


def _concept_instruction(existing_concepts: list[str]) -> str:
    """Concept-curation paragraph, including the declaration-line contract.

    Lives in code rather than in the writable prompt template: the declaration
    line is what the parser enforces, so overriding the template from settings
    cannot silently break section recovery.
    """
    reuse = ""
    if existing_concepts:
        reuse = (
            "Reuse these existing concept names verbatim when they fit: "
            f"{', '.join(existing_concepts)}.\n\n"
        )
    separator = f"{_CONCEPT_DECLARATION_SEPARATOR} "
    return (
        "First, identify 3-5 CONCEPTS: abstract topics or domain terms "
        "from the source that deserve a standalone wiki page. Do NOT include "
        "pronouns, articles, or generic nouns.\n\n"
        f"{reuse}"
        f"Declare them on a single line as `{_CONCEPT_DECLARATION_PREFIX} "
        f"first{separator}second{separator}third` before writing anything else. "
        "A section whose heading is neither a declared concept nor a listed "
        "entity is discarded.\n\n"
        "Then write a wiki section for each of the concepts you identified, "
        "PLUS one section for each NER ENTITY listed below.\n\n"
    )


def _build_batch_prompt(
    source: str,
    entities: list[ExtractedEntity],
    chunks_text: str,
    extract_concepts: bool,
    wiki_root: Path,
    config: Config,
) -> str:
    """Render :attr:`Config.wiki_entity_batch_prompt` for one source call.

    ``extract_concepts`` controls whether the concept-curation
    paragraph is injected: True adds the "identify 3-5 concepts" block
    and its declaration-line contract; False leaves
    ``{concept_instruction}`` empty so the LLM writes entity sections
    only. Keeps the per-source batched call the single entry point
    whether or not concepts are requested.
    """
    entity_labels = ", ".join(clean_label_for_display(e.label) for e in entities) or "(none)"
    concept_instruction = (
        _concept_instruction(_existing_concept_labels(wiki_root)) if extract_concepts else ""
    )
    return config.wiki_entity_batch_prompt.format(
        source=source,
        entity_list=entity_labels,
        chunks_text=chunks_text,
        concept_instruction=concept_instruction,
    )


def group_entities_by_primary_source(
    entities: list[ExtractedEntity],
) -> dict[str, list[ExtractedEntity]]:
    """Group entities under the source that mentions them most.

    Primary source = source with the highest chunk-ref count;
    lexicographic tiebreak. An entity with no refs is dropped
    silently (defensive: extractor always attaches refs, but a
    future extractor might not).
    """
    grouped: dict[str, list[ExtractedEntity]] = {}
    for entity in entities:
        if not entity.chunk_refs:
            continue
        counts: dict[str, int] = {}
        for ref in entity.chunk_refs:
            counts[ref.source] = counts.get(ref.source, 0) + 1
        primary = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        grouped.setdefault(primary, []).append(entity)
    return grouped


def generate_source_batch(
    source: str,
    entities: list[ExtractedEntity],
    chunks: list[SearchChunk],
    provider: LLMProvider,
    store: Store,
    config: Config,
    *,
    extract_concepts: bool,
    written_concept_slugs: dict[str, str],
) -> list[Path]:
    """Issue one LLM call for *source* and finalize every recovered section.

    Returns the list of page paths written (entities + concepts
    combined). Every expected entity that produced no page leaves a
    PENDING marker under ``wiki/drafts/`` so the next build can retry,
    whether the parser missed its section, a downstream gate rejected
    it, or the LLM call itself failed. Concept slugs already written by
    an earlier source produce a PENDING-COLLISION marker on the losing
    side (see :func:`divert_concept_collision`).

    ``written_concept_slugs`` is the per-build ledger of
    slug → first_source. Callers share one dict across the per-source
    loop. The second source to propose a slug is the one that gets
    diverted to a collision marker.
    """
    if not chunks:
        return []
    wiki_root = config.data_root / config.wiki_dir
    drafts_dir = wiki_root / WikiSubdir.DRAFTS
    budgeted = truncate_chunks_to_budget(chunks, config)
    prompt = _build_batch_prompt(
        source, entities, chunks_to_text(budgeted), extract_concepts, wiki_root, config
    )
    text = _request_batch_sections(source, prompt, provider, config)
    if text is None:
        _write_pending_markers(entities, set(), source, drafts_dir)
        return []

    declared_concepts = _parse_declared_concepts(text) if extract_concepts else set()
    parsed = _split_batched_output(text, {e.label for e in entities}, declared_concepts)

    source_names = [source]
    finalize = functools.partial(
        finalize_section,
        chunks=budgeted,
        citation_resolver=functools.partial(
            resolve_multi_source_citations,
            source_names=source_names,
            source_hashes=hash_existing_sources(source_names),
            chunks_by_source={source: budgeted},
        ),
        source_names=source_names,
        store=store,
        config=config,
        source=source,
        written_concept_slugs=written_concept_slugs,
        drafts_dir=drafts_dir,
        # Citation definitions live in the trailing block of the WHOLE response,
        # not inside any one section body. Parse once and replay for every
        # section, so pages other than the last still resolve their footnotes.
        shared_parsed_citations=parse_wiki_citations(text),
    )
    pages, written_labels = _finalize_sections(parsed, finalize)
    _write_pending_markers(entities, written_labels, source, drafts_dir)
    return pages


def _request_batch_sections(
    source: str,
    prompt: str,
    provider: LLMProvider,
    config: Config,
) -> str | None:
    """Issue the batched LLM call; None when it raised or came back empty."""
    messages = build_wiki_messages(prompt, provider, config)
    try:
        response = provider.chat(messages, stream=False, options=wiki_generation_options(config))
    except Exception as exc:
        log.warning("Batched LLM call failed for source %s: %s", source, exc)
        return None
    text = strip_reasoning(response.text).strip()
    if not text:
        log.warning("Batched LLM call returned empty response for source %s", source)
        return None
    return text


def _finalize_sections(
    parsed: dict[str, tuple[EntityKind, str]],
    finalize: Callable[..., Path | None],
) -> tuple[list[Path], set[str]]:
    """Finalize each recovered section; return the pages written and labels covered.

    A label counts as covered only once ``finalize`` returns a path, so a
    section dropped by the citation or slug gate still earns a PENDING marker.
    """
    pages: list[Path] = []
    written: set[str] = set()
    for header_label, (kind, body) in parsed.items():
        page = finalize(header_label=header_label, kind=kind, body=body)
        if page is not None:
            pages.append(page)
            written.add(header_label)
    return pages, written


def _write_pending_markers(
    entities: list[ExtractedEntity],
    written_labels: set[str],
    source: str,
    drafts_dir: Path,
) -> None:
    """Write a PENDING-PARSE marker for every expected entity that produced no page."""
    for entity in entities:
        if entity.label in written_labels:
            continue
        marker = (
            f"{_PENDING_PARSE_MARKER_PREFIX} for source {source}, "
            f"entity/concept {entity.label} - "
            "run wiki build again or manually accept via wiki drafts accept -->"
        )
        # Route through ``yaml.safe_dump`` so a label or source containing a
        # colon, quote, or newline does not produce a frontmatter block that
        # ``parse_frontmatter`` silently drops.
        frontmatter_body = yaml.safe_dump(
            {
                "pending_source": source,
                "pending_label": entity.label,
                "pending_kind": PendingKind.PARSE.value,
            },
            sort_keys=False,
        )
        path = write_pending_marker(
            drafts_dir, entity.slug, marker, f"---\n{frontmatter_body}---\n"
        )
        log.info("Wrote PENDING-PARSE marker for %s -> %s", entity.slug, path)
