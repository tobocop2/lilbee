"""Cross-source synthesis pages and per-source batched generation.

Two related orchestrators live here:

- ``_generate_synthesis_page`` and friends produce a single
  cross-source page from a concept cluster spanning 3+ documents.
- ``_generate_source_batch`` issues one LLM call per source that
  emits sections for every pre-extracted entity plus 3-5 LLM-curated
  concepts; the response is split into per-section bodies and each
  section is finalized via :func:`_finalize_section`.

The shared output-parsing helpers (``_split_batched_output``,
``_prefix_heading``, ``_match_label``) cover both paths.
"""

from __future__ import annotations

import functools
import logging
import re
from pathlib import Path
from typing import cast

import yaml

from lilbee.core.config import Config
from lilbee.data.store import CitationRecord, SearchChunk, Store
from lilbee.providers.base import LLMProvider
from lilbee.retrieval.reasoning import strip_reasoning
from lilbee.wiki.batch import (
    _finalize_section,
    _hash_existing_sources,
    _match_label,
)
from lilbee.wiki.citation import ParsedCitation, parse_wiki_citations
from lilbee.wiki.citations import _resolve_multi_source_citations
from lilbee.wiki.entity_extractor import EntityKind, ExtractedEntity
from lilbee.wiki.page import (
    _build_wiki_messages,
    _chunks_to_text,
    _generate_page,
    _truncate_chunks_to_budget,
)
from lilbee.wiki.persistence import _write_pending_marker
from lilbee.wiki.shared import (
    DRAFTS_SUBDIR,
    PENDING_KIND_PARSE,
    PENDING_MARKER_KEYWORD_PARSE,
    SYNTHESIS_SUBDIR,
    clean_label_for_display,
    make_slug,
)

log = logging.getLogger(__name__)

# Regex that matches section headers the batch parser recognizes:
# H1 (``# Name``), H2 (``## Name``), or a bold-line heading
# (``**Name**``) at line start. The name capture is anchored to the
# rest of the line (stripped of trailing whitespace) so labels like
# ``## Brake System (hydraulic)`` still parse.
_SECTION_HEADER_RE = re.compile(
    r"^(?:(?:##?)\s+(?P<hashname>[^\n]+)|\*\*(?P<boldname>[^\*\n]+)\*\*)\s*$",
    re.MULTILINE,
)

_PENDING_PARSE_MARKER_PREFIX = f"<!-- {PENDING_MARKER_KEYWORD_PARSE}"


def _generate_synthesis_page(
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
    # circular: ingest tooling for source hashing lives outside this module.
    from lilbee.data.ingest import file_hash

    all_chunks = [c for cs in chunks_by_source.values() for c in cs]
    if not all_chunks:
        log.warning("No chunks for synthesis topic %r, skipping", topic)
        return None

    all_chunks = _truncate_chunks_to_budget(all_chunks, config)
    chunks_text = _chunks_to_text(all_chunks)
    source_list = "\n".join(f"- {name}" for name in sorted(source_names))
    template = config.wiki_synthesis_prompt
    display_topic = clean_label_for_display(topic)
    prompt = template.format(topic=display_topic, source_list=source_list, chunks_text=chunks_text)
    slug = make_slug(topic)

    source_hashes: dict[str, str] = {}
    for name in source_names:
        source_path = config.documents_dir / name
        if source_path.exists():
            source_hashes[name] = file_hash(source_path)

    def resolver(parsed: list[ParsedCitation]) -> list[CitationRecord]:
        return _resolve_multi_source_citations(
            parsed, source_names, source_hashes, chunks_by_source
        )

    return _generate_page(
        label=topic,
        prompt=prompt,
        chunks=all_chunks,
        citation_resolver=resolver,
        page_type=SYNTHESIS_SUBDIR,
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
    """Best-effort parse of the batched LLM response into per-label bodies.

    Splits on H1/H2/bold-line headers, then matches each header
    against the expected entity and concept label sets via
    case-insensitive substring. Known labels are tagged with the
    right ``EntityKind``; unknown headers are dropped. Labels whose
    section could not be recovered at all are surfaced to the caller
    (they show up as *missing from the return dict* rather than a
    separate list: caller loops over the expected sets to write
    PENDING markers).
    """
    concepts = expected_concept_labels or set()
    recovered: dict[str, tuple[EntityKind, str]] = {}
    matches = list(_SECTION_HEADER_RE.finditer(text))
    if not matches:
        return recovered
    for i, match in enumerate(matches):
        name = match.group("hashname") or match.group("boldname") or ""
        name = name.strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        lowered = name.lower()
        kind_label = _match_label(lowered, expected_entity_labels, EntityKind.ENTITY)
        if kind_label is None:
            kind_label = _match_label(lowered, concepts, EntityKind.CONCEPT)
        if kind_label is None:
            # Concept labels come from the LLM itself: tag any
            # unmatched section as CONCEPT only when the caller is
            # expecting concept curation; otherwise drop it as
            # noise.
            if concepts is not None and expected_concept_labels is not None:
                recovered.setdefault(name, (EntityKind.CONCEPT, _prefix_heading(name, body)))
            continue
        kind, label = kind_label
        recovered[label] = (kind, _prefix_heading(name, body))
    return recovered


def _prefix_heading(name: str, body: str) -> str:
    """Ensure the extracted body starts with a ``# Name`` H1.

    The batched prompt instructs the model to emit ``## Name`` per
    section. After splitting, the per-section body has lost its
    header. Rebuild an H1 so the B3 title/body coherence gate still
    has a heading to match.
    """
    stripped = body.lstrip()
    if stripped.startswith("# "):
        return body
    return f"# {name}\n\n{body}"


def _build_batch_prompt(
    source: str,
    entities: list[ExtractedEntity],
    chunks_text: str,
    extract_concepts: bool,
    config: Config,
) -> str:
    """Render :attr:`Config.wiki_entity_batch_prompt` for one source call.

    ``extract_concepts`` controls whether the concept-curation
    paragraph is injected: True adds a "identify 3-5 concepts" block;
    False leaves ``{concept_instruction}`` empty so the LLM writes
    entity sections only. Keeps the per-source batched call the
    single entry point whether or not concepts are requested.
    """
    entity_labels = ", ".join(clean_label_for_display(e.label) for e in entities) or "(none)"
    if extract_concepts:
        concept_instruction = (
            "First, identify 3-5 CONCEPTS: abstract topics or domain terms "
            "from the source that deserve a standalone wiki page. Do NOT include "
            "pronouns, articles, or generic nouns.\n\n"
            "Then write a wiki section for each of the concepts you identified, "
            "PLUS one section for each NER ENTITY listed below.\n\n"
        )
    else:
        concept_instruction = ""
    return config.wiki_entity_batch_prompt.format(
        source=source,
        entity_list=entity_labels,
        chunks_text=chunks_text,
        concept_instruction=concept_instruction,
    )


def _group_entities_by_primary_source(
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


def _generate_source_batch(
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
    combined). Labels not recovered by the parser become PENDING
    markers under ``wiki/drafts/`` so the next build can retry.
    Concept slugs already written by an earlier source produce a
    PENDING-COLLISION marker on the losing side (see
    :func:`_handle_concept_write`).

    ``written_concept_slugs`` is the per-build ledger of
    slug → first_source. Callers share one dict across the per-source
    loop. The second source to propose a slug is the one that gets
    diverted to a collision marker.
    """
    if not chunks:
        return []
    budgeted = _truncate_chunks_to_budget(chunks, config)
    chunks_text = _chunks_to_text(budgeted)
    prompt = _build_batch_prompt(source, entities, chunks_text, extract_concepts, config)
    messages = _build_wiki_messages(prompt, provider, config)
    options = config.generation_options(
        temperature=config.wiki_temperature,
        max_tokens=config.wiki_summary_max_tokens,
    )
    try:
        response = provider.chat(messages, stream=False, options=options)
        text = strip_reasoning(cast(str, response)).strip()
    except Exception as exc:
        log.warning("Batched LLM call failed for source %s: %s", source, exc)
        return []

    if not text:
        log.warning("Batched LLM call returned empty response for source %s", source)
        return []

    expected_entity_labels = {e.label for e in entities}
    expected_concepts: set[str] | None = set() if extract_concepts else None
    parsed = _split_batched_output(text, expected_entity_labels, expected_concepts)

    wiki_root = config.data_root / config.wiki_dir
    drafts_dir = wiki_root / DRAFTS_SUBDIR
    source_names = [source]
    source_hashes = _hash_existing_sources(source_names, config.documents_dir)
    chunks_by_source = {source: budgeted}

    # Citation definitions live in the trailing block of the WHOLE
    # response, not inside any one section body. Parse once over the
    # full text and replay the same list for every section, so each
    # page sees its own citations even when only the last section
    # carries the definition trailer.
    shared_parsed_citations = parse_wiki_citations(text)

    pages: list[Path] = []
    seen_labels: set[str] = set()
    for header_label, (kind, body) in parsed.items():
        seen_labels.add(header_label)
        resolver = functools.partial(
            _resolve_multi_source_citations,
            source_names=source_names,
            source_hashes=source_hashes,
            chunks_by_source=chunks_by_source,
        )
        page = _finalize_section(
            header_label=header_label,
            kind=kind,
            body=body,
            chunks=budgeted,
            citation_resolver=resolver,
            source_names=source_names,
            store=store,
            config=config,
            source=source,
            written_concept_slugs=written_concept_slugs,
            drafts_dir=drafts_dir,
            shared_parsed_citations=shared_parsed_citations,
        )
        if page is not None:
            pages.append(page)

    for entity in entities:
        if entity.label not in seen_labels:
            marker = (
                f"{_PENDING_PARSE_MARKER_PREFIX} for source {source}, "
                f"entity/concept {entity.label} - "
                "run wiki build again or manually accept via wiki drafts accept -->"
            )
            # Route through ``yaml.safe_dump`` so a label or source
            # containing a colon, quote, or newline does not produce a
            # frontmatter block that ``parse_frontmatter`` silently drops.
            frontmatter_body = yaml.safe_dump(
                {
                    "pending_source": source,
                    "pending_label": entity.label,
                    "pending_kind": PENDING_KIND_PARSE,
                },
                sort_keys=False,
            )
            frontmatter = f"---\n{frontmatter_body}---\n"
            path = _write_pending_marker(drafts_dir, entity.slug, marker, frontmatter)
            log.info("Wrote PENDING-PARSE marker for %s -> %s", entity.slug, path)

    return pages
