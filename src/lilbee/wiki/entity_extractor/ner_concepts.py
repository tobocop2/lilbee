"""spaCy NER entity extractor (default strategy).

Phase D removed the noun-chunk "concept" path from this extractor. The
per-source batched call in :mod:`lilbee.wiki.generation` now proposes concept
pages through the LLM. This module produces typed NER entities only.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from lilbee.wiki.entity_extractor.base import (
    ChunkRef,
    EntityKind,
    ExtractedEntity,
)
from lilbee.wiki.shared import is_valid_label, make_slug

if TYPE_CHECKING:
    from lilbee.config import Config
    from lilbee.providers.base import LLMProvider
    from lilbee.store import SearchChunk

log = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")

# Pre-spaCy markdown-noise strippers. Compiled once at module scope so
# the extractor's hot path does not recompile them per chunk. Match on
# line boundaries via re.MULTILINE; each sub() empties the matched
# line so downstream line-joins collapse the hole to a single newline.
_TABLE_ROW_RE = re.compile(r"^\|.*\|\s*$", re.MULTILINE)
_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)
_NAV_CHROME_RE = re.compile(
    r"^\s*(?:Home|Menu|Navigation|Edit this page|Jump to navigation|Jump to search)\s*$",
    re.MULTILINE,
)


def _normalize(text: str) -> str:
    """Lowercase, strip, and collapse internal whitespace for dedup keys."""
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def pre_clean_for_ner(text: str) -> str:
    """Strip markdown-structural noise before handing text to spaCy.

    Removes whole-line markdown-table rows (``| Designer | Irv ... |``),
    standalone page-number lines from PDF extraction (``42``), and
    Wikipedia / CMS navigation chrome (``Edit this page``). Leaves
    prose untouched: every regex anchors to a full line and emits an
    empty line in place of the match, which spaCy treats as a sentence
    break.

    Only targets the noise patterns actually observed in the bb-8b7s
    QA corpus. Fuller markdown parsing is deferred; a regex pre-clean
    is sufficient for the current signal-to-noise ratio.
    """
    text = _TABLE_ROW_RE.sub("", text)
    text = _PAGE_NUMBER_RE.sub("", text)
    return _NAV_CHROME_RE.sub("", text)


class NerConceptsExtractor:
    """Emit typed NER entities (``EntityKind.ENTITY`` only).

    Phase D removed the noun-chunk concept loop: LLM-curated concept
    pages are produced downstream by the per-source batched call in
    :mod:`lilbee.wiki.generation`. The class name is kept for backwards
    compatibility at the factory dispatch site; the implementation
    emits only ``EntityKind.ENTITY`` records now.
    """

    def __init__(self, provider: LLMProvider, config: Config) -> None:
        self._provider = provider
        self._config = config

    def extract(self, chunks: list[SearchChunk]) -> list[ExtractedEntity]:
        if not chunks:
            return []
        nlp = _load_spacy()
        if nlp is None:
            return []

        entity_records: dict[str, _Aggregate] = {}
        allowed_ent_types = self._config.concept_allowed_ent_types

        debug_enabled = log.isEnabledFor(logging.DEBUG)
        # Per-pass funnel counters; emitted once after the loop so the
        # DEBUG trace captures the whole corpus in one line instead of
        # one per chunk.
        funnel = {
            "raw_ents": 0,
            "type_filter_dropped": 0,
            "label_sanity_dropped_entities": 0,
            "kept_entity_surfaces": 0,
        }
        cleaned_texts = (pre_clean_for_ner(c.chunk) for c in chunks)
        for chunk, doc in zip(chunks, nlp.pipe(cleaned_texts), strict=True):
            ref = ChunkRef(source=chunk.source, chunk_index=chunk.chunk_index)
            for ent in doc.ents:
                funnel["raw_ents"] += 1
                if ent.label_ not in allowed_ent_types:
                    funnel["type_filter_dropped"] += 1
                    continue
                surface = ent.text.strip()
                if not is_valid_label(surface):
                    funnel["label_sanity_dropped_entities"] += 1
                    if debug_enabled:
                        log.debug("label-sanity: rejected entity %r", surface)
                    continue
                key = _normalize(surface)
                rec = entity_records.setdefault(
                    key, _Aggregate(label=surface, type_hint=ent.label_)
                )
                rec.refs.add(ref)
                funnel["kept_entity_surfaces"] += 1

        if debug_enabled:
            log.debug(
                "ner funnel: raw_ents=%(raw_ents)d "
                "type_filter_dropped=%(type_filter_dropped)d "
                "label_sanity_dropped_entities=%(label_sanity_dropped_entities)d "
                "kept_entity_surfaces=%(kept_entity_surfaces)d",
                funnel,
            )

        min_mentions = self._config.wiki_entity_min_mentions
        results: list[ExtractedEntity] = []
        for agg in entity_records.values():
            record = _make_record(agg, EntityKind.ENTITY, min_mentions)
            if record is not None:
                results.append(record)
        results.sort(key=lambda e: (e.kind.value, e.slug))
        return results


class _Aggregate:
    """Mutable accumulator used only while folding per-chunk hits."""

    __slots__ = ("label", "refs", "type_hint")

    def __init__(self, label: str, type_hint: str) -> None:
        self.label = label
        self.type_hint = type_hint
        self.refs: set[ChunkRef] = set()


def _sorted_refs(refs: set[ChunkRef]) -> tuple[ChunkRef, ...]:
    return tuple(sorted(refs, key=lambda r: (r.source, r.chunk_index)))


def _make_record(agg: _Aggregate, kind: EntityKind, min_mentions: int) -> ExtractedEntity | None:
    """Turn an aggregate into an ``ExtractedEntity`` or drop it.

    Filters records below the mention threshold and records whose label
    slug-cleans to an empty string (e.g. labels of only punctuation);
    without the empty-slug guard those would try to write files named
    just ``.md`` on disk.
    """
    if len(agg.refs) < min_mentions:
        return None
    slug = make_slug(agg.label)
    if not slug:
        return None
    return ExtractedEntity(
        slug=slug,
        kind=kind,
        label=agg.label,
        type_hint=agg.type_hint,
        chunk_refs=_sorted_refs(agg.refs),
    )


def _load_spacy() -> Any | None:
    """Load the shared spaCy pipeline, or return None if unavailable."""
    try:
        from lilbee.concepts import load_spacy_pipeline
    except ImportError:
        log.warning("Entity extraction disabled: lilbee.concepts unavailable")
        return None
    try:
        return load_spacy_pipeline()
    except ImportError:
        log.warning("Entity extraction disabled: spaCy model unavailable")
        return None
