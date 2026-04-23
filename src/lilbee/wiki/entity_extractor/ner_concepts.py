"""spaCy NER + noun-phrase concept extractor (default strategy)."""

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

_ALLOWED_NER_LABELS: frozenset[str] = frozenset(
    {"PERSON", "ORG", "GPE", "LOC", "EVENT", "WORK_OF_ART", "PRODUCT"}
)
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercase, strip, and collapse internal whitespace for dedup keys."""
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


class NerConceptsExtractor:
    """Combine spaCy NER and noun-phrase concepts into one entity set.

    NER surface forms (PERSON/ORG/etc.) become ``EntityKind.ENTITY``
    records. Noun-phrase concepts become ``EntityKind.CONCEPT`` records.
    A concept whose normalized form matches an entity's normalized form
    is folded into the entity record so one topic never splits across
    two pages.
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
        concept_records: dict[str, _Aggregate] = {}

        debug_enabled = log.isEnabledFor(logging.DEBUG)
        for chunk, doc in zip(chunks, nlp.pipe(c.chunk for c in chunks), strict=True):
            ref = ChunkRef(source=chunk.source, chunk_index=chunk.chunk_index)
            for ent in doc.ents:
                if ent.label_ not in _ALLOWED_NER_LABELS:
                    continue
                surface = ent.text.strip()
                if not is_valid_label(surface):
                    if debug_enabled:
                        log.debug("label-sanity: rejected entity %r", surface)
                    continue
                key = _normalize(surface)
                rec = entity_records.setdefault(
                    key, _Aggregate(label=surface, type_hint=ent.label_)
                )
                rec.refs.add(ref)
            for noun_chunk in doc.noun_chunks:
                surface = noun_chunk.text.strip()
                if not is_valid_label(surface):
                    if debug_enabled:
                        log.debug("label-sanity: rejected noun-chunk %r", surface)
                    continue
                key = _normalize(surface)
                rec = concept_records.setdefault(
                    key, _Aggregate(label=key, type_hint="noun_phrase")
                )
                rec.refs.add(ref)

        for key, entity_agg in entity_records.items():
            if key in concept_records:
                entity_agg.refs.update(concept_records.pop(key).refs)

        min_mentions = self._config.wiki_entity_min_mentions
        results: list[ExtractedEntity] = []
        for agg in entity_records.values():
            record = _make_record(agg, EntityKind.ENTITY, min_mentions)
            if record is not None:
                results.append(record)
        for agg in concept_records.values():
            record = _make_record(agg, EntityKind.CONCEPT, min_mentions)
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
