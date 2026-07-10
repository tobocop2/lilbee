"""Two-phase typed entity extraction.

Phase 1 (cheap): an LLM reads a stratified sample of chunks and proposes the
corpus-specific type schema, persisted as a reviewable artifact before any
expensive pass runs.

Phase 2 (scales with corpus): each type is found by the cheapest extractor
that can serve it: compiled regex for identifier-shaped types, spaCy labels
for the general ones, and an LLM only for types neither can catch. Cost is
therefore dominated by how many LLM-kind types the reviewed schema keeps.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from lilbee.retrieval.entities.schema import EntitySchema, EntityType, ExtractorKind
from lilbee.retrieval.reasoning import strip_reasoning

if TYPE_CHECKING:
    from lilbee.providers.base import LLMProvider

log = logging.getLogger(__name__)

INDUCTION_SAMPLE_SIZE = 40
INDUCTION_MAX_TOKENS = 1200
# Chunks per LLM call in phase 2; larger batches save round-trips, smaller
# ones keep each response comfortably parseable.
LLM_EXTRACTION_BATCH = 8
LLM_EXTRACTION_MAX_TOKENS = 800

_CONFIDENCE = {ExtractorKind.REGEX: 1.0, ExtractorKind.SPACY: 0.8, ExtractorKind.LLM: 0.6}

INDUCTION_PROMPT = (
    "You are designing an entity-extraction schema for a document collection. "
    "Below are sample passages. Propose the 3-8 entity TYPES most useful for "
    "counting and cross-referencing in this collection.\n"
    "Return ONLY a JSON object of the form:\n"
    '{{"types": [{{"name": "snake_case_name", "kind": "regex|spacy|llm", '
    '"pattern": "<regex for regex kinds, spaCy label like PERSON/ORG/DATE for '
    'spacy kinds, empty for llm kinds>", "description": "one line", '
    '"synonyms": ["words a question would use"]}}]}}\n'
    "Prefer regex for identifier-shaped types (codes, numbered records), spacy "
    "for people/organizations/dates, llm only when unavoidable.\n\n"
    "Passages:\n{sample}"
)

LLM_EXTRACTION_PROMPT = (
    "Extract entities of these types from each numbered passage.\n"
    "Types:\n{types}\n"
    "Return ONLY a JSON object mapping passage number to a list of "
    '{{"type": ..., "text": ...}} objects; use an empty list when a passage '
    "has none.\n\nPassages:\n{passages}"
)


def _first_json_object(text: str) -> dict | None:
    """The first balanced JSON object in *text*, or None."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def normalize_value(text: str) -> str:
    """Canonical form for grouping: casefold, collapse spaces, and strip
    leading zeros from purely numeric values so 00482 and 482 group together."""
    value = " ".join(text.split()).casefold()
    if value.isdigit():
        value = value.lstrip("0") or "0"
    return value


def induce_schema(sample_texts: list[str], provider: LLMProvider) -> EntitySchema | None:
    """Phase 1: propose a schema from sampled chunk texts. None on failure."""
    if not sample_texts:
        return None
    sample = "\n---\n".join(t[:600] for t in sample_texts[:INDUCTION_SAMPLE_SIZE])
    prompt = INDUCTION_PROMPT.format(sample=sample)
    try:
        response = provider.chat(
            [{"role": "user", "content": prompt}],
            stream=False,
            options={"num_predict": INDUCTION_MAX_TOKENS},
        )
    except Exception:
        log.warning("Entity schema induction failed at the provider", exc_info=True)
        return None
    payload = _first_json_object(strip_reasoning(response.text))
    if payload is None:
        log.warning("Entity schema induction returned no parseable JSON")
        return None
    types: list[EntityType] = []
    for raw in payload.get("types", []):
        try:
            entity_type = EntityType.model_validate(raw)
        except Exception:
            log.warning("Dropping invalid induced type: %r", raw)
            continue
        if entity_type.kind is ExtractorKind.REGEX:
            try:
                re.compile(entity_type.pattern)
            except re.error:
                log.warning("Dropping induced type %s: bad regex", entity_type.name)
                continue
        types.append(entity_type)
    return EntitySchema(types=types) if types else None


def _extract_regex(entity_type: EntityType, text: str) -> list[str]:
    return [m.group(0) for m in re.finditer(entity_type.pattern, text)]


def _extract_spacy(types: list[EntityType], text: str, nlp: Any) -> list[tuple[EntityType, str]]:
    wanted = {t.pattern.upper(): t for t in types}
    doc = nlp(text)
    found: list[tuple[EntityType, str]] = []
    for ent in doc.ents:
        entity_type = wanted.get(ent.label_)
        if entity_type is not None:
            found.append((entity_type, ent.text))
    return found


def _extract_llm_batch(
    types: list[EntityType],
    texts: list[str],
    provider: LLMProvider,
) -> list[list[tuple[EntityType, str]]]:
    by_name = {t.name: t for t in types}
    type_lines = "\n".join(f"- {t.name}: {t.description or t.name}" for t in types)
    passages = "\n".join(f"[{i}] {t[:800]}" for i, t in enumerate(texts))
    prompt = LLM_EXTRACTION_PROMPT.format(types=type_lines, passages=passages)
    empty: list[list[tuple[EntityType, str]]] = [[] for _ in texts]
    try:
        response = provider.chat(
            [{"role": "user", "content": prompt}],
            stream=False,
            options={"num_predict": LLM_EXTRACTION_MAX_TOKENS},
        )
    except Exception:
        log.warning("LLM entity extraction failed for a batch", exc_info=True)
        return empty
    payload = _first_json_object(strip_reasoning(response.text))
    if payload is None:
        return empty
    results = empty
    for key, items in payload.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        if not (0 <= index < len(texts)) or not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            entity_type = by_name.get(str(item.get("type", "")))
            mention = str(item.get("text", "")).strip()
            if entity_type is not None and mention:
                results[index].append((entity_type, mention))
    return results


def extract_entities(
    chunks: list[Mapping[str, Any]],
    schema: EntitySchema,
    *,
    provider: LLMProvider | None = None,
    nlp: Any = None,
) -> list[dict]:
    """Phase 2 over ingest-shaped chunk records; returns entities-table rows.

    Each chunk dict needs ``chunk`` (text), ``source``, ``chunk_index``, and
    ``page_start``. Extractor kinds degrade independently: regex always runs,
    spaCy kinds are skipped without a loaded model, LLM kinds without a
    provider, so a partial toolchain yields partial extraction, never failure.
    """
    regex_types = [t for t in schema.types if t.kind is ExtractorKind.REGEX]
    spacy_types = [t for t in schema.types if t.kind is ExtractorKind.SPACY]
    llm_types = [t for t in schema.types if t.kind is ExtractorKind.LLM]

    per_chunk: list[list[tuple[EntityType, str]]] = []
    for record in chunks:
        text = record["chunk"]
        found: list[tuple[EntityType, str]] = []
        for entity_type in regex_types:
            found.extend((entity_type, m) for m in _extract_regex(entity_type, text))
        if spacy_types and nlp is not None:
            found.extend(_extract_spacy(spacy_types, text, nlp))
        per_chunk.append(found)

    if llm_types and provider is not None:
        for start in range(0, len(chunks), LLM_EXTRACTION_BATCH):
            batch = chunks[start : start + LLM_EXTRACTION_BATCH]
            batch_found = _extract_llm_batch(llm_types, [r["chunk"] for r in batch], provider)
            for offset, found in enumerate(batch_found):
                per_chunk[start + offset].extend(found)

    return [
        row
        for record, found in zip(chunks, per_chunk, strict=True)
        for row in _rows_for_chunk(record, found)
    ]


def _rows_for_chunk(record: Mapping[str, Any], found: list[tuple[EntityType, str]]) -> list[dict]:
    """Deduplicated entities-table rows for one chunk's findings."""
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for entity_type, mention in found:
        normalized = normalize_value(mention)
        if not normalized:
            continue
        key = (entity_type.name, normalized)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "entity": mention,
                "type": entity_type.name,
                "normalized_value": normalized,
                "source": record["source"],
                "page": int(record.get("page_start") or 0),
                "chunk_index": int(record["chunk_index"]),
                "confidence": _CONFIDENCE[entity_type.kind],
            }
        )
    return rows
