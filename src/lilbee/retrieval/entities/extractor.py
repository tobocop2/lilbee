"""Two-phase typed entity extraction.

Phase 1 (cheap): an LLM reads a stratified sample of chunks and proposes the
corpus-specific type schema, persisted as an editable artifact before any
expensive pass runs.

Phase 2 (scales with corpus): each type is found by the cheapest extractor
that can serve it: compiled regex for identifier-shaped types, spaCy labels
for the general ones, and an LLM only for types neither can catch. Cost is
therefore dominated by how many LLM-kind types the schema keeps.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lilbee.retrieval.entities.schema import (
    EntitySchema,
    EntityType,
    ExtractorKind,
    extractor_key,
)
from lilbee.retrieval.reasoning import strip_reasoning

if TYPE_CHECKING:
    from lilbee.providers.base import LLMProvider

log = logging.getLogger(__name__)

INDUCTION_SAMPLE_SIZE = 40
# Thinking models spend most of their budget reasoning before the JSON
# appears (the reasoning is stripped afterward); 1200 starved them into
# emitting nothing parseable.
INDUCTION_MAX_TOKENS = 4096
# Chunks per LLM call in phase 2; larger batches save round-trips, smaller
# ones keep each response comfortably parseable.
LLM_EXTRACTION_BATCH = 8
LLM_EXTRACTION_MAX_TOKENS = 800

_CONFIDENCE = {ExtractorKind.REGEX: 1.0, ExtractorKind.SPACY: 0.8, ExtractorKind.LLM: 0.6}


@dataclass
class ExtractionStats:
    """Mutable counters :func:`extract_entities` fills for its caller.

    Lets the full pass tell a clean zero-entity result from batches the
    provider failed outright -- the latter must not count as a completed
    pass, or the schema gets marked applied with rows silently missing.
    """

    llm_batches: int = 0
    llm_batches_failed: int = 0

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
    "for people/organizations/dates, llm only when unavoidable. Regex patterns "
    "must match identifiers INLINE in running text: describe the identifier "
    "token itself and never use ^ or $ anchors.\n\n"
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
            # think=False: a small thinking model can loop inside <think>
            # until the budget is gone and emit no JSON at all. temperature 0:
            # induction wants one deterministic, well-formed schema, not a
            # creative sample that parses only some of the time.
            options={"num_predict": INDUCTION_MAX_TOKENS, "think": False, "temperature": 0},
        )
    except Exception:
        log.warning("Entity schema induction failed at the provider", exc_info=True)
        return None
    payload = _first_json_object(strip_reasoning(response.text))
    if payload is None:
        log.warning("Entity schema induction returned no parseable JSON")
        return None
    types: list[EntityType] = []
    seen_extractors: set[tuple[ExtractorKind, str]] = set()
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
        # Small models sometimes propose several names for one extractor
        # (three types sharing a regex triple the table with identical rows);
        # the first name wins.
        key = extractor_key(entity_type)
        if key in seen_extractors:
            log.warning("Dropping induced type %s: duplicate extractor", entity_type.name)
            continue
        seen_extractors.add(key)
        types.append(entity_type)
    return EntitySchema(types=types) if types else None


# Tokens for anchored-pattern matching: identifier-shaped runs of word chars
# (hyphens allowed inside), so punctuation adjacent to an inline identifier
# never defeats the match.
_IDENTIFIER_TOKEN_RE = re.compile(r"[0-9A-Za-z][0-9A-Za-z-]*")


def _extract_regex(entity_type: EntityType, text: str) -> list[str]:
    """Regex mentions, treating anchored patterns as per-token full matches.

    Schema authors (and the induction model) tend to write ^...$ patterns
    that describe an identifier's whole shape. Applied to running text those
    match almost nothing, since identifiers appear inline with adjacent
    punctuation; a ^...$ pattern therefore full-matches each identifier
    token instead of the chunk.
    """
    pattern = entity_type.pattern
    inner_text = pattern[1:-1]
    # Only a single fully-anchored pattern converts; an alternation of
    # anchored branches (^A$|^B$) would be mangled by stripping the outer
    # pair, so it falls through to finditer unchanged.
    if (
        pattern.startswith("^")
        and pattern.endswith("$")
        and "^" not in inner_text
        and "$" not in inner_text
    ):
        inner = re.compile(inner_text)
        return [
            token.group(0)
            for token in _IDENTIFIER_TOKEN_RE.finditer(text)
            if inner.fullmatch(token.group(0))
        ]
    return [m.group(0) for m in re.finditer(pattern, text)]


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
) -> list[list[tuple[EntityType, str]]] | None:
    """One LLM extraction call over a batch of texts.

    Returns ``None`` when the provider call itself fails (model down or
    unloaded), so the caller can tell a failed batch from a batch with no
    entities. A response that parses to nothing usable is an empty result,
    not a failure.
    """
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
        return None
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
    stats: ExtractionStats | None = None,
) -> list[dict]:
    """Phase 2 over ingest-shaped chunk records; returns entities-table rows.

    Each chunk dict needs ``chunk`` (text), ``source``, ``chunk_index``, and
    ``page_start``. Extractor kinds degrade independently: regex always runs,
    spaCy kinds are skipped without a loaded model, LLM kinds without a
    provider, so a partial toolchain yields partial extraction, never failure.
    Pass ``stats`` to observe how many LLM batches ran and how many the
    provider failed; a failed batch contributes no rows either way.
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
            if stats is not None:
                stats.llm_batches += 1
                if batch_found is None:
                    stats.llm_batches_failed += 1
            if batch_found is None:
                continue
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
