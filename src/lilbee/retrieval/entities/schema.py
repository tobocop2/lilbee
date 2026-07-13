"""Typed entity extraction: the schema artifact and the entities table.

The extraction taxonomy is induced from the corpus, not fixed: a general NER
tag set has no notion of the identifier types a specific corpus carries.
Sync induces a schema from a corpus sample and applies it automatically; the
JSON artifact it writes is the contract for later tuning: a human (or agent)
can edit types, patterns, and synonyms after the fact, and the next sync
detects the edit and re-applies it across the corpus.
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path

import pyarrow as pa
from pydantic import BaseModel, Field, field_validator

log = logging.getLogger(__name__)

SCHEMA_FILENAME = "entity_schema.json"


class ExtractorKind(Enum):
    """How a type's mentions are found, cheapest first."""

    REGEX = "regex"
    SPACY = "spacy"
    LLM = "llm"


class EntityType(BaseModel):
    """One induced type: how to find it and what to call it in questions."""

    name: str = Field(min_length=1, max_length=64)
    kind: ExtractorKind
    # REGEX kinds compile this; SPACY kinds name a spaCy label (PERSON, ORG,
    # DATE, ...); LLM kinds carry a one-line description for the prompt.
    pattern: str = ""
    description: str = ""
    # Question nouns that mean this type ("part number", "part numbers").
    synonyms: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _slugify(cls, v: str) -> str:
        slug = "_".join(v.strip().lower().split())
        if not slug.replace("_", "").isalnum():
            raise ValueError(f"type name must be alphanumeric words: {v!r}")
        return slug


# Plural forms the suffix rules below can't produce, mapped both ways.
_IRREGULAR_PLURALS = {
    "person": "people",
    "man": "men",
    "woman": "women",
    "child": "children",
    "foot": "feet",
    "tooth": "teeth",
    "mouse": "mice",
    "goose": "geese",
}
_IRREGULAR_SINGULARS = {plural: singular for singular, plural in _IRREGULAR_PLURALS.items()}


def noun_variants(noun: str) -> set[str]:
    """Normalized spelling variants of a noun phrase: itself plus
    singular/plural forms of its last word ("tail numbers" ~ "tail number",
    "people" ~ "person"). Over-generated junk forms match nothing; a missed
    form only fails to resolve, never resolves wrongly.
    """
    normalized = " ".join(noun.strip().lower().split())
    if not normalized:
        return set()
    head, _, last = normalized.rpartition(" ")
    prefix = head + " " if head else ""
    forms = {last}
    if last in _IRREGULAR_PLURALS:
        forms.add(_IRREGULAR_PLURALS[last])
    if last in _IRREGULAR_SINGULARS:
        forms.add(_IRREGULAR_SINGULARS[last])
    if last.endswith("ies") and len(last) > len("ies"):
        forms.add(last[:-3] + "y")
    if last.endswith("y"):
        forms.add(last[:-1] + "ies")
    if last.endswith(("ses", "xes", "zes", "ches", "shes")):
        forms.add(last[:-2])
    forms.add(last[:-1] if last.endswith("s") else last + "s")
    return {prefix + form for form in forms}


class EntitySchema(BaseModel):
    """The editable extraction contract for one corpus."""

    types: list[EntityType]

    def type_for_noun(self, noun: str) -> EntityType | None:
        """Resolve a question noun (singular or plural) to a type, if any."""
        candidates = noun_variants(noun)
        for entity_type in self.types:
            names = {entity_type.name, entity_type.name.replace("_", " ")}
            names.update(entity_type.synonyms)
            expanded: set[str] = set()
            for name in names:
                expanded |= noun_variants(name)
            if candidates & expanded:
                return entity_type
        return None


def schema_path(data_dir: Path) -> Path:
    """Where the corpus's schema artifact lives."""
    return data_dir / SCHEMA_FILENAME


def load_schema(data_dir: Path) -> EntitySchema | None:
    """Read the schema artifact, or ``None`` when absent or unreadable.

    Unreadable is logged, not raised: a hand-edited artifact with a typo
    should degrade to "extraction off" rather than break sync.
    """
    path = schema_path(data_dir)
    if not path.is_file():
        return None
    try:
        return EntitySchema.model_validate_json(path.read_text())
    except Exception:
        log.warning("Entity schema at %s is unreadable; extraction skipped", path, exc_info=True)
        return None


def save_schema(schema: EntitySchema, data_dir: Path) -> Path:
    """Write the schema artifact (pretty, stable order) and return its path."""
    path = schema_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = schema.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _entities_schema(dim_unused: int | None = None) -> pa.Schema:
    return pa.schema(
        [
            pa.field("entity", pa.utf8()),
            pa.field("type", pa.utf8()),
            pa.field("normalized_value", pa.utf8()),
            pa.field("source", pa.utf8()),
            pa.field("page", pa.int32()),
            pa.field("chunk_index", pa.int32()),
            pa.field("confidence", pa.float32()),
        ]
    )
