"""Opt-in typed entity extraction: schema induction, extraction, storage."""

from lilbee.retrieval.entities.extractor import (
    extract_entities,
    induce_schema,
    normalize_value,
)
from lilbee.retrieval.entities.lifecycle import ensure_entities
from lilbee.retrieval.entities.schema import (
    EntitySchema,
    EntityType,
    ExtractorKind,
    load_schema,
    save_schema,
)

__all__ = [
    "EntitySchema",
    "EntityType",
    "ExtractorKind",
    "ensure_entities",
    "extract_entities",
    "induce_schema",
    "load_schema",
    "normalize_value",
    "save_schema",
]
