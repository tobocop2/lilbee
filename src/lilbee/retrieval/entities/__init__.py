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
    extractor_key,
    load_schema,
    merge_schemas,
    parse_schema,
    save_schema,
)

__all__ = [
    "EntitySchema",
    "EntityType",
    "ExtractorKind",
    "ensure_entities",
    "extract_entities",
    "extractor_key",
    "induce_schema",
    "load_schema",
    "merge_schemas",
    "normalize_value",
    "parse_schema",
    "save_schema",
]
