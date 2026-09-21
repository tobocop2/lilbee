"""Per-field type and value-set metadata, read off the JSON schema pydantic builds."""

from __future__ import annotations

from functools import cache
from typing import Any

from .model import Config

# Where pydantic puts the named sub-schema a field's type resolves to.
_DEFS_PREFIX = "#/$defs/"

# JSON schema type name -> the short name every lilbee surface reports.
_WIRE_TYPE_NAMES: dict[str, str] = {
    "boolean": "bool",
    "integer": "int",
    "number": "float",
    "string": "str",
    "array": "list",
    "object": "dict",
    "null": "null",
}


@cache
def _properties() -> dict[str, dict[str, Any]]:
    """Each Config field's JSON schema, keyed by field name."""
    return dict(Config.model_json_schema()["properties"])


@cache
def _definitions() -> dict[str, dict[str, Any]]:
    """The named sub-schemas a field's ``$ref`` points at, one per enum type."""
    return dict(Config.model_json_schema().get("$defs", {}))


def _resolved(schema: dict[str, Any]) -> dict[str, Any]:
    """The definition a ``$ref`` names, or the schema itself when it holds no ref."""
    ref = schema.get("$ref")
    if ref is None:
        return schema
    return _definitions()[ref.removeprefix(_DEFS_PREFIX)]


def _type_name(schema: dict[str, Any]) -> str:
    branches = schema.get("anyOf")
    if branches is not None:
        return "|".join(_type_name(branch) for branch in branches)
    return _WIRE_TYPE_NAMES[_resolved(schema)["type"]]


def field_type_name(key: str) -> str:
    """Render a field's wire type: ``int``, ``str``, ``list``, ``str|null``."""
    return _type_name(_properties()[key])


def field_value_set(key: str) -> tuple[str, ...] | None:
    """The closed set of values a field accepts, or None when the set is open.

    Only a scalar field has one: a collection's schema holds its member type
    under ``items``, which this does not descend into, so a list of enum values
    stays an open field rather than rendering as a single-select.
    """
    schema = _properties()[key]
    for branch in schema.get("anyOf", [schema]):
        values = _resolved(branch).get("enum")
        if values is not None:
            return tuple(str(value) for value in values)
    return None
