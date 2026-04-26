"""Metadata derived from :class:`lilbee.config.Config` field declarations.

These field sets used to live in ``lilbee.server.handlers``, but the CLI and
MCP entry points now need them too (to know which fields a per-vault
``config.toml`` is allowed to overlay). Keeping them next to the model they
describe, and away from the FastAPI bootstrap, lets non-server modules
import them without paying the handlers import cost.
"""

from __future__ import annotations

import types
from typing import Union, get_args, get_origin

from pydantic.fields import FieldInfo

from lilbee.config import Config

MODEL_ROLE_FIELDS: frozenset[str] = frozenset(
    {"chat_model", "embedding_model", "vision_model", "reranker_model"}
)


def _get_extra(info: FieldInfo, key: str, default: bool = False) -> bool:
    """Read a boolean flag from a field's ``json_schema_extra``."""
    extra = info.json_schema_extra
    if isinstance(extra, dict):
        return bool(extra.get(key, default))
    return default


def _is_nullable(info: FieldInfo) -> bool:
    """Return True if ``None`` is part of the field's type union."""
    origin = get_origin(info.annotation)
    if origin is Union or origin is types.UnionType:
        return type(None) in get_args(info.annotation)
    return False


def _derive_field_sets() -> tuple[
    types.MappingProxyType[str, bool], frozenset[str], frozenset[str]
]:
    """Derive writable, reindex, and public field sets from Config metadata.

    Model-role fields are public even when not writable: they are set via
    ``PUT /api/models/<role>`` but must still surface in ``GET /api/config``.
    """
    writable: dict[str, bool] = {}
    reindex: set[str] = set()
    public: set[str] = set()
    for name, info in Config.model_fields.items():
        if _get_extra(info, "writable"):
            writable[name] = _is_nullable(info)
            if not _get_extra(info, "write_only") and _get_extra(info, "public", default=True):
                public.add(name)
            if _get_extra(info, "reindex"):
                reindex.add(name)
        elif name in MODEL_ROLE_FIELDS:
            public.add(name)
    return types.MappingProxyType(writable), frozenset(reindex), frozenset(public)


WRITABLE_CONFIG_FIELDS, REINDEX_FIELDS, PUBLIC_CONFIG_FIELDS = _derive_field_sets()
