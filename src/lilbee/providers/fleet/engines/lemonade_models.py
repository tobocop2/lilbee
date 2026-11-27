"""Catalog rows for the NPU models that the Lemonade engine runs, read from the publisher's list."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from lilbee.catalog.models import CatalogModel, estimate_min_ram_gb
from lilbee.catalog.types import ModelFormat, ModelTask
from lilbee.catalog.xdna import XDNA_TAG_RE

log = logging.getLogger(__name__)

_PLATFORM_XDNA2 = "aie2p"
_NAME_SUFFIX_XDNA2 = "-NPU2"
_LABEL_EMBEDDINGS = "embeddings"
_LABELS_NOT_LISTED = frozenset({"decision", "transcription", "realtime-transcription"})
"""A decision model writes no text and a speech model transcribes audio: neither chats."""
_PARAMS_RE = re.compile(r"(\d+(?:\.\d+)?)([BM])")
_PARAMS_PER_UNIT = {"B": 10**9, "M": 10**6}
_BYTES_PER_GIB = 1024**3
_CONTEXT_UNIT = 1024
_DESCRIPTION_LIMIT = 120


def _number(value: Any) -> bool:
    """True for a finite number; a bool, ``NaN`` and infinity are not one."""
    # isinstance: the value is untyped JSON from the network
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _on_xdna2(entry: Mapping[str, Any]) -> bool:
    """Whether the entry targets XDNA2: by its platform list, else by its published name."""
    platforms = entry.get("supported_platforms")
    if platforms is None:
        return str(entry.get("name", "")).endswith(_NAME_SUFFIX_XDNA2)
    return _PLATFORM_XDNA2 in platforms


def _parameters(entry: Mapping[str, Any]) -> int:
    """The count in ``details.parameter_size`` (``35B-A3B`` reads 35B); ``ValueError`` if none."""
    found = _PARAMS_RE.match(entry["details"]["parameter_size"])
    if found is None:
        raise ValueError("no parameter count")
    return int(Decimal(found.group(1)) * _PARAMS_PER_UNIT[found.group(2)])


def _description(name: str, context: int) -> str:
    """The published name, its spoken form, the silicon and the context window."""
    window = f"{context // _CONTEXT_UNIT}k" if context >= _CONTEXT_UNIT else str(context)
    spoken = name.lower().replace("-", " ").replace("_", " ")
    return f"{name}, {spoken}, AMD NPU, {window} context"[:_DESCRIPTION_LIMIT]


def _row(ref: str, entry: Mapping[str, Any]) -> CatalogModel:
    """The row of one listed entry; an invalid entry raises ``ValueError`` or ``KeyError``."""
    name = entry["name"]
    size = entry["size"]
    context = entry["default_context_length"]
    footprint = entry.get("footprint")
    # isinstance: the name is untyped JSON from the network
    if not (isinstance(name, str) and name and _number(size) and size > 0 and _number(context)):
        raise ValueError("bad name, size or context")
    if footprint is not None and not (_number(footprint) and footprint > 0):
        raise ValueError("bad footprint")
    size_gb = round(size / _BYTES_PER_GIB, 1)
    is_embedding = _LABEL_EMBEDDINGS in (entry.get("label") or [])
    return CatalogModel(
        hf_repo=ref,
        gguf_filename="",
        size_gb=size_gb,
        min_ram_gb=estimate_min_ram_gb(size_gb) if footprint is None else round(footprint, 1),
        footprint_gb=0.0 if footprint is None else float(footprint),
        description=_description(name, int(context)),
        featured=False,
        downloads=0,
        task=ModelTask.EMBEDDING if is_embedding else ModelTask.CHAT,
        params=_parameters(entry),
        context_length=int(context),
        format=ModelFormat.XDNA,
        formats=frozenset({ModelFormat.XDNA}),
    )


def parse_models(payload: Any) -> list[CatalogModel]:
    """The rows for the XDNA2 chat and embedding entries of the publisher's list, in file order.

    ``ValueError`` when the payload is not a list of families. An entry that is not listed
    or is invalid is skipped and the others stay.
    """
    # isinstance: the payload is untyped JSON from the network
    families = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(families, dict) or not all(isinstance(t, dict) for t in families.values()):
        raise ValueError("The model list has no families of models.")
    rows: list[CatalogModel] = []
    for family, tags in families.items():
        for tag, entry in tags.items():
            ref = f"{family}:{tag}"
            if not XDNA_TAG_RE.fullmatch(ref):
                log.debug("Skipping NPU model %r: its ref is not a model tag", ref)
                continue
            try:
                if not _on_xdna2(entry) or _LABELS_NOT_LISTED & set(entry.get("label") or []):
                    continue
                rows.append(_row(ref, entry))
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                log.debug("Skipping NPU model %r: %s", ref, exc)
    return rows
