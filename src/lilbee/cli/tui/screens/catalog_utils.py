"""Catalog data types, row builders, and formatting helpers.

The catalog renders two distinct row shapes side by side: locally
installed / installable GGUFs (``LocalCatalogRow``) and cloud chat
models accessed through a provider's API (``FrontierCatalogRow``).
They share enough surface area that grouping and search reuse the
same helpers, but they carry different metadata and pull from
different sources, so they're separate types under a sealed
``CatalogRow`` union rather than a single optional-fields dataclass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from lilbee.catalog import PARAM_COUNT_RE, CatalogModel, ModelFamily, ModelVariant, extract_quant
from lilbee.modelhub.model_manager import RemoteModel

# SI thresholds for short download counts ("12.3M" / "456K") and binary
# thresholds for sizes ("4.2 GB" / "768 MB").
_DOWNLOADS_PER_M = 1_000_000
_DOWNLOADS_PER_K = 1_000
_MB_PER_GB = 1024


@dataclass
class LocalCatalogRow:
    """A row in the catalog backed by a local GGUF (installable or installed).

    ``name`` is the human-readable display label (e.g. "Qwen3 0.6B").
    ``ref`` is the canonical identifier used for config persistence:
    ``hf_repo`` for catalog rows, ``hf_repo/filename`` for installed
    native models, and the provider's ref shape for remote/API rows.
    """

    name: str
    task: str
    params: str
    size: str
    quant: str
    downloads: str
    featured: bool
    installed: bool
    sort_downloads: int
    sort_size: float
    ref: str = ""
    backend: str = ""
    variant: ModelVariant | None = None
    family: ModelFamily | None = None
    catalog_model: CatalogModel | None = None
    remote_model: RemoteModel | None = None


class KeyStatus(Enum):
    """Whether the user has the API key needed to use a frontier model."""

    READY = "ready"
    MISSING_KEY = "missing_key"


@dataclass
class FrontierCatalogRow:
    """A row in the catalog backed by a cloud provider's chat API.

    Frontier rows skip the local-model fields (size on disk, quant,
    GGUF filename) because they don't apply: the model lives on the
    provider's infrastructure.
    """

    name: str
    ref: str
    task: str
    provider: str  # Display label, e.g. "Gemini" / "OpenAI" / "Anthropic".
    provider_id: str  # Canonical id used for the API key field, e.g. "gemini".
    key_status: KeyStatus


# Sealed union: any catalog renderer dispatches on isinstance of these
# two types and exhaustively handles both.
CatalogRow = LocalCatalogRow | FrontierCatalogRow


def parse_param_label(name: str) -> str:
    """Extract parameter count label from model name (e.g. '8B', '0.6B')."""
    from lilbee.catalog import PARAM_COUNT_RE

    match = PARAM_COUNT_RE.search(name)
    return match.group(1).upper() if match else "--"


def _format_downloads(n: int) -> str:
    if n >= _DOWNLOADS_PER_M:
        return f"{n / _DOWNLOADS_PER_M:.1f}M"
    if n >= _DOWNLOADS_PER_K:
        return f"{n / _DOWNLOADS_PER_K:.0f}K"
    return str(n)


def _format_size_mb(size_mb: int) -> str:
    """Format size in MB to a human-readable string."""
    if size_mb == 0:
        return "--"
    if size_mb >= _MB_PER_GB:
        return f"{size_mb / _MB_PER_GB:.1f} GB"
    return f"{size_mb} MB"


def format_size_gb(size_gb: float) -> str:
    """Format size in GB to a human-readable string."""
    if size_gb <= 0:
        return "--"
    return f"{size_gb:.1f} GB"


def _is_param_count(label: str) -> bool:
    """True when label looks like a parameter count (e.g. '8B', '0.6B')."""
    return bool(PARAM_COUNT_RE.fullmatch(label))


def variant_to_row(v: ModelVariant, f: ModelFamily, installed: bool) -> LocalCatalogRow:
    """Convert a ModelVariant + family to a LocalCatalogRow."""
    # Avoid duplicating the param count when the family name already ends with it.
    if v.param_count and not f.name.endswith(v.param_count):
        label = f"{f.name} {v.param_count}"
    else:
        label = f.name
    params = v.param_count if _is_param_count(v.param_count) else "--"
    return LocalCatalogRow(
        name=label,
        task=f.task,
        params=params,
        size=_format_size_mb(v.size_mb),
        quant=v.quant or "--",
        downloads="--",
        featured=True,
        installed=installed,
        sort_downloads=0,
        sort_size=v.size_mb / 1024,
        ref=v.hf_repo,
        backend="native",
        variant=v,
        family=f,
    )


def catalog_to_row(m: CatalogModel, installed: bool) -> LocalCatalogRow:
    """Convert a CatalogModel to a LocalCatalogRow."""
    quant = extract_quant(m.gguf_filename)
    return LocalCatalogRow(
        name=m.display_name,
        task=m.task,
        params=parse_param_label(m.display_name),
        size=format_size_gb(m.size_gb),
        quant=quant or "--",
        downloads=_format_downloads(m.downloads) if m.downloads > 0 else "--",
        featured=m.featured,
        installed=installed,
        sort_downloads=m.downloads,
        sort_size=m.size_gb,
        ref=m.ref,
        backend="native",
        catalog_model=m,
    )


def remote_to_row(rm: RemoteModel) -> LocalCatalogRow:
    """Convert a RemoteModel to a LocalCatalogRow.

    Remote/Ollama models live on a local-network backend and present the
    same operational shape as installed GGUFs (no API key gating, name
    suffices as ref). They render through the local row path.
    """
    return LocalCatalogRow(
        name=rm.name,
        task=rm.task,
        params=rm.parameter_size or "--",
        size="--",
        quant="--",
        downloads="--",
        featured=False,
        installed=True,
        sort_downloads=0,
        sort_size=0.0,
        ref=rm.name,
        backend=rm.provider.lower(),
        remote_model=rm,
    )


def frontier_row_from_remote(
    rm: RemoteModel, *, provider_id: str, key_status: KeyStatus
) -> FrontierCatalogRow:
    """Convert a discovered cloud chat model to a FrontierCatalogRow."""
    return FrontierCatalogRow(
        name=rm.name,
        ref=rm.name,
        task=rm.task,
        provider=rm.provider,
        provider_id=provider_id,
        key_status=key_status,
    )


# Column sort key extractors. Frontier rows fold into Name sort (the
# only sort the picker exposes today); the other keys read fields that
# only LocalCatalogRow carries, so the catalog screen sorts local and
# frontier rows independently and concatenates them.
SORT_KEYS = {
    "Name": lambda r: r.name.lower(),
    "Task": lambda r: getattr(r, "task", ""),
    "Backend": lambda r: getattr(r, "backend", "").lower(),
    "Params": lambda r: _param_sort_value(getattr(r, "params", "")),
    "Size": lambda r: getattr(r, "sort_size", 0.0),
    "Quant": lambda r: getattr(r, "quant", ""),
    "Downloads": lambda r: getattr(r, "sort_downloads", 0),
}


def _param_sort_value(params: str) -> float:
    """Convert param label to sortable float (e.g. '8B' -> 8.0)."""
    match = re.search(r"(\d+\.?\d*)", params)
    return float(match.group(1)) if match else 0.0


def matches_search(row: CatalogRow, search: str) -> bool:
    """Return True if the row matches the search text (hyphen/underscore-insensitive).

    Local rows match against name/task/params/quant/backend; frontier
    rows match against name + provider so users can type "gemini" and
    see every Gemini model regardless of suffix.
    """
    if not search:
        return True
    needle = _normalize_for_search(search)
    if isinstance(row, FrontierCatalogRow):
        return any(
            needle in _normalize_for_search(field)
            for field in (row.name, row.provider, row.provider_id)
        )
    return any(
        needle in _normalize_for_search(field)
        for field in (row.name, row.task, row.params, row.quant, row.backend)
    )


def _normalize_for_search(value: str) -> str:
    return value.lower().replace("-", " ").replace("_", " ")
