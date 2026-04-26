"""Catalog data types, row builders, and formatting helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from lilbee.catalog import PARAM_COUNT_RE, CatalogModel, ModelFamily, ModelVariant, extract_quant
from lilbee.model_manager import RemoteModel


@dataclass
class TableRow:
    """A row in the catalog grid or list view with source metadata.

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


def parse_param_label(name: str) -> str:
    """Extract parameter count label from model name (e.g. '8B', '0.6B')."""
    from lilbee.catalog import PARAM_COUNT_RE

    match = PARAM_COUNT_RE.search(name)
    return match.group(1).upper() if match else "--"


def _format_downloads(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _format_size_mb(size_mb: int) -> str:
    """Format size in MB to a human-readable string."""
    if size_mb == 0:
        return "--"
    if size_mb >= 1024:
        return f"{size_mb / 1024:.1f} GB"
    return f"{size_mb} MB"


def format_size_gb(size_gb: float) -> str:
    """Format size in GB to a human-readable string."""
    if size_gb <= 0:
        return "--"
    return f"{size_gb:.1f} GB"


def _is_param_count(label: str) -> bool:
    """True when label looks like a parameter count (e.g. '8B', '0.6B')."""
    return bool(PARAM_COUNT_RE.fullmatch(label))


def variant_to_row(v: ModelVariant, f: ModelFamily, installed: bool) -> TableRow:
    """Convert a ModelVariant + family to a TableRow."""
    # Avoid duplicating the param count when the family name already ends with it.
    if v.param_count and not f.name.endswith(v.param_count):
        label = f"{f.name} {v.param_count}"
    else:
        label = f.name
    params = v.param_count if _is_param_count(v.param_count) else "--"
    return TableRow(
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


def catalog_to_row(m: CatalogModel, installed: bool) -> TableRow:
    """Convert a CatalogModel to a TableRow."""
    quant = extract_quant(m.gguf_filename)
    return TableRow(
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


def remote_to_row(rm: RemoteModel) -> TableRow:
    """Convert a RemoteModel to a TableRow."""
    return TableRow(
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


# Column sort key extractors
SORT_KEYS = {
    "Name": lambda r: r.name.lower(),
    "Task": lambda r: r.task,
    "Backend": lambda r: r.backend.lower(),
    "Params": lambda r: _param_sort_value(r.params),
    "Size": lambda r: r.sort_size,
    "Quant": lambda r: r.quant,
    "Downloads": lambda r: r.sort_downloads,
}


def _param_sort_value(params: str) -> float:
    """Convert param label to sortable float (e.g. '8B' -> 8.0)."""
    match = re.search(r"(\d+\.?\d*)", params)
    return float(match.group(1)) if match else 0.0


def matches_search(row: TableRow, search: str) -> bool:
    """Return True if the row matches the search text (hyphen/underscore-insensitive)."""
    if not search:
        return True
    needle = _normalize_for_search(search)
    return any(
        needle in _normalize_for_search(field)
        for field in (row.name, row.task, row.params, row.quant, row.backend)
    )


def _normalize_for_search(value: str) -> str:
    return value.lower().replace("-", " ").replace("_", " ")
