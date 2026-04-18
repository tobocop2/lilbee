"""Catalog data types, row builders, and formatting helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from lilbee.catalog import PARAM_COUNT_RE, CatalogModel, ModelFamily, ModelVariant
from lilbee.model_manager import RemoteModel
from lilbee.models import FEATURED_STAR


@dataclass
class TableRow:
    """A row in the catalog DataTable with source metadata.
    ``name`` is the human-readable display label (e.g. "Qwen3 0.6B").
    ``ref`` is the canonical name:tag identifier used for config persistence
    (e.g. "qwen3:0.6b").  When ``ref`` is empty, fall back to ``name``.
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
    prefix = "* " if v.recommended else ""
    # Avoid duplicating the tag when the family name already ends with it.
    if f.name.endswith(v.param_count):
        label = f"{prefix}{f.name}"
    else:
        label = f"{prefix}{f.name} {v.param_count}"
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
        ref=f"{f.slug}:{v.tag}",
        backend="native",
        variant=v,
        family=f,
    )


def catalog_to_row(m: CatalogModel, installed: bool) -> TableRow:
    """Convert a CatalogModel to a TableRow."""
    quant = _extract_quant_from_filename(m.gguf_filename)
    return TableRow(
        name=m.display_name,
        task=m.task,
        params=parse_param_label(m.tag),
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


def _extract_quant_from_filename(filename: str) -> str:
    """Extract quantization label from a GGUF filename pattern."""
    m = re.search(r"(Q\d[A-Z0-9_]*)", filename, re.IGNORECASE)
    return m.group(1).upper() if m else ""


def row_display_name(row: TableRow) -> str:
    """Build the display name with featured/installed markers."""
    parts: list[str] = []
    if row.featured:
        parts.append(FEATURED_STAR)
    parts.append(row.name)
    if row.installed:
        parts.append("[installed]")
    return " ".join(parts)


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
    """Return True if the row matches the search text."""
    if not search:
        return True
    return (
        search in row.name.lower()
        or search in row.task.lower()
        or search in row.params.lower()
        or search in row.quant.lower()
        or search in row.backend.lower()
    )
