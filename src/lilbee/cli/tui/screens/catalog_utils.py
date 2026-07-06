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
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from lilbee.catalog import PARAM_COUNT_RE, CatalogModel, ModelFamily, ModelVariant, extract_quant
from lilbee.catalog.types import KeyStatus, ModelCompat, ModelTask
from lilbee.modelhub.model_manager import RemoteModel
from lilbee.providers.model_ref import format_remote_ref
from lilbee.runtime.hardware import FitChip

# Backend label for local GGUF models. Renderers drop this pill (the backend is
# implied for local models) and only show it for non-native SDK backends.
NATIVE_BACKEND = "native"


class CatalogRowKind(StrEnum):
    """Discriminator for the sealed CatalogRow union."""

    LOCAL = "local"
    FRONTIER = "frontier"


# Tab IDs for the 6-tab catalog shell. Discover is the curated landing,
# the four task tabs each render a single per-task grid, Library is the
# personal-encyclopedia view of installed local + activated cloud APIs.
TAB_DISCOVER = "discover"
TAB_CHAT = "chat"
TAB_EMBED = "embed"
TAB_VISION = "vision"
TAB_RERANK = "rerank"
TAB_LIBRARY = "library"

# Order matters: numbered shortcuts 1-6 follow this sequence.
ALL_TAB_IDS: tuple[str, ...] = (
    TAB_DISCOVER,
    TAB_CHAT,
    TAB_EMBED,
    TAB_VISION,
    TAB_RERANK,
    TAB_LIBRARY,
)
TASK_TAB_IDS: tuple[str, ...] = (TAB_CHAT, TAB_EMBED, TAB_VISION, TAB_RERANK)

# Maps ModelTask -> the per-task tab id that renders its rows. Featured
# items still appear pinned at the top of their task tab; cross-task
# discovery happens on the Discover landing.
TASK_TO_TAB_ID: dict[ModelTask, str] = {
    ModelTask.CHAT: TAB_CHAT,
    ModelTask.EMBEDDING: TAB_EMBED,
    ModelTask.VISION: TAB_VISION,
    ModelTask.RERANK: TAB_RERANK,
}
TAB_ID_TO_TASK: dict[str, ModelTask] = {v: k for k, v in TASK_TO_TAB_ID.items()}


class SourceMode(StrEnum):
    """Per-task tab filter for which row source backs the visible cards.

    LOCAL hides frontier rows (the default; mirrors the legacy mega-grid
    behavior). CLOUD shows only frontier rows for the active task. BOTH
    unions them so users can compare a local Llama with a cloud Llama
    side by side. Cycled via the ``c`` keybinding.
    """

    LOCAL = "local"
    CLOUD = "cloud"
    BOTH = "both"


_SOURCE_MODE_CYCLE: tuple[SourceMode, ...] = (SourceMode.LOCAL, SourceMode.CLOUD, SourceMode.BOTH)


def next_source_mode(current: SourceMode) -> SourceMode:
    """Return the next SourceMode in the LOCAL -> CLOUD -> BOTH -> LOCAL cycle."""
    idx = _SOURCE_MODE_CYCLE.index(current)
    return _SOURCE_MODE_CYCLE[(idx + 1) % len(_SOURCE_MODE_CYCLE)]


def task_to_tab_id(task: ModelTask | str) -> str:
    """Return the per-task tab id for a ModelTask or its string value.

    Accepts the string form because catalog rows carry ``task`` as a
    raw string (matching how HF API and the row builders return it),
    while the routing tables are keyed on the enum.
    """
    if isinstance(task, ModelTask):
        return TASK_TO_TAB_ID[task]
    try:
        return TASK_TO_TAB_ID[ModelTask(task)]
    except (KeyError, ValueError) as exc:
        raise KeyError(f"unknown task: {task!r}") from exc


# SI thresholds for short download counts ("12.3M" / "456K") and binary
# thresholds for sizes ("4.2 GB" / "768 MB").
_DOWNLOADS_PER_M = 1_000_000
_DOWNLOADS_PER_K = 1_000
_MB_PER_GB = 1024


@dataclass(frozen=True)
class SizeVariant:
    """One size/quant variant of a model family for the family-as-card strip.

    ``label`` renders inline on the card (e.g. "8B Q4_K_M"). ``ref`` is
    the canonical pull target for this specific variant. ``fit`` is the
    fit chip computed against the host's available memory; ``None``
    when the hardware probe has not yet run.
    """

    label: str
    quant: str
    size_gb: float
    ref: str
    fit: FitChip | None = None


@dataclass
class LocalCatalogRow:
    """A row in the catalog backed by a local GGUF (installable or installed).

    ``name`` is the human-readable display label (e.g. "Qwen3 0.6B").
    ``ref`` is the canonical identifier used for config persistence:
    ``hf_repo`` for catalog rows, ``hf_repo/filename`` for installed
    native models, and the provider's ref shape for remote/API rows.
    ``size_variants`` carries every quant for a family-aggregated row,
    so the card can render an inline chip strip and the detail drawer
    can list all sizes. ``fit`` is the chip for the row's primary
    variant.
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
    size_variants: list[SizeVariant] = field(default_factory=list)
    fit: FitChip | None = None
    compat: ModelCompat = ModelCompat.UNKNOWN
    kind: Literal[CatalogRowKind.LOCAL] = CatalogRowKind.LOCAL


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
    kind: Literal[CatalogRowKind.FRONTIER] = CatalogRowKind.FRONTIER


# Sealed union discriminated on .kind. Pattern-match (or compare) on row.kind
# to dispatch instead of isinstance, so adding a new row type is one place.
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


def family_to_size_variants(family: ModelFamily) -> list[SizeVariant]:
    """Build the size-chip strip for a featured ModelFamily.

    Variants are returned in increasing size order so the chip strip
    reads compact-to-large left-to-right. ``fit`` is left ``None``;
    the catalog screen fills it in once the hardware probe has run.
    """
    variants = sorted(family.variants, key=lambda v: v.size_mb)
    return [
        SizeVariant(
            label=_size_variant_label(v),
            quant=v.quant or "--",
            size_gb=v.size_mb / 1024,
            ref=v.hf_repo,
            fit=None,
        )
        for v in variants
    ]


def _size_variant_label(v: ModelVariant) -> str:
    """Render a compact label for a ModelVariant chip (e.g. '8B Q4_K_M')."""
    pieces = [p for p in (v.param_count, v.quant) if p]
    return " ".join(pieces) if pieces else "--"


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
        backend=NATIVE_BACKEND,
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
        backend=NATIVE_BACKEND,
        catalog_model=m,
        compat=m.compat,
    )


def remote_to_row(rm: RemoteModel) -> LocalCatalogRow:
    """Convert a RemoteModel to a LocalCatalogRow.

    ``ref`` is the canonical ``provider/name`` form so it round-trips
    through ``Config.chat_model``'s validator without a per-call-site
    fixup.
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
        ref=format_remote_ref(rm.name, rm.provider),
        backend=rm.provider.lower(),
        remote_model=rm,
    )


def frontier_row_from_remote(
    rm: RemoteModel, *, provider_id: str, key_status: KeyStatus
) -> FrontierCatalogRow:
    """Convert a discovered cloud chat model to a FrontierCatalogRow.

    ``ref`` is the canonical ``provider/name`` form so callers pass it
    straight to ``Config.chat_model`` without re-prefixing.
    """
    return FrontierCatalogRow(
        name=rm.name,
        ref=format_remote_ref(rm.name, rm.provider),
        task=rm.task,
        provider=rm.provider,
        provider_id=provider_id,
        key_status=key_status,
    )


# Column sort key extractors. Local-only because every column except
# Name reads a field FrontierCatalogRow doesn't carry, and the catalog
# screen sorts local and frontier rows independently before concat.
SORT_KEYS: dict[str, Callable[[LocalCatalogRow], Any]] = {
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


def row_delete_id(row: CatalogRow) -> str | None:
    """Return the model_manager-compatible identifier for *row*.

    Remote rows hand back the bare ``RemoteModel.name`` because the
    Ollama HTTP API keys models by bare name, while ``ref`` carries the
    canonical ``ollama/<name>`` chat_model form.
    """
    if row.kind == CatalogRowKind.FRONTIER:
        return row.ref or None
    if row.remote_model is not None:
        return row.remote_model.name or None
    return row.ref or None


def matches_search(row: CatalogRow, search: str) -> bool:
    """Return True if the row matches the search text (hyphen/underscore-insensitive).

    Local rows match against name/task/params/quant/backend; frontier
    rows match against name + provider so users can type "gemini" and
    see every Gemini model regardless of suffix.
    """
    if not search:
        return True
    needle = _normalize_for_search(search)
    if row.kind == CatalogRowKind.FRONTIER:
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
