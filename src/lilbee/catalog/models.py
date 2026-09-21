"""Catalog dataclasses and pydantic types. Imports only the catalog's leaf modules."""

import functools
import re
from dataclasses import dataclass

from pydantic import BaseModel

from lilbee.catalog.refs import ggml_bytes_per_param, ggml_quant_block_sizes, quant_label
from lilbee.catalog.types import ModelCompat, ModelTask

# Minimum recommended floor so a tiny model still reports a sane RAM ask.
_MIN_RAM_FLOOR_GB = 2.0
# Working-set multiple over the on-disk size (weights + KV cache + overhead).
_RAM_OVER_SIZE_FACTOR = 1.5

_BYTES_PER_GB = 1024**3


@functools.cache
def _default_bytes_per_param() -> float:
    """Bytes per weight of Q4_K, the type a filename naming no quant is sized as.

    Q4_K heads the pull path's quant preference, so it is the type a pull would
    most likely land on.
    """
    block, type_size = ggml_quant_block_sizes()["Q4_K"]
    return type_size / block


# A quant ggml does not name still says how many bits it packs. One fp16 scale
# per group costs an eighth on top, whatever the width, because a group is sized
# to the width: 1-bit in groups of 128, 2-bit in 64, 4-bit in 32 all carry two
# bytes per group. Reading the width beats falling back to Q4_K_M, which reports
# a 2-bit file at more than twice its size.
_SCALE_OVERHEAD = 1.125
_BITS_PER_BYTE = 8

_WIDTH_RE = re.compile(r"I?Q(\d)")


def _width_bytes_per_param(quant: str) -> float | None:
    """Bytes per weight from the bit width *quant* names, or None if it names none."""
    match = _WIDTH_RE.match(quant)
    if match is None:
        return None
    return int(match.group(1)) / _BITS_PER_BYTE * _SCALE_OVERHEAD


def _quant_bytes_per_param(gguf_filename: str) -> float:
    """Bytes per weight of the ggml type *gguf_filename* names, or Q4_K when it names none.

    The type's own block arithmetic answers first, so a label cannot read under
    what its tensors physically cost. The bit width is the last resort, for a
    publisher's own naming that ggml has no type for.
    """
    quant = quant_label(gguf_filename)
    rate = ggml_bytes_per_param(quant)
    if rate is not None:
        return rate
    width = _width_bytes_per_param(quant)
    return width if width is not None else _default_bytes_per_param()


def estimate_min_ram_gb(size_gb: float) -> float:
    """Estimate the RAM a model needs from its on-disk size (single source)."""
    return round(max(_MIN_RAM_FLOOR_GB, size_gb * _RAM_OVER_SIZE_FACTOR), 1)


def estimate_size_gb(params: int, gguf_filename: str) -> float:
    """Approximate the on-disk GB of *gguf_filename* from a model's parameter count.

    A lower bound, and the browse list renders it as approximate. llama.cpp
    promotes ``output.weight`` and an untied ``token_embd`` above the ftype and
    leaves the norms in F32, so a published file costs more per weight than the
    type its name carries; how much more needs the header's vocabulary and
    embedding lengths, which a listing row does not have.

    This is the one place a size cannot be read: the HF listing API reports a
    parameter count (``gguf.total``) and no per-file bytes, and getting the real
    figure for a 50-row page means 50 more requests. Every path that acts on a
    size resolves the exact one for the single file in play.
    """
    if params <= 0:
        return 0.0  # unknown: display as "?" in UI
    return round(params * _quant_bytes_per_param(gguf_filename) / _BYTES_PER_GB, 1)


class HfGgufMeta(BaseModel):
    """GGUF metadata returned by the HF API when expand=gguf is requested.

    ModelInfo.gguf is typed as ``dict | None`` upstream, so we validate it ourselves.

    ``total`` is the model's parameter count, not a byte size; ``totalFileSize``
    holds bytes. Verified against repos that name their own parameter count:
    Qwen3-8B-GGUF reports ``total=8_190_000_000`` against 4.7 GB of files.
    """

    total: int = 0
    architecture: str = ""
    context_length: int = 0


@dataclass
class DownloadProgress:
    """Human-readable snapshot of download progress.

    ``percent`` is a float (0.0 to 100.0) so the ProgressBar renders smooth
    fractional movement during multi-GB downloads. Call sites that need
    an integer for display format it themselves.
    """

    percent: float
    detail: str
    is_cache_hit: bool


@dataclass(frozen=True)
class CatalogModel:
    """One catalog entry, keyed by HuggingFace repo. ``gguf_filename`` may be a glob."""

    hf_repo: str
    gguf_filename: str
    size_gb: float
    min_ram_gb: float
    description: str
    featured: bool
    downloads: int
    task: ModelTask
    architecture: str = ""
    compat: ModelCompat = ModelCompat.UNKNOWN
    # Parameter count. Size buckets key off this rather than on-disk bytes so a
    # model keeps its bucket across quants. 0 when the repo publishes no GGUF
    # metadata.
    params: int = 0
    # HuggingFace trending rank. 0 when the listing omits it.
    trending_score: int = 0
    # Safety-stripped (abliterated/uncensored) per the repo's HF tags. Browse
    # rows carry it; recommendation rails exclude rows that set it.
    safety_stripped: bool = False

    @property
    def ref(self) -> str:
        """Browse-time ref (the HF repo); concrete filename is resolved at install."""
        return self.hf_repo

    @property
    def display_name(self) -> str:
        """Human-readable label derived from the HuggingFace repo id."""
        # circular: models -> formatting via clean_display_name
        from lilbee.catalog.formatting import clean_display_name

        return clean_display_name(self.hf_repo)


@dataclass(frozen=True)
class CatalogResult:
    """Paginated catalog result."""

    total: int | None
    limit: int
    offset: int
    models: list[CatalogModel]
    has_more: bool = False


@dataclass(frozen=True)
class PageWindow:
    """The part of a page left for the rows paged after the ones held locally."""

    rest_offset: int
    rest_limit: int


def page_window(leading_count: int, offset: int, limit: int) -> PageWindow:
    """The window left of ``[offset, offset + limit)`` after *leading_count* local rows."""
    covered = min(offset + limit, leading_count) - min(offset, leading_count)
    return PageWindow(rest_offset=max(0, offset - leading_count), rest_limit=limit - covered)


@dataclass(frozen=True)
class HfPage:
    """One page of HuggingFace API results."""

    models: list[CatalogModel]
    has_more: bool


def dedupe_models(models: list[CatalogModel]) -> list[CatalogModel]:
    """Models in first-seen order, later repeats dropped."""
    seen: set[str] = set()
    unique: list[CatalogModel] = []
    for model in models:
        if model.hf_repo not in seen:
            seen.add(model.hf_repo)
            unique.append(model)
    return unique


@dataclass(frozen=True)
class ModelVariant:
    """One quantization within a model family. ``filename`` may be a glob."""

    hf_repo: str
    filename: str
    param_count: str
    quant: str
    size_mb: int
    mmproj_filename: str = ""
    compat: ModelCompat = ModelCompat.UNKNOWN
    safety_stripped: bool = False


@dataclass(frozen=True)
class ModelFamily:
    """A group of related model variants (e.g. Qwen3 in multiple sizes)."""

    slug: str  # family slug for building refs: "qwen3"
    name: str  # display name: "Qwen3"
    task: ModelTask
    description: str
    variants: tuple[ModelVariant, ...]
