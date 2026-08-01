"""Catalog dataclasses and pydantic types: leaf module, no sibling imports."""

from dataclasses import dataclass

from pydantic import BaseModel

from lilbee.catalog.types import ModelCompat, ModelTask

# Minimum recommended floor so a tiny model still reports a sane RAM ask.
_MIN_RAM_FLOOR_GB = 2.0
# Working-set multiple over the on-disk size (weights + KV cache + overhead).
_RAM_OVER_SIZE_FACTOR = 1.5

_BYTES_PER_GB = 1024**3

# On-disk bytes per parameter for each GGUF quantization. llama.cpp quants carry
# per-block scale metadata, so the real cost per weight exceeds the nominal bit
# width: Q4_K_M is ~4.9 bits, not 4. Values verified against Qwen3-8B-GGUF, whose
# published quants measure 0.614 (Q4_K_M), 0.699 (Q5_0), 0.714 (Q5_K_M),
# 0.821 (Q6_K) and 1.063 (Q8_0).
_BYTES_PER_PARAM: dict[str, float] = {
    "Q2_K": 0.325,
    "Q3_K_S": 0.45,
    "Q3_K_M": 0.488,
    "Q3_K_L": 0.53,
    "IQ4_XS": 0.53,
    "Q4_0": 0.569,
    "Q4_K_S": 0.575,
    "Q4_K_M": 0.614,
    "Q5_0": 0.699,
    "Q5_K_S": 0.688,
    "Q5_K_M": 0.714,
    "Q6_K": 0.821,
    "Q8_0": 1.063,
    "F16": 2.0,
    "BF16": 2.0,
    "F32": 4.0,
}

# Q4_K_M is what ``pick_best_gguf`` prefers, so an unrecognized quant label
# estimates as if it were the quant a pull would most likely land on.
_DEFAULT_BYTES_PER_PARAM = _BYTES_PER_PARAM["Q4_K_M"]


def estimate_min_ram_gb(size_gb: float) -> float:
    """Estimate the RAM a model needs from its on-disk size (single source)."""
    return round(max(_MIN_RAM_FLOOR_GB, size_gb * _RAM_OVER_SIZE_FACTOR), 1)


def estimate_size_gb(params: int, gguf_filename: str) -> float:
    """Estimate the on-disk GB of *gguf_filename* from a model's parameter count.

    The HF listing API reports a parameter count (``gguf.total``) but no
    per-file byte size; siblings carry no ``size`` on either the list or the
    detail endpoint, and ``gguf.totalFileSize`` sums every quant in the repo
    rather than the one file a pull fetches. Per-file bytes are only available
    from ``/tree/main``, which is one extra request per repo and unaffordable
    for a catalog page. Parameters times the quant's bytes-per-weight gets
    within a few percent for a fraction of the cost.
    """
    if params <= 0:
        return 0.0  # unknown: display as "?" in UI
    # Local import keeps models.py a leaf (no sibling imports at module top).
    from lilbee.catalog.formatting import extract_quant

    quant = extract_quant(gguf_filename)
    bytes_per_param = _BYTES_PER_PARAM.get(quant, _DEFAULT_BYTES_PER_PARAM)
    return round(params * bytes_per_param / _BYTES_PER_GB, 1)


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
    recommended: bool = False
    architecture: str = ""
    compat: ModelCompat = ModelCompat.UNKNOWN
    # Parameter count. Size buckets key off this rather than on-disk bytes so a
    # model keeps its bucket across quants. 0 when the repo publishes no GGUF
    # metadata.
    params: int = 0

    @property
    def ref(self) -> str:
        """Browse-time ref (the HF repo); concrete filename is resolved at install."""
        return self.hf_repo

    @property
    def display_name(self) -> str:
        """Human-readable label derived from the HuggingFace repo id."""
        # Local import keeps models.py a leaf (no sibling imports at module top).
        from lilbee.catalog.formatting import clean_display_name

        return clean_display_name(self.hf_repo)


@dataclass(frozen=True)
class CatalogResult:
    """Paginated catalog result."""

    total: int
    limit: int
    offset: int
    models: list[CatalogModel]
    has_more: bool = False


@dataclass(frozen=True)
class HfPage:
    """One page of HuggingFace API results."""

    models: list[CatalogModel]
    has_more: bool


@dataclass(frozen=True)
class ModelVariant:
    """One quantization within a model family. ``filename`` may be a glob."""

    hf_repo: str
    filename: str
    param_count: str
    quant: str
    size_mb: int
    recommended: bool
    mmproj_filename: str = ""


@dataclass(frozen=True)
class ModelFamily:
    """A group of related model variants (e.g. Qwen3 in multiple sizes)."""

    slug: str  # family slug for building refs: "qwen3"
    name: str  # display name: "Qwen3"
    task: ModelTask
    description: str
    variants: tuple[ModelVariant, ...]
