"""Catalog dataclasses and pydantic types. Imports only the catalog's leaf modules."""

import re
from dataclasses import dataclass

from pydantic import BaseModel

from lilbee.catalog.refs import quant_label
from lilbee.catalog.types import ModelCompat, ModelTask

# Minimum recommended floor so a tiny model still reports a sane RAM ask.
_MIN_RAM_FLOOR_GB = 2.0
# Working-set multiple over the on-disk size (weights + KV cache + overhead).
_RAM_OVER_SIZE_FACTOR = 1.5

_BYTES_PER_GB = 1024**3

# Whole-file bytes per parameter for each llama.cpp quantization.
#
# Not the same quantity as ``gguf.GGML_QUANT_SIZES``, which gives the block size
# of one ggml tensor type. llama.cpp never writes a homogeneous file: it promotes
# ``output.weight`` and tied ``token_embd`` to Q6_K/Q8_0 whatever the ftype, and
# leaves norms in F32, so a real file always costs more per weight than its
# nominal type. These are measured file sizes; Qwen3-8B-GGUF publishes 0.614
# (Q4_K_M), 0.699 (Q5_0), 0.714 (Q5_K_M), 0.821 (Q6_K) and 1.063 (Q8_0).
#
# GGML_QUANT_SIZES is still used, as the physical floor: no file can be smaller
# than its base type, so _quant_bytes_per_param clamps to it and a typo below the
# floor cannot survive.
_BYTES_PER_PARAM: dict[str, float] = {
    "Q2_K": 0.33,
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

# Q4_K_M heads the pull path's quant preference, so a label naming no bit width
# at all estimates as if it were the quant a pull would most likely land on.
_DEFAULT_BYTES_PER_PARAM = _BYTES_PER_PARAM["Q4_K_M"]

# A quant the table does not name still says how many bits it packs. One fp16
# scale per group costs an eighth on top, whatever the width, because a group is
# sized to the width: 1-bit in groups of 128, 2-bit in 64, 4-bit in 32 all carry
# two bytes per group. Reading the width beats falling back to Q4_K_M, which
# reports a 2-bit file at more than twice its size.
_SCALE_OVERHEAD = 1.125
_BITS_PER_BYTE = 8


def _ggml_floor(quant: str) -> float | None:
    """Bytes per weight of *quant*'s base ggml type, or None if it names none."""
    from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

    base = quant.split("_")[0] if quant.startswith(("F", "BF")) else quant
    for name in (quant, base, "_".join(quant.split("_")[:2])):
        try:
            block, type_size = GGML_QUANT_SIZES[GGMLQuantizationType[name]]
        except KeyError:
            continue
        return type_size / block
    return None


def _packed_bits(quant: str) -> int | None:
    """The bit width *quant* packs each weight into, or None if it names none."""
    match = re.match(r"I?Q(\d)", quant)
    return int(match.group(1)) if match else None


def _quant_bytes_per_param(gguf_filename: str) -> float:
    """Bytes per weight for the quant *gguf_filename* names, never below its floor."""
    quant = quant_label(gguf_filename)
    measured = _BYTES_PER_PARAM.get(quant)
    if measured is None:
        bits = _packed_bits(quant)
        measured = (
            bits / _BITS_PER_BYTE * _SCALE_OVERHEAD
            if bits is not None
            else _DEFAULT_BYTES_PER_PARAM
        )
    floor = _ggml_floor(quant)
    return max(measured, floor) if floor is not None else measured


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
    mmproj_filename: str = ""
    compat: ModelCompat = ModelCompat.UNKNOWN


@dataclass(frozen=True)
class ModelFamily:
    """A group of related model variants (e.g. Qwen3 in multiple sizes)."""

    slug: str  # family slug for building refs: "qwen3"
    name: str  # display name: "Qwen3"
    task: ModelTask
    description: str
    variants: tuple[ModelVariant, ...]
