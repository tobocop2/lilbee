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
# leaves norms in F32, so a real file usually costs more per weight than its
# nominal type; five published IQ2_S files sit at 0.93-0.94 of their type instead,
# because an ftype of that name may mix a cheaper type into some tensors.
#
# These are measured file sizes: each entry is the highest published rate seen
# for that label, plus a small margin, over a corpus of published GGUF files
# from repos of 7B parameters or more (the catalog's typical download size), so
# no entry reads under a file actually shipped at that scale. A smaller or
# architecturally unusual repo can still publish under an entry, because a
# fixed-size vocabulary head is a larger share of a small model's weight, and
# that residual is accepted rather than chased with a bigger margin that would
# over-read every ordinary download instead. Qwen3-8B-GGUF alone publishes
# 0.614 (Q4_K_M), 0.699 (Q5_0), 0.714 (Q5_K_M), 0.821 (Q6_K) and 1.063 (Q8_0),
# each already under its table entry.
#
# An entry moves only where the corpus backs it with at least three distinct
# repos; fewer than that, including none, keeps the prior figure. Q5_0 has no
# corpus rows and keeps 0.699 on that ground.
# ``test_table_entries_cover_the_measured_corpus`` replays the checked-in
# corpus in ``tests/fixtures/quant_file_rates.json`` (fetch date and filter
# recorded inside the fixture) against every entry it has rows for.
# Coverage wins over the three-repo minimum: a published file under an entry
# fails that test on one backing row same as on ten, because the minimum
# governs when an entry may rise, not whether an under-read is safe to ship.
#
# No entry may sit below its base type's bytes per weight, which is physically
# impossible; ``test_measured_quants_are_above_their_ggml_floor`` checks each one
# against ``gguf.constants.GGML_QUANT_SIZES`` so a typo cannot survive review.
_BYTES_PER_PARAM: dict[str, float] = {
    "Q2_K": 0.399,
    "Q3_K_S": 0.461,
    "Q3_K_M": 0.503,
    "Q3_K_L": 0.541,
    "IQ4_XS": 0.564,
    "Q4_0": 0.587,
    "Q4_K_S": 0.589,
    "Q4_K_M": 0.619,
    "Q5_0": 0.699,
    "Q5_K_S": 0.702,
    "Q5_K_M": 0.719,
    "Q6_K": 0.826,
    "Q8_0": 1.069,
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

# A ggml type size usually floors a file, and it never sizes one: the promoted
# output and embedding tensors and the F32 norms are not of the type the filename
# names. How much they add is the publisher's choice, so it is measured, not
# derived. Over published files whose parameter count checks out against a float
# copy of the same model, a label that states no bit width runs 1.02
# (gpt-oss-120b MXFP4) to 1.09 (gpt-oss-20b MXFP4) times its type, with
# Ternary-Bonsai-2-27B TQ1_0 at 1.05. This covers the largest with a little room.
# The direction is deliberately high: a size read too low tells someone a model
# fits in their RAM when it does not, which is the reading this estimate exists
# to prevent.
_PROMOTION_OVERHEAD = 1.10


def _ggml_bytes_per_param(quant: str) -> float | None:
    """Bytes per weight of the ggml type named *quant*, or None if ggml has no such type.

    What one tensor of that type costs, which is the floor a file of that type
    sits on. Not exact in both directions: an ftype of the same name may mix a
    cheaper type into some tensors, so a published file can come in under it.
    """
    # heavy: gguf pulls numpy, 58 ms by importtime
    from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

    try:
        block, type_size = GGML_QUANT_SIZES[GGMLQuantizationType[quant]]
    except KeyError:
        return None
    return type_size / block


def _width_bytes_per_param(quant: str) -> float | None:
    """Bytes per weight from the bit width *quant* names, or None if it names none."""
    match = re.match(r"I?Q(\d)", quant)
    if match is None:
        return None
    return int(match.group(1)) / _BITS_PER_BYTE * _SCALE_OVERHEAD


def _quant_bytes_per_param(gguf_filename: str) -> float:
    """Bytes per weight for the quant *gguf_filename* names.

    The measured table first. Then the bit width the label states, floored by
    ggml's type: the width rule carries a scale term already measured against
    published files, so it estimates, and the type only stops it reading far
    under what the tensors cost. The type is not an exact floor: an ftype that
    mixes a cheaper type into some tensors publishes under it, and five IQ2_S
    files sit at 0.93 of theirs, which bounds the over-read at 1.07. A label
    stating no width leaves the type as the only figure there is, so that one
    takes the promotion term. Then the default.
    """
    quant = quant_label(gguf_filename)
    measured = _BYTES_PER_PARAM.get(quant)
    if measured is not None:
        return measured
    floor = _ggml_bytes_per_param(quant)
    width = _width_bytes_per_param(quant)
    if width is not None:
        return width if floor is None else max(width, floor)
    if floor is not None:
        return floor * _PROMOTION_OVERHEAD
    return _DEFAULT_BYTES_PER_PARAM


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
