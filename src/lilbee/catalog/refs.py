"""HuggingFace ref helpers: parse and format ``<org>/<repo>/<file>.gguf`` strings."""

from __future__ import annotations

import re
from collections.abc import Iterable
from enum import IntEnum

# A native GGUF ref ``<org>/<repo>/<file>.gguf`` has at least two ``/`` separators;
# the filename may add more when a quant lives in a repo subdir (``Q4_K_M/...``).
NATIVE_GGUF_REF_MIN_SLASHES = 2

WILDCARD = "*"
GGUF_SUFFIX = ".gguf"
GGUF_GLOB = f"{WILDCARD}{GGUF_SUFFIX}"

# Vision models need both the main GGUF and an mmproj (CLIP projection) file.
# Resolved by glob rather than a per-repo table: every mainstream VL repo names
# its projector this way, and a table would be one more thing to maintain.
DEFAULT_MMPROJ_PATTERN = f"{WILDCARD}mmproj{WILDCARD}{GGUF_SUFFIX}"

# Quantization labels in descending order of preference, best size/quality
# balance first. This orders the candidates a pull tries; which one is really
# the model is settled by the GGUF header, not by the name.
_QUANT_PREFERENCE = (
    "Q4_K_M",
    "Q4_K_S",
    "Q5_K_M",
    "Q5_K_S",
    "Q8_0",
    "Q6_K",
    "Q3_K_M",
    "IQ4_XS",
    "Q4_0",
    "Q5_0",
    "Q3_K_L",
    "Q3_K_S",
    "Q2_K",
)

# Unquantized types. A repo that publishes one beside quantized packs offers it
# as the reference copy, so it ranks below every quant including unrecognized
# ones: picking it turns a 7 GB pull into a 54 GB one.
FLOAT_QUANTS = frozenset({"F16", "BF16", "F32"})

# ggml's own quantized type table: name -> (block size, type size in bytes), for
# every type whose block packs more than one weight (a block of one is a scalar
# type, not a quantization; the float alternation below names the three scalar
# types a GGUF filename actually carries). This is the product's own copy of
# ``gguf.constants.GGML_QUANT_SIZES``: the library is not a product dependency
# (see the ``gguf`` entry in pyproject.toml's dev group), so the product cannot
# read it at run time. ``TestGgmlQuantTableMatchesLibrary`` in
# tests/test_catalog.py checks this table against the library on every test
# run, which is where the dependency's job stays: catching an upstream type
# addition or a transcription typo, not sizing anything at run time.
GGML_QUANT_BLOCK_SIZES: dict[str, tuple[int, int]] = {
    "Q4_0": (32, 18),
    "Q4_1": (32, 20),
    "Q5_0": (32, 22),
    "Q5_1": (32, 24),
    "Q8_0": (32, 34),
    "Q8_1": (32, 40),
    "Q2_K": (256, 84),
    "Q3_K": (256, 110),
    "Q4_K": (256, 144),
    "Q5_K": (256, 176),
    "Q6_K": (256, 210),
    "Q8_K": (256, 292),
    "IQ2_XXS": (256, 66),
    "IQ2_XS": (256, 74),
    "IQ3_XXS": (256, 98),
    "IQ1_S": (256, 50),
    "IQ4_NL": (32, 18),
    "IQ3_S": (256, 110),
    "IQ2_S": (256, 82),
    "IQ4_XS": (256, 136),
    "IQ1_M": (256, 56),
    "TQ1_0": (256, 54),
    "TQ2_0": (256, 66),
    "MXFP4": (32, 17),
    "NVFP4": (64, 36),
    "Q1_0": (128, 18),
}


# The pattern that reads a quant label out of a GGUF filename. ggml's own
# quantized type names come from ``GGML_QUANT_BLOCK_SIZES`` because a name like
# ``TQ1_0`` or ``MXFP4`` states no bit width, so no pattern over a label's shape
# can find it. The leading bit-width alternative already matches every
# ``Q<digit>`` and ``IQ<digit>`` name in that table; the names go in whole so
# that no upstream name depends on that coincidence.
#
# Sorting longest first and escaping the names are both future-proofing: no
# ggml name is a prefix of another today, and a type name is always a plain
# identifier, so neither can change a match until upstream adds a name that
# breaks one of those.
#
# A quant label occupies a whole ``-``/``_``/``.``/``/``-delimited segment of
# the filename. Matching it as a bare substring makes ``Q8_0`` match inside
# ``mmproj-Q8_0`` and ``F16`` inside ``BF16``.
#
# Built at module import, not lazily: the type table above is a plain dict
# literal, so compiling the pattern costs microseconds. The lazy form existed
# only to defer ``gguf``'s numpy pull past CLI startup; there is no heavy
# import left to defer.
_QUANT_NAMES_BY_LENGTH = sorted(
    (re.escape(name) for name in GGML_QUANT_BLOCK_SIZES),
    key=lambda name: (-len(name), name),
)
_QUANT_TOKEN_RE = re.compile(
    r"(?:^|[-_./])P?(I?Q\d[A-Za-z0-9_]*|"
    + "|".join(_QUANT_NAMES_BY_LENGTH)
    + r"|BF16|F16|F32)(?=$|[-_./])",
    re.IGNORECASE,
)


_SPLIT_SHARD_RE = re.compile(r"^(?P<base>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$")
_SHARD_NUMBER_WIDTH = 5
_FIRST_SHARD_INDEX = 1

_PREFERENCE_INDEX: dict[str, int] = {quant: i for i, quant in enumerate(_QUANT_PREFERENCE)}


class _QuantTier(IntEnum):
    """Ordering tier of a GGUF candidate, best first."""

    PREFERRED = 0
    UNRANKED = 1
    FLOAT = 2


def quant_label(filename: str) -> str:
    """The GGUF quantization label *filename* names, uppercased, or empty string.

    Reads the last labelled segment: a quant stored in a repo subdir repeats the
    label in both the directory and the file, and a mismatched pair names the
    real type on the file.
    """
    matches = _QUANT_TOKEN_RE.findall(filename)
    return matches[-1].upper() if matches else ""


def ggml_bytes_per_param(quant: str) -> float | None:
    """Bytes per weight of the ggml type named *quant*, or None if ggml has no such type."""
    sizes = GGML_QUANT_BLOCK_SIZES.get(quant)
    if sizes is None:
        return None
    block, type_size = sizes
    return type_size / block


def _shard_name(base: str, index: int, total: int) -> str:
    """Render one part of a split GGUF's ``<base>-<i>-of-<n>.gguf`` naming."""
    return f"{base}-{index:0{_SHARD_NUMBER_WIDTH}d}-of-{total:0{_SHARD_NUMBER_WIDTH}d}{GGUF_SUFFIX}"


def split_shard_filenames(filename: str) -> list[str]:
    """Return every shard of a split GGUF in order, or ``[filename]`` if it isn't split.

    A split GGUF names its parts ``<base>-00001-of-0000N.gguf`` through
    ``<base>-0000N-of-0000N.gguf``. llama.cpp loads the whole set from the first
    shard but needs every part on disk, so the catalog must fetch all of them and
    only consider the model installed once the full set is present.
    """
    match = _SPLIT_SHARD_RE.match(filename)
    if match is None:
        return [filename]
    base = match.group("base")
    total = int(match.group("total"))
    return [_shard_name(base, index, total) for index in range(_FIRST_SHARD_INDEX, total + 1)]


def _first_shard(filename: str) -> str:
    """The shard llama.cpp loads a split GGUF from, or *filename* when it isn't split.

    Only the first shard carries the full metadata header, so it is the one a
    header probe can read and the only part worth ranking as a candidate.
    """
    match = _SPLIT_SHARD_RE.match(filename)
    if match is None:
        return filename
    return _shard_name(match.group("base"), _FIRST_SHARD_INDEX, int(match.group("total")))


def _rank_key(filename: str) -> tuple[int, int, str]:
    """Sort key placing the best-quantized candidate first, ties broken by name."""
    quant = quant_label(filename)
    preference = _PREFERENCE_INDEX.get(quant)
    if preference is not None:
        return (_QuantTier.PREFERRED, preference, filename)
    tier = _QuantTier.FLOAT if quant in FLOAT_QUANTS else _QuantTier.UNRANKED
    return (tier, 0, filename)


def rank_gguf_candidates(filenames: Iterable[str]) -> list[str]:
    """A repo's GGUF files in the order a pull should try them, best quant first.

    Split shards collapse to their first part. An unrecognized quant outranks an
    unquantized copy, so a repo that publishes only exotic packs beside an F16
    still resolves to a pack rather than the full-precision weights.

    Hand-rolled because neither huggingface_hub nor gguf-py exposes a
    "choose a file from this repo" API; both stop at listing and reading.
    """
    candidates = {_first_shard(name) for name in filenames if name.endswith(GGUF_SUFFIX)}
    return sorted(candidates, key=_rank_key)


def is_bare_hf_repo(ref: str) -> bool:
    """True if *ref* has the bare ``<org>/<repo>`` shape (no filename segment)."""
    return ref.count("/") == 1 and not ref.endswith(GGUF_SUFFIX)


def hf_repo_from_ref(ref: str) -> str:
    """Return the ``<org>/<repo>`` portion of a native GGUF ref.

    Native GGUF refs have the form ``<org>/<repo>/<filename>.gguf``, where the
    filename may itself include repo subdirectories (unsloth stores quants under
    e.g. ``Q4_K_M/...gguf``). The repo is always the first two segments.
    Provider-prefixed refs (``openai/gpt-4``, ``ollama/llama3:8b``) and bare
    repos lack the ``.gguf`` suffix and are returned unchanged.
    """
    if ref.endswith(GGUF_SUFFIX) and ref.count("/") >= NATIVE_GGUF_REF_MIN_SLASHES:
        return "/".join(ref.split("/")[:NATIVE_GGUF_REF_MIN_SLASHES])
    return ref


def gguf_filename_from_ref(ref: str) -> str:
    """Return the filename portion of a native GGUF ref (after ``<org>/<repo>/``).

    The filename may include repo subdirectories (a quant stored under e.g.
    ``Q4_K_M/``), so everything past the first two segments is kept.
    Returns empty string for non-native refs (bare repos, provider-prefixed).
    """
    if ref.endswith(GGUF_SUFFIX) and ref.count("/") >= NATIVE_GGUF_REF_MIN_SLASHES:
        return "/".join(ref.split("/")[NATIVE_GGUF_REF_MIN_SLASHES:])
    return ""


def format_native_gguf_ref(hf_repo: str, gguf_filename: str) -> str:
    """Render the canonical ``<hf_repo>/<gguf_filename>`` native GGUF ref."""
    return f"{hf_repo}/{gguf_filename}"
