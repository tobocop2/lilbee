"""Display-name, quantization, and enrichment helpers."""

import re
from dataclasses import dataclass

from lilbee.catalog.models import CatalogModel, CatalogResult
from lilbee.catalog.refs import hf_repo_from_ref
from lilbee.catalog.types import ModelSource

PARAM_COUNT_RE = re.compile(r"(\d+\.?\d*B)", re.IGNORECASE)

# One alternation strips every kind of trailing noise from a display name:
# name suffixes (anywhere they precede ``-`` or end-of-string), trailing GGUF
# quant tokens (``-Q4_K_M``, ``-F16`` ...), and trailing date stamps (``-2507``).
_DISPLAY_NAME_NOISE = re.compile(
    r"-(?:GGUF|Instruct|Chat|Embedding|Embed|qat)(?=-|$)"
    r"|-(?:Q\d[A-Z0-9_]*|F16|F32)$"
    r"|-\d{4}$",
    re.IGNORECASE,
)
_DISPLAY_NAME_META_PREFIX = re.compile(r"^Meta-", re.IGNORECASE)

# A native GGUF ref of the form ``<owner>/<repo>/<file>.gguf`` has at least
# two ``/`` separators; one-slash refs are bare repo IDs.
_NATIVE_GGUF_REF_MIN_SLASHES = 2


def clean_display_name(repo_id: str) -> str:
    """Derive a human-friendly display name from a HuggingFace repo ID.

    Examples:
        "Qwen/Qwen2.5-7B-Instruct-GGUF"           -> "Qwen2.5 7B"
        "meta-llama/Meta-Llama-3-8B"              -> "Llama 3 8B"
        "unsloth/embeddinggemma-300M-qat-GGUF"    -> "embeddinggemma 300M"
        "ggml-org/all-MiniLM-L6-v2-Embedding-Q8_0" -> "all MiniLM L6 v2"
    """
    name = repo_id.split("/")[-1]
    while True:
        stripped = _DISPLAY_NAME_NOISE.sub("", name)
        if stripped == name:
            break
        name = stripped
    name = _DISPLAY_NAME_META_PREFIX.sub("", name)
    name = name.replace("-", " ").strip()
    return re.sub(r"\s+", " ", name)


def display_label_for_ref(ref: str) -> str:
    """Render any model ref as a short, human-friendly UI label.

    - Native HF ref (``<repo>/<file>.gguf``): cleaned repo name.
    - Provider-prefixed (``ollama/``, ``openai/`` ...): the part after the prefix.
    - Anything else: returned unchanged.
    """
    if not ref:
        return ""
    if ref.endswith(".gguf") and ref.count("/") >= _NATIVE_GGUF_REF_MIN_SLASHES:
        return clean_display_name(ref.rsplit("/", 1)[0])
    if "/" in ref:
        return ref.split("/", 1)[1]
    return ref


def extract_quant(filename: str) -> str:
    """Extract the GGUF quantization label (e.g. ``Q4_K_M``) from a filename."""
    m = re.search(r"(Q\d[A-Z0-9_]*)", filename, re.IGNORECASE)
    return m.group(1).upper() if m else ""


QUANT_TIERS: dict[str, str] = {
    "Q2_K": "compact",
    "Q3_K_S": "compact",
    "Q3_K_M": "compact",
    "Q3_K_L": "compact",
    "Q4_K_S": "balanced",
    "Q4_K_M": "balanced",
    "Q4_0": "balanced",
    "Q5_K_S": "high quality",
    "Q5_K_M": "high quality",
    "Q6_K": "high quality",
    "Q8_0": "full precision",
    "F16": "unquantized",
    "F32": "unquantized",
}


def quant_tier(quant: str) -> str:
    """Map a quantization label to a human-readable quality tier."""
    if not quant:
        return "--"
    return QUANT_TIERS.get(quant, "--")


def derive_param_count(model: CatalogModel) -> str:
    """Parse the ``7B``-style param count from the display name; ``""`` if absent."""
    match = PARAM_COUNT_RE.search(model.display_name)
    return match.group(1) if match else ""


@dataclass(frozen=True)
class EnrichedModel:
    """A catalog model enriched with display metadata and install status."""

    hf_repo: str
    gguf_filename: str
    size_gb: float
    min_ram_gb: float
    description: str
    featured: bool
    downloads: int
    task: str
    display_name: str
    param_count: str
    quality_tier: str
    installed: bool
    source: str


def enrich_catalog(result: CatalogResult, installed_refs: set[str]) -> list[EnrichedModel]:
    """Enrich catalog models with display names, quality tiers, and install status.

    *installed_refs* contains the ``hf_repo/filename`` refs returned by
    ``model_manager.list_installed()``. A repo is considered installed
    when at least one of its quants has a manifest.
    """
    installed_repos = {hf_repo_from_ref(ref) for ref in installed_refs}
    enriched: list[EnrichedModel] = []
    for m in result.models:
        enriched.append(
            EnrichedModel(
                hf_repo=m.hf_repo,
                gguf_filename=m.gguf_filename,
                size_gb=m.size_gb,
                min_ram_gb=m.min_ram_gb,
                description=m.description,
                featured=m.featured,
                downloads=m.downloads,
                task=m.task,
                display_name=m.display_name,
                param_count=derive_param_count(m),
                quality_tier=quant_tier(extract_quant(m.gguf_filename)),
                installed=m.hf_repo in installed_repos,
                source=ModelSource.NATIVE.value,
            )
        )
    return enriched
