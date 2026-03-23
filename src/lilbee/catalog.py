"""Model catalog — discovers available GGUF models from HuggingFace.

Three levels:
1. Featured models — curated favorites (hardcoded, always available)
2. HF API models — fetched from HuggingFace API, paginated and filterable
3. Combined catalog — featured first, then HF results
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from lilbee.config import cfg

log = logging.getLogger(__name__)

HF_API_URL = "https://huggingface.co/api/models"
HF_DOWNLOAD_URL = "https://huggingface.co/{repo}/resolve/main/{filename}"
_DEFAULT_TIMEOUT = 30.0


@dataclass(frozen=True)
class CatalogModel:
    """A model entry in the catalog."""

    name: str
    hf_repo: str
    gguf_filename: str
    size_gb: float
    min_ram_gb: float
    description: str
    featured: bool
    downloads: int
    task: str


@dataclass(frozen=True)
class CatalogResult:
    """Paginated catalog result."""

    total: int
    limit: int
    offset: int
    models: list[CatalogModel]


FEATURED_CHAT: tuple[CatalogModel, ...] = (
    CatalogModel(
        "Qwen3 0.6B",
        "Qwen/Qwen3-0.6B-GGUF",
        "*Q4_K_M.gguf",
        0.5,
        2,
        "Tiny — runs on anything",
        True,
        0,
        "chat",
    ),
    CatalogModel(
        "Qwen3 4B",
        "Qwen/Qwen3-4B-GGUF",
        "*Q4_K_M.gguf",
        2.5,
        8,
        "Small — good balance for 8 GB RAM",
        True,
        0,
        "chat",
    ),
    CatalogModel(
        "Qwen3 8B",
        "Qwen/Qwen3-8B-GGUF",
        "*Q4_K_M.gguf",
        5.0,
        8,
        "Medium — strong general purpose",
        True,
        0,
        "chat",
    ),
    CatalogModel(
        "Mistral 7B Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3-GGUF",
        "*Q4_K_M.gguf",
        4.4,
        8,
        "Fast 7B, 32K context",
        True,
        0,
        "chat",
    ),
    CatalogModel(
        "Qwen3-Coder 30B A3B",
        "Qwen/Qwen3-Coder-30B-A3B-GGUF",
        "*Q4_K_M.gguf",
        18.0,
        32,
        "Extra large — best quality",
        True,
        0,
        "chat",
    ),
)

FEATURED_EMBEDDING: tuple[CatalogModel, ...] = (
    CatalogModel(
        "Nomic Embed Text v1.5",
        "nomic-ai/nomic-embed-text-v1.5-GGUF",
        "nomic-embed-text-v1.5.Q4_K_M.gguf",
        0.3,
        2,
        "Fast, high quality — default for lilbee",
        True,
        0,
        "embedding",
    ),
)

FEATURED_VISION: tuple[CatalogModel, ...] = (
    CatalogModel(
        "LLaVA 1.6 7B",
        "mys/ggml-model-llava-v1.6-7b",
        "*Q4_K_M.gguf",
        4.1,
        8,
        "Strong vision model",
        True,
        0,
        "vision",
    ),
)

FEATURED_ALL: tuple[CatalogModel, ...] = FEATURED_CHAT + FEATURED_EMBEDDING + FEATURED_VISION

_SIZE_RANGES: dict[str, tuple[float, float]] = {
    "small": (0.0, 3.0),
    "medium": (3.0, 10.0),
    "large": (10.0, float("inf")),
}


# TTL cache for HuggingFace API results (5 minutes)
_HF_CACHE_TTL = 300
_hf_cache: dict[str, tuple[float, list["CatalogModel"]]] = {}


def _fetch_hf_models(
    pipeline_tag: str = "text-generation",
    tags: str = "gguf",
    sort: str = "downloads",
    limit: int = 50,
) -> list[CatalogModel]:
    """Fetch models from HuggingFace API with 5-minute cache. Returns empty list on error."""
    cache_key = f"{pipeline_tag}:{tags}:{sort}:{limit}"
    now = time.monotonic()
    cached = _hf_cache.get(cache_key)
    if cached and now - cached[0] < _HF_CACHE_TTL:
        return cached[1]

    params: dict[str, str | int] = {
        "pipeline_tag": pipeline_tag,
        "tags": tags,
        "sort": sort,
        "limit": limit,
    }
    try:
        resp = httpx.get(HF_API_URL, params=params, timeout=_DEFAULT_TIMEOUT)
        if resp.status_code >= 400:
            log.warning("HuggingFace API returned HTTP %d", resp.status_code)
            return []
        data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("Failed to fetch models from HuggingFace: %s", exc)
        return []

    models: list[CatalogModel] = []
    for item in data:
        repo_id = item.get("id", "")
        if not repo_id:
            continue
        downloads = item.get("downloads", 0)
        card_data = item.get("cardData", {}) or {}
        model_desc = item.get("description") or card_data.get("description") or ""
        # Estimate size from siblings — find largest GGUF file
        size_gb = _estimate_size_from_siblings(item.get("siblings", []))
        models.append(
            CatalogModel(
                name=repo_id.split("/")[-1],
                hf_repo=repo_id,
                gguf_filename="*.gguf",
                size_gb=size_gb,
                min_ram_gb=max(2.0, size_gb * 1.5),
                description=model_desc[:120] if model_desc else "",
                featured=False,
                downloads=downloads,
                task="chat",
            )
        )
    _hf_cache[cache_key] = (now, models)
    return models


def _estimate_size_from_siblings(siblings: list[dict[str, Any]]) -> float:
    """Estimate model size in GB from the largest GGUF file in siblings."""
    max_bytes = 0
    for sib in siblings:
        filename = sib.get("rfilename", "")
        if filename.endswith(".gguf"):
            size = sib.get("size", 0) or 0
            max_bytes = max(max_bytes, size)
    if max_bytes > 0:
        return round(max_bytes / (1024**3), 1)
    return 0.0  # unknown — display as "?" in UI


def get_catalog(
    task: str | None = None,
    *,
    search: str = "",
    size: str | None = None,
    installed: bool | None = None,
    featured: bool | None = None,
    sort: str = "featured",
    limit: int = 20,
    offset: int = 0,
    model_manager: Any = None,
) -> CatalogResult:
    """Get paginated, filtered catalog of models."""
    # Start with featured models
    all_models = list(FEATURED_ALL)

    # Optionally fetch from HF API
    if not featured:
        hf_task = _task_to_pipeline(task)
        hf_models = _fetch_hf_models(pipeline_tag=hf_task, limit=50)
        # Deduplicate: skip HF models whose repo matches a featured model
        featured_repos = {m.hf_repo for m in FEATURED_ALL}
        hf_models = [m for m in hf_models if m.hf_repo not in featured_repos]
        all_models.extend(hf_models)

    # Filter by task
    if task:
        all_models = [m for m in all_models if m.task == task]

    # Filter by search
    if search:
        search_lower = search.lower()
        all_models = [
            m
            for m in all_models
            if search_lower in m.name.lower()
            or search_lower in m.hf_repo.lower()
            or search_lower in m.description.lower()
        ]

    # Filter by size
    if size and size in _SIZE_RANGES:
        lo, hi = _SIZE_RANGES[size]
        all_models = [m for m in all_models if lo <= m.size_gb < hi]

    # Filter by installed status
    if installed is not None and model_manager is not None:
        installed_models = _get_installed_models(model_manager)
        if installed:
            all_models = [m for m in all_models if m.name in installed_models]
        else:
            all_models = [m for m in all_models if m.name not in installed_models]

    # Filter by featured status
    if featured is not None:
        all_models = [m for m in all_models if m.featured == featured]

    # Sort
    all_models = _sort_models(all_models, sort)

    total = len(all_models)
    paginated = all_models[offset : offset + limit]

    return CatalogResult(total=total, limit=limit, offset=offset, models=paginated)


def _task_to_pipeline(task: str | None) -> str:
    """Map task name to HuggingFace pipeline tag."""
    mapping = {
        "chat": "text-generation",
        "embedding": "feature-extraction",
        "vision": "image-text-to-text",
    }
    return mapping.get(task or "chat", "text-generation")


def _get_installed_models(model_manager: Any) -> set[str]:
    """Get set of installed model names from model_manager."""
    try:
        return {m.name for m in model_manager.list_models()}
    except Exception:
        return set()


_SORT_KEYS: dict[str, tuple] = {
    "downloads": (lambda m: m.downloads, True),
    "size_asc": (lambda m: m.size_gb, False),
    "size_desc": (lambda m: m.size_gb, True),
    "featured": (lambda m: (not m.featured, -m.downloads), False),
}


def _sort_models(models: list[CatalogModel], sort: str) -> list[CatalogModel]:
    """Sort models according to the specified sort order."""
    key_fn, reverse = _SORT_KEYS.get(sort, _SORT_KEYS["featured"])
    return sorted(models, key=key_fn, reverse=reverse)


# Maps Ollama-style names to catalog display names for lookup
def find_catalog_entry(name: str) -> CatalogModel | None:
    """Find a featured model by display name (case-insensitive)."""
    name_lower = name.lower()
    for model in FEATURED_ALL:
        if model.name.lower() == name_lower:
            return model
    return None


def download_model(entry: CatalogModel, *, on_progress: Any = None) -> Path:
    """Download a GGUF model from HuggingFace to cfg.models_dir."""
    cfg.models_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the filename pattern to an actual file
    filename = _resolve_filename(entry)
    dest = cfg.models_dir / filename

    if dest.exists():
        log.info("Model already downloaded: %s", dest)
        return dest

    url = HF_DOWNLOAD_URL.format(repo=entry.hf_repo, filename=filename)
    log.info("Downloading %s → %s", url, dest)

    try:
        with httpx.stream("GET", url, timeout=None, follow_redirects=True) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if on_progress and total > 0:
                        on_progress(downloaded, total)
    except httpx.HTTPError:
        if dest.exists():
            dest.unlink()
        raise

    return dest


_QUANT_PREFERENCE = ("Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q8_0", "Q6_K", "Q3_K_M")


def _resolve_filename(entry: CatalogModel) -> str:
    """Resolve a GGUF filename pattern to the best concrete filename.

    For exact filenames, return as-is. For wildcards, query the HF API
    and pick the best quantization (prefer Q4_K_M for balance of size/quality).
    """
    if "*" not in entry.gguf_filename:
        return entry.gguf_filename

    try:
        resp = httpx.get(
            f"https://huggingface.co/api/models/{entry.hf_repo}",
            timeout=_DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        siblings = resp.json().get("siblings", [])
    except Exception as exc:
        raise RuntimeError(f"Cannot query files for {entry.hf_repo}: {exc}") from exc

    gguf_files = [
        s.get("rfilename", "") for s in siblings if s.get("rfilename", "").endswith(".gguf")
    ]
    if not gguf_files:
        raise RuntimeError(f"No GGUF files found in {entry.hf_repo}")

    return _pick_best_gguf(gguf_files)


def _pick_best_gguf(filenames: list[str]) -> str:
    """Pick the best GGUF file by quantization preference."""
    for quant in _QUANT_PREFERENCE:
        for f in filenames:
            if quant in f:
                return f
    return filenames[0]
