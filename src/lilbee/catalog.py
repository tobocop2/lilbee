"""Model catalog — discovers available GGUF models from HuggingFace.

Three levels:
1. Featured models — curated favorites (hardcoded, always available)
2. HF API models — fetched from HuggingFace API, paginated and filterable
3. Combined catalog — featured first, then HF results
"""

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from lilbee.config import cfg

log = logging.getLogger(__name__)

HF_API_URL = "https://huggingface.co/api/models"
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


@dataclass(frozen=True)
class ModelVariant:
    """A specific quantization/size variant within a model family."""

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

    name: str
    task: str
    description: str
    variants: tuple[ModelVariant, ...]


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
        "Fast 7B, 32K context (requires login)",
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
        "LightOnOCR-2",
        "LightOnIO/LightOnOCR-2-0.5B-GGUF",
        "*Q4_K_M.gguf",
        1.5,
        4,
        "Fast OCR — clean markdown output, tiny footprint",
        True,
        0,
        "vision",
    ),
)

# Maps vision catalog entries to their mmproj (CLIP projection) filenames.
# Vision models need both the main GGUF and the mmproj file to work.
# Keys are hf_repo identifiers; values are glob patterns resolved at download time.
VISION_MMPROJ_FILES: dict[str, str] = {
    "LightOnIO/LightOnOCR-2-0.5B-GGUF": "*mmproj*.gguf",
}

FEATURED_ALL: tuple[CatalogModel, ...] = FEATURED_CHAT + FEATURED_EMBEDDING + FEATURED_VISION

_FAMILY_NAME_RE = re.compile(r"^(.+?)\s+\d")


def _extract_family_name(model_name: str) -> str:
    """Extract the family name by stripping the trailing parameter count.

    "Qwen3 8B" -> "Qwen3", "Qwen3-Coder 30B A3B" -> "Qwen3-Coder",
    "Nomic Embed Text v1.5" -> "Nomic Embed Text v1.5" (no trailing number pattern).
    """
    m = _FAMILY_NAME_RE.match(model_name)
    return m.group(1) if m else model_name


def _extract_quant(filename: str) -> str:
    """Extract quantization label from a GGUF filename pattern.

    "*Q4_K_M.gguf" -> "Q4_K_M", "nomic-embed-text-v1.5.Q4_K_M.gguf" -> "Q4_K_M".
    """
    m = re.search(r"(Q\d[A-Z0-9_]*)", filename, re.IGNORECASE)
    return m.group(1).upper() if m else ""


def _catalog_to_variant(model: CatalogModel, *, recommended: bool = False) -> ModelVariant:
    """Convert a CatalogModel to a ModelVariant."""
    m = re.search(r"(\d+\.?\d*B)", model.name, re.IGNORECASE)
    param_count = m.group(1) if m else ""
    return ModelVariant(
        hf_repo=model.hf_repo,
        filename=model.gguf_filename,
        param_count=param_count,
        quant=_extract_quant(model.gguf_filename),
        size_mb=int(model.size_gb * 1024),
        recommended=recommended,
    )


def _build_families(models: tuple[CatalogModel, ...], task: str) -> list[ModelFamily]:
    """Group CatalogModels into families by extracted family name."""
    groups: dict[str, list[CatalogModel]] = {}
    order: list[str] = []
    for m in models:
        family = _extract_family_name(m.name)
        if family not in groups:
            order.append(family)
        groups.setdefault(family, []).append(m)

    families: list[ModelFamily] = []
    for name in order:
        members = groups[name]
        variants: list[ModelVariant] = []
        for i, m in enumerate(members):
            recommended = len(members) > 1 and i == len(members) - 1
            variants.append(_catalog_to_variant(m, recommended=recommended))
        description = members[0].description
        families.append(
            ModelFamily(
                name=name,
                task=task,
                description=description,
                variants=tuple(variants),
            )
        )
    return families


def get_families() -> list[ModelFamily]:
    """Get all featured models grouped into families.

    Returns families ordered: chat families, then embedding, then vision.
    Within each family, variants are ordered smallest to largest, with
    the largest marked as recommended (for multi-variant families).
    """
    return (
        _build_families(FEATURED_CHAT, "chat")
        + _build_families(FEATURED_EMBEDDING, "embedding")
        + _build_families(FEATURED_VISION, "vision")
    )


_SIZE_RANGES: dict[str, tuple[float, float]] = {
    "small": (0.0, 3.0),
    "medium": (3.0, 10.0),
    "large": (10.0, float("inf")),
}


def _hf_token() -> str | None:
    """Read HuggingFace token from LILBEE_HF_TOKEN or HF_TOKEN env vars."""
    return os.environ.get("LILBEE_HF_TOKEN") or os.environ.get("HF_TOKEN") or None


def _hf_headers() -> dict[str, str]:
    """Build HTTP headers for HuggingFace API requests."""
    token = _hf_token()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


# TTL cache for HuggingFace API results (5 minutes)
_HF_CACHE_TTL = 300
_hf_cache: dict[str, tuple[float, list["CatalogModel"]]] = {}


def _fetch_hf_models(
    pipeline_tag: str = "text-generation",
    tags: str = "gguf",
    sort: str = "downloads",
    limit: int = 50,
    offset: int = 0,
) -> list[CatalogModel]:
    """Fetch models from HuggingFace API with 5-minute cache. Returns empty list on error."""
    cache_key = f"{pipeline_tag}:{tags}:{sort}:{limit}:{offset}"
    now = time.monotonic()
    # Evict expired entries
    expired = [k for k, (ts, _) in _hf_cache.items() if now - ts >= _HF_CACHE_TTL]
    for k in expired:
        del _hf_cache[k]

    cached = _hf_cache.get(cache_key)
    if cached and now - cached[0] < _HF_CACHE_TTL:
        return cached[1]

    params: dict[str, str | int] = {
        "pipeline_tag": pipeline_tag,
        "tags": tags,
        "sort": sort,
        "limit": limit,
        "skip": offset,
    }
    try:
        resp = httpx.get(HF_API_URL, params=params, timeout=_DEFAULT_TIMEOUT, headers=_hf_headers())
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
        task = _pipeline_to_task(item.get("pipeline_tag", ""))
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
                task=task,
            )
        )
    _hf_cache[cache_key] = (now, models)
    # Cap cache size at 50 entries, evicting oldest on overflow
    _HF_CACHE_MAX_ENTRIES = 50
    if len(_hf_cache) > _HF_CACHE_MAX_ENTRIES:
        oldest_key = min(_hf_cache, key=lambda k: _hf_cache[k][0])
        del _hf_cache[oldest_key]
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
    # Featured models only on the first page
    all_models = list(FEATURED_ALL) if offset == 0 else []

    # Optionally fetch from HF API
    if not featured:
        hf_task = _task_to_pipeline(task)
        hf_models = _fetch_hf_models(pipeline_tag=hf_task, limit=limit, offset=offset)
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

    # When HF API pagination is active (offset passed to API), skip local slicing
    # to avoid double-applying the offset. Only slice for featured-only requests.
    paginated = all_models[offset : offset + limit] if featured else all_models[:limit]

    return CatalogResult(total=total, limit=limit, offset=offset, models=paginated)


def _task_to_pipeline(task: str | None) -> str:
    """Map task name to HuggingFace pipeline tag."""
    mapping = {
        "chat": "text-generation",
        "embedding": "feature-extraction",
        "vision": "image-text-to-text",
    }
    return mapping.get(task or "chat", "text-generation")


_PIPELINE_TO_TASK: dict[str, str] = {
    "text-generation": "chat",
    "feature-extraction": "embedding",
    "image-text-to-text": "vision",
    "image-to-text": "vision",
}


def _pipeline_to_task(pipeline_tag: str) -> str:
    """Map HuggingFace pipeline tag to internal task name."""
    return _PIPELINE_TO_TASK.get(pipeline_tag, "chat")


def _get_installed_models(model_manager: Any) -> set[str]:
    """Get set of installed model names from model_manager."""
    try:
        return set(model_manager.list_installed())
    except Exception:
        return set()


_SORT_KEYS: dict[str, tuple] = {
    "downloads": (lambda m: m.downloads, True),
    "name": (lambda m: m.name.lower(), False),
    "size_asc": (lambda m: m.size_gb, False),
    "size_desc": (lambda m: m.size_gb, True),
    "featured": (lambda m: (not m.featured, -m.downloads), False),
}


def _sort_models(models: list[CatalogModel], sort: str) -> list[CatalogModel]:
    """Sort models according to the specified sort order."""
    key_fn, reverse = _SORT_KEYS.get(sort, _SORT_KEYS["featured"])
    return sorted(models, key=key_fn, reverse=reverse)


# Maps model names to catalog display names for lookup
def find_catalog_entry(name: str) -> CatalogModel | None:
    """Find a featured model by display name (case-insensitive)."""
    name_lower = name.lower()
    for model in FEATURED_ALL:
        if model.name.lower() == name_lower:
            return model
    return None


def _make_progress_tqdm(callback: Any) -> type:
    """Create a minimal tqdm-compatible class that routes progress to a callback.

    Does NOT inherit from real tqdm — avoids multiprocessing lock issues
    that crash in Textual worker threads. Only implements the interface
    that huggingface_hub actually calls.
    """

    class _CallbackProgress:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.total: int = kwargs.get("total", 0) or 0
            self.n: int = kwargs.get("initial", 0) or 0

        def update(self, n: int = 1) -> None:
            self.n += n
            if callback and self.total > 0:
                callback(self.n, self.total)

        def close(self) -> None:
            pass

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: Any) -> None:
            self.close()

    return _CallbackProgress


def download_model(entry: CatalogModel, *, on_progress: Any = None) -> Path:
    """Download a GGUF model from HuggingFace to cfg.models_dir.

    Uses huggingface_hub for resumable downloads, caching, and auth.
    The optional *on_progress(downloaded, total)* callback receives byte counts.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    cfg.models_dir.mkdir(parents=True, exist_ok=True)

    filename = _resolve_filename(entry)
    dest = cfg.models_dir / filename
    if dest.exists():
        log.info("Model already downloaded: %s", dest)
        return dest

    log.info("Downloading %s/%s → %s", entry.hf_repo, filename, cfg.models_dir)
    token = _hf_token()

    kwargs: dict[str, Any] = {
        "repo_id": entry.hf_repo,
        "filename": filename,
        "local_dir": cfg.models_dir,
        "token": token,
    }
    if on_progress is not None:
        kwargs["tqdm_class"] = _make_progress_tqdm(on_progress)

    try:
        path = hf_hub_download(**kwargs)
    except GatedRepoError:
        raise PermissionError(
            f"{entry.name} requires HuggingFace authentication. "
            "Set HF_TOKEN env var or visit the repo page to request access."
        ) from None
    except RepositoryNotFoundError:
        raise RuntimeError(f"Repository {entry.hf_repo!r} not found on HuggingFace.") from None

    return Path(path)


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
            headers=_hf_headers(),
        )
        if resp.status_code == 401:
            raise PermissionError(
                f"{entry.hf_repo} requires HuggingFace authentication. "
                "Set HF_TOKEN env var or visit the repo page to request access."
            )
        resp.raise_for_status()
        siblings = resp.json().get("siblings", [])
    except PermissionError:
        raise
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


def fetch_model_file_size(hf_repo: str) -> float:
    """Fetch the best GGUF file size from HuggingFace tree API.

    Returns size in GB, or 0.0 if unavailable.
    """
    try:
        resp = httpx.get(
            f"https://huggingface.co/api/models/{hf_repo}/tree/main",
            timeout=_DEFAULT_TIMEOUT,
            headers=_hf_headers(),
        )
        resp.raise_for_status()
        files = resp.json()
    except Exception:
        return 0.0

    gguf_files = [
        (f.get("path", ""), f.get("size", 0) or f.get("lfs", {}).get("size", 0))
        for f in files
        if isinstance(f, dict) and f.get("path", "").endswith(".gguf")
    ]
    if not gguf_files:
        return 0.0

    best_name = _pick_best_gguf([name for name, _ in gguf_files])
    size_bytes = next((s for n, s in gguf_files if n == best_name), 0)
    return round(size_bytes / (1024**3), 1) if size_bytes else 0.0
