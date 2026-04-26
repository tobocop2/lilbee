"""Model catalog — discovers available GGUF models from HuggingFace.

Three levels:
1. Featured models — curated favorites (hardcoded, always available)
2. HF API models — fetched from HuggingFace API, paginated and filterable
3. Combined catalog — featured first, then HF results
"""

import fnmatch
import functools
import io
import logging
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import httpx
from huggingface_hub import ModelInfo
from huggingface_hub.hf_api import RepoSibling
from huggingface_hub.utils import HFValidationError, validate_repo_id
from pydantic import BaseModel
from tqdm.auto import tqdm as _base_tqdm

from lilbee.cancellation import TaskCancelled
from lilbee.model_manager import ModelSource
from lilbee.models import ModelTask
from lilbee.registry import ModelManifest, ModelRegistry

# circular: config.py -> catalog (via the per-role task validator). cfg is
# imported lazily so this module can load before Config() finishes init.

log = logging.getLogger(__name__)


def _cfg() -> Any:
    """Lazy accessor for the global ``cfg`` singleton (see circular-import note)."""
    from lilbee.config import cfg

    return cfg


def __getattr__(name: str) -> Any:
    """Expose ``catalog.cfg`` lazily so ``monkeypatch.setattr(catalog.cfg, ...)`` still works."""
    if name == "cfg":
        from lilbee.config import cfg

        return cfg
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


HF_API_URL = "https://huggingface.co/api/models"


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


ProgressCallback = Callable[[int, int], None]
_BYTES_PER_MB = 1024 * 1024


def make_download_callback(
    on_update: Callable[[DownloadProgress], None],
    *,
    throttle_interval: float = 0.1,
) -> ProgressCallback:
    """Build a download progress callback that converts bytes to human-readable state.
    *on_update(progress: DownloadProgress)* is called at most once per
    ``throttle_interval`` seconds with a float percentage (0.0 to 100.0), a
    ``"<done>/<total> MB"`` detail string, and a cache-hit flag. Both the
    catalog and setup screens use this so byte-to-MB conversion and
    cache-hit detection aren't duplicated.
    """
    last_update_time = 0.0
    seen_partial = False

    def _on_progress(downloaded: int, total: int) -> None:
        nonlocal last_update_time, seen_partial

        if total > 0 and downloaded >= total and not seen_partial:
            on_update(
                DownloadProgress(percent=100.0, detail="already downloaded", is_cache_hit=True)
            )
            return
        seen_partial = True

        now = time.monotonic()
        if now - last_update_time < throttle_interval:
            return
        last_update_time = now

        mb_done = downloaded / _BYTES_PER_MB
        if total > 0:
            pct = min(downloaded * 100.0 / total, 100.0)
            mb_total = total / _BYTES_PER_MB
            on_update(
                DownloadProgress(
                    percent=pct,
                    detail=f"{mb_done:.0f}/{mb_total:.0f} MB",
                    is_cache_hit=False,
                )
            )
        else:
            on_update(DownloadProgress(percent=0.0, detail=f"{mb_done:.0f} MB", is_cache_hit=False))

    return _on_progress


class _CallbackProgressBar(_base_tqdm):
    """tqdm subclass that forwards progress to a plain callback.
    Fully suppresses terminal output by disabling tqdm rendering and redirecting
    its file handle to a devnull sink — prevents ANSI escape sequences from leaking
    into Textual's managed terminal.

    Overrides ``get_lock`` to return a threading lock instead of tqdm's default
    multiprocessing lock. Vanilla tqdm acquires ``self._lock`` even on the
    ``disable=True`` path (std.py:988), and the multiprocessing lock's lazy init
    raises ``ValueError`` when ``sys.stderr.fileno() == -1`` (Textual, Jupyter,
    pytest capture). A thread lock sidesteps that fd handling entirely.
    """

    _lock = threading.RLock()
    _callback: Any = None

    @classmethod
    def get_lock(cls) -> threading.RLock:
        return cls._lock

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs["disable"] = True
        kwargs["file"] = io.StringIO()  # absorb any accidental tqdm output
        super().__init__(*args, **kwargs)
        self._cumulative = 0

    def update(self, n: float = 1) -> bool | None:
        self._cumulative += int(n)
        if self._callback is not None:
            total = self.total if self.total is not None else 0
            self._callback(int(self._cumulative), int(total))
        return None


class _ProgressTracker:
    """Wraps a tqdm_class to detect whether progress updates actually fired."""

    def __init__(self, callback: Any) -> None:
        self.was_used = False
        self._callback = callback

    def make_tqdm_class(self) -> type[_base_tqdm]:
        tracker = self

        class _Cls(_CallbackProgressBar):
            _callback = staticmethod(tracker._callback)

            def update(self, n: float = 1) -> bool | None:
                tracker.was_used = True
                return super().update(n)

        return _Cls


class _HfGgufMeta(BaseModel):
    """GGUF metadata returned by the HF API when expand=gguf is requested.

    ModelInfo.gguf is typed as ``dict | None`` upstream, so we validate it ourselves.
    """

    total: int = 0
    architecture: str = ""
    context_length: int = 0


class DownloadConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    repo_id: str
    filename: str
    token: str | None
    force_download: bool = False
    cache_dir: str | None = None
    tqdm_class: Any = None


_DEFAULT_TIMEOUT = 30.0

# Fields to request from the HF listing API via ?expand=.
# Without expand, the default response omits siblings, cardData, and gguf.
_HF_EXPAND_FIELDS: list[str] = ["gguf", "siblings", "downloads", "pipeline_tag", "cardData"]


@dataclass(frozen=True)
class CatalogModel:
    """A model entry in the catalog.

    Identity is the HuggingFace repo. ``gguf_filename`` may be a glob
    (e.g. ``*Q4_K_M.gguf``) for featured entries; the concrete filename
    is resolved at install time and recorded on the manifest.
    """

    hf_repo: str
    gguf_filename: str
    size_gb: float
    min_ram_gb: float
    description: str
    featured: bool
    downloads: int
    task: str
    recommended: bool = False

    @property
    def ref(self) -> str:
        """Browse-time ref (the HF repo); concrete filename is resolved at install."""
        return self.hf_repo

    @property
    def display_name(self) -> str:
        """Human-readable label derived from the HuggingFace repo id."""
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
class _HfPage:
    """Internal: one page of HuggingFace API results."""

    models: list[CatalogModel]
    has_more: bool


@dataclass(frozen=True)
class ModelVariant:
    """A specific quantization/size variant within a model family.

    Identity is ``(hf_repo, filename)``. ``filename`` may be a glob for
    featured entries; the concrete file is resolved at install time.
    """

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
    task: str
    description: str
    variants: tuple[ModelVariant, ...]


def _load_featured() -> tuple[
    tuple[CatalogModel, ...],
    tuple[CatalogModel, ...],
    tuple[CatalogModel, ...],
    tuple[CatalogModel, ...],
]:
    """Load featured models from the TOML file, cached after first call."""
    import tomllib

    toml_path = Path(__file__).parent / "featured_models.toml"
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    def _build(task: ModelTask) -> tuple[CatalogModel, ...]:
        return tuple(
            CatalogModel(
                hf_repo=m["hf_repo"],
                gguf_filename=m["gguf_filename"],
                size_gb=m["size_gb"],
                min_ram_gb=m["min_ram_gb"],
                description=m["description"],
                featured=True,
                downloads=0,
                task=task,
                recommended=m.get("recommended", False),
            )
            for m in data.get(task, [])
        )

    return (
        _build(ModelTask.CHAT),
        _build(ModelTask.EMBEDDING),
        _build(ModelTask.VISION),
        _build(ModelTask.RERANK),
    )


FEATURED_CHAT, FEATURED_EMBEDDING, FEATURED_VISION, FEATURED_RERANK = _load_featured()

# Maps vision catalog entries to their mmproj (CLIP projection) filenames.
# Vision models need both the main GGUF and the mmproj file to work.
# Keys are hf_repo identifiers; values are glob patterns resolved at download time.
# Every FEATURED_VISION entry MUST have a corresponding key here.
_DEFAULT_MMPROJ_PATTERN = "*mmproj*.gguf"

VISION_MMPROJ_FILES: dict[str, str] = {
    "noctrex/LightOnOCR-2-1B-GGUF": _DEFAULT_MMPROJ_PATTERN,
}

FEATURED_ALL: tuple[CatalogModel, ...] = (
    FEATURED_CHAT + FEATURED_EMBEDDING + FEATURED_VISION + FEATURED_RERANK
)

_FAMILY_NAME_RE = re.compile(r"^(.+?)\s+\d")
PARAM_COUNT_RE = re.compile(r"(\d+\.?\d*B)", re.IGNORECASE)


def _extract_family_name(model_name: str) -> str:
    """Extract the family name by stripping the trailing parameter count.
    Applies clean_display_name first to strip -GGUF, -Instruct, etc.

    "Qwen3 8B" -> "Qwen3", "Qwen3-Coder 30B A3B" -> "Qwen3-Coder",
    "Nomic Embed Text v1.5" -> "Nomic Embed Text v1.5" (no trailing number pattern).
    """
    cleaned = clean_display_name(model_name)
    m = _FAMILY_NAME_RE.match(cleaned)
    return m.group(1) if m else cleaned


def _extract_quant(filename: str) -> str:
    """Extract quantization label from a GGUF filename pattern.
    "*Q4_K_M.gguf" -> "Q4_K_M", "nomic-embed-text-v1.5.Q4_K_M.gguf" -> "Q4_K_M".
    """
    m = re.search(r"(Q\d[A-Z0-9_]*)", filename, re.IGNORECASE)
    return m.group(1).upper() if m else ""


def _derive_param_count(model: CatalogModel) -> str:
    """Extract the parameter-count label (e.g. ``7B``) from a catalog model.

    Reads the count from the cleaned HF repo name. Returns an empty
    string when the display has no numeric suffix (e.g. embedding models
    like ``Nomic Embed Text v1.5``); callers fall back to ``"--"``.
    """
    match = PARAM_COUNT_RE.search(model.display_name)
    return match.group(1) if match else ""


def _catalog_to_variant(model: CatalogModel) -> ModelVariant:
    """Convert a CatalogModel to a ModelVariant."""
    return ModelVariant(
        hf_repo=model.hf_repo,
        filename=model.gguf_filename,
        param_count=_derive_param_count(model),
        quant=_extract_quant(model.gguf_filename),
        size_mb=int(model.size_gb * 1024),
        recommended=model.recommended,
    )


def _family_slug(display_name: str) -> str:
    """Stable slug for a family, derived from its display name."""
    return _extract_family_name(display_name).lower().replace(" ", "-")


def _build_families(models: tuple[CatalogModel, ...], task: str) -> list[ModelFamily]:
    """Group CatalogModels into families by display-derived family name."""
    groups: dict[str, list[CatalogModel]] = {}
    order: list[str] = []
    for m in models:
        family = _extract_family_name(m.display_name)
        if family not in groups:
            order.append(family)
        groups.setdefault(family, []).append(m)

    families: list[ModelFamily] = []
    for family_name in order:
        members = groups[family_name]
        representative = next((m for m in members if m.recommended), members[0])
        variants = [_catalog_to_variant(m) for m in members]
        families.append(
            ModelFamily(
                slug=_family_slug(representative.display_name),
                name=family_name,
                task=task,
                description=representative.description,
                variants=tuple(variants),
            )
        )
    return families


def get_families() -> list[ModelFamily]:
    """Get all featured models grouped into families.
    Returns families ordered: chat, then embedding, then vision, then reranker.
    Within each family, variants are ordered smallest to largest, with
    the largest marked as recommended (for multi-variant families).
    """
    return (
        _build_families(FEATURED_CHAT, ModelTask.CHAT)
        + _build_families(FEATURED_EMBEDDING, ModelTask.EMBEDDING)
        + _build_families(FEATURED_VISION, ModelTask.VISION)
        + _build_families(FEATURED_RERANK, ModelTask.RERANK)
    )


_SIZE_RANGES: dict[str, tuple[float, float]] = {
    "small": (0.0, 3.0),
    "medium": (3.0, 10.0),
    "large": (10.0, float("inf")),
}


def _hf_token() -> str | None:
    """Read HuggingFace token from env vars or huggingface_hub login cache."""
    token = os.environ.get("LILBEE_HF_TOKEN") or os.environ.get("HF_TOKEN") or None
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def _hf_headers() -> dict[str, str]:
    """Build HTTP headers for HuggingFace API requests."""
    token = _hf_token()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


# TTL cache for HuggingFace API results (5 minutes). The lock guards the
# evict-then-insert path so concurrent TUI workers can't race and hit
# ``RuntimeError: dictionary changed size during iteration``.
_HF_CACHE_TTL = 300
_HF_CACHE_MAX_ENTRIES = 50
_hf_cache: dict[str, tuple[float, _HfPage]] = {}
_hf_cache_lock = threading.Lock()

_EMPTY_HF_PAGE = _HfPage(models=[], has_more=False)

# HF ``?search=`` is a single space-tokenized substring match on the model id.
# Multiple ``search=`` params are silently ignored, so the user's query is
# space-joined onto the GGUF filter into one param value.
_HF_GGUF_SEARCH_TERM = "GGUF"


def _hf_search_value(search: str) -> str:
    """Build the HF ``search=`` value: GGUF plus the user's tokens, space-joined."""
    tokens = [_HF_GGUF_SEARCH_TERM, *search.split()]
    return " ".join(tokens)


def _fetch_hf_models(
    pipeline_tag: str = "text-generation",
    sort: str = "downloads",
    limit: int = 50,
    offset: int = 0,
    library: str | None = None,
    search: str = "",
) -> _HfPage:
    """Fetch GGUF models from HuggingFace API with 5-minute cache.

    Returns an ``_HfPage`` with a ``has_more`` flag derived from the
    ``Link: <...>; rel="next"`` response header (RFC 5988), the same
    mechanism the ``huggingface_hub`` library uses internally.
    """
    search_value = _hf_search_value(search)
    cache_key = f"{pipeline_tag}:{sort}:{limit}:{offset}:{library}:{search_value}"
    now = time.monotonic()
    with _hf_cache_lock:
        expired = [k for k, (ts, _) in _hf_cache.items() if now - ts >= _HF_CACHE_TTL]
        for k in expired:
            del _hf_cache[k]

        cached = _hf_cache.get(cache_key)
        if cached and now - cached[0] < _HF_CACHE_TTL:
            return cached[1]

    params = httpx.QueryParams(
        pipeline_tag=pipeline_tag,
        search=search_value,
        sort=sort,
        limit=limit,
        skip=offset,
        expand=_HF_EXPAND_FIELDS,
    )
    if library:
        params = params.add("library", library)
    try:
        resp = httpx.get(HF_API_URL, params=params, timeout=_DEFAULT_TIMEOUT, headers=_hf_headers())
        if resp.status_code >= 400:
            log.warning("HuggingFace API returned HTTP %d", resp.status_code)
            return _EMPTY_HF_PAGE
        data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("Failed to fetch models from HuggingFace: %s", exc)
        return _EMPTY_HF_PAGE

    has_more = "next" in resp.links

    models: list[CatalogModel] = []
    for raw in data:
        if not raw.get("id"):
            continue
        item = ModelInfo(**raw)
        card_desc = item.card_data.get("description", "") if item.card_data else ""
        model_desc = card_desc
        gguf_meta = _HfGgufMeta(**(item.gguf or {}))
        if gguf_meta.total > 0:
            size_gb = round(gguf_meta.total / (1024**3), 1)
        else:
            size_gb = _estimate_size_from_siblings(item.siblings or [])
        task = _pipeline_to_task(item.pipeline_tag or "")
        models.append(
            CatalogModel(
                hf_repo=item.id,
                gguf_filename="*.gguf",
                size_gb=size_gb,
                min_ram_gb=max(2.0, size_gb * 1.5),
                description=model_desc[:120] if model_desc else "",
                featured=False,
                downloads=item.downloads or 0,
                task=task,
            )
        )
    page = _HfPage(models=models, has_more=has_more)
    with _hf_cache_lock:
        _hf_cache[cache_key] = (now, page)
        if len(_hf_cache) > _HF_CACHE_MAX_ENTRIES:
            oldest_key = min(_hf_cache, key=lambda k: _hf_cache[k][0])
            del _hf_cache[oldest_key]
    return page


def _has_gguf_siblings(siblings: list[RepoSibling]) -> bool:
    """Return True if the sibling list contains at least one .gguf file."""
    return any(s.rfilename.endswith(".gguf") for s in siblings)


def _estimate_size_from_siblings(siblings: list[RepoSibling]) -> float:
    """Estimate model size in GB from the largest GGUF file in siblings."""
    max_bytes = 0
    for sib in siblings:
        if sib.rfilename.endswith(".gguf"):
            max_bytes = max(max_bytes, sib.size or 0)
    if max_bytes > 0:
        return round(max_bytes / (1024**3), 1)
    return 0.0  # unknown — display as "?" in UI


def _search_blob(m: CatalogModel) -> str:
    """Lowercased join of searchable fields on a catalog row.

    Null char joins the fields so a search term never straddles them.
    """
    return f"{m.display_name}\0{m.hf_repo}\0{m.description}".lower()


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
    hf_has_more = False

    # Optionally fetch from HF API
    if not featured:
        hf_task, hf_library = _task_to_pipeline(task)
        hf_page = _fetch_hf_models(
            pipeline_tag=hf_task,
            limit=limit,
            offset=offset,
            library=hf_library,
            search=search,
        )
        hf_has_more = hf_page.has_more
        # Deduplicate: skip HF models whose repo matches a featured model
        featured_repos = {m.hf_repo for m in FEATURED_ALL}
        hf_models = [m for m in hf_page.models if m.hf_repo not in featured_repos]
        all_models.extend(hf_models)

    # Filter by task
    if task:
        all_models = [m for m in all_models if m.task == task]

    # Filter by search. Single join+lower per model per keystroke instead
    # of four separate lowers + substring checks; the no-match path
    # (the common case) runs four times fewer ``str.lower()`` calls.
    if search:
        search_lower = search.lower()
        all_models = [m for m in all_models if search_lower in _search_blob(m)]

    # Filter by size
    if size and size in _SIZE_RANGES:
        lo, hi = _SIZE_RANGES[size]
        all_models = [m for m in all_models if lo <= m.size_gb < hi]

    # Filter by installed status. Installed manifests have refs of the
    # form ``hf_repo/filename``; catalog entries identify by ``hf_repo``.
    # A repo is "installed" if any quant of it has a manifest.
    if installed is not None and model_manager is not None:
        installed_repos = {ref.rsplit("/", 1)[0] for ref in _get_installed_models(model_manager)}
        if installed:
            all_models = [m for m in all_models if m.hf_repo in installed_repos]
        else:
            all_models = [m for m in all_models if m.hf_repo not in installed_repos]

    # Filter by featured status
    if featured is not None:
        all_models = [m for m in all_models if m.featured == featured]

    # Sort
    all_models = _sort_models(all_models, sort)

    total = len(all_models)

    # When HF API pagination is active (offset passed to API), skip local slicing
    # to avoid double-applying the offset. Only slice for featured-only requests.
    paginated = all_models[offset : offset + limit] if featured else all_models[:limit]

    return CatalogResult(
        total=total, limit=limit, offset=offset, models=paginated, has_more=hf_has_more
    )


def _task_to_pipeline(task: str | None) -> tuple[str, str | None]:
    """Map task name to HuggingFace pipeline tag and library filter."""
    mapping: dict[str, tuple[str, str | None]] = {
        ModelTask.CHAT: ("text-generation", None),
        ModelTask.EMBEDDING: ("feature-extraction", "sentence-transformers"),
        ModelTask.VISION: ("image-text-to-text", None),
        ModelTask.RERANK: ("text-classification", None),
    }
    return mapping.get(task or ModelTask.CHAT, ("text-generation", None))


_PIPELINE_TO_TASK: dict[str, str] = {
    "text-generation": ModelTask.CHAT,
    "feature-extraction": ModelTask.EMBEDDING,
    "sentence-similarity": ModelTask.EMBEDDING,
    "image-text-to-text": ModelTask.VISION,
    "image-to-text": ModelTask.VISION,
    "text-classification": ModelTask.RERANK,
    "text-ranking": ModelTask.RERANK,
}


def _pipeline_to_task(pipeline_tag: str) -> str:
    """Map HuggingFace pipeline tag to internal task name."""
    return _PIPELINE_TO_TASK.get(pipeline_tag, ModelTask.CHAT)


def _get_installed_models(model_manager: Any) -> set[str]:
    """Get set of installed model names from model_manager."""
    try:
        return set(model_manager.list_installed())
    except Exception:
        return set()


_SORT_KEYS: dict[str, tuple] = {
    "downloads": (lambda m: m.downloads, True),
    "name": (lambda m: m.display_name.lower(), False),
    "size_asc": (lambda m: m.size_gb, False),
    "size_desc": (lambda m: m.size_gb, True),
    "featured": (lambda m: (not m.featured, -m.downloads), False),
}


def _sort_models(models: list[CatalogModel], sort: str) -> list[CatalogModel]:
    """Sort models according to the specified sort order."""
    key_fn, reverse = _SORT_KEYS.get(sort, _SORT_KEYS["featured"])
    return sorted(models, key=key_fn, reverse=reverse)


class CatalogIndex(NamedTuple):
    """Case-insensitive lookup indexes for find_catalog_entry."""

    by_hf_repo: dict[str, CatalogModel]
    by_full_ref: dict[str, CatalogModel]  # repo + concrete filename


@functools.cache
def _build_catalog_index() -> CatalogIndex:
    """Build case-insensitive lookup indexes for find_catalog_entry."""
    by_hf_repo: dict[str, CatalogModel] = {}
    by_full_ref: dict[str, CatalogModel] = {}
    for m in FEATURED_ALL:
        by_hf_repo.setdefault(m.hf_repo.lower(), m)
        if "*" not in m.gguf_filename:
            by_full_ref[f"{m.hf_repo}/{m.gguf_filename}".lower()] = m
    return CatalogIndex(by_hf_repo, by_full_ref)


def find_catalog_entry(query: str) -> CatalogModel | None:
    """Find a featured model by hf_repo or by full ``hf_repo/filename`` ref.

    Lookups, in order:
    1. Full canonical ref ``<repo>/<filename>`` (only featured entries
       with a concrete, non-glob filename live in this index).
    2. Bare ``<repo>``: strip the trailing ``.gguf`` segment if present.
    3. Provider-prefixed refs (``ollama/...``, ``openai/...``): strip
       the first segment when it is not itself a known HF owner.

    Case-insensitive. Returns ``None`` on miss.
    """
    if not query:
        return None
    idx = _build_catalog_index()
    q = query.lower()
    candidates = [q]
    # Strip the filename for ``<repo>/<filename>.gguf`` queries so the
    # bare-repo index catches featured entries whose gguf_filename is a
    # glob (most are).
    if q.endswith(".gguf") and q.count("/") >= 2:
        candidates.append(q.rsplit("/", 1)[0])
    if "/" in q:
        prefix, rest = q.split("/", 1)
        hf_owners = {r.split("/", 1)[0] for r in idx.by_hf_repo if "/" in r}
        if prefix not in hf_owners:
            candidates.append(rest)
    for c in candidates:
        hit = idx.by_full_ref.get(c) or idx.by_hf_repo.get(c)
        if hit is not None:
            return hit
    return None


def is_rerank_ref(model_ref: str) -> bool:
    """Return True iff *model_ref* resolves to a rerank catalog entry."""
    if not model_ref:
        return False
    entry = find_catalog_entry(model_ref)
    return entry is not None and entry.task == ModelTask.RERANK


def _is_hf_repo_id(value: str) -> bool:
    """True if *value* is a well-formed ``owner/name`` HuggingFace repo id."""
    if "/" not in value:
        return False
    try:
        validate_repo_id(value)
    except HFValidationError:
        return False
    return True


def build_adhoc_entry(hf_repo: str, *, task: str = ModelTask.CHAT) -> CatalogModel:
    """Minimal CatalogModel for a non-featured HuggingFace GGUF repo."""
    return CatalogModel(
        hf_repo=hf_repo,
        gguf_filename="*.gguf",
        size_gb=0.0,
        min_ram_gb=2.0,
        description="",
        featured=False,
        downloads=0,
        task=task,
    )


def resolve_pull_target(model: str) -> CatalogModel | None:
    """Resolve *model* to a pullable entry: featured first, then ad-hoc HF."""
    featured = find_catalog_entry(model)
    if featured is not None:
        return featured
    return build_adhoc_entry(model) if _is_hf_repo_id(model) else None


def download_model(entry: CatalogModel, *, on_progress: ProgressCallback | None = None) -> Path:
    """Download a GGUF model from HuggingFace to cfg.models_dir.
    Uses huggingface_hub for resumable downloads, caching, and auth.
    The optional *on_progress(downloaded, total)* callback receives byte counts.
    For vision models, also downloads the mmproj (CLIP projection) file.

    Raises:
        PermissionError: gated repo requiring authentication
        RuntimeError: repo not found or download failure with details
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    _cfg().models_dir.mkdir(parents=True, exist_ok=True)

    filename = resolve_filename(entry)
    dest = _cfg().models_dir / filename
    if dest.exists():
        log.info("Model already downloaded: %s", dest)
        if on_progress is not None:
            size = dest.stat().st_size
            on_progress(size, size)  # Report 100% immediately
        return _finalize_download(entry, dest, on_progress=on_progress)

    log.info("Downloading %s/%s → %s", entry.hf_repo, filename, _cfg().models_dir)
    token = _hf_token()

    tracker = _ProgressTracker(on_progress) if on_progress else None
    config = DownloadConfig(
        repo_id=entry.hf_repo,
        filename=filename,
        token=token,
        cache_dir=str(_cfg().models_dir),
        tqdm_class=tracker.make_tqdm_class() if tracker else None,
    )

    try:
        # HF_HUB_DISABLE_XET is set in lilbee/__init__.py at import time.
        # Setting it here is too late — huggingface_hub.constants already
        # captured the value when this module first imported it.
        cached = Path(hf_hub_download(**config.model_dump(exclude_none=True)))
    except TaskCancelled:
        raise
    except GatedRepoError:
        raise PermissionError(
            f"{entry.hf_repo} requires HuggingFace authentication. "
            "Set HF_TOKEN env var or visit the repo page to request access."
        ) from None
    except RepositoryNotFoundError:
        raise RuntimeError(f"Repository {entry.hf_repo!r} not found on HuggingFace.") from None
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        raise RuntimeError(f"Network error downloading {entry.hf_repo}: {exc}") from None
    except OSError as exc:
        raise RuntimeError(f"I/O error downloading {entry.hf_repo}: {exc}") from None
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download {entry.hf_repo}: {type(exc).__name__}: {exc}"
        ) from None

    if on_progress:
        actual_size = cached.stat().st_size
        if not tracker or not tracker.was_used:
            log.info("Model found in HuggingFace cache: %s", cached)
        on_progress(actual_size, actual_size)
    dest = cached
    return _finalize_download(entry, dest, on_progress=on_progress)


def _finalize_download(
    entry: CatalogModel,
    dest: Path,
    *,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Register the model in the manifest and download mmproj for vision models."""
    _register_model(entry, dest)
    if entry.task == ModelTask.VISION:
        _download_mmproj(entry, on_progress=on_progress)
    return dest


def _register_model(entry: CatalogModel, file_path: Path) -> None:
    """Create a registry manifest for a downloaded model."""
    registry = ModelRegistry(_cfg().models_dir)
    manifest = ModelManifest(
        hf_repo=entry.hf_repo,
        gguf_filename=file_path.name,
        size_bytes=file_path.stat().st_size,
        task=entry.task,
        downloaded_at=datetime.now(UTC).isoformat(),
    )
    try:
        registry.install(entry.hf_repo, file_path.name, file_path, manifest)
        log.info("Registered %s/%s in manifest", entry.hf_repo, file_path.name)
    except Exception:
        log.warning("Failed to register manifest for %s", entry.hf_repo, exc_info=True)


def _download_mmproj(
    entry: CatalogModel,
    *,
    on_progress: ProgressCallback | None = None,
) -> Path | None:
    """Download the mmproj (CLIP projection) file for a vision model.
    Returns the path to the downloaded file, or None if no mmproj is configured.
    The optional ``on_progress`` callback receives ``(downloaded, total)`` byte
    counts and is wired through the same tqdm hook used by the main download.
    """
    mmproj_pattern = VISION_MMPROJ_FILES.get(entry.hf_repo, _DEFAULT_MMPROJ_PATTERN)

    mmproj_filename = _resolve_mmproj_filename(entry.hf_repo, mmproj_pattern)
    if not mmproj_filename:
        log.warning("Could not resolve mmproj file for %s", entry.hf_repo)
        return None

    from huggingface_hub import hf_hub_download

    tracker = _ProgressTracker(on_progress) if on_progress else None
    log.info("Downloading mmproj %s/%s → %s", entry.hf_repo, mmproj_filename, _cfg().models_dir)
    path = Path(
        hf_hub_download(
            repo_id=entry.hf_repo,
            filename=mmproj_filename,
            cache_dir=str(_cfg().models_dir),
            token=_hf_token(),
            tqdm_class=tracker.make_tqdm_class() if tracker else None,
        )
    )
    if on_progress is not None and (not tracker or not tracker.was_used):
        # Cache hit — HF returned the cached path without invoking tqdm.
        size = path.stat().st_size
        on_progress(size, size)
    return path


def _resolve_mmproj_filename(hf_repo: str, pattern: str) -> str | None:
    """Resolve an mmproj filename pattern to a concrete filename via the HF API."""
    if "*" not in pattern:
        return pattern

    try:
        resp = httpx.get(
            f"https://huggingface.co/api/models/{hf_repo}",
            timeout=_DEFAULT_TIMEOUT,
            headers=_hf_headers(),
        )
        resp.raise_for_status()
        siblings = resp.json().get("siblings", [])
    except Exception as exc:
        log.warning("Cannot query mmproj files for %s: %s", hf_repo, exc)
        return None

    mmproj_files: list[str] = [
        s.get("rfilename", "") for s in siblings if fnmatch.fnmatch(s.get("rfilename", ""), pattern)
    ]
    if not mmproj_files:
        return None

    # Prefer F16 over F32 (smaller), and any over BF16
    for preference in ("f16", "F16"):
        for f in mmproj_files:
            if preference in f:
                return f
    return mmproj_files[0]


def _mmproj_in_models_dir_matching(pattern: str) -> Path | None:
    """Return the first ``*.gguf`` under ``_cfg().models_dir`` that matches."""
    models_dir: Path = _cfg().models_dir
    for p in models_dir.rglob("*.gguf"):
        if fnmatch.fnmatch(p.name, pattern) or "mmproj" in p.name.lower():
            return p
    return None


def find_mmproj_file(model_ref: str) -> Path | None:
    """Find the mmproj for a ``FEATURED_VISION`` entry under ``_cfg().models_dir``.

    *model_ref* is matched against each featured vision entry's
    ``hf_repo``. Returns ``None`` when nothing matches. Never falls back
    to an arbitrary mmproj: that cross-contaminates non-vision chat
    models (e.g. a chat model would inherit a vision model's mmproj and
    be misreported as vision-capable).
    """
    if not _cfg().models_dir.exists():
        return None
    for entry in FEATURED_VISION:
        if model_ref not in entry.hf_repo and entry.hf_repo not in model_ref:
            continue
        pattern = VISION_MMPROJ_FILES.get(entry.hf_repo, _DEFAULT_MMPROJ_PATTERN)
        match = _mmproj_in_models_dir_matching(pattern)
        if match is not None:
            return match
    return None


_QUANT_PREFERENCE = ("Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q8_0", "Q6_K", "Q3_K_M")


def resolve_filename(entry: CatalogModel) -> str:
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


_DISPLAY_NAME_SUFFIXES = re.compile(r"-(GGUF|Instruct|Chat)(?=-|$)", re.IGNORECASE)
_DISPLAY_NAME_DATE_SUFFIX = re.compile(r"-\d{4}$")
_DISPLAY_NAME_META_PREFIX = re.compile(r"^Meta-", re.IGNORECASE)


def clean_display_name(repo_id: str) -> str:
    """Derive a human-friendly display name from a HuggingFace repo ID.
    Strips org prefix, -GGUF/-Instruct/-Chat suffixes, date suffixes (-2507),
    and Meta- prefix. Replaces hyphens with spaces.

    Examples:
        "Qwen/Qwen2.5-7B-Instruct-GGUF" -> "Qwen2.5 7B"
        "meta-llama/Meta-Llama-3-8B"     -> "Llama 3 8B"
    """
    name = repo_id.split("/")[-1]
    name = _DISPLAY_NAME_SUFFIXES.sub("", name)
    name = _DISPLAY_NAME_DATE_SUFFIX.sub("", name)
    name = _DISPLAY_NAME_META_PREFIX.sub("", name)
    name = name.replace("-", " ").strip()
    return re.sub(r"\s+", " ", name)


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
        return "—"
    return QUANT_TIERS.get(quant, "—")


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
    installed_repos = {ref.rsplit("/", 1)[0] for ref in installed_refs}
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
                param_count=_derive_param_count(m),
                quality_tier=quant_tier(_extract_quant(m.gguf_filename)),
                installed=m.hf_repo in installed_repos,
                source=ModelSource.NATIVE.value,
            )
        )
    return enriched
