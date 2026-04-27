"""HuggingFace API client, TTL cache, and download progress plumbing."""

import io
import logging
import os
import threading
import time
from collections.abc import Callable
from http import HTTPStatus
from typing import Any

import httpx
from huggingface_hub import ModelInfo
from huggingface_hub.hf_api import RepoSibling
from tqdm.auto import tqdm as _base_tqdm

from lilbee.catalog.models import CatalogModel, DownloadProgress, _HfGgufMeta, _HfPage

log = logging.getLogger(__name__)

HF_API_URL = "https://huggingface.co/api/models"

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
    its file handle to a devnull sink: prevents ANSI escape sequences from leaking
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


_DEFAULT_TIMEOUT = 30.0

# Fields to request from the HF listing API via ?expand=.
# Without expand, the default response omits siblings, cardData, and gguf.
_HF_EXPAND_FIELDS: list[str] = ["gguf", "siblings", "downloads", "pipeline_tag", "cardData"]


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
# Module-private TTL cache; not in Services because it's request state, not an instance handle.
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
    # Local import to avoid circular: query imports hf_client; hf_client needs
    # _pipeline_to_task which lives in query.
    from lilbee.catalog.query import _pipeline_to_task

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
        if resp.status_code >= HTTPStatus.BAD_REQUEST:
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
    return 0.0  # unknown: display as "?" in UI
