"""HuggingFace API client with TTL cache."""

from __future__ import annotations

import logging
import os
import threading
import time
from http import HTTPStatus

import httpx
from huggingface_hub import ModelInfo
from huggingface_hub.hf_api import RepoSibling

from lilbee.catalog.compat import classify
from lilbee.catalog.models import CatalogModel, HfGgufMeta, HfPage
from lilbee.catalog.refs import GGUF_GLOB, pick_best_gguf
from lilbee.core.config import cfg

log = logging.getLogger(__name__)

# Substrings dropped from huggingface_hub's request / file-download loggers.
# These advisories aren't actionable in a local TUI: HF prints an
# unauthenticated-requests notice on every public pull, and the file_download
# logger re-warns on every retry the library schedules. The catalog surfaces
# the final download failure with a clear message, so per-attempt warnings
# are noise.
_HF_SUPPRESS_SUBSTRINGS = (
    "unauthenticated requests to the HF Hub",
    "Error while downloading from",
    "Trying to resume download",
)

_HF_FILTERED_LOGGER_NAMES = (
    "huggingface_hub.utils._http",
    "huggingface_hub.file_download",
)


class _HfSubstringFilter(logging.Filter):
    """Drop huggingface_hub log records whose message contains a suppressed substring."""

    def __init__(self, needles: tuple[str, ...]) -> None:
        super().__init__()
        self._needles = needles

    def filter(self, record: logging.LogRecord) -> bool:
        return not any(n in record.getMessage() for n in self._needles)


def install_hf_log_filter() -> None:
    """Attach the substring filter to huggingface_hub's chatty loggers.

    Called automatically when this module is imported (see the module-top
    invocation below) so the filter is in place before any catalog HTTP
    call can emit a warning. Exposed as a function so tests can re-apply.
    """
    hf_filter = _HfSubstringFilter(_HF_SUPPRESS_SUBSTRINGS)
    for name in _HF_FILTERED_LOGGER_NAMES:
        logging.getLogger(name).addFilter(hf_filter)


# Install the filter at module import. All HF HTTP traffic in lilbee
# routes through this module, so installing here always beats the first
# huggingface_hub warning to the punch.
install_hf_log_filter()

HF_API_URL = "https://huggingface.co/api/models"

DEFAULT_TIMEOUT = 30.0

# Fields requested from the HF listing API via ``?expand=``. Without this
# expand, the default response omits siblings, cardData, and gguf.
_HF_EXPAND_FIELDS: list[str] = ["gguf", "siblings", "downloads", "pipeline_tag", "cardData"]

# HF ``?search=`` is a single space-tokenized substring match on the model id.
# Multiple ``search=`` params are silently ignored, so the user's query is
# space-joined onto the GGUF filter into one param value.
_HF_GGUF_SEARCH_TERM = "GGUF"

_EMPTY_HF_PAGE = HfPage(models=[], has_more=False)

_BYTES_PER_GB = 1024**3


def hf_token() -> str | None:
    """Resolve the HuggingFace token in priority order: env > cfg > hub cache."""
    token = os.environ.get("LILBEE_HF_TOKEN") or os.environ.get("HF_TOKEN") or None
    if token:
        return token
    if cfg.hf_token:
        return cfg.hf_token
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def hf_headers() -> dict[str, str]:
    """Build HTTP headers for HuggingFace API requests."""
    token = hf_token()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _hf_search_value(search: str) -> str:
    """Build the HF ``search=`` value: GGUF plus the user's tokens, space-joined."""
    tokens = [_HF_GGUF_SEARCH_TERM, *search.split()]
    return " ".join(tokens)


def _resolve_sibling_gguf(siblings: list[RepoSibling]) -> str:
    """Concrete GGUF filename for a repo's sibling list, or ``GGUF_GLOB``.

    Uses the same quant picker as the pull path so the filename a catalog
    row carries always names the file a pull of that row produces.
    """
    gguf_files = [s.rfilename for s in siblings if s.rfilename.endswith(".gguf")]
    if not gguf_files:
        return GGUF_GLOB
    return pick_best_gguf(gguf_files)


def _estimate_size_from_siblings(siblings: list[RepoSibling]) -> float:
    """Estimate model size in GB from the largest GGUF file in siblings."""
    max_bytes = 0
    for sib in siblings:
        if sib.rfilename.endswith(".gguf"):
            max_bytes = max(max_bytes, sib.size or 0)
    if max_bytes > 0:
        return round(max_bytes / _BYTES_PER_GB, 1)
    return 0.0  # unknown: display as "?" in UI


class HfClient:
    """HuggingFace catalog API client with a per-instance TTL cache.

    Holds the per-process cache of catalog pages keyed by query
    parameters. The cache TTL and capacity are class-level so tests can
    override them via subclassing if needed; the cache state itself is
    per-instance so ``reset_services()`` discards a stale instance
    along with its cache.
    """

    CACHE_TTL: float = 300.0
    CACHE_MAX_ENTRIES: int = 50
    # Rate-limit the "Failed to fetch models" warning so an offline user
    # doesn't see one line per UI tick. First failure surfaces immediately;
    # repeats within the window stay at DEBUG.
    FETCH_FAILURE_WARN_INTERVAL_S: float = 300.0

    def __init__(self) -> None:
        self._cache: dict[str, tuple[float, HfPage]] = {}
        self._cache_lock = threading.Lock()
        self._arch_cache: dict[str, str] = {}
        # -inf, not 0.0: on a freshly booted machine ``time.monotonic()`` can be
        # smaller than the window, which would push the first failure to DEBUG.
        self._last_fetch_failure_warn: float = float("-inf")

    def get_cached_arch(self, ref: str) -> str | None:
        """Return the cached `general.architecture` for *ref*, or None if not cached."""
        return self._arch_cache.get(ref)

    def cache_arch(self, ref: str, architecture: str) -> None:
        """Record *architecture* for *ref* in the per-instance cache."""
        self._arch_cache[ref] = architecture

    def fetch_models(
        self,
        pipeline_tag: str = "text-generation",
        sort: str = "downloads",
        limit: int = 50,
        offset: int = 0,
        library: str | None = None,
        search: str = "",
    ) -> HfPage:
        """Fetch GGUF models from HuggingFace API with TTL cache.

        Returns an ``HfPage`` with a ``has_more`` flag derived from the
        ``Link: <...>; rel="next"`` response header (RFC 5988), the same
        mechanism the ``huggingface_hub`` library uses internally.
        """
        # Local import to avoid a cycle: query imports hf_client (this
        # module), and hf_client uses pipeline_to_task from query.
        from lilbee.catalog.query import pipeline_to_task

        search_value = _hf_search_value(search)
        cache_key = f"{pipeline_tag}:{sort}:{limit}:{offset}:{library}:{search_value}"
        now = time.monotonic()
        with self._cache_lock:
            expired = [k for k, (ts, _) in self._cache.items() if now - ts >= self.CACHE_TTL]
            for k in expired:
                del self._cache[k]

            cached = self._cache.get(cache_key)
            if cached and now - cached[0] < self.CACHE_TTL:
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
            resp = httpx.get(
                HF_API_URL, params=params, timeout=DEFAULT_TIMEOUT, headers=hf_headers()
            )
            if resp.status_code >= HTTPStatus.BAD_REQUEST:
                log.warning("HuggingFace API returned HTTP %d", resp.status_code)
                return _EMPTY_HF_PAGE
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            self._log_fetch_failure(exc)
            return _EMPTY_HF_PAGE

        has_more = "next" in resp.links

        models: list[CatalogModel] = []
        for raw in data:
            if not raw.get("id"):
                continue
            item = ModelInfo(**raw)
            card_desc = item.card_data.get("description", "") if item.card_data else ""
            gguf_meta = HfGgufMeta(**(item.gguf or {}))
            if gguf_meta.total > 0:
                size_gb = round(gguf_meta.total / _BYTES_PER_GB, 1)
            else:
                size_gb = _estimate_size_from_siblings(item.siblings or [])
            task = pipeline_to_task(item.pipeline_tag or "")
            models.append(
                CatalogModel(
                    hf_repo=item.id,
                    gguf_filename=_resolve_sibling_gguf(item.siblings or []),
                    size_gb=size_gb,
                    min_ram_gb=round(max(2.0, size_gb * 1.5), 1),
                    description=card_desc[:120] if card_desc else "",
                    featured=False,
                    downloads=item.downloads or 0,
                    task=task,
                    architecture=gguf_meta.architecture,
                    compat=classify(gguf_meta.architecture),
                )
            )
            self.cache_arch(item.id, gguf_meta.architecture)
        page = HfPage(models=models, has_more=has_more)
        with self._cache_lock:
            self._cache[cache_key] = (now, page)
            if len(self._cache) > self.CACHE_MAX_ENTRIES:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
                del self._cache[oldest_key]
        return page

    def _log_fetch_failure(self, exc: Exception) -> None:
        """Log an HF fetch failure, rate-limited so offline use doesn't spam.

        First failure of each ``FETCH_FAILURE_WARN_INTERVAL_S`` window logs
        at WARNING; repeats within the window log at DEBUG. The interval
        starts from the last WARNING so a flapping network produces one
        line every five minutes, not one per UI tick.
        """
        now = time.monotonic()
        if now - self._last_fetch_failure_warn >= self.FETCH_FAILURE_WARN_INTERVAL_S:
            log.warning("Failed to fetch models from HuggingFace: %s", exc)
            self._last_fetch_failure_warn = now
        else:
            log.debug("Suppressed repeat HF fetch failure: %s", exc)
