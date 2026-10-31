"""HuggingFace API client with TTL cache."""

from __future__ import annotations

import fnmatch
import functools
import logging
import os
import threading
import time
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Literal

import httpx
from huggingface_hub import HfApi, ModelInfo
from huggingface_hub.errors import HfHubHTTPError, NotASafetensorsRepoError, SafetensorsParsingError
from huggingface_hub.hf_api import RepoSibling

from lilbee.catalog.compat import classify
from lilbee.catalog.content import is_safety_stripped
from lilbee.catalog.formats import hf_listing_params, select_files
from lilbee.catalog.models import (
    CatalogModel,
    HfGgufMeta,
    HfPage,
    estimate_min_ram_gb,
    estimate_size_gb,
)
from lilbee.catalog.refs import GGUF_GLOB, rank_gguf_candidates
from lilbee.catalog.types import ModelCompat, ModelFormat
from lilbee.providers.base import is_header_safe

if TYPE_CHECKING:
    from lilbee.providers.fleet.engines.base import EngineAdapter

log = logging.getLogger(__name__)

_HF_TOKEN_ENV = ("LILBEE_HF_TOKEN", "HF_TOKEN")

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
_HF_EXPAND_FIELDS: list[str] = [
    "gguf",
    "siblings",
    "downloads",
    "pipeline_tag",
    "cardData",
    "tags",
]

# Query parameter carrying the listing position in the ``Link rel=next`` URL.
_HF_CURSOR_PARAM = "cursor"

_EMPTY_HF_PAGE = HfPage(models=[])


# A resolved token: the token text, None when the library may look for one, or False
# when a refused token must make the library send none (it re-reads HF_TOKEN on None).
HfToken = str | Literal[False] | None


def hf_token() -> HfToken:
    """Resolve the HuggingFace token in priority order: env > cfg > hub cache.

    A token that cannot travel in a header resolves to False, which the library reads as
    anonymous. None means no token was found.
    """
    # circular: a module-level cfg import makes Config()'s model-ref validator
    # circular (config -> model_ref -> catalog -> here -> config).
    from lilbee.core.config import cfg

    token: str | None
    env_name = next((n for n in _HF_TOKEN_ENV if os.environ.get(n)), None)
    if env_name is not None:
        token, source = os.environ[env_name], f"the {env_name} environment variable"
    elif cfg.hf_token:
        token, source = cfg.hf_token, "the hf_token setting"
    else:
        try:
            from huggingface_hub import get_token

            token = get_token()
        except Exception:
            return None
        source = "the Hugging Face token file"
    if token is not None and not is_hf_token_sendable(token):
        _warn_unsendable_token(source)
        return False
    return token


def is_hf_token_sendable(token: str) -> bool:
    """True when *token* can travel in a header and holds no space."""
    return is_header_safe(token) and " " not in token


@functools.cache
def _warn_unsendable_token(source: str) -> None:
    """Tell the user once per source that the token is not sent."""
    log.warning(
        "The token from %s holds a character that cannot be sent, "
        "so requests run without a token. Set it again.",
        source,
    )


def hf_headers() -> dict[str, str]:
    """Build HTTP headers for HuggingFace API requests."""
    token = hf_token()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _next_cursor(resp: httpx.Response) -> str | None:
    """The cursor of the page after *resp*; None on the last page or an unusable next link."""
    link = resp.links.get("next")
    if link is None:
        return None
    try:
        cursor: str | None = httpx.URL(link["url"]).params.get(_HF_CURSOR_PARAM)
    except httpx.InvalidURL:
        log.debug("Ignoring an unparseable HuggingFace next link: %s", link["url"])
        return None
    return cursor or None


def _hf_search_value(search: str, fmt: ModelFormat = ModelFormat.GGUF) -> str:
    """Build the HF ``search=`` value: the format's search word plus the user's tokens."""
    return hf_listing_params(fmt, search)["search"]


@functools.lru_cache(maxsize=64)
def repo_has_mmproj(hf_repo: str) -> bool:
    """True when *hf_repo* ships a multimodal projector (``mmproj*.gguf`` sibling).

    A projector sibling marks the repo's model as a vision loader regardless of
    its text architecture or name; mainstream VL repos (Qwen-VL, InternVL,
    SmolVLM, gemma-3) match no vision name pattern. Fails open to False (any
    error: the probe is advisory) so an offline pull degrades to name-based
    classification.
    """
    from huggingface_hub import HfApi

    from lilbee.catalog.refs import DEFAULT_MMPROJ_PATTERN

    try:
        siblings = HfApi(token=hf_token()).model_info(hf_repo).siblings or []
    except Exception as exc:
        log.debug("mmproj sibling probe failed for %s: %s", hf_repo, exc)
        return False
    return any(fnmatch.fnmatch(s.rfilename, DEFAULT_MMPROJ_PATTERN) for s in siblings)


def read_parameter_count(repo: str, *, token: str | bool | None = None) -> dict[str, int]:
    """The per-dtype tensor count of a Hub repo, read from the safetensors headers alone.

    Returns an empty dict when the repo has no readable header, so the caller
    reaches the unknown-size reserve instead of an error. A header that parses but
    has the wrong shape (a list where a mapping belongs) counts as unreadable.
    """
    try:
        counts = HfApi().get_safetensors_metadata(repo, token=token).parameter_count
        return {str(dtype): count for dtype, count in counts.items()}
    except (
        HfHubHTTPError,
        NotASafetensorsRepoError,
        SafetensorsParsingError,
        OSError,
        AttributeError,
        TypeError,
        ValueError,
        KeyError,
    ) as exc:
        log.debug("No safetensors header for %s: %s", repo, exc)
        return {}


def _resolve_sibling_gguf(siblings: list[RepoSibling]) -> str:
    """Best-guess GGUF filename for a repo's sibling list, or ``GGUF_GLOB``.

    Ranks by quant label alone: a listing covers up to a full page of repos and
    cannot afford a header probe per row. A pull re-checks the choice against the
    file's header, so a row may name a file the pull then rejects in favour of a
    better one.
    """
    ranked = rank_gguf_candidates(s.rfilename for s in siblings)
    return next(iter(ranked), GGUF_GLOB)


def _resolve_sibling_file(fmt: ModelFormat, siblings: list[RepoSibling]) -> str:
    """The entry file a row of *fmt* names, or an empty string when the repo has none."""
    if fmt == ModelFormat.GGUF:
        return _resolve_sibling_gguf(siblings)
    try:
        selected = select_files(fmt, (s.rfilename for s in siblings))
    except ValueError:  # an unusable file set: the row has no entry file to name
        return ""
    return selected[0] if selected else ""


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

    def __init__(self, engine: EngineAdapter) -> None:
        self._engine = engine
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
        library: str | None = None,
        search: str = "",
        cursor: str | None = None,
        fmt: ModelFormat = ModelFormat.GGUF,
    ) -> HfPage:
        """Fetch one page of *fmt* models from the HuggingFace API with TTL cache.

        *cursor* names the page to fetch (None is the first). The returned page
        carries the next page's cursor from the ``Link: <...>; rel="next"``
        response header (RFC 5988), the pagination the HuggingFace API documents.
        """
        search_value = _hf_search_value(search, fmt)
        cache_key = f"{pipeline_tag}:{sort}:{limit}:{cursor}:{library}:{search_value}:{fmt}"
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
            filter=fmt.value,
            search=search_value,
            sort=sort,
            limit=limit,
            expand=_HF_EXPAND_FIELDS,
        )
        if library:
            params = params.add("library", library)
        if cursor:
            params = params.add(_HF_CURSOR_PARAM, cursor)
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

        next_cursor = _next_cursor(resp)

        models: list[CatalogModel] = []
        for raw in data:
            model = self._catalog_row(raw, fmt)
            if model is not None:
                models.append(model)
        page = HfPage(models=models, next_cursor=next_cursor)
        with self._cache_lock:
            self._cache[cache_key] = (now, page)
            if len(self._cache) > self.CACHE_MAX_ENTRIES:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
                del self._cache[oldest_key]
        return page

    def _catalog_row(self, raw: dict[str, Any], fmt: ModelFormat) -> CatalogModel | None:
        """One catalog row for a Hub repo, or None when it has no id or no file of *fmt*."""
        # Local import to avoid a cycle: query imports hf_client (this
        # module), and hf_client uses pipeline_to_task from query.
        from lilbee.catalog.query import pipeline_to_task

        if not raw.get("id"):
            return None
        item = ModelInfo(**raw)
        filename = _resolve_sibling_file(fmt, item.siblings or [])
        if not filename:
            return None
        card_desc = item.card_data.get("description", "") if item.card_data else ""
        gguf_meta = HfGgufMeta(**(item.gguf or {}))
        size_gb = estimate_size_gb(gguf_meta.total, filename)
        self.cache_arch(item.id, gguf_meta.architecture)
        return CatalogModel(
            hf_repo=item.id,
            gguf_filename=filename,
            size_gb=size_gb,
            min_ram_gb=estimate_min_ram_gb(size_gb),
            description=card_desc[:120] if card_desc else "",
            featured=False,
            downloads=item.downloads or 0,
            trending_score=item.trending_score or 0,
            task=pipeline_to_task(item.pipeline_tag or ""),
            architecture=gguf_meta.architecture,
            compat=self._compat(fmt, gguf_meta.architecture),
            params=gguf_meta.total,
            safety_stripped=is_safety_stripped(item.tags or []),
            format=fmt,
        )

    def _compat(self, fmt: ModelFormat, architecture: str) -> ModelCompat:
        """The engine verdict for a GGUF row; other formats stay unknown until an engine rules."""
        if fmt == ModelFormat.GGUF:
            return classify(architecture, self._engine)
        return ModelCompat.UNKNOWN

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
