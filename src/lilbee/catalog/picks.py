"""Model picks: the most popular models of each parameter tier, from HuggingFace."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import replace

from lilbee.app.services import get_services
from lilbee.catalog.hf_client import repo_has_mmproj
from lilbee.catalog.models import CatalogModel
from lilbee.catalog.refs import hf_repo_from_ref
from lilbee.catalog.types import CatalogSize, ModelCompat, ModelTask

log = logging.getLogger(__name__)

# The ranking behind huggingface.co's Trending tab. The API exposes no
# "downloads in the last 24 hours" field.
TRENDING_SORT = "trendingScore"

# Every role shows the same number of picks. Chat reaches it by taking an equal
# share from each parameter tier; the other roles have no tier spread.
_PICKS_PER_ROLE = 8
_CHAT_PICKS_PER_TIER = _PICKS_PER_ROLE // len(CatalogSize)

_UNTIERED_ROLES = (ModelTask.EMBEDDING, ModelTask.VISION, ModelTask.RERANK)

# Wide enough to populate every tier. A live trending fetch at 200 holds
# 34 / 59 / 64 / 42 candidates across the four tiers.
_CANDIDATE_WINDOW = 200

# Scanned past the quota because mistagged and unsupported entries get dropped;
# the scan short-circuits once the quota is met.
_UNTIERED_WINDOW = 100

# Minimum gap between resolution attempts after one comes back short, so a
# degraded network cannot turn every read into a fresh fan-out.
_RETRY_BACKOFF_S = 30.0


def _serves_role(model: CatalogModel, task: ModelTask) -> bool:
    """True when *model* serves *task*, ignoring its HF pipeline tag.

    Publishers mistag: a live fetch returned a Llama-2 chat model under
    ``text-classification`` and a 35B MoE chat model under ``feature-extraction``.
    Vision is settled by the mmproj sibling, which also catches VL repos no name
    pattern matches.
    """
    # circular: query -> picks via get_picks
    from lilbee.catalog.query import reclassify_by_name

    if model.compat is not ModelCompat.SUPPORTED:
        # A pick is a recommendation. Offering an architecture the bundled
        # engine cannot load turns one click into a failed download.
        return False
    if task == ModelTask.VISION:
        return repo_has_mmproj(model.hf_repo)
    return reclassify_by_name(model.hf_repo, ModelTask.CHAT) == task


def _fetch_trending(task: ModelTask, limit: int, needed: int | None = None) -> list[CatalogModel]:
    """Trending models serving *task*, most popular first. Empty on fetch failure.

    Stops at *needed* so the vision probe costs one request per candidate
    examined, not per candidate fetched.
    """
    # circular: query -> picks via get_picks
    from lilbee.catalog.query import task_to_pipeline

    pipeline_tag, library = task_to_pipeline(task)
    page = get_services().hf_client.fetch_models(
        pipeline_tag=pipeline_tag,
        sort=TRENDING_SORT,
        limit=limit,
        library=library,
    )
    qualified: list[CatalogModel] = []
    for model in page.models:
        if model.task != task or not _serves_role(model, task):
            continue
        qualified.append(model)
        if needed is not None and len(qualified) >= needed:
            break
    return qualified


def _chat_picks() -> list[CatalogModel]:
    """The most popular chat models of each parameter tier, in tier order."""
    # circular: query -> picks via get_picks
    from lilbee.catalog.query import size_bucket

    candidates = _fetch_trending(ModelTask.CHAT, _CANDIDATE_WINDOW)
    by_tier: dict[CatalogSize, list[CatalogModel]] = {}
    for model in candidates:
        tier = size_bucket(model.params)  # None when the repo publishes no count
        if tier is not None:
            by_tier.setdefault(tier, []).append(model)

    picks: list[CatalogModel] = []
    for tier in CatalogSize:
        # A short tier contributes what it has; topping up from another tier
        # would defeat the spread.
        picks.extend(by_tier.get(tier, [])[:_CHAT_PICKS_PER_TIER])
    return picks


def _resolve_picks() -> tuple[CatalogModel, ...]:
    """One full set of picks across every role, flagged for the picks section."""
    picks = list(_chat_picks())
    for task in _UNTIERED_ROLES:
        picks.extend(_fetch_trending(task, _UNTIERED_WINDOW, needed=_PICKS_PER_ROLE))
    # The flag is what puts a row in the picks section and keeps the browse
    # list from duplicating it.
    return tuple(replace(m, featured=True) for m in picks)


def _is_complete(picks: tuple[CatalogModel, ...]) -> bool:
    """True when every role has at least one pick."""
    roles = {m.task for m in picks}
    return ModelTask.CHAT in roles and all(task in roles for task in _UNTIERED_ROLES)


class ModelPicks:
    """Process-lifetime memo of the resolved picks.

    Not a TTL cache: one draw serves the session so rows do not reshuffle while
    the user is reading them. Owns its state and lock like
    :class:`~lilbee.modelhub.model_manager.discovery.KnownModelCache`.
    """

    def __init__(self) -> None:
        self._picks: tuple[CatalogModel, ...] | None = None
        self._complete = False
        self._next_attempt_at = 0.0
        self._lock = threading.Lock()

    def all(self) -> tuple[CatalogModel, ...]:
        """Every pick across every role.

        A set missing a role is served but not treated as final: each role is
        fetched independently, so one failure would otherwise leave that role
        empty for the process lifetime. Re-resolution is rate-limited by
        ``_RETRY_BACKOFF_S`` so a degraded network cannot turn every read into a
        fresh fan-out. Resolution runs off the lock, which would otherwise
        serialize every reader behind the slowest HTTP call.
        """
        with self._lock:
            if self._picks is not None and (
                self._complete or time.monotonic() < self._next_attempt_at
            ):
                return self._picks
            if self._picks is None and time.monotonic() < self._next_attempt_at:
                return ()

        try:
            resolved = _resolve_picks()
        except Exception:
            log.warning("Could not fetch model picks from HuggingFace", exc_info=True)
            resolved = ()

        with self._lock:
            if self._complete:  # another thread landed a full set while fetching
                return self._picks or ()
            if resolved:
                self._picks = resolved
                self._complete = _is_complete(resolved)
            if not self._complete:
                self._next_attempt_at = time.monotonic() + _RETRY_BACKOFF_S
            return self._picks or ()

    def seed(self, picks: tuple[CatalogModel, ...]) -> None:
        """Install *picks* directly, skipping resolution. For tests."""
        with self._lock:
            self._picks = picks
            self._complete = True
            self._next_attempt_at = 0.0

    def reset(self) -> None:
        """Drop the memo so the next read resolves again."""
        with self._lock:
            self._picks = None
            self._complete = False
            self._next_attempt_at = 0.0


_PICKS = ModelPicks()


def get_picks() -> tuple[CatalogModel, ...]:
    """Every pick across every role, resolved once per process."""
    return _PICKS.all()


def picks_for(task: ModelTask) -> tuple[CatalogModel, ...]:
    """Picks for a single role."""
    return tuple(m for m in get_picks() if m.task == task)


def find_pick(ref: str) -> CatalogModel | None:
    """The pick matching *ref* by repo id, or None. Case-insensitive."""
    if not ref:
        return None
    wanted = hf_repo_from_ref(ref).lower()
    return next((m for m in get_picks() if m.hf_repo.lower() == wanted), None)


def seed_picks(picks: tuple[CatalogModel, ...]) -> None:
    """Install *picks* directly, skipping resolution. For tests."""
    _PICKS.seed(picks)


def reset_picks() -> None:
    """Drop the memoized picks so the next read resolves again. For tests."""
    _PICKS.reset()
