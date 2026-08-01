"""Popular-model picks, sourced live from HuggingFace's trending ranking.

Replaces a hand-curated list. Nothing about which models exist is written down
here: chat picks are the most popular models of each parameter-count tier, and
the other roles take the head of their own trending list.
"""

from __future__ import annotations

import logging
import threading

from lilbee.catalog.models import CatalogModel
from lilbee.catalog.types import CatalogSize, ModelTask

log = logging.getLogger(__name__)

# HF's trending ranking, the signal behind the Trending tab on huggingface.co.
# The API exposes no raw "downloads in the last 24 hours" field, so this is the
# available expression of "popular right now".
TRENDING_SORT = "trendingScore"

# Chat picks per parameter tier. Spanning tiers is the point: the trending head
# is dominated by 20B+ models, and a flat take would leave an 8 GB machine with
# nothing it can run.
_CHAT_PICKS_PER_TIER = 2

# Picks for roles that aren't tiered. Embedders and rerankers are uniformly
# small, and one vision pick has no spread to distribute.
_UNTIERED_PICKS: dict[ModelTask, int] = {
    ModelTask.EMBEDDING: 3,
    ModelTask.VISION: 1,
    ModelTask.RERANK: 1,
}

# Wide enough that every parameter tier is populated. Measured against a live
# trending fetch: at 200 the tiers hold 34 / 59 / 64 / 42 candidates.
_CANDIDATE_WINDOW = 200

# Mistagged and misnamed entries get dropped, so scan well past the quota. The
# scan short-circuits once the quota is met, so a wide window costs nothing when
# the head of the list already qualifies.
_UNTIERED_WINDOW = 50


def _tier_of(params: int) -> CatalogSize | None:
    """Parameter-count tier for *params*, or None when the repo publishes no count."""
    if params <= 0:
        return None
    from lilbee.catalog.query import size_bucket

    return size_bucket(params)


def _serves_role(model: CatalogModel, task: ModelTask) -> bool:
    """True when *model* really serves *task*, not just per its HF pipeline tag.

    Publishers mistag constantly: a live trending fetch put a Llama-2 chat model
    under ``text-classification`` (so it surfaced as a reranker) and a 35B MoE
    chat model under ``feature-extraction``. Picking one of those wires a broken
    model into the role, so the tag alone cannot be trusted.

    Vision is settled by probing for an mmproj sibling, which is definitive and
    catches VL repos no name pattern matches (Qwen-VL, InternVL, SmolVLM). The
    other roles go by name, which is what ``reclassify_by_name`` already does
    for manifests and remote discovery.
    """
    from lilbee.catalog.query import reclassify_by_name

    if task == ModelTask.VISION:
        from lilbee.catalog.hf_client import repo_has_mmproj

        return repo_has_mmproj(model.hf_repo)
    return reclassify_by_name(model.hf_repo, ModelTask.CHAT) == task


def _fetch_trending(task: ModelTask, limit: int, needed: int | None = None) -> list[CatalogModel]:
    """Trending models that genuinely serve *task*, most popular first.

    Stops once *needed* models qualify, so the vision probe issues one request
    per candidate examined rather than one per candidate fetched. Empty on any
    fetch failure.
    """
    from lilbee.app.services import get_services
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
    """Most popular chat models of each parameter tier, in tier order."""
    candidates = _fetch_trending(ModelTask.CHAT, _CANDIDATE_WINDOW)
    by_tier: dict[CatalogSize, list[CatalogModel]] = {}
    for model in candidates:
        tier = _tier_of(model.params)
        if tier is not None:
            by_tier.setdefault(tier, []).append(model)

    picks: list[CatalogModel] = []
    for tier in CatalogSize:
        # Candidates keep their trending order, so the head of each tier is that
        # tier's most popular. A tier with fewer than the quota contributes what
        # it has; the total is not topped up from other tiers, which would
        # defeat the spread.
        picks.extend(by_tier.get(tier, [])[:_CHAT_PICKS_PER_TIER])
    return picks


def _resolve_picks() -> tuple[CatalogModel, ...]:
    """Fetch one full set of picks across every role, flagged for the picks section."""
    from dataclasses import replace

    picks = list(_chat_picks())
    for task, count in _UNTIERED_PICKS.items():
        picks.extend(_fetch_trending(task, _UNTIERED_WINDOW, needed=count))
    # fetch_models builds plain browse rows; the flag is what puts a row in the
    # picks section, stars it, and keeps the browse list from duplicating it.
    return tuple(replace(m, featured=True) for m in picks)


_picks_lock = threading.Lock()
_picks: tuple[CatalogModel, ...] | None = None


def get_picks() -> tuple[CatalogModel, ...]:
    """Every pick across every role, resolved once per process.

    Held for the life of the process rather than behind a TTL: rows must not
    reshuffle while the user types in the catalog search box or moves between
    tabs. A new set appears on relaunch.

    An empty result is not memoized, so an offline launch that later regains
    network still fills in.
    """
    global _picks
    with _picks_lock:
        if _picks:
            return _picks
        try:
            resolved = _resolve_picks()
        except Exception:
            log.warning("Could not fetch model picks from HuggingFace", exc_info=True)
            return ()
        if resolved:
            _picks = resolved
        return resolved


def picks_for(task: ModelTask) -> tuple[CatalogModel, ...]:
    """Picks for a single role."""
    return tuple(m for m in get_picks() if m.task == task)


def find_pick(ref: str) -> CatalogModel | None:
    """The pick matching *ref* by repo id, or None.

    Case-insensitive, and tolerates a trailing ``/<file>.gguf`` so a full
    native ref resolves to the repo it names.
    """
    if not ref:
        return None
    from lilbee.catalog.refs import hf_repo_from_ref

    wanted = hf_repo_from_ref(ref).lower()
    return next((m for m in get_picks() if m.hf_repo.lower() == wanted), None)


def reset_picks() -> None:
    """Drop the memoized picks. For tests and ``reset_services()``."""
    global _picks
    with _picks_lock:
        _picks = None


# Vision models need both the main GGUF and an mmproj (CLIP projection) file.
# Resolved by glob rather than a per-repo table: every mainstream VL repo names
# its projector this way, and a table would be one more thing to maintain.
DEFAULT_MMPROJ_PATTERN = "*mmproj*.gguf"
