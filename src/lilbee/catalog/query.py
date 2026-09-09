"""Catalog filtering, sorting, lookup, and ad-hoc HF resolution."""

import logging
from collections.abc import Callable
from typing import Any

from huggingface_hub.utils import HFValidationError, validate_repo_id

from lilbee.app.services import get_services
from lilbee.catalog.models import CatalogModel, CatalogResult, HfPage, PageWindow, page_window
from lilbee.catalog.picks import get_picks
from lilbee.catalog.refs import (
    GGUF_GLOB,
    GGUF_SUFFIX,
    NATIVE_GGUF_REF_MIN_SLASHES,
    hf_repo_from_ref,
)
from lilbee.catalog.types import CatalogSize, CatalogSort, ModelTask

log = logging.getLogger(__name__)


def _search_blob(m: CatalogModel) -> str:
    """Lowercased join of searchable fields on a catalog row.

    Null char joins the fields so a search term never straddles them.
    """
    return f"{m.display_name}\0{m.hf_repo}\0{m.description}".lower()


# Upper bound in billions of parameters for each bucket below HUGE, which takes
# everything above the last one. Keyed on parameters rather than on-disk bytes so
# a model keeps its bucket whichever quant is picked, and so buckets match how
# model sizes are actually talked about ("a 70B"). HUGE starts where consumer
# hardware stops.
_PARAM_TIER_CEILINGS: tuple[tuple[float, CatalogSize], ...] = (
    (4.0, CatalogSize.SMALL),
    (20.0, CatalogSize.MEDIUM),
    (70.0, CatalogSize.LARGE),
)

_PARAMS_PER_BILLION = 1e9


def size_bucket(params: int) -> CatalogSize | None:
    """Bucket a parameter count. None when the repo publishes no count."""
    if params <= 0:
        return None
    billions = params / _PARAMS_PER_BILLION
    for ceiling, bucket in _PARAM_TIER_CEILINGS:
        if billions < ceiling:
            return bucket
    return CatalogSize.HUGE


def get_catalog(
    task: ModelTask | None = None,
    *,
    search: str = "",
    size: CatalogSize | None = None,
    installed: bool | None = None,
    featured: bool | None = None,
    fit_filter: Callable[[CatalogModel], bool] | None = None,
    sort: CatalogSort = CatalogSort.FEATURED,
    limit: int = 20,
    offset: int = 0,
    model_manager: Any = None,
) -> CatalogResult:
    """Get paginated, filtered catalog of models.

    The picks lead the listing and the HuggingFace rows follow them, so one
    page window covers both: the HuggingFace request is shifted by the picks
    that precede it, and a window that ends inside the picks makes no request.
    """
    picks = get_picks()
    installed_filter = _installed_filter(installed, model_manager)

    def keep(models: list[CatalogModel]) -> list[CatalogModel]:
        return _filter_models(
            models,
            task=task,
            search=search,
            size=size,
            installed_filter=installed_filter,
            fit_filter=fit_filter,
            featured=featured,
        )

    leading = _sort_models(keep(list(picks)), sort)
    window = page_window(len(leading), offset, limit)
    page = leading[offset : offset + limit]
    hf_models: list[CatalogModel] = []
    if featured:
        has_more = offset + limit < len(leading)
    elif window.rest_limit == 0:
        has_more = True  # the HuggingFace rows start on the next page
    else:
        hf_page = _fetch_hf_page(task, search, window)
        # Deduplicate: skip HF models already shown as a pick
        pick_repos = {m.hf_repo for m in picks}
        hf_models = keep([m for m in hf_page.models if m.hf_repo not in pick_repos])
        page.extend(_sort_models(hf_models, sort))
        has_more = hf_page.has_more

    return CatalogResult(
        total=len(leading) + len(hf_models),
        limit=limit,
        offset=offset,
        models=page,
        has_more=has_more,
    )


def _fetch_hf_page(task: ModelTask | None, search: str, window: PageWindow) -> HfPage:
    """The HuggingFace rows that fill the rest of *window*."""
    hf_task, hf_library = task_to_pipeline(task)
    return get_services().hf_client.fetch_models(
        pipeline_tag=hf_task,
        limit=window.rest_limit,
        offset=window.rest_offset,
        library=hf_library,
        search=search,
    )


def _installed_filter(
    installed: bool | None, model_manager: Any
) -> Callable[[CatalogModel], bool] | None:
    """Row predicate for the installed filter, or None when it is off."""
    if installed is None or model_manager is None:
        return None
    # A repo is installed if any of its quants has a manifest.
    installed_repos = {hf_repo_from_ref(ref) for ref in _get_installed_models(model_manager)}
    return lambda m: (m.hf_repo in installed_repos) == installed


def _filter_models(
    models: list[CatalogModel],
    *,
    task: ModelTask | None,
    search: str,
    size: CatalogSize | None,
    installed_filter: Callable[[CatalogModel], bool] | None,
    fit_filter: Callable[[CatalogModel], bool] | None,
    featured: bool | None,
) -> list[CatalogModel]:
    """The rows of *models* that pass every requested filter."""
    if task:
        models = [m for m in models if m.task == task]
    if search:
        search_lower = search.lower()
        models = [m for m in models if search_lower in _search_blob(m)]
    if size is not None:
        models = [m for m in models if size_bucket(m.params) == size]
    if installed_filter is not None:
        models = [m for m in models if installed_filter(m)]
    if fit_filter is not None:
        models = [m for m in models if fit_filter(m)]
    if featured is not None:
        models = [m for m in models if m.featured == featured]
    return models


def task_to_pipeline(task: ModelTask | None) -> tuple[str, str | None]:
    """Map task name to HuggingFace pipeline tag and library filter."""
    mapping: dict[ModelTask, tuple[str, str | None]] = {
        ModelTask.CHAT: ("text-generation", None),
        ModelTask.EMBEDDING: ("feature-extraction", "sentence-transformers"),
        ModelTask.VISION: ("image-text-to-text", None),
        ModelTask.RERANK: ("text-classification", None),
    }
    return mapping.get(task or ModelTask.CHAT, ("text-generation", None))


_PIPELINE_TO_TASK: dict[str, ModelTask] = {
    "text-generation": ModelTask.CHAT,
    "feature-extraction": ModelTask.EMBEDDING,
    "sentence-similarity": ModelTask.EMBEDDING,
    "image-text-to-text": ModelTask.VISION,
    "image-to-text": ModelTask.VISION,
    "text-classification": ModelTask.RERANK,
    "text-ranking": ModelTask.RERANK,
}


def pipeline_to_task(pipeline_tag: str) -> ModelTask:
    """Map HuggingFace pipeline tag to internal task name."""
    return _PIPELINE_TO_TASK.get(pipeline_tag, ModelTask.CHAT)


def _get_installed_models(model_manager: Any) -> set[str]:
    """Get set of installed model names from model_manager.

    Treats a manager failure as "nothing installed" so the browse list still
    renders, but logs it: silently swallowing would hide a broken registry that
    makes every model look uninstalled.
    """
    try:
        return set(model_manager.list_installed())
    except Exception:
        log.warning("Could not read installed models; treating as none installed", exc_info=True)
        return set()


_SORT_KEYS: dict[CatalogSort, tuple] = {
    CatalogSort.DOWNLOADS: (lambda m: m.downloads, True),
    CatalogSort.NAME: (lambda m: m.display_name.lower(), False),
    CatalogSort.SIZE_ASC: (lambda m: m.size_gb, False),
    CatalogSort.SIZE_DESC: (lambda m: m.size_gb, True),
    CatalogSort.FEATURED: (lambda m: (not m.featured, -m.downloads), False),
}


def _sort_models(models: list[CatalogModel], sort: CatalogSort) -> list[CatalogModel]:
    """Sort models according to the specified sort order."""
    key_fn, reverse = _SORT_KEYS[sort]
    return sorted(models, key=key_fn, reverse=reverse)


def is_rerank_ref(model_ref: str) -> bool:
    """Return True iff *model_ref* names a reranker."""
    if not model_ref:
        return False
    return reclassify_by_name(model_ref, ModelTask.CHAT) == ModelTask.RERANK


def _is_hf_repo_id(value: str) -> bool:
    """True if *value* is a well-formed ``owner/name`` HuggingFace repo id."""
    if "/" not in value:
        return False
    try:
        validate_repo_id(value)
    except HFValidationError:
        return False
    return True


def build_adhoc_entry(
    hf_repo: str,
    *,
    gguf_filename: str = GGUF_GLOB,
    task: ModelTask = ModelTask.CHAT,
) -> CatalogModel:
    """Minimal CatalogModel for a HuggingFace GGUF repo.

    *gguf_filename* defaults to the ``*.gguf`` glob (bare-repo pull picks the best
    quant); pass a concrete filename, which may include a repo subdirectory, to
    pin the exact file the user named.
    """
    return CatalogModel(
        hf_repo=hf_repo,
        gguf_filename=gguf_filename,
        size_gb=0.0,
        min_ram_gb=2.0,
        description="",
        featured=False,
        downloads=0,
        task=task,
    )


def resolve_pull_target(model: str) -> CatalogModel | None:
    """Resolve *model* to a pullable entry, HF-first.

    A ref naming a concrete ``.gguf`` file (flat or in a repo subdir) is honored
    exactly. A bare ``owner/name`` repo pulls through the ``*.gguf`` glob, which
    picks the best quant. Returns None when *model* is not a usable repo id.
    """
    # circular: modelhub.registry imports catalog.query at top
    from lilbee.modelhub.registry import parse_hf_ref

    if model.endswith(GGUF_SUFFIX) and model.count("/") >= NATIVE_GGUF_REF_MIN_SLASHES:
        try:
            hf_repo, gguf_filename = parse_hf_ref(model)
        except ValueError:
            return None
        task = ModelTask(reclassify_by_name(model, ModelTask.CHAT))
        return build_adhoc_entry(hf_repo, gguf_filename=gguf_filename, task=task)
    if not _is_hf_repo_id(model):
        return None
    return build_adhoc_entry(model, task=ModelTask(reclassify_by_name(model, ModelTask.CHAT)))


# Embedding detection by name, for servers (LM Studio) that report ids but no
# family. Trailing hyphens keep chat models that merely contain the letters out.
EMBEDDING_NAME_PATTERNS: frozenset[str] = frozenset({"embed", "bge-", "e5-", "gte-"})
VISION_NAME_PATTERNS: frozenset[str] = frozenset(
    {"llava", "vision", "moondream", "ocr", "minicpm-v"}
)
# Reranker detection runs before embedding detection so ``bge-reranker-*`` is
# not misclassified as EMBEDDING.
RERANKER_NAME_PATTERNS: frozenset[str] = frozenset({"reranker", "rerank", "cross-encoder"})


def reclassify_by_name(ref: str, declared_task: str) -> str:
    """Override declared_task to RERANK / VISION / EMBEDDING when ref names a known role.

    Defends against manifests that stored ``task="chat"`` for models whose ref
    obviously identifies them as rerankers (e.g. ``bge-reranker-*``), vision
    loaders, or embedders. Embedders on a chat decoder arch (e.g.
    ``Qwen3-Embedding-*``, a qwen3 backbone + pooling head) classify as chat by
    architecture, so the name is the only signal short of probing the GGUF
    pooling type.

    Check order (rerank, embedding, vision) matches
    :func:`lilbee.modelhub.model_manager.discovery._classify_remote_task` so the
    manifest and remote-discovery paths never disagree. Reranker is checked first
    so ``bge-reranker`` (which also matches the ``bge-`` embedder pattern) stays a
    reranker; embedding is checked before vision so an image embedder like
    ``nomic-embed-vision`` (matching both ``embed`` and ``vision``) stays an
    embedder.
    """
    name_lower = ref.lower()
    if any(rp in name_lower for rp in RERANKER_NAME_PATTERNS):
        return ModelTask.RERANK
    if any(ep in name_lower for ep in EMBEDDING_NAME_PATTERNS):
        return ModelTask.EMBEDDING
    if any(vp in name_lower for vp in VISION_NAME_PATTERNS):
        return ModelTask.VISION
    return declared_task
