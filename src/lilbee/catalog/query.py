"""Catalog filtering, sorting, lookup, and ad-hoc HF resolution."""

import logging
from typing import Any

from huggingface_hub.utils import HFValidationError, validate_repo_id

from lilbee.app.services import get_services
from lilbee.catalog.models import CatalogModel, CatalogResult
from lilbee.catalog.picks import get_picks
from lilbee.catalog.refs import GGUF_GLOB, hf_repo_from_ref
from lilbee.catalog.types import CatalogSize, CatalogSort, ModelTask

log = logging.getLogger(__name__)


def _search_blob(m: CatalogModel) -> str:
    """Lowercased join of searchable fields on a catalog row.

    Null char joins the fields so a search term never straddles them.
    """
    return f"{m.display_name}\0{m.hf_repo}\0{m.description}".lower()


# Parameter-count bounds per bucket, in billions. Keyed on parameters rather
# than on-disk bytes so a model keeps its bucket whichever quant is picked, and
# so buckets match how model sizes are actually talked about ("a 70B"). HUGE
# starts where consumer hardware stops.
_PARAM_RANGES: dict[CatalogSize, tuple[float, float]] = {
    CatalogSize.SMALL: (0.0, 4.0),
    CatalogSize.MEDIUM: (4.0, 20.0),
    CatalogSize.LARGE: (20.0, 70.0),
    CatalogSize.HUGE: (70.0, float("inf")),
}

_PARAMS_PER_BILLION = 1e9


def size_bucket(params: int) -> CatalogSize | None:
    """Bucket a parameter count. None when the repo publishes no count."""
    if params <= 0:
        return None
    billions = params / _PARAMS_PER_BILLION
    for bucket, (lo, hi) in _PARAM_RANGES.items():
        if lo <= billions < hi:
            return bucket
    return None


# A native GGUF ref of the form ``<owner>/<repo>/<file>.gguf`` has at least
# two ``/`` separators; one-slash refs are bare repo IDs.
_NATIVE_GGUF_REF_MIN_SLASHES = 2


def get_catalog(
    task: ModelTask | None = None,
    *,
    search: str = "",
    size: CatalogSize | None = None,
    installed: bool | None = None,
    featured: bool | None = None,
    sort: CatalogSort = CatalogSort.FEATURED,
    limit: int = 20,
    offset: int = 0,
    model_manager: Any = None,
) -> CatalogResult:
    """Get paginated, filtered catalog of models."""
    picks = get_picks()
    # Picks only on the first page
    all_models = list(picks) if offset == 0 else []
    hf_has_more = False

    # Optionally fetch from HF API
    if not featured:
        hf_task, hf_library = task_to_pipeline(task)
        hf_page = get_services().hf_client.fetch_models(
            pipeline_tag=hf_task,
            limit=limit,
            offset=offset,
            library=hf_library,
            search=search,
        )
        hf_has_more = hf_page.has_more
        # Deduplicate: skip HF models already shown as a pick
        pick_repos = {m.hf_repo for m in picks}
        hf_models = [m for m in hf_page.models if m.hf_repo not in pick_repos]
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
    if size is not None:
        all_models = [m for m in all_models if size_bucket(m.params) == size]

    # A repo is "installed" if any of its quants has a manifest.
    if installed is not None and model_manager is not None:
        installed_repos = {hf_repo_from_ref(ref) for ref in _get_installed_models(model_manager)}
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

    if model.endswith(".gguf") and model.count("/") >= _NATIVE_GGUF_REF_MIN_SLASHES:
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
