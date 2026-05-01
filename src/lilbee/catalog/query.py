"""Catalog filtering, sorting, lookup, and ad-hoc HF resolution."""

import functools
from typing import Any, NamedTuple

from huggingface_hub.utils import HFValidationError, validate_repo_id

from lilbee.catalog.featured import FEATURED_ALL
from lilbee.catalog.models import CatalogModel, CatalogResult
from lilbee.core.services import get_services
from lilbee.modelhub.models import ModelTask


def _search_blob(m: CatalogModel) -> str:
    """Lowercased join of searchable fields on a catalog row.

    Null char joins the fields so a search term never straddles them.
    """
    return f"{m.display_name}\0{m.hf_repo}\0{m.description}".lower()


_SIZE_RANGES: dict[str, tuple[float, float]] = {
    "small": (0.0, 3.0),
    "medium": (3.0, 10.0),
    "large": (10.0, float("inf")),
}

# A native GGUF ref of the form ``<owner>/<repo>/<file>.gguf`` has at least
# two ``/`` separators; one-slash refs are bare repo IDs.
_NATIVE_GGUF_REF_MIN_SLASHES = 2


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
        hf_page = get_services().hf_client.fetch_models(
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

    # A repo is "installed" if any of its quants has a manifest.
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


def pipeline_to_task(pipeline_tag: str) -> str:
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
    """Find a featured model by hf_repo or by ``hf_repo/filename`` ref.

    Tries the query as-is, then strips a trailing ``/<filename>.gguf``,
    then strips a leading non-HF provider prefix (``ollama/``, etc.).
    Case-insensitive; returns ``None`` on miss.
    """
    if not query:
        return None
    idx = _build_catalog_index()
    q = query.lower()
    candidates = [q]
    # Strip the filename for ``<repo>/<filename>.gguf`` queries so the
    # bare-repo index catches featured entries whose gguf_filename is a
    # glob (most are).
    if q.endswith(".gguf") and q.count("/") >= _NATIVE_GGUF_REF_MIN_SLASHES:
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
