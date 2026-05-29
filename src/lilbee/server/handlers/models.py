"""Model catalog, role assignment, install/delete, and external listing handlers."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from lilbee.app.services import get_services
from lilbee.app.settings import apply_settings_update
from lilbee.catalog import (
    FEATURED_CHAT,
    FEATURED_EMBEDDING,
    FEATURED_RERANK,
    FEATURED_VISION,
    ModelFamily,
    enrich_catalog,
    find_catalog_entry,
    get_catalog,
    get_families,
)
from lilbee.catalog.refs import hf_repo_from_ref
from lilbee.catalog.types import CatalogSize, CatalogSort, ModelSource, ModelTask
from lilbee.core.config import cfg
from lilbee.modelhub.role_validator import _MODEL_FIELD_TO_TASK, validate_model_task_assignment
from lilbee.providers.model_ref import parse_model_ref
from lilbee.runtime.hardware import (
    FitLevel,
    SizeVariantInfo,
    available_memory_for_fit,
    compute_fit,
    family_size_variants,
)
from lilbee.runtime.progress import SseEvent
from lilbee.server.handlers.sse import SseStream, sse_error, sse_event
from lilbee.server.models import (
    CatalogEntryResponse,
    ExternalModelsResponse,
    InstalledModelEntry,
    ModelsCatalogResponse,
    ModelsDeleteResponse,
    ModelsInstalledResponse,
    ModelsShowResponse,
    SetModelResponse,
)

if TYPE_CHECKING:
    from lilbee.catalog import CatalogModel
    from lilbee.catalog.formatting import EnrichedModel

log = logging.getLogger(__name__)


class ModelCatalogEntry(BaseModel):
    """A single model in the catalog."""

    name: str
    size_gb: float
    min_ram_gb: float
    description: str
    installed: bool


class ModelCatalogSection(BaseModel):
    """A single-role catalog section with active model and installed list."""

    active: str
    catalog: list[ModelCatalogEntry]
    installed: list[str]


class ModelsResponse(BaseModel):
    """Response for GET /api/models: one catalog section per role."""

    chat: ModelCatalogSection
    embedding: ModelCatalogSection
    vision: ModelCatalogSection
    reranker: ModelCatalogSection


# ``ModelTask.RERANK.value`` is ``"rerank"`` but the route is ``/api/models/reranker``,
# so this mapping is needed to build correct redirect URLs in 422 responses.
TASK_ENDPOINT_PATH: dict[ModelTask, str] = {
    ModelTask.CHAT: "chat",
    ModelTask.EMBEDDING: "embedding",
    ModelTask.VISION: "vision",
    ModelTask.RERANK: "reranker",
}


def format_task_mismatch(ref: str, entry_task: ModelTask, expected_task: ModelTask) -> str:
    """Build the 422 body when a role slot is assigned a model of the wrong task."""
    endpoint = TASK_ENDPOINT_PATH[entry_task]
    return (
        f"Model '{ref}' is a {entry_task} model, not {expected_task}. "
        f"Set it via PUT /api/models/{endpoint} instead."
    )


def _catalog_section(
    featured: tuple[CatalogModel, ...],
    active: str,
    installed: set[str],
) -> ModelCatalogSection:
    """Build a ModelCatalogSection from a featured-catalog tuple.

    A featured row is "installed" when at least one quant of its
    ``hf_repo`` has a manifest. Installed refs are full
    ``hf_repo/filename`` strings, so the membership test compares the
    leading ``hf_repo`` segment. Bare ``hf_repo`` entries are accepted
    too (e.g. older clients that report just the repo).
    """
    installed_repos = {hf_repo_from_ref(ref) for ref in installed}
    return ModelCatalogSection(
        active=active,
        catalog=[
            ModelCatalogEntry(
                name=m.display_name,
                size_gb=m.size_gb,
                min_ram_gb=m.min_ram_gb,
                description=m.description,
                installed=m.hf_repo in installed_repos,
            )
            for m in featured
        ],
        installed=sorted(installed),
    )


async def list_models() -> ModelsResponse:
    """Return per-role catalogs (chat, embedding, vision, reranker) with active selections.

    Uses the unfiltered installed set so a single ref lights up in every
    catalog section it legitimately matches.
    """
    installed = set(get_services().model_manager.list_installed())

    return ModelsResponse(
        chat=_catalog_section(FEATURED_CHAT, cfg.chat_model, installed),
        embedding=_catalog_section(FEATURED_EMBEDDING, cfg.embedding_model, installed),
        vision=_catalog_section(FEATURED_VISION, cfg.vision_model, installed),
        reranker=_catalog_section(FEATURED_RERANK, cfg.reranker_model, installed),
    )


async def _set_model(
    field: Literal["chat_model", "embedding_model", "vision_model", "reranker_model"],
    model: str,
) -> SetModelResponse:
    """Persist a model field through the shared write boundary."""
    apply_settings_update({field: model})
    return SetModelResponse(model=model)


def _resolve_via_catalog(model: str, available: set[str]) -> str | None:
    """Resolve a bare ``hf_repo`` to whichever quant of it is in *available*."""
    entry = find_catalog_entry(model)
    if entry is None:
        return None
    return next((ref for ref in available if ref.startswith(f"{entry.hf_repo}/")), None)


def _resolve_via_parse(model: str, available: set[str]) -> str | None:
    """Resolve a provider-prefixed ref to its bare provider name in *available*."""
    try:
        parsed = parse_model_ref(model)
    except ValueError:
        return None
    return parsed.name if parsed.name in available else None


def _require_model_available(model: str) -> str:
    """Return the installed-and-routable form of *model*, or raise."""
    not_available = ValueError(
        f"Model '{model}' is not available. Pull it first or check the name."
    )
    if not model:
        raise not_available
    available = set(get_services().provider.list_models())
    if model in available:
        return model
    hit = _resolve_via_catalog(model, available) or _resolve_via_parse(model, available)
    if hit is None:
        raise not_available
    return hit


def _build_task_to_field() -> dict[ModelTask, str]:
    """Invert config's ``_MODEL_FIELD_TO_TASK`` so the two maps stay in sync."""
    return {ModelTask(task): field for field, task in _MODEL_FIELD_TO_TASK.items()}


_TASK_TO_FIELD: dict[ModelTask, str] = _build_task_to_field()


def _require_model_for_task(model: str, expected: ModelTask, *, allow_empty: bool = False) -> str:
    """Validate *model* is installed locally AND passes the catalog task check.

    Empty string unsets the role when *allow_empty* is True. Catalog +
    task validation delegates to ``validate_model_task_assignment`` so
    the handler and config paths share a single implementation.
    """
    if allow_empty and not model.strip():
        return ""
    normalized = _require_model_available(model)
    return validate_model_task_assignment(_TASK_TO_FIELD[expected], normalized, allow_bypass=False)


async def set_chat_model(model: str) -> SetModelResponse:
    """Switch active chat model. Validates installation and catalog task."""
    normalized = _require_model_for_task(model, ModelTask.CHAT)
    return await _set_model("chat_model", normalized)


async def set_embedding_model(model: str) -> SetModelResponse:
    """Switch embedding model. Validates installation and catalog task.

    Returns ``reindex_required=True`` when the new model differs from the
    embedding model that built the persisted vector store. The caller is
    expected to trigger a rebuild (``lilbee rebuild`` or ``POST /api/sync``
    with ``force_rebuild=true``). Search and ingest will refuse to operate
    until that happens. The settings boundary pins legacy store meta to
    the OLD ref before the write and computes ``reindex_required`` after.
    """
    normalized = _require_model_for_task(model, ModelTask.EMBEDDING)
    result = apply_settings_update({"embedding_model": normalized})
    return SetModelResponse(model=normalized, reindex_required=result.reindex_required)


async def set_vision_model(model: str) -> SetModelResponse:
    """Switch vision OCR model. Empty string unsets it (vision OCR disabled)."""
    normalized = _require_model_for_task(model, ModelTask.VISION, allow_empty=True)
    return await _set_model("vision_model", normalized)


async def set_reranker_model(model: str) -> SetModelResponse:
    """Switch reranker model. Empty string unsets it (reranking disabled)."""
    normalized = _require_model_for_task(model, ModelTask.RERANK, allow_empty=True)
    return await _set_model("reranker_model", normalized)


async def models_show(model: str) -> ModelsShowResponse:
    """Return model metadata/parameters. Returns empty model if unavailable."""
    provider = get_services().provider
    result = provider.show_model(model)
    return ModelsShowResponse(**(result or {}))


def _parse_source(source: str) -> ModelSource:
    """Convert a source string to ModelSource enum."""
    return ModelSource(source)


_BYTES_PER_GB = 1024**3


def _row_fit(enriched: EnrichedModel, available_bytes: int | None) -> FitLevel | None:
    """Fit level for *enriched*, or None when host memory or row size can't be measured."""
    if available_bytes is None:
        return None
    if enriched.source != ModelSource.NATIVE.value:
        return None
    if enriched.size_gb <= 0:
        return None
    return compute_fit(int(enriched.size_gb * _BYTES_PER_GB), available_bytes).level


def _families_by_repo() -> dict[str, ModelFamily]:
    """Index featured ModelFamilies by every variant's ``hf_repo`` for size-variant lookup."""
    index: dict[str, ModelFamily] = {}
    for family in get_families():
        for variant in family.variants:
            index[variant.hf_repo] = family
    return index


def _row_size_variants(
    enriched: EnrichedModel, families_by_repo: dict[str, ModelFamily]
) -> list[SizeVariantInfo]:
    """Size-variant strip for *enriched*; empty when the row isn't part of a family."""
    family = families_by_repo.get(enriched.hf_repo)
    if family is None:
        return []
    return family_size_variants(family)


def _build_catalog_entry(
    enriched: EnrichedModel,
    *,
    available_bytes: int | None,
    families_by_repo: dict[str, ModelFamily],
) -> CatalogEntryResponse:
    """Translate one enriched catalog model into its HTTP response row."""
    return CatalogEntryResponse(
        hf_repo=enriched.hf_repo,
        gguf_filename=enriched.gguf_filename,
        task=enriched.task,
        display_name=enriched.display_name,
        param_count=enriched.param_count,
        size_gb=enriched.size_gb,
        min_ram_gb=enriched.min_ram_gb,
        description=enriched.description,
        quality_tier=enriched.quality_tier,
        featured=enriched.featured,
        downloads=enriched.downloads,
        installed=enriched.installed,
        source=enriched.source,
        fit=_row_fit(enriched, available_bytes),
        size_variants=_row_size_variants(enriched, families_by_repo),
        architecture=enriched.architecture,
        compat=enriched.compat,
    )


async def models_catalog(
    task: str | None = None,
    search: str = "",
    size: str | None = None,
    installed: bool | None = None,
    featured: bool | None = None,
    sort: str = "featured",
    limit: int = 20,
    offset: int = 0,
) -> ModelsCatalogResponse:
    """Return paginated model catalog with installed status."""
    # Validate every closed-set param at the HTTP boundary instead of
    # letting unknown values silently short-circuit the filter inside.
    parsed_task = ModelTask(task) if task else None
    parsed_size = CatalogSize(size) if size else None
    parsed_sort = CatalogSort(sort)
    result = get_catalog(
        task=parsed_task,
        search=search,
        size=parsed_size,
        installed=installed,
        featured=featured,
        sort=parsed_sort,
        limit=limit,
        offset=offset,
    )

    registry = get_services().registry
    installed_refs = {m.ref for m in registry.list_installed()}
    enriched = enrich_catalog(result, installed_refs)

    available_bytes = available_memory_for_fit()
    families_by_repo = _families_by_repo()

    return ModelsCatalogResponse(
        total=result.total,
        limit=result.limit,
        offset=result.offset,
        has_more=result.has_more,
        models=[
            _build_catalog_entry(
                e, available_bytes=available_bytes, families_by_repo=families_by_repo
            )
            for e in enriched
        ],
    )


async def models_installed() -> ModelsInstalledResponse:
    """Return list of installed models with their source."""
    manager = get_services().model_manager
    names = manager.list_installed()
    models = []
    for name in names:
        src = manager.get_source(name)
        models.append(InstalledModelEntry(name=name, source=src or ModelSource.REMOTE))
    return ModelsInstalledResponse(models=models)


async def models_pull(
    model: str, *, source: str = "native", allow_unsupported: bool = False
) -> AsyncGenerator[str, None]:
    """Yield SSE progress events while pulling a model in real time.
    Sets a cancel event on client disconnect so the pull stops.

    Pre-checks architecture compatibility BEFORE opening the SSE stream so
    refusals surface as an HTTP 409 from the route, not an in-stream error.
    """
    from litestar.exceptions import HTTPException

    from lilbee.catalog.compat import SUPPORTED_ARCHS, UnsupportedArchError

    manager = get_services().model_manager
    src = _parse_source(source)

    if src is ModelSource.NATIVE and not allow_unsupported:
        try:
            await asyncio.to_thread(manager._enforce_arch_compat, model)
        except UnsupportedArchError as exc:
            raise HTTPException(
                status_code=409,
                detail="unsupported_arch",
                extra={
                    "code": "unsupported_arch",
                    "arch": exc.architecture,
                    "ref": exc.ref,
                    "supported_examples": sorted(SUPPORTED_ARCHS)[:5],
                    "total_supported": len(SUPPORTED_ARCHS),
                },
            ) from exc

    sse = SseStream()

    def _pull_blocking() -> None:
        def _on_bytes(downloaded: int, total: int) -> None:
            if sse.cancel.is_set():
                return
            payload = sse_event(SseEvent.PROGRESS, {"current": downloaded, "total": total})
            sse.loop.call_soon_threadsafe(sse.queue.put_nowait, payload)

        try:
            manager.pull(
                model,
                src,
                on_bytes=_on_bytes,
                allow_unsupported=allow_unsupported,
            )
        except Exception as exc:
            sse.loop.call_soon_threadsafe(sse.queue.put_nowait, sse_error(str(exc)))
        finally:
            sse.loop.call_soon_threadsafe(sse.queue.put_nowait, None)

    task = asyncio.ensure_future(asyncio.to_thread(_pull_blocking))
    async for event in sse.drain(task, "Model pull stream"):
        yield event


async def models_delete(model: str, *, source: str = "native") -> ModelsDeleteResponse:
    """Delete a model. Returns deletion status, model name, and freed space."""
    manager = get_services().model_manager
    src = _parse_source(source)
    deleted = manager.remove(model, src)
    return ModelsDeleteResponse(deleted=deleted, model=model, freed_gb=0.0)


_EXTERNAL_MODELS_TTL = 60


class _ExternalModelsCache:
    """TTL cache for external model listings (no module-level mutable global)."""

    def __init__(self) -> None:
        self._time: float = 0.0
        self._key: str = ""
        self._result: ExternalModelsResponse | None = None

    def get(self, key: str) -> ExternalModelsResponse | None:
        now = time.monotonic()
        if self._result and key == self._key and (now - self._time) < _EXTERNAL_MODELS_TTL:
            return self._result
        return None

    def set(self, key: str, result: ExternalModelsResponse) -> None:
        self._time = time.monotonic()
        self._key = key
        self._result = result


_external_cache = _ExternalModelsCache()


async def list_external_models() -> ExternalModelsResponse:
    """Query the provider for available models via its list_models() API."""
    key = f"{cfg.remote_base_url}:{cfg.llm_api_key or ''}"
    cached = _external_cache.get(key)
    if cached:
        return cached

    try:
        models = await asyncio.to_thread(get_services().provider.list_models)
        result = ExternalModelsResponse(models=models)
        _external_cache.set(key, result)
        return result
    except Exception as exc:
        log.warning("Failed to list external models: %s", exc)
        return ExternalModelsResponse(models=[], error=str(exc))
