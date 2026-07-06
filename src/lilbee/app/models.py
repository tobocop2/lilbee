"""Surface-agnostic model lifecycle use-cases (list / show / pull / remove)."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from lilbee.app.services import get_services
from lilbee.catalog.types import ModelCompat, ModelTask
from lilbee.core.config import cfg
from lilbee.modelhub.registry import ModelRegistry

if TYPE_CHECKING:
    from collections.abc import Callable

    from lilbee.catalog import CatalogModel, DownloadProgress
    from lilbee.catalog.types import ModelSource
    from lilbee.modelhub.model_manager import RemoteModel
    from lilbee.modelhub.registry import ModelManifest


_BYTES_PER_GB = 1024**3  # Model sizes are reported to users in GiB.
_BACKEND_LIST_TIMEOUT_S = 2.0  # Keep `model list` snappy when backend is down.


def _bytes_to_gb(n: int) -> float:
    """Convert bytes to GiB rounded to 2 decimals for user display."""
    return round(n / _BYTES_PER_GB, 2)


class ModelCommand(StrEnum):
    """Command field values for model sub-app JSON output."""

    LIST = "model list"
    SHOW = "model show"
    PULL = "model pull"
    RM = "model rm"


class PullStatus(StrEnum):
    OK = "ok"
    ALREADY_INSTALLED = "already_installed"


class AdoptStatus(StrEnum):
    ADOPTED = "adopted"
    ALREADY_ACTIVE = "already_active"


class PullEvent(StrEnum):
    PROGRESS = "progress"
    DONE = "done"


class ModelEntry(BaseModel):
    """One row of `lilbee model list` output."""

    name: str
    source: str
    task: ModelTask | None = None
    size_gb: float | None = None
    display_name: str = ""

    @classmethod
    def from_native(cls, ref: str, manifest: ModelManifest | None) -> ModelEntry:
        # heavy: lilbee.catalog (>50ms; huggingface_hub) + lilbee.modelhub.model_manager (>50ms)
        from lilbee.catalog import clean_display_name
        from lilbee.catalog.types import ModelSource

        return cls(
            name=ref,
            source=ModelSource.NATIVE.value,
            task=manifest.task if manifest else None,
            size_gb=_bytes_to_gb(manifest.disk_size_bytes) if manifest else None,
            display_name=clean_display_name(manifest.hf_repo) if manifest else "",
        )

    @classmethod
    def from_backend(cls, ref: str, remote: RemoteModel | None, source: ModelSource) -> ModelEntry:
        from lilbee.providers.local_servers import canonical_local_ref

        return cls(
            name=canonical_local_ref(ref, source.value),
            source=source.value,
            task=remote.task if remote else None,
            size_gb=None,
            display_name=remote.parameter_size if remote else "",
        )


class ListModelsResult(BaseModel):
    command: str = ModelCommand.LIST
    models: list[ModelEntry]
    total: int


class CatalogEntryData(BaseModel):
    ref: str
    display_name: str
    hf_repo: str
    gguf_filename: str
    size_gb: float
    min_ram_gb: float
    description: str
    task: ModelTask
    featured: bool
    recommended: bool
    architecture: str = ""
    compat: ModelCompat = ModelCompat.UNKNOWN

    @classmethod
    def from_catalog_model(cls, entry: CatalogModel) -> CatalogEntryData:
        return cls(
            ref=entry.ref,
            display_name=entry.display_name,
            hf_repo=entry.hf_repo,
            gguf_filename=entry.gguf_filename,
            size_gb=entry.size_gb,
            min_ram_gb=entry.min_ram_gb,
            description=entry.description,
            task=entry.task,
            featured=entry.featured,
            recommended=entry.recommended,
            architecture=entry.architecture,
            compat=entry.compat,
        )


class ManifestData(BaseModel):
    ref: str
    display_name: str
    task: ModelTask
    size_gb: float
    size_bytes: int
    hf_repo: str
    gguf_filename: str
    downloaded_at: str

    @classmethod
    def from_manifest(cls, manifest: ModelManifest) -> ManifestData:
        from lilbee.catalog import clean_display_name

        return cls(
            ref=manifest.ref,
            display_name=clean_display_name(manifest.hf_repo),
            task=manifest.task,
            size_gb=_bytes_to_gb(manifest.disk_size_bytes),
            size_bytes=manifest.disk_size_bytes,
            hf_repo=manifest.hf_repo,
            gguf_filename=manifest.gguf_filename,
            downloaded_at=manifest.downloaded_at,
        )


class ShowModelResult(BaseModel):
    command: str = ModelCommand.SHOW
    model: str
    catalog: CatalogEntryData | None = None
    installed: bool = False
    source: str | None = None
    path: str | None = None
    manifest: ManifestData | None = None


class PullResult(BaseModel):
    command: str = ModelCommand.PULL
    model: str
    source: str
    status: PullStatus
    path: str | None = None


class PullProgressEvent(BaseModel):
    command: str = ModelCommand.PULL
    event: str = PullEvent.PROGRESS
    model: str
    percent: float
    detail: str
    cache_hit: bool


class RemoveResult(BaseModel):
    command: str = ModelCommand.RM
    model: str
    deleted: bool
    freed_gb: float = Field(default=0.0)


class AdoptResult(BaseModel):
    """Outcome of adopting a downloaded index's embedder."""

    model: str
    status: AdoptStatus
    reindex_required: bool = False


def adopt_embedder(ref: str) -> AdoptResult:
    """Switch lilbee to embedder *ref*, downloading it first if missing.

    Makes a downloaded index searchable under its own embedder without a
    rebuild: the persisted vectors already match *ref*, so the switch routes
    through the settings boundary and ``reindex_required`` stays false.
    """
    from lilbee.app.settings import apply_settings_update
    from lilbee.catalog.types import ModelSource

    manager = get_services().model_manager
    installed = manager.is_installed(ref, ModelSource.NATIVE)
    already_active = cfg.embedding_model == ref and installed
    if not installed:
        pull_model_data(ref, ModelSource.NATIVE)
    result = apply_settings_update({"embedding_model": ref})
    return AdoptResult(
        model=ref,
        status=AdoptStatus.ALREADY_ACTIVE if already_active else AdoptStatus.ADOPTED,
        reindex_required=result.reindex_required,
    )


def _native_manifest_index() -> dict[str, ModelManifest]:
    """Map ref string ('hf_repo/filename') to manifest for every installed native model."""
    registry = ModelRegistry(cfg.models_dir)
    return {m.ref: m for m in registry.list_installed()}


def _resolve_native_path(ref: str) -> str | None:
    """Return the on-disk path of an installed native model, if resolvable.

    Swallows ``KeyError`` (manifest present but blob missing) and
    ``ValueError`` (malformed ref) so callers can treat the path as
    optional metadata.
    """
    try:
        return str(ModelRegistry(cfg.models_dir).resolve(ref))
    except (KeyError, ValueError):
        return None


def _collect_native_entries() -> list[ModelEntry]:
    # heavy: lilbee.modelhub.model_manager (>50ms; huggingface_hub fanout)
    from lilbee.catalog.types import ModelSource

    manifests = _native_manifest_index()
    refs = get_services().model_manager.list_installed(source=ModelSource.NATIVE)
    return [ModelEntry.from_native(ref, manifests.get(ref)) for ref in refs]


def _collect_backend_entries() -> list[ModelEntry]:
    # heavy: lilbee.modelhub.model_manager (>50ms; huggingface_hub fanout)
    from lilbee.catalog.types import ModelSource
    from lilbee.modelhub.model_manager import classify_all_remote_models
    from lilbee.providers.local_servers import local_server_for_label

    def _source(remote: RemoteModel) -> ModelSource:
        spec = local_server_for_label(remote.provider)
        return ModelSource(spec.key) if spec is not None else ModelSource.REMOTE

    remote_by_name = {
        rm.name: rm for rm in classify_all_remote_models(timeout=_BACKEND_LIST_TIMEOUT_S)
    }
    return [
        ModelEntry.from_backend(name, rm, _source(rm))
        for name, rm in sorted(remote_by_name.items())
    ]


def list_models_data(
    source: ModelSource | None = None,
    task: ModelTask | None = None,
) -> ListModelsResult:
    """Build the list of installed models with source and task metadata.

    Discovers remote models via a single HTTP call with a short timeout
    so the command stays responsive when the backend is down.
    """
    # heavy: lilbee.modelhub.model_manager (>50ms; huggingface_hub fanout)
    from lilbee.catalog.types import ModelSource

    entries: list[ModelEntry] = []
    if source is None or source is ModelSource.NATIVE:
        entries.extend(_collect_native_entries())
    if source is not ModelSource.NATIVE:
        backend = _collect_backend_entries()
        # A specific local-server source (ollama/lm_studio/frontier) narrows the
        # backend list; REMOTE and None keep every backend entry.
        if source is not None and source is not ModelSource.REMOTE:
            backend = [e for e in backend if e.source == source.value]
        entries.extend(backend)
    if task:
        entries = [e for e in entries if e.task == task]
    return ListModelsResult(models=entries, total=len(entries))


def show_model_data(ref: str) -> ShowModelResult:
    """Return catalog and install metadata for *ref*.

    Raises :class:`~lilbee.modelhub.model_manager.ModelNotFoundError` if the ref
    is unknown to both the catalog and the installed set.
    """
    # heavy: lilbee.catalog (>50ms; huggingface_hub) + lilbee.modelhub.model_manager (>50ms)
    from lilbee.catalog import find_catalog_entry
    from lilbee.modelhub.model_manager import ModelNotFoundError

    entry = find_catalog_entry(ref)
    source = get_services().model_manager.get_source(ref)
    if entry is None and source is None:
        raise ModelNotFoundError(f"model not found: {ref}")
    manifest = _native_manifest_index().get(ref)
    return ShowModelResult(
        model=ref,
        catalog=CatalogEntryData.from_catalog_model(entry) if entry else None,
        installed=source is not None,
        source=source.value if source else None,
        manifest=ManifestData.from_manifest(manifest) if manifest else None,
        path=_resolve_native_path(ref) if manifest is not None else None,
    )


def _ensure_vision_projector(ref: str) -> None:
    """Fetch a vision model's mmproj projector when a cached install lacks it (bb-7yd).

    No-op for non-vision refs. ``download_mmproj`` is idempotent against the HF
    cache, so this is cheap when the projector is already present.
    """
    from lilbee.catalog import download_mmproj, resolve_pull_target

    entry = resolve_pull_target(ref)
    if entry is not None and entry.task is ModelTask.VISION:
        download_mmproj(entry)


def pull_model_data(
    ref: str,
    source: ModelSource,
    *,
    on_update: Callable[[DownloadProgress], None] | None = None,
    allow_unsupported: bool = False,
) -> PullResult:
    """Pull *ref* from *source* and return a typed result.

    Only native models are downloadable; a non-native *source* is refused by
    :meth:`ModelManager.pull`. Progress updates are throttled by
    :func:`~lilbee.catalog.make_download_callback`, so callers see at most
    roughly 10 Hz of progress events.
    """
    # heavy: lilbee.catalog (>50ms; huggingface_hub fanout)
    from lilbee.catalog import make_download_callback

    manager = get_services().model_manager

    if manager.is_installed(ref, source):
        # A cached vision install may carry the main GGUF but not its mmproj
        # projector (bb-7yd); without it llama-server can't serve OCR, so ensure
        # it before reporting already-installed.
        _ensure_vision_projector(ref)
        return PullResult(model=ref, source=source.value, status=PullStatus.ALREADY_INSTALLED)

    bytes_cb = make_download_callback(on_update) if on_update is not None else None
    path = manager.pull(
        ref,
        source,
        on_bytes=bytes_cb,
        allow_unsupported=allow_unsupported,
    )
    return PullResult(
        model=ref,
        source=source.value,
        status=PullStatus.OK,
        path=str(path) if path is not None else None,
    )


def _legacy_disk_size(ref: str, *, fallback: int) -> int:
    """Sum a split GGUF's shard sizes on disk; *fallback* on any failure/single file."""
    import contextlib

    with contextlib.suppress(Exception):
        shards = get_services().registry.shard_paths(ref)
        if len(shards) > 1:
            return sum(path.stat().st_size for path in shards)
    return fallback


def remove_model_data(
    ref: str,
    source: ModelSource | None = None,
) -> RemoveResult:
    """Remove *ref* and return a typed result with freed size."""
    manager = get_services().model_manager
    manifests = _native_manifest_index()
    # disk_size_bytes is the full multi-shard total; size_bytes alone would report
    # only the first shard for a split GGUF.
    manifest = manifests.get(ref)
    size_bytes = manifest.disk_size_bytes if manifest is not None else 0
    if manifest is not None and manifest.total_size_bytes is None:
        # Legacy manifest without shard accounting: removal still frees every
        # shard, so recover the true on-disk total from the shards for the report.
        size_bytes = _legacy_disk_size(ref, fallback=size_bytes)
    removed = manager.remove(ref, source=source)
    return RemoveResult(
        model=ref,
        deleted=removed,
        freed_gb=_bytes_to_gb(size_bytes),
    )
