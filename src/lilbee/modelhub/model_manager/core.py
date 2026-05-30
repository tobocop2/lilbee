"""ModelManager: native and SDK-backed model lifecycle operations."""

import logging
import time
from collections.abc import Callable
from pathlib import Path

from lilbee.catalog.types import ModelSource
from lilbee.core.config import DEFAULT_HTTP_TIMEOUT
from lilbee.core.security import validate_path_within
from lilbee.modelhub.model_manager.types import ModelNotFoundError
from lilbee.modelhub.registry import ModelRegistry
from lilbee.providers.local_servers import LOCAL_SERVERS, detect_local_server
from lilbee.providers.model_ref import parse_model_ref

log = logging.getLogger(__name__)

_INSTALLED_CACHE_TTL_SECONDS = 30.0


def _prefixed_source(model: str) -> ModelSource | None:
    """Map a provider-prefixed ref to its source, or ``None`` for a bare ref.

    Local-server prefixes (``ollama/``, ``lm_studio/``) map to that server's
    source; API-provider prefixes are FRONTIER. A bare name returns ``None``
    so the caller falls back to backend membership.
    """
    for spec in LOCAL_SERVERS:
        if model.startswith(spec.wire_prefix):
            return ModelSource(spec.key)
    try:
        ref = parse_model_ref(model)
    except ValueError:
        return None
    return ModelSource.FRONTIER if ref.is_api else None


class ModelManager:
    """Manages model lifecycle with distinct sources."""

    def __init__(self, models_dir: Path, remote_base_url: str = "http://localhost:11434") -> None:
        self._models_dir = models_dir
        self._remote_base_url = remote_base_url.rstrip("/")
        self._registry = ModelRegistry(self._models_dir)
        # Memoize list_installed results to avoid walking the registry
        # filesystem and hitting the backend HTTP endpoint on every call.
        # The catalog filter path fires this per request. Time-based TTL
        # plus explicit invalidation on pull/remove keeps freshness.
        self._installed_cache: dict[ModelSource | None, tuple[float, list[str]]] = {}
        # Identity cache: refs + hf_repos of installed natives. The catalog
        # screen reads this to mark rows as installed without re-walking
        # the registry on every screen mount (~150-300 ms saved).
        self._native_identities_cache: tuple[float, frozenset[str]] | None = None

    def list_installed(self, source: ModelSource | None = None) -> list[str]:
        """List installed model names. ``source=None`` lists all sources.

        Memoized with a ``_INSTALLED_CACHE_TTL_SECONDS`` TTL and
        invalidated eagerly by ``pull``/``remove``.
        """
        now = time.monotonic()
        cached = self._installed_cache.get(source)
        if cached is not None:
            cached_at, cached_result = cached
            if now - cached_at < _INSTALLED_CACHE_TTL_SECONDS:
                return cached_result

        if source is None:
            native = set(self._list_native())
            remote = set(self._list_remote())
            result = sorted(native | remote)
        elif source is ModelSource.NATIVE:
            result = self._list_native()
        else:
            result = self._list_remote()

        self._installed_cache[source] = (now, result)
        return result

    def list_native_identities(self) -> frozenset[str]:
        """Return refs + hf_repos of installed native models.

        Same TTL as ``list_installed``. The catalog screen reads this to
        mark catalog rows as installed without re-walking the registry
        on every screen mount.
        """
        now = time.monotonic()
        if self._native_identities_cache is not None:
            cached_at, cached_result = self._native_identities_cache
            if now - cached_at < _INSTALLED_CACHE_TTL_SECONDS:
                return cached_result
        identities: set[str] = set()
        try:
            for m in self._registry.list_installed():
                identities.add(m.ref)
                identities.add(m.hf_repo)
        except Exception:
            log.debug("ModelRegistry.list_installed failed", exc_info=True)
        result = frozenset(identities)
        self._native_identities_cache = (now, result)
        return result

    def _invalidate_installed_cache(self) -> None:
        """Drop all cached list_installed results."""
        self._installed_cache.clear()
        self._native_identities_cache = None

    def _list_native(self) -> list[str]:
        """List native models from the registry only."""
        return sorted(m.ref for m in self._registry.list_installed())

    def _list_remote(self) -> list[str]:
        """List model names from the configured local server (Ollama or LM Studio).

        Reuses the discovery dispatch so the listing endpoint matches the
        detected server (Ollama ``/api/tags`` vs LM Studio ``/v1/models``).
        Returns ``[]`` when the backend is unreachable.
        """
        # circular: discovery -> app.services -> model_manager.__init__ -> core
        from lilbee.modelhub.model_manager.discovery import classify_remote_models

        models = classify_remote_models(self._remote_base_url, timeout=DEFAULT_HTTP_TIMEOUT)
        return [m.name for m in models]

    def is_installed(self, model: str, source: ModelSource | None = None) -> bool:
        """Check if model exists in specified source."""
        if source is None:
            return self._is_native(model) or self._is_remote(model)
        if source is ModelSource.NATIVE:
            return self._is_native(model)
        return self._is_remote(model)

    def _is_native(self, model: str) -> bool:
        if self._registry.is_installed(model):
            return True
        try:
            validate_path_within(self._models_dir / model, self._models_dir)
        except ValueError:
            return False
        return (self._models_dir / model).is_file()

    def _is_remote(self, model: str) -> bool:
        return model in self.list_installed(ModelSource.REMOTE)

    def get_source(self, model: str) -> ModelSource | None:
        """Return the granular source a model lives in. Native takes precedence.

        A provider-prefixed ref classifies without a network call; a bare name
        gets the configured local server's source when that backend reports it
        installed. ``None`` when the model is in no known source.
        """
        if self._is_native(model):
            return ModelSource.NATIVE
        prefixed = _prefixed_source(model)
        if prefixed is not None:
            return prefixed
        if self._is_remote(model):
            server = detect_local_server(self._remote_base_url)
            return ModelSource(server.key) if server is not None else ModelSource.REMOTE
        return None

    def pull(
        self,
        model: str,
        source: ModelSource,
        *,
        on_bytes: Callable[[int, int], None] | None = None,
        allow_unsupported: bool = False,
    ) -> Path | None:
        """Download a native GGUF model and return its path.

        lilbee pulls native models only. Local servers (Ollama, LM Studio)
        are read-only: their models are managed in their own app and surface
        here once present, so a non-native *source* is refused.

        Native pulls of architectures the bundled llama.cpp doesn't support
        are refused with ``UnsupportedArchError`` unless *allow_unsupported*
        is True. *on_bytes* receives (downloaded_bytes, total_bytes) progress.
        """
        if source is not ModelSource.NATIVE:
            server = detect_local_server(self._remote_base_url)
            where = server.display_name if server is not None else "the configured server"
            raise ValueError(
                f"lilbee runs {where} models but doesn't download them. "
                f"Add the model in {where}, then pick it here."
            )
        if not allow_unsupported:
            self._enforce_arch_compat(model)
        try:
            return self._pull_native(model, on_bytes=on_bytes)
        finally:
            self._invalidate_installed_cache()

    def _enforce_arch_compat(self, ref: str) -> None:
        """Raise UnsupportedArchError if *ref*'s architecture isn't in the supported set."""
        from lilbee.app.services import get_services
        from lilbee.catalog.compat import (
            ModelCompat,
            UnsupportedArchError,
            classify,
            resolve_arch_for_pull,
        )

        arch = resolve_arch_for_pull(ref, get_services().hf_client)
        if classify(arch) is ModelCompat.UNSUPPORTED:
            raise UnsupportedArchError(ref, arch)

    def _pull_native(
        self,
        model: str,
        *,
        on_bytes: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Download a featured or ad-hoc HuggingFace model to the native GGUF directory."""
        # heavy: lilbee.catalog (>50ms; huggingface_hub fanout)
        from lilbee.catalog import download_model, resolve_pull_target
        from lilbee.modelhub.registry import register_downloaded_model

        entry = resolve_pull_target(model)
        if entry is None:
            raise ModelNotFoundError(
                f"Model '{model}' not recognized. "
                "Pass a HuggingFace repo id (owner/name) or a featured model name."
            )
        path = download_model(entry, on_progress=on_bytes, on_complete=register_downloaded_model)
        log.info("Downloaded %s to %s", model, path)
        return path

    def remove(self, model: str, source: ModelSource | None = None) -> bool:
        """Remove an installed native model. Returns True if removed.

        lilbee removes only native GGUF models it downloaded. Local servers
        (Ollama, LM Studio) are read-only: a model that lives on one is refused
        (mirrors ``pull``), since its lifecycle is managed in that app. A bare
        ``source`` is resolved so a local-server ref is caught either way.
        """
        effective = source if source is not None else self.get_source(model)
        if effective is not None and effective is not ModelSource.NATIVE:
            server = detect_local_server(self._remote_base_url)
            where = server.display_name if server is not None else "the configured server"
            raise ValueError(
                f"lilbee runs {where} models but doesn't remove them. "
                f"Manage them in {where} instead."
            )
        try:
            return self._remove_native(model)
        finally:
            self._invalidate_installed_cache()

    def _remove_native(self, model: str) -> bool:
        if self._registry.remove(model):
            log.info("Removed native model %s from registry", model)
            return True
        try:
            path = validate_path_within(self._models_dir / model, self._models_dir)
        except ValueError:
            log.warning("Path traversal blocked: %s escapes %s", model, self._models_dir)
            return False
        if path.is_file():
            path.unlink()
            log.info("Removed native model %s", model)
            return True
        return False
