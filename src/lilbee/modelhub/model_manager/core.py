"""ModelManager: native and SDK-backed model lifecycle operations."""

import json
import logging
import time
from collections.abc import Callable
from http import HTTPStatus
from pathlib import Path

import httpx

from lilbee.catalog.types import ModelSource
from lilbee.core.config import DEFAULT_HTTP_TIMEOUT
from lilbee.core.security import validate_path_within
from lilbee.modelhub.model_manager.types import ModelNotFoundError
from lilbee.modelhub.registry import ModelRegistry

log = logging.getLogger(__name__)

_INSTALLED_CACHE_TTL_SECONDS = 30.0


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
        """List models from the SDK backend via its HTTP API."""
        url = f"{self._remote_base_url}/api/tags"
        try:
            resp = httpx.get(url, timeout=DEFAULT_HTTP_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except httpx.HTTPStatusError as exc:
            log.warning("SDK backend HTTP error listing models: %s", exc)
            return []
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            log.debug("SDK backend not reachable: %s", exc)
            return []

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
        """Find which source a model lives in. Native takes precedence."""
        if self._is_native(model):
            return ModelSource.NATIVE
        if self._is_remote(model):
            return ModelSource.REMOTE
        return None

    def pull(
        self,
        model: str,
        source: ModelSource,
        *,
        on_progress: Callable[[dict], None] | None = None,
        on_bytes: Callable[[int, int], None] | None = None,
    ) -> Path | None:
        """Pull/download model to specified source.

        Returns the Path for native downloads, None for backend-managed pulls.

        *on_progress* receives dict events from the SDK backend.
        *on_bytes* receives (downloaded_bytes, total_bytes) from native
        HuggingFace downloads. The two sources report progress in different
        shapes, so callers pass whichever matches the chosen source.
        """
        try:
            if source is ModelSource.NATIVE:
                return self._pull_native(model, on_bytes=on_bytes)
            self._pull_remote(model, on_progress=on_progress)
            return None
        finally:
            self._invalidate_installed_cache()

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

    def _pull_remote(
        self, model: str, *, on_progress: Callable[[dict], None] | None = None
    ) -> None:
        """Pull model via the SDK backend's HTTP API with streaming progress."""
        url = f"{self._remote_base_url}/api/pull"
        try:
            with (
                # Model pulls stream progress over minutes; an overall
                # timeout would cut the download. Connect/write timeouts
                # still apply via httpx defaults when timeout=None.
                httpx.Client(timeout=None) as client,  # noqa: S113
                client.stream("POST", url, json={"name": model, "stream": True}) as resp,
            ):
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if "error" in data:
                        raise RuntimeError(f"Failed to pull '{model}': {data['error']}")
                    if on_progress is not None:
                        on_progress(data)
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Cannot connect to SDK backend: {exc}. Is the server running?"
            ) from exc
        log.info("Pulled %s via SDK backend", model)

    def remove(self, model: str, source: ModelSource | None = None) -> bool:
        """Remove installed model. Returns True if removed."""
        try:
            if source is None:
                native_removed = self._remove_native(model)
                backend_removed = self._remove_remote(model)
                return native_removed or backend_removed
            if source is ModelSource.NATIVE:
                return self._remove_native(model)
            return self._remove_remote(model)
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

    def _remove_remote(self, model: str) -> bool:
        url = f"{self._remote_base_url}/api/delete"
        try:
            resp = httpx.request(
                "DELETE",
                url,
                content=json.dumps({"model": model}).encode(),
                headers={"Content-Type": "application/json"},
                timeout=DEFAULT_HTTP_TIMEOUT,
            )
            if resp.status_code == HTTPStatus.OK:
                log.info("Removed backend model %s", model)
                return True
            if resp.status_code == HTTPStatus.NOT_FOUND:
                return False
            log.warning("Unexpected status %d removing %s", resp.status_code, model)
            return False
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Cannot connect to SDK backend: {exc}. Is the server running?"
            ) from exc
