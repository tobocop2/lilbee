"""Model lifecycle management across native GGUF and SDK-backed sources."""

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import httpx

from lilbee.config import DEFAULT_HTTP_TIMEOUT
from lilbee.models import ModelTask
from lilbee.providers.sdk_backend import (
    PROVIDER_KEYS,
    REMOTE_BACKEND_NAME,
    detect_backend_name,
)
from lilbee.security import validate_path_within

# lilbee.config imported lazily; see lilbee/catalog.py for the rationale.

log = logging.getLogger(__name__)


class ModelSource(Enum):
    """Where a model is stored."""

    NATIVE = "native"  # lilbee's GGUF files in cfg.models_dir
    BACKEND = "backend"  # Models managed by an external SDK backend

    @classmethod
    def parse(cls, value: str | None) -> "ModelSource | None":
        """Parse a user-supplied source string. Empty or None means 'all'.

        Raises ValueError on any other non-empty input so callers get a
        consistent error type regardless of entry point (CLI, MCP, server).
        """
        if value is None or value == "":
            return None
        try:
            return cls(value)
        except ValueError as exc:
            valid = ", ".join(s.value for s in cls)
            raise ValueError(f"invalid source {value!r}; expected one of: {valid}") from exc


class ModelNotFoundError(RuntimeError):
    """Raised when a model ref is not found in any source or the catalog.

    Subclasses RuntimeError so pre-existing `except RuntimeError` call
    sites still catch it.
    """


_INSTALLED_CACHE_TTL_SECONDS = 30.0


class ModelManager:
    """Manages model lifecycle with distinct sources."""

    def __init__(self, models_dir: Path, backend_base_url: str = "http://localhost:11434") -> None:
        self._models_dir = models_dir
        self._backend_base_url = backend_base_url.rstrip("/")

        from lilbee.registry import ModelRegistry

        self._registry = ModelRegistry(self._models_dir)
        # Memoize list_installed results to avoid walking the registry
        # filesystem and hitting the backend HTTP endpoint on every call.
        # The catalog filter path fires this per request. Time-based TTL
        # plus explicit invalidation on pull/remove keeps freshness.
        self._installed_cache: dict[ModelSource | None, tuple[float, list[str]]] = {}

    def list_installed(self, source: ModelSource | None = None) -> list[str]:
        """List installed model names. source=None lists all sources.

        Result is cached for ``_INSTALLED_CACHE_TTL_SECONDS`` and invalidated
        eagerly on any ``pull``/``remove`` mutation so callers only see
        stale data during the narrow window when no local change has
        happened and the cached backend response has not yet aged out.
        """
        now = time.monotonic()
        cached = self._installed_cache.get(source)
        if cached is not None:
            cached_at, cached_result = cached
            if now - cached_at < _INSTALLED_CACHE_TTL_SECONDS:
                return cached_result

        if source is None:
            native = set(self._list_native())
            remote = set(self._list_backend())
            result = sorted(native | remote)
        elif source is ModelSource.NATIVE:
            result = self._list_native()
        else:
            result = self._list_backend()

        self._installed_cache[source] = (now, result)
        return result

    def _invalidate_installed_cache(self) -> None:
        """Drop all cached list_installed results."""
        self._installed_cache.clear()

    def _list_native(self) -> list[str]:
        """List native models from the registry only."""
        return sorted(f"{m.name}:{m.tag}" for m in self._registry.list_installed())

    def _list_backend(self) -> list[str]:
        """List models from the SDK backend via its HTTP API."""
        url = f"{self._backend_base_url}/api/tags"
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
            return self._is_native(model) or self._is_backend(model)
        if source is ModelSource.NATIVE:
            return self._is_native(model)
        return self._is_backend(model)

    def _is_native(self, model: str) -> bool:
        if self._registry.is_installed(model):
            return True
        try:
            validate_path_within(self._models_dir / model, self._models_dir)
        except ValueError:
            return False
        return (self._models_dir / model).is_file()

    def _is_backend(self, model: str) -> bool:
        return model in self._list_backend()

    def get_source(self, model: str) -> ModelSource | None:
        """Find which source a model lives in. Native takes precedence."""
        if self._is_native(model):
            return ModelSource.NATIVE
        if self._is_backend(model):
            return ModelSource.BACKEND
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
            self._pull_backend(model, on_progress=on_progress)
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
        from lilbee.catalog import download_model, resolve_pull_target

        entry = resolve_pull_target(model)
        if entry is None:
            raise ModelNotFoundError(
                f"Model '{model}' not recognized. "
                "Pass a HuggingFace repo id (owner/name) or a featured model name."
            )
        path = download_model(entry, on_progress=on_bytes)
        log.info("Downloaded %s to %s", model, path)
        return path

    def _pull_backend(
        self, model: str, *, on_progress: Callable[[dict], None] | None = None
    ) -> None:
        """Pull model via the SDK backend's HTTP API with streaming progress."""
        url = f"{self._backend_base_url}/api/pull"
        try:
            with (
                httpx.Client(timeout=None) as client,
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
                backend_removed = self._remove_backend(model)
                return native_removed or backend_removed
            if source is ModelSource.NATIVE:
                return self._remove_native(model)
            return self._remove_backend(model)
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

    def _remove_backend(self, model: str) -> bool:
        url = f"{self._backend_base_url}/api/delete"
        try:
            resp = httpx.request(
                "DELETE",
                url,
                content=json.dumps({"model": model}).encode(),
                headers={"Content-Type": "application/json"},
                timeout=DEFAULT_HTTP_TIMEOUT,
            )
            if resp.status_code == 200:
                log.info("Removed backend model %s", model)
                return True
            if resp.status_code == 404:
                return False
            log.warning("Unexpected status %d removing %s", resp.status_code, model)
            return False
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Cannot connect to SDK backend: {exc}. Is the server running?"
            ) from exc


_EMBEDDING_FAMILIES = frozenset({"bert", "nomic-bert", "e5", "bge"})
_VISION_NAME_PATTERNS = frozenset({"llava", "vision", "moondream", "ocr", "minicpm-v"})
# Reranker detection runs BEFORE embedding detection so ``bge-reranker-*``
# (family "bge" but clearly a reranker) does not get misclassified as
# EMBEDDING. ``cross-encoder`` covers the sentence-transformers SBERT
# naming convention for cross-encoder rerankers.
_RERANKER_NAME_PATTERNS = frozenset({"reranker", "rerank", "cross-encoder"})


@dataclass
class RemoteModel:
    """A model from the SDK backend with inferred task classification."""

    name: str
    task: str  # "chat", "embedding", "vision", "rerank"
    family: str
    parameter_size: str
    provider: str = REMOTE_BACKEND_NAME


def _classify_remote_task(name: str, family: str) -> str:
    """Classify a remote model as chat, embedding, vision, or rerank.

    Reranker detection runs first so ``bge-reranker-base`` (family
    ``bge``) does not get dragged into the embedding bucket by the
    family check. After reranker: embedding by family tag, vision by
    name pattern, else chat.
    """
    name_lower = name.lower()
    if any(rp in name_lower for rp in _RERANKER_NAME_PATTERNS):
        return ModelTask.RERANK
    family_lower = family.lower()
    if any(ef in family_lower for ef in _EMBEDDING_FAMILIES):
        return ModelTask.EMBEDDING
    if any(vp in name_lower for vp in _VISION_NAME_PATTERNS):
        return ModelTask.VISION
    return ModelTask.CHAT


_CLASSIFY_DEFAULT_TIMEOUT_S = 5.0


def classify_remote_models(
    base_url: str = "http://localhost:11434",
    *,
    timeout: float = _CLASSIFY_DEFAULT_TIMEOUT_S,
) -> list[RemoteModel]:
    """Discover and classify all models from the SDK backend by task.

    Uses /api/tags family metadata for embedding detection and name
    patterns for reranker and vision detection. Returns an empty list
    on any error (including timeout) so callers in read-only code paths
    can stay responsive when the backend is down.
    """
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=timeout)
        resp.raise_for_status()
        raw_models = resp.json().get("models", [])
    except Exception:
        log.debug("Failed to classify remote models", exc_info=True)
        return []

    provider = detect_backend_name(base_url)
    result: list[RemoteModel] = []
    for model in raw_models:
        name = model.get("name", "")
        details = model.get("details", {})
        family = details.get("family", "")
        param_size = details.get("parameter_size", "")
        task = _classify_remote_task(name, family)
        result.append(
            RemoteModel(
                name=name, task=task, family=family, parameter_size=param_size, provider=provider
            )
        )
    return result


def _has_provider_key(cfg_field: str, env_var: str) -> bool:
    """Return True if a usable API key exists via env var or lilbee config."""
    if os.environ.get(env_var):
        return True
    # circular: model_manager -> config via cfg; config.py's catalog
    # imports pull model_manager in transitively, so we defer cfg access
    # to call time to avoid a module-init cycle.
    from lilbee.config import cfg

    return bool(getattr(cfg, cfg_field, ""))


def discover_api_models() -> dict[str, list[RemoteModel]]:
    """Return frontier chat models grouped by provider.

    Checks which provider API keys are available (env vars or config),
    then asks the active provider's backend for each provider's chat
    catalog. Going through ``SdkLLMProvider.list_chat_models`` ensures
    ``cfg.json_mode`` suppression is applied before the SDK import,
    so ``lilbee --json status`` does not leak a debug banner.

    Short-circuits before touching the SDK when no keys are present.
    """
    active = [
        (prov, cfg_f, env, label)
        for prov, cfg_f, env, label in PROVIDER_KEYS
        if _has_provider_key(cfg_f, env)
    ]
    if not active:
        return {}

    from lilbee.services import get_services

    provider = get_services().provider

    result: dict[str, list[RemoteModel]] = {}
    for prov, _cfg_field, _env_var, display_name in active:
        chat_models = [
            RemoteModel(
                name=model_name,
                task=ModelTask.CHAT,
                family="",
                parameter_size="",
                provider=display_name,
            )
            for model_name in provider.list_chat_models(prov)
        ]
        if chat_models:
            result[display_name] = chat_models
    return result


def detect_remote_embedding_models(base_url: str = "http://localhost:11434") -> list[str]:
    """Return names of models classified as embedding from the SDK backend."""
    return [m.name for m in classify_remote_models(base_url) if m.task == ModelTask.EMBEDDING]


class _ManagerHolder:
    """Encapsulates the ModelManager singleton (no module-level mutable global)."""

    def __init__(self) -> None:
        self._instance: ModelManager | None = None

    def get(self) -> ModelManager:
        if self._instance is None:
            from lilbee.config import cfg

            self._instance = ModelManager(cfg.models_dir, cfg.backend_base_url)
        return self._instance

    def reset(self) -> None:
        self._instance = None


_holder = _ManagerHolder()


def get_model_manager() -> ModelManager:
    """Get or create the singleton ModelManager."""
    return _holder.get()


def reset_model_manager() -> None:
    """Clear the singleton (for testing)."""
    _holder.reset()
