"""Model lifecycle management across native GGUF and litellm-backed sources."""

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import httpx

from lilbee.config import cfg
from lilbee.models import ModelTask
from lilbee.security import validate_path_within

log = logging.getLogger(__name__)

_LITELLM_TIMEOUT = 30.0


class ModelSource(Enum):
    """Where a model is stored."""

    NATIVE = "native"  # lilbee's GGUF files in cfg.models_dir
    LITELLM = "litellm"  # Models managed by an external backend

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


class ModelManager:
    """Manages model lifecycle with distinct sources."""

    def __init__(self, models_dir: Path, litellm_base_url: str = "http://localhost:11434") -> None:
        self._models_dir = models_dir
        self._litellm_base_url = litellm_base_url.rstrip("/")

        from lilbee.registry import ModelRegistry

        self._registry = ModelRegistry(self._models_dir)

    def list_installed(self, source: ModelSource | None = None) -> list[str]:
        """List installed model names. source=None lists all sources."""
        if source is None:
            native = set(self._list_native())
            remote = set(self._list_litellm())
            return sorted(native | remote)
        if source is ModelSource.NATIVE:
            return self._list_native()
        return self._list_litellm()

    def _list_native(self) -> list[str]:
        """List native models from the registry only."""
        return sorted(f"{m.name}:{m.tag}" for m in self._registry.list_installed())

    def _list_litellm(self) -> list[str]:
        """List models from the litellm backend via its HTTP API."""
        url = f"{self._litellm_base_url}/api/tags"
        try:
            resp = httpx.get(url, timeout=_LITELLM_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except httpx.HTTPStatusError as exc:
            log.warning("litellm backend HTTP error listing models: %s", exc)
            return []
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            log.debug("litellm backend not reachable: %s", exc)
            return []

    def is_installed(self, model: str, source: ModelSource | None = None) -> bool:
        """Check if model exists in specified source."""
        if source is None:
            return self._is_native(model) or self._is_litellm(model)
        if source is ModelSource.NATIVE:
            return self._is_native(model)
        return self._is_litellm(model)

    def _is_native(self, model: str) -> bool:
        if self._registry.is_installed(model):
            return True
        try:
            validate_path_within(self._models_dir / model, self._models_dir)
        except ValueError:
            return False
        return (self._models_dir / model).is_file()

    def _is_litellm(self, model: str) -> bool:
        return model in self._list_litellm()

    def get_source(self, model: str) -> ModelSource | None:
        """Find which source a model lives in. Native takes precedence."""
        if self._is_native(model):
            return ModelSource.NATIVE
        if self._is_litellm(model):
            return ModelSource.LITELLM
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

        Returns the Path for native downloads, None for litellm-backed pulls.

        *on_progress* receives dict events from the litellm backend.
        *on_bytes* receives (downloaded_bytes, total_bytes) from native
        HuggingFace downloads. The two sources report progress in different
        shapes, so callers pass whichever matches the chosen source.
        """
        if source is ModelSource.NATIVE:
            return self._pull_native(model, on_bytes=on_bytes)
        self._pull_litellm(model, on_progress=on_progress)
        return None

    def _pull_native(
        self,
        model: str,
        *,
        on_bytes: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Download model to the native GGUF directory via catalog."""
        from lilbee.catalog import download_model, find_catalog_entry

        entry = find_catalog_entry(model)
        if entry is None:
            raise ModelNotFoundError(f"Model '{model}' not found in catalog")
        path = download_model(entry, on_progress=on_bytes)
        log.info("Downloaded %s to %s", model, path)
        return path

    def _pull_litellm(
        self, model: str, *, on_progress: Callable[[dict], None] | None = None
    ) -> None:
        """Pull model via the litellm backend's HTTP API with streaming progress."""
        url = f"{self._litellm_base_url}/api/pull"
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
                f"Cannot connect to litellm backend: {exc}. Is the server running?"
            ) from exc
        log.info("Pulled %s via litellm backend", model)

    def remove(self, model: str, source: ModelSource | None = None) -> bool:
        """Remove installed model. Returns True if removed."""
        if source is None:
            native_removed = self._remove_native(model)
            litellm_removed = self._remove_litellm(model)
            return native_removed or litellm_removed
        if source is ModelSource.NATIVE:
            return self._remove_native(model)
        return self._remove_litellm(model)

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

    def _remove_litellm(self, model: str) -> bool:
        url = f"{self._litellm_base_url}/api/delete"
        try:
            resp = httpx.request(
                "DELETE",
                url,
                content=json.dumps({"model": model}).encode(),
                headers={"Content-Type": "application/json"},
                timeout=_LITELLM_TIMEOUT,
            )
            if resp.status_code == 200:
                log.info("Removed litellm model %s", model)
                return True
            if resp.status_code == 404:
                return False
            log.warning("Unexpected status %d removing %s", resp.status_code, model)
            return False
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Cannot connect to litellm backend: {exc}. Is the server running?"
            ) from exc


_EMBEDDING_FAMILIES = frozenset({"bert", "nomic-bert", "e5", "bge"})
_VISION_NAME_PATTERNS = frozenset({"llava", "vision", "moondream", "ocr", "minicpm-v"})

_vision_cache: dict[str, bool] = {}


def is_vision_capable(model: str) -> bool:
    """Check whether *model* supports vision/image input.

    Resolution order (first match wins):
    1. Provider ``get_capabilities`` (mmproj for llama-cpp, /api/show for Ollama).
    2. Featured catalog: model task is ``vision``.
    3. Name-pattern fallback: model name contains a known vision keyword.

    Results are cached per model name for the lifetime of the process.
    """
    if not model:
        return False

    if model in _vision_cache:
        return _vision_cache[model]

    result = _check_vision_capable(model)
    _vision_cache[model] = result
    return result


def _check_vision_capable(model: str) -> bool:
    """Uncached implementation of vision capability detection."""
    from lilbee.services import get_services

    try:
        provider = get_services().provider
        caps = provider.get_capabilities(model)
        if "vision" in caps:
            return True
    except Exception:
        log.debug("Provider capability check failed for %s", model, exc_info=True)

    from lilbee.catalog import FEATURED_VISION

    model_lower = model.lower()
    if any(
        model_lower in entry.name.lower() or model_lower in entry.hf_repo.lower()
        for entry in FEATURED_VISION
    ):
        return True

    return any(vp in model_lower for vp in _VISION_NAME_PATTERNS)


def reset_vision_cache() -> None:
    """Clear the vision capability cache (for testing)."""
    _vision_cache.clear()


@dataclass
class RemoteModel:
    """A model from the litellm backend with inferred task classification."""

    name: str
    task: str  # "chat", "embedding", "vision"
    family: str
    parameter_size: str
    provider: str = "Remote"  # "Ollama", "OpenAI", "Anthropic", or "Remote"


_PROVIDER_PATTERNS: tuple[tuple[str, str], ...] = (
    ("localhost:11434", "Ollama"),
    ("ollama", "Ollama"),
    ("openai", "OpenAI"),
    ("anthropic", "Anthropic"),
)


def detect_provider(base_url: str) -> str:
    """Detect the remote provider name from a litellm base URL."""
    url_lower = base_url.lower()
    for pattern, provider in _PROVIDER_PATTERNS:
        if pattern in url_lower:
            return provider
    return "Remote"


def _classify_remote_task(name: str, family: str) -> str:
    """Classify a remote model as chat, embedding, or vision."""
    family_lower = family.lower()
    if any(ef in family_lower for ef in _EMBEDDING_FAMILIES):
        return ModelTask.EMBEDDING
    name_lower = name.lower()
    if any(vp in name_lower for vp in _VISION_NAME_PATTERNS):
        return ModelTask.VISION
    return ModelTask.CHAT


_CLASSIFY_DEFAULT_TIMEOUT_S = 5.0


def classify_remote_models(
    base_url: str = "http://localhost:11434",
    *,
    timeout: float = _CLASSIFY_DEFAULT_TIMEOUT_S,
) -> list[RemoteModel]:
    """Discover and classify all models from the litellm backend by task.

    Uses /api/tags family metadata for embedding detection and name
    patterns for vision detection. Returns an empty list on any error
    (including timeout) so callers in read-only code paths can stay
    responsive when the backend is down.
    """
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=timeout)
        resp.raise_for_status()
        raw_models = resp.json().get("models", [])
    except Exception:
        log.debug("Failed to classify remote models", exc_info=True)
        return []

    provider = detect_provider(base_url)
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


def _has_provider_key(provider_name: str, cfg_field: str, env_var: str) -> bool:
    """Return True if a usable API key exists for *provider_name*."""
    if os.environ.get(env_var):
        return True
    return bool(getattr(cfg, cfg_field, ""))


def discover_api_models() -> dict[str, list[RemoteModel]]:
    """Return frontier chat models grouped by provider.

    Checks which provider API keys are available (env vars or config),
    then queries ``litellm.models_by_provider`` for each. Only chat
    models are returned. Returns an empty dict when litellm is not
    installed or no keys are configured.

    Short-circuits before importing litellm when no keys are present,
    avoiding the expensive import on CI or headless environments.
    """
    from lilbee.providers.litellm_provider import PROVIDER_KEYS

    # Check for any configured key before paying the litellm import cost.
    active = [
        (prov, cfg_f, env, label)
        for prov, cfg_f, env, label in PROVIDER_KEYS
        if _has_provider_key(prov, cfg_f, env)
    ]
    if not active:
        return {}

    try:
        import litellm
    except ImportError:
        return {}

    result: dict[str, list[RemoteModel]] = {}
    for provider, _cfg_field, _env_var, display_name in active:
        models = litellm.models_by_provider.get(provider, set())
        chat_models: list[RemoteModel] = []
        for model_name in sorted(models):
            info = litellm.model_cost.get(model_name, {})
            if info.get("mode") != "chat":
                continue
            chat_models.append(
                RemoteModel(
                    name=model_name,
                    task=ModelTask.CHAT,
                    family="",
                    parameter_size="",
                    provider=display_name,
                )
            )
        if chat_models:
            result[display_name] = chat_models
    return result


def detect_remote_embedding_models(base_url: str = "http://localhost:11434") -> list[str]:
    """Return names of models classified as embedding from the litellm backend."""
    return [m.name for m in classify_remote_models(base_url) if m.task == ModelTask.EMBEDDING]


class _ManagerHolder:
    """Encapsulates the ModelManager singleton (no module-level mutable global)."""

    def __init__(self) -> None:
        self._instance: ModelManager | None = None

    def get(self) -> ModelManager:
        if self._instance is None:
            self._instance = ModelManager(cfg.models_dir, cfg.litellm_base_url)
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
