"""Model lifecycle management across native GGUF and litellm-backed sources."""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import httpx

from lilbee.security import validate_path_within

log = logging.getLogger(__name__)

_LITELLM_TIMEOUT = 30.0


class ModelSource(Enum):
    """Where a model is stored."""

    NATIVE = "native"  # lilbee's GGUF files in cfg.models_dir
    LITELLM = "litellm"  # Models managed by an external tool (Ollama, etc.)


class ModelManager:
    """Manages model lifecycle with distinct sources."""

    def __init__(self, models_dir: Path, litellm_base_url: str = "http://localhost:11434") -> None:
        self._models_dir = models_dir
        self._litellm_base_url = litellm_base_url.rstrip("/")

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
        """List .gguf files in the models directory."""
        if not self._models_dir.is_dir():
            return []
        return sorted(p.name for p in self._models_dir.iterdir() if p.suffix == ".gguf")

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
    ) -> Path | None:
        """Pull/download model to specified source.

        Returns the Path for native downloads, None for litellm-backed pulls.
        """
        if source is ModelSource.NATIVE:
            return self._pull_native(model)
        self._pull_litellm(model, on_progress=on_progress)
        return None

    def _pull_native(self, model: str) -> Path:
        """Download model to the native GGUF directory via catalog."""
        from lilbee.catalog import download_model, find_catalog_entry

        entry = find_catalog_entry(model)
        if entry is None:
            raise RuntimeError(f"Model '{model}' not found in catalog")
        path = download_model(entry)
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


@dataclass
class RemoteModel:
    """A model from the litellm backend with inferred task classification."""

    name: str
    task: str  # "chat", "embedding", "vision"
    family: str
    parameter_size: str


def _classify_remote_task(name: str, family: str) -> str:
    """Classify a remote model as chat, embedding, or vision."""
    family_lower = family.lower()
    if any(ef in family_lower for ef in _EMBEDDING_FAMILIES):
        return "embedding"
    name_lower = name.lower()
    if any(vp in name_lower for vp in _VISION_NAME_PATTERNS):
        return "vision"
    return "chat"


def classify_remote_models(base_url: str = "http://localhost:11434") -> list[RemoteModel]:
    """Discover and classify all models from the litellm backend by task.

    Uses /api/tags family metadata for embedding detection and
    name patterns for vision detection.
    """
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        resp.raise_for_status()
        raw_models = resp.json().get("models", [])
    except Exception:
        return []

    result: list[RemoteModel] = []
    for model in raw_models:
        name = model.get("name", "")
        details = model.get("details", {})
        family = details.get("family", "")
        param_size = details.get("parameter_size", "")
        task = _classify_remote_task(name, family)
        result.append(RemoteModel(name=name, task=task, family=family, parameter_size=param_size))
    return result


def detect_remote_embedding_models(base_url: str = "http://localhost:11434") -> list[str]:
    """Return names of models classified as embedding from the litellm backend."""
    return [m.name for m in classify_remote_models(base_url) if m.task == "embedding"]


_manager: ModelManager | None = None


def get_model_manager() -> ModelManager:
    """Get or create the singleton ModelManager."""
    global _manager
    if _manager is None:
        from lilbee.config import cfg

        _manager = ModelManager(cfg.models_dir, cfg.litellm_base_url)
    return _manager


def reset_model_manager() -> None:
    """Clear the singleton (for testing)."""
    global _manager
    _manager = None
