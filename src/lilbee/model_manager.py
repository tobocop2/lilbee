"""Model lifecycle management across native GGUF and Ollama sources."""

import json
import logging
from collections.abc import Callable
from enum import Enum
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

_OLLAMA_TIMEOUT = 30.0


class ModelSource(Enum):
    """Where a model is stored."""

    NATIVE = "native"  # lilbee's GGUF files in cfg.models_dir
    OLLAMA = "ollama"  # Ollama's model store


class ModelManager:
    """Manages model lifecycle with distinct sources."""

    def __init__(self, models_dir: Path, ollama_base_url: str = "http://localhost:11434") -> None:
        self._models_dir = models_dir
        self._ollama_base_url = ollama_base_url.rstrip("/")

    # ── Listing ──────────────────────────────────────────────────────────

    def list_installed(self, source: ModelSource | None = None) -> list[str]:
        """List installed model names. source=None lists all sources."""
        if source is None:
            native = set(self._list_native())
            ollama = set(self._list_ollama())
            return sorted(native | ollama)
        if source is ModelSource.NATIVE:
            return self._list_native()
        return self._list_ollama()

    def _list_native(self) -> list[str]:
        """List .gguf files in the models directory."""
        if not self._models_dir.is_dir():
            return []
        return sorted(p.name for p in self._models_dir.iterdir() if p.suffix == ".gguf")

    def _list_ollama(self) -> list[str]:
        """List models from Ollama via its HTTP API."""
        url = f"{self._ollama_base_url}/api/tags"
        try:
            resp = httpx.get(url, timeout=_OLLAMA_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except httpx.HTTPStatusError as exc:
            log.warning("Ollama HTTP error listing models: %s", exc)
            return []
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            log.debug("Ollama not reachable: %s", exc)
            return []

    # ── Query ────────────────────────────────────────────────────────────

    def is_installed(self, model: str, source: ModelSource | None = None) -> bool:
        """Check if model exists in specified source."""
        if source is None:
            return self._is_native(model) or self._is_ollama(model)
        if source is ModelSource.NATIVE:
            return self._is_native(model)
        return self._is_ollama(model)

    def _is_native(self, model: str) -> bool:
        return (self._models_dir / model).is_file()

    def _is_ollama(self, model: str) -> bool:
        return model in self._list_ollama()

    def get_source(self, model: str) -> ModelSource | None:
        """Find which source a model lives in. Native takes precedence."""
        if self._is_native(model):
            return ModelSource.NATIVE
        if self._is_ollama(model):
            return ModelSource.OLLAMA
        return None

    # ── Pull / Download ──────────────────────────────────────────────────

    def pull(
        self,
        model: str,
        source: ModelSource,
        *,
        on_progress: Callable[[dict], None] | None = None,
    ) -> Path | None:
        """Pull/download model to specified source.

        Returns the Path for native downloads, None for Ollama.
        """
        if source is ModelSource.NATIVE:
            return self._pull_native(model)
        self._pull_ollama(model, on_progress=on_progress)
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

    def _pull_ollama(
        self, model: str, *, on_progress: Callable[[dict], None] | None = None
    ) -> None:
        """Pull model via Ollama's HTTP API with streaming progress."""
        url = f"{self._ollama_base_url}/api/pull"
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
            raise RuntimeError(f"Cannot connect to Ollama: {exc}. Is Ollama running?") from exc
        log.info("Pulled %s via Ollama", model)

    # ── Remove ───────────────────────────────────────────────────────────

    def remove(self, model: str, source: ModelSource | None = None) -> bool:
        """Remove installed model. Returns True if removed."""
        if source is None:
            native_removed = self._remove_native(model)
            ollama_removed = self._remove_ollama(model)
            return native_removed or ollama_removed
        if source is ModelSource.NATIVE:
            return self._remove_native(model)
        return self._remove_ollama(model)

    def _remove_native(self, model: str) -> bool:
        path = self._models_dir / model
        if path.is_file():
            path.unlink()
            log.info("Removed native model %s", model)
            return True
        return False

    def _remove_ollama(self, model: str) -> bool:
        url = f"{self._ollama_base_url}/api/delete"
        try:
            resp = httpx.request(
                "DELETE",
                url,
                content=json.dumps({"model": model}).encode(),
                headers={"Content-Type": "application/json"},
                timeout=_OLLAMA_TIMEOUT,
            )
            if resp.status_code == 200:
                log.info("Removed Ollama model %s", model)
                return True
            if resp.status_code == 404:
                return False
            log.warning("Unexpected status %d removing %s", resp.status_code, model)
            return False
        except httpx.ConnectError as exc:
            raise RuntimeError(f"Cannot connect to Ollama: {exc}. Is Ollama running?") from exc


# ── Singleton ─────────────────────────────────────────────────────────────

_manager: ModelManager | None = None


def get_model_manager() -> ModelManager:
    """Get or create the singleton ModelManager."""
    global _manager
    if _manager is None:
        from lilbee.config import cfg

        _manager = ModelManager(cfg.models_dir, cfg.llm_base_url)
    return _manager


def reset_model_manager() -> None:
    """Clear the singleton (for testing)."""
    global _manager
    _manager = None
