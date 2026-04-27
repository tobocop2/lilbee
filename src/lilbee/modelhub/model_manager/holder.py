"""Singleton holder for the ModelManager instance."""

from lilbee.core.config.model import cfg
from lilbee.modelhub.model_manager.core import ModelManager


class _ManagerHolder:
    """Encapsulates the ModelManager singleton (no module-level mutable global)."""

    def __init__(self) -> None:
        self._instance: ModelManager | None = None

    def get(self) -> ModelManager:
        if self._instance is None:
            self._instance = ModelManager(cfg.models_dir, cfg.remote_base_url)
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
