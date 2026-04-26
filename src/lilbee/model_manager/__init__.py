"""Model lifecycle management across native GGUF and SDK-backed sources."""

from lilbee.model_manager.core import ModelManager
from lilbee.model_manager.discovery import (
    _classify_remote_task,
    _has_provider_key,
    classify_remote_models,
    detect_remote_embedding_models,
    discover_api_models,
)
from lilbee.model_manager.holder import get_model_manager, reset_model_manager
from lilbee.model_manager.types import ModelNotFoundError, ModelSource, RemoteModel

__all__ = [
    "ModelManager",
    "ModelNotFoundError",
    "ModelSource",
    "RemoteModel",
    "_classify_remote_task",
    "_has_provider_key",
    "classify_remote_models",
    "detect_remote_embedding_models",
    "discover_api_models",
    "get_model_manager",
    "reset_model_manager",
]
