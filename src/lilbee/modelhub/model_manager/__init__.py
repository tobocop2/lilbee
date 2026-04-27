"""Model lifecycle management across native GGUF and SDK-backed sources."""

from lilbee.modelhub.model_manager.core import ModelManager
from lilbee.modelhub.model_manager.discovery import (
    classify_remote_models,
    detect_remote_embedding_models,
    discover_api_models,
)
from lilbee.modelhub.model_manager.holder import get_model_manager, reset_model_manager
from lilbee.modelhub.model_manager.types import ModelNotFoundError, ModelSource, RemoteModel

__all__ = [
    "ModelManager",
    "ModelNotFoundError",
    "ModelSource",
    "RemoteModel",
    "classify_remote_models",
    "detect_remote_embedding_models",
    "discover_api_models",
    "get_model_manager",
    "reset_model_manager",
]
