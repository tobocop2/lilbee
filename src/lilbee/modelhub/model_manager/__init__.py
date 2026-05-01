"""Model lifecycle management across native GGUF and SDK-backed sources."""

from lilbee.modelhub.model_manager.core import ModelManager
from lilbee.modelhub.model_manager.discovery import (
    classify_remote_models,
    detect_remote_embedding_models,
    discover_api_models,
)
from lilbee.modelhub.model_manager.types import (
    ModelNotFoundError,
    ModelSource,
    RemoteModel,
    ValidationResult,
)
from lilbee.modelhub.model_manager.validation import (
    CanonicalRef,
    canonicalize_chat_model,
    canonicalize_embedding_model,
    validate_persisted_model,
)

__all__ = [
    "CanonicalRef",
    "ModelManager",
    "ModelNotFoundError",
    "ModelSource",
    "RemoteModel",
    "ValidationResult",
    "canonicalize_chat_model",
    "canonicalize_embedding_model",
    "classify_remote_models",
    "detect_remote_embedding_models",
    "discover_api_models",
    "validate_persisted_model",
]
