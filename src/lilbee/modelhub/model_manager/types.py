"""Public value types for model lifecycle management."""

from dataclasses import dataclass
from enum import Enum

from lilbee.catalog.types import KeyStatus, ModelTask
from lilbee.providers.backend_names import BackendName


class ModelNotFoundError(RuntimeError):
    """Raised when a model ref is not found in any source or the catalog.

    Subclasses RuntimeError so pre-existing `except RuntimeError` call
    sites still catch it.
    """


@dataclass
class RemoteModel:
    """A model from the SDK backend with inferred task classification."""

    name: str
    task: ModelTask
    family: str
    parameter_size: str
    provider: str = BackendName.REMOTE


@dataclass(frozen=True)
class ApiModelGroup:
    """One hosted provider's chat models and the status of its API key."""

    provider: str
    display_name: str
    key_status: KeyStatus
    models: list[RemoteModel]


class ValidationResult(Enum):
    """Outcome of validating a persisted model ref against current state."""

    OK = "ok"
    NOT_INSTALLED = "not_installed"  # Local ref but no GGUF on disk.
    NO_KEY = "no_key"  # API ref but no provider key configured.
    INVALID_KEY = "invalid_key"  # API ref whose provider rejected the key.
    UNKNOWN = "unknown"  # Ref string is malformed or its provider is unknown.
