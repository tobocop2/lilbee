"""Public value types for model lifecycle management."""

from dataclasses import dataclass
from enum import Enum

from lilbee.catalog.types import ModelTask
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


class ValidationResult(Enum):
    """Outcome of validating a persisted model ref against current state."""

    OK = "ok"
    NOT_INSTALLED = "not_installed"  # Local ref but no GGUF on disk.
    NO_KEY = "no_key"  # API ref but no provider key configured.
    UNKNOWN = "unknown"  # Ref string is malformed or its provider is unknown.
