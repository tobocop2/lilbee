"""Public value types for model lifecycle management."""

from dataclasses import dataclass
from enum import Enum

from lilbee.providers.sdk_backend import REMOTE_BACKEND_NAME


class ModelSource(Enum):
    """Where a model is stored."""

    NATIVE = "native"  # lilbee's GGUF files in cfg.models_dir
    REMOTE = "remote"  # Models managed by a remote SDK-backed service

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


@dataclass
class RemoteModel:
    """A model from the SDK backend with inferred task classification."""

    name: str
    task: str  # "chat", "embedding", "vision", "rerank"
    family: str
    parameter_size: str
    provider: str = REMOTE_BACKEND_NAME


class ValidationResult(Enum):
    """Outcome of validating a persisted model ref against current state."""

    OK = "ok"
    NOT_INSTALLED = "not_installed"  # Local ref but no GGUF on disk.
    NO_KEY = "no_key"  # API ref but no provider key configured.
    UNKNOWN = "unknown"  # Ref string is malformed or its provider is unknown.
