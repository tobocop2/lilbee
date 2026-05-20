"""Architecture compatibility classification for catalog entries."""

from __future__ import annotations

from gguf import MODEL_ARCH_NAMES

from lilbee.catalog.types import ModelCompat

SUPPORTED_ARCHS: frozenset[str] = frozenset(MODEL_ARCH_NAMES.values())


def classify(architecture: str) -> ModelCompat:
    """Map a `general.architecture` string to a `ModelCompat` verdict."""
    if not architecture:
        return ModelCompat.UNKNOWN
    return ModelCompat.SUPPORTED if architecture in SUPPORTED_ARCHS else ModelCompat.UNSUPPORTED


class UnsupportedArchError(Exception):
    """Raised when a pull is attempted for a model whose architecture isn't supported."""

    def __init__(self, ref: str, architecture: str) -> None:
        self.ref = ref
        self.architecture = architecture
        super().__init__(
            f"Model {ref!r} uses architecture {architecture!r}, not supported by this lilbee build."
        )
