"""Architecture compatibility classification for catalog entries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gguf import MODEL_ARCH_NAMES
from huggingface_hub import hf_hub_url

from lilbee.catalog.header_probe import probe_architecture
from lilbee.catalog.types import ModelCompat

if TYPE_CHECKING:
    from lilbee.catalog.hf_client import HfClient

SUPPORTED_ARCHS: frozenset[str] = frozenset(MODEL_ARCH_NAMES.values())


def classify(architecture: str) -> ModelCompat:
    """Map a `general.architecture` string to a `ModelCompat` verdict."""
    if not architecture:
        return ModelCompat.UNKNOWN
    return ModelCompat.SUPPORTED if architecture in SUPPORTED_ARCHS else ModelCompat.UNSUPPORTED


def resolve_arch_for_pull(ref: str, hf_client: HfClient) -> str:
    """Resolve general.architecture for *ref*: cache hit > Range-GET probe > empty (UNKNOWN)."""
    cached = hf_client._arch_cache.get(ref)
    if cached is not None:
        return cached
    url = _resolve_blob_url(ref)
    if not url:
        return ""
    arch = probe_architecture(url)
    hf_client._arch_cache[ref] = arch
    return arch


def _resolve_blob_url(ref: str) -> str:
    """Return a probable .gguf blob URL for *ref*, or empty string if unresolvable."""
    if ":" in ref:
        repo, filename = ref.split(":", 1)
    else:
        repo, filename = ref, ""
    if not filename or "*" in filename:
        return ""
    try:
        return hf_hub_url(repo, filename)
    except Exception:
        return ""


class UnsupportedArchError(Exception):
    """Raised when a pull is attempted for a model whose architecture isn't supported."""

    def __init__(self, ref: str, architecture: str) -> None:
        self.ref = ref
        self.architecture = architecture
        super().__init__(
            f"Model {ref!r} uses architecture {architecture!r}, not supported by this lilbee build."
        )
