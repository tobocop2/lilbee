"""Architecture compatibility classification for catalog entries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from huggingface_hub import hf_hub_url
from huggingface_hub.utils import HFValidationError

from lilbee._generated.engine_archs import SUPPORTED_ARCHS
from lilbee.catalog.header_probe import GgufHeader, probe_header
from lilbee.catalog.refs import (
    GGUF_SUFFIX,
    NATIVE_GGUF_REF_MIN_SLASHES,
    WILDCARD,
    gguf_filename_from_ref,
    hf_repo_from_ref,
)
from lilbee.catalog.types import ModelCompat

if TYPE_CHECKING:
    from lilbee.catalog.hf_client import HfClient


def classify(architecture: str) -> ModelCompat:
    """Map a `general.architecture` string to a `ModelCompat` verdict."""
    if not architecture:
        return ModelCompat.UNKNOWN
    return ModelCompat.SUPPORTED if architecture in SUPPORTED_ARCHS else ModelCompat.UNSUPPORTED


def resolve_arch_for_pull(ref: str, hf_client: HfClient) -> str:
    """Resolve general.architecture for *ref*: cache hit > Range-GET probe > empty (UNKNOWN).

    ``probe_header`` returns empty fields on any failure (network, non-200, parse),
    so an empty result means "undetermined", never a real verdict. Only a non-empty
    arch is cached; otherwise a transient probe failure would be cached permanently
    and disable the unsupported-arch guard for this ref on every later pull.
    """
    cached = hf_client.get_cached_arch(ref)
    if cached is not None:
        return cached
    url = _resolve_blob_url(ref)
    if not url:
        return ""
    arch = probe_header(url).architecture
    if arch:
        hf_client.cache_arch(ref, arch)
    return arch


def file_header(hf_repo: str, filename: str) -> GgufHeader:
    """The GGUF header of one repo file, empty when it cannot be read.

    An unreadable header is not a verdict: every field stays empty, which reads
    as "a model of undetermined architecture" so an offline or gated repo
    resolves the way it did before the probe existed.
    """
    try:
        url = hf_hub_url(hf_repo, filename)
    except (HFValidationError, ValueError):
        return GgufHeader()
    return probe_header(url)


def _resolve_blob_url(ref: str) -> str:
    """Return a probable .gguf blob URL for *ref*, or empty string if unresolvable.

    lilbee's canonical native refs are slash-delimited ``<org>/<repo>/<file>.gguf``
    (the filename may add subdirs for a quant), so the repo and filename are split
    on that shape; an ollama-style ``repo:tag`` ref is split on the colon.
    """
    if ref.endswith(GGUF_SUFFIX) and ref.count("/") >= NATIVE_GGUF_REF_MIN_SLASHES:
        repo, filename = hf_repo_from_ref(ref), gguf_filename_from_ref(ref)
    elif ":" in ref:
        repo, filename = ref.split(":", 1)
    else:
        repo, filename = ref, ""
    if not filename or WILDCARD in filename:
        return ""
    try:
        return hf_hub_url(repo, filename)
    except (HFValidationError, ValueError):
        return ""


class UnsupportedArchError(Exception):
    """Raised when a pull is attempted for a model whose architecture isn't supported."""

    def __init__(self, ref: str, architecture: str) -> None:
        self.ref = ref
        self.architecture = architecture
        super().__init__(
            f"Model {ref!r} uses architecture {architecture!r}, not supported by this lilbee build."
        )


class UnsupportedQuantError(RuntimeError):
    """Raised when a GGUF's tensors carry a type the bundled engine cannot decode.

    A RuntimeError so every pull surface renders it the way it already renders a
    failed pull. Unlike an unsupported architecture there is no preflight route
    and no list of accepted types to offer, so there is nothing structured for a
    client to read beyond the message.
    """

    def __init__(self, ref: str, quant: str) -> None:
        self.ref = ref
        self.quant = quant
        super().__init__(
            f"Model {ref!r} is quantized as {quant}, which this lilbee build cannot load. "
            "Pass --allow-unsupported to download it anyway."
        )
