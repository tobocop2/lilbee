"""GGUF metadata helpers: header reads, mmproj sidecar lookup, projector type."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from gguf import GGUFReader, GGUFValueType

from lilbee.catalog.header_probe import GGUF_ARCH_KEY
from lilbee.providers.base import ProviderError

log = logging.getLogger(__name__)

_HF_BLOBS_DIR_NAME = "blobs"
_HF_SNAPSHOTS_DIR_NAME = "snapshots"
_CLIP_PROJECTOR_TYPE_KEY = "clip.projector_type"


def train_ctx_from_meta(
    meta: dict[str, str] | None,
    *,
    fallback: int,
    model_path: Path,
) -> int:
    """Resolve ``<arch>.context_length`` from GGUF metadata, clamping junk to ``fallback``.

    Some published GGUFs (nomic-embed, certain Qwen3 and vision builds)
    report ``context_length=0`` in their headers. Passing zero into
    ``Llama(n_ctx=...)`` cascades into ``n_batch=0`` / ``n_ubatch=0``,
    which trips ggml's Vulkan dispatch into undefined behaviour and
    surfaces as STATUS_HEAP_CORRUPTION on Windows. Unparseable values
    and non-positive integers both route to ``fallback``.
    """
    if not meta:
        return fallback
    raw = meta.get("context_length", str(fallback))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        log.warning(
            "GGUF %s has unparseable context_length=%r; using %d",
            model_path.name,
            raw,
            fallback,
        )
        return fallback
    if value <= 0:
        log.warning(
            "GGUF %s reports context_length=%d; using %d to avoid n_batch=0 crash",
            model_path.name,
            value,
            fallback,
        )
        return fallback
    return value


@contextmanager
def _gguf_reader(path: Path) -> Iterator[GGUFReader]:
    """Open ``GGUFReader`` and release its memmap deterministically.

    ``GGUFReader.__init__`` mmaps the file via ``numpy.memmap``. On Windows
    the OS handle blocks unlink/rename until the underlying mmap is closed;
    ``del reader.data`` alone isn't enough because field views into the
    array keep refcounts alive, so close the mmap object explicitly.
    """
    reader = GGUFReader(str(path))
    try:
        yield reader
    finally:
        if hasattr(reader, "data"):
            backing = getattr(reader.data, "_mmap", None)
            if backing is not None:
                backing.close()
            del reader.data


def read_gguf_metadata(model_path: Path) -> dict[str, str] | None:
    """Read GGUF header metadata. Returns ``None`` on any read failure.

    Reads via ``gguf.GGUFReader`` (pure-Python binary parser) instead of
    ``Llama(vocab_only=True)`` so the metadata pass stays robust against
    GGUFs whose embedded Jinja chat template uses tags llama-cpp-python's
    bundled Jinja can't compile (e.g. SmolLM3's ``{% generation %}``).
    """
    try:
        with _gguf_reader(model_path) as reader:
            return _collect_metadata(reader)
    except Exception:
        log.debug("read_gguf_metadata failed for %s", model_path, exc_info=True)
        return None


def _collect_metadata(reader: GGUFReader) -> dict[str, str] | None:
    """Pull the fields lilbee consults from one open ``GGUFReader``."""

    def field_value(name: str) -> Any:
        field = reader.fields.get(name)
        return field.contents() if field is not None else None

    arch = field_value(GGUF_ARCH_KEY) or "llama"
    fields_we_want: tuple[tuple[str, str], ...] = (
        (GGUF_ARCH_KEY, "architecture"),
        ("general.file_type", "file_type"),
        ("general.name", "name"),
        ("tokenizer.chat_template", "chat_template"),
        (f"{arch}.context_length", "context_length"),
        (f"{arch}.embedding_length", "embedding_length"),
        (f"{arch}.block_count", "block_count"),
        (f"{arch}.attention.head_count_kv", "head_count_kv"),
        (f"{arch}.attention.head_count", "head_count"),
        (f"{arch}.attention.key_length", "key_length"),
        (f"{arch}.attention.value_length", "value_length"),
    )
    result: dict[str, str] = {}
    for gguf_key, out_key in fields_we_want:
        value = field_value(gguf_key)
        if value is not None:
            result[out_key] = str(value)
    return result or None


def _find_mmproj_in_hf_snapshots(model_dir: Path) -> Path | None:
    """Walk an HF-cache ``blobs/`` dir up to its sibling ``snapshots/`` tree."""
    if model_dir.name != _HF_BLOBS_DIR_NAME:
        return None
    snapshots_dir = model_dir.parent / _HF_SNAPSHOTS_DIR_NAME
    if not snapshots_dir.is_dir():
        return None
    for snapshot in snapshots_dir.iterdir():
        candidates = sorted(snapshot.glob("*mmproj*.gguf"))
        if candidates:
            return candidates[0]
    return None


def _find_mmproj_in_flat_dir(model_dir: Path) -> Path | None:
    """Glob ``*mmproj*.gguf`` siblings of a model GGUF (sideloaded layout)."""
    candidates = sorted(model_dir.glob("*mmproj*.gguf"))
    return candidates[0] if candidates else None


def find_mmproj_for_model(model_path: Path) -> Path:
    """Find the mmproj (CLIP projection) file for a vision model.

    Resolution order: (1) catalog lookup scoped to ``FEATURED_VISION``,
    (2) HuggingFace-cache ``snapshots/`` sibling of ``blobs/``,
    (3) same-directory glob for flat sideloaded layouts.
    Raises ``ProviderError`` if none find a file.
    """
    from lilbee.catalog import find_mmproj_file

    found = (
        find_mmproj_file(model_path.stem)
        or _find_mmproj_in_hf_snapshots(model_path.parent)
        or _find_mmproj_in_flat_dir(model_path.parent)
    )
    if found is not None:
        return found

    raise ProviderError(
        f"No mmproj (CLIP projection) file found for vision model {model_path.name}. "
        f"Download the mmproj file to {model_path.parent} or re-download the vision "
        "model through the catalog to get both files.",
        provider="llama-cpp",
    )


def read_mmproj_projector_type(mmproj_path: Path) -> str | None:
    """Read ``clip.projector_type`` from a GGUF mmproj without loading the model."""
    try:
        with _gguf_reader(mmproj_path) as reader:
            field = reader.get_field(_CLIP_PROJECTOR_TYPE_KEY)
            if field is None or field.types[-1] != GGUFValueType.STRING:
                return None
            return bytes(field.parts[field.data[0]]).decode("utf-8", errors="replace")
    except Exception:
        log.debug("Failed to read mmproj metadata from %s", mmproj_path, exc_info=True)
        return None
