"""GGUF metadata helpers: header reads, mmproj sidecar lookup, projector type.

Reads GGUF headers with the standalone ``gguf`` parser (no native binding), so
metadata is available to any provider without loading a model into the engine.
"""

from __future__ import annotations

import logging
from pathlib import Path

from gguf import GGUFReader, GGUFValueType

from lilbee.catalog.header_probe import GGUF_ARCH_KEY, gguf_scalar_str
from lilbee.providers.base import ProviderError

log = logging.getLogger(__name__)

_HF_BLOBS_DIR_NAME = "blobs"
_HF_SNAPSHOTS_DIR_NAME = "snapshots"
_CLIP_PROJECTOR_TYPE_KEY = "clip.projector_type"
_DEFAULT_ARCH = "llama"
_CHAT_TEMPLATE_KEY = "tokenizer.chat_template"
_FILE_TYPE_KEY = "general.file_type"
_NAME_KEY = "general.name"

# Arch-prefixed metadata key suffix -> the lilbee field name it maps to. The
# prefix is the GGUF's general.architecture value (e.g. "qwen3.context_length").
_ARCH_FIELD_SUFFIXES: dict[str, str] = {
    "context_length": "context_length",
    "embedding_length": "embedding_length",
    "block_count": "block_count",
    "attention.head_count_kv": "head_count_kv",
    "attention.head_count": "head_count",
    "attention.key_length": "key_length",
    "attention.value_length": "value_length",
}


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


def read_gguf_metadata(model_path: Path) -> dict[str, str] | None:
    """Read header metadata from a GGUF file with the ``gguf`` parser.

    Returns a dict with keys like ``architecture``, ``context_length``,
    ``embedding_length``, ``chat_template``, ``file_type``, plus the
    KV-cache-shape fields (``block_count``, ``head_count_kv``,
    ``head_count``, ``key_length``, ``value_length``) used to size n_ctx
    against host memory. ``None`` when the file carries none of them.
    """
    reader = GGUFReader(str(model_path))
    fields = reader.fields
    result: dict[str, str] = {}

    arch = gguf_scalar_str(fields.get(GGUF_ARCH_KEY))
    if arch is not None:
        result["architecture"] = arch
    arch = arch or _DEFAULT_ARCH

    for suffix, out_key in _ARCH_FIELD_SUFFIXES.items():
        value = gguf_scalar_str(fields.get(f"{arch}.{suffix}"))
        if value is not None:
            result[out_key] = value
    for raw_key, out_key in (
        (_CHAT_TEMPLATE_KEY, "chat_template"),
        (_FILE_TYPE_KEY, "file_type"),
        (_NAME_KEY, "name"),
    ):
        value = gguf_scalar_str(fields.get(raw_key))
        if value is not None:
            result[out_key] = value
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
        provider="llama-server",
    )


def read_mmproj_projector_type(mmproj_path: Path) -> str | None:
    """Read ``clip.projector_type`` from a GGUF mmproj without loading the model."""
    try:
        reader = GGUFReader(str(mmproj_path))
        field = reader.get_field(_CLIP_PROJECTOR_TYPE_KEY)
    except Exception:
        log.debug("Failed to read mmproj metadata from %s", mmproj_path, exc_info=True)
        return None
    if field is None or not field.types or field.types[-1] != GGUFValueType.STRING:
        return None
    return bytes(field.parts[field.data[0]]).decode("utf-8", errors="replace")
