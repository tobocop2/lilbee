"""GGUF metadata helpers: header reads, mmproj sidecar lookup, projector type."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from gguf import GGUFReader, GGUFValueType

from lilbee.providers.base import ProviderError
from lilbee.providers.llama_cpp.abort_signal import abort_callback, clear_abort
from lilbee.providers.llama_cpp.log_dispatch import (
    import_llama_cpp,
    install_llama_log_handler,
    suppress_native_stderr,
)

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
    """Resolve ``<arch>.context_length`` from GGUF metadata, with a safe floor.

    Several published GGUFs (nomic-embed, some Qwen3 variants, certain
    vision models) report ``context_length=0`` in their headers. Passing
    zero into ``Llama(n_ctx=...)`` for an embedding or vision loader
    cascades into ``n_batch=0`` / ``n_ubatch=0`` and trips ggml's Vulkan
    dispatch into undefined behaviour, surfacing as STATUS_HEAP_CORRUPTION
    on Windows. Unparseable values (any non-integer string) raised
    ``ValueError`` in three independent call sites before this helper
    consolidated them.

    Args:
        meta: GGUF metadata dict from :func:`read_gguf_metadata`, or
            ``None`` when the read itself failed.
        fallback: Value to return when metadata is missing, unparseable,
            or non-positive. Each loader picks the value appropriate for
            its task (chat / embed / vision).
        model_path: Used purely for the warning message so the user knows
            which model produced the junk metadata.
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
    """Read metadata from a GGUF file's headers via llama-cpp-python.

    Returns a dict with keys like ``architecture``, ``context_length``,
    ``embedding_length``, ``chat_template``, ``file_type``, plus the
    KV-cache-shape fields (``block_count``, ``head_count_kv``,
    ``head_count``, ``key_length``, ``value_length``) used to size n_ctx
    against host memory.
    """
    Llama = import_llama_cpp().Llama  # noqa: N806

    # Fresh abort flag: a prior request_abort() must not latch and break
    # this metadata read, which is on the path of every model swap.
    clear_abort()
    install_llama_log_handler()
    kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "vocab_only": True,
        "verbose": False,
        "n_gpu_layers": 0,
    }
    kwargs.setdefault("abort_callback", abort_callback)
    llm = suppress_native_stderr(Llama, **kwargs)
    try:
        raw = llm.metadata or {}
        result: dict[str, str] = {}
        if "general.architecture" in raw:
            result["architecture"] = str(raw["general.architecture"])
        arch = raw.get("general.architecture", "llama")
        ctx_key = f"{arch}.context_length"
        if ctx_key in raw:
            result["context_length"] = str(raw[ctx_key])
        emb_key = f"{arch}.embedding_length"
        if emb_key in raw:
            result["embedding_length"] = str(raw[emb_key])
        for arch_key, out_key in (
            (f"{arch}.block_count", "block_count"),
            (f"{arch}.attention.head_count_kv", "head_count_kv"),
            (f"{arch}.attention.head_count", "head_count"),
            (f"{arch}.attention.key_length", "key_length"),
            (f"{arch}.attention.value_length", "value_length"),
        ):
            if arch_key in raw:
                result[out_key] = str(raw[arch_key])
        if "tokenizer.chat_template" in raw:
            result["chat_template"] = str(raw["tokenizer.chat_template"])
        if "general.file_type" in raw:
            result["file_type"] = str(raw["general.file_type"])
        if "general.name" in raw:
            result["name"] = str(raw["general.name"])
        return result or None
    finally:
        llm.close()


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
        reader = GGUFReader(str(mmproj_path))
        field = reader.get_field(_CLIP_PROJECTOR_TYPE_KEY)
    except Exception:
        log.debug("Failed to read mmproj metadata from %s", mmproj_path, exc_info=True)
        return None
    if field is None or field.types[-1] != GGUFValueType.STRING:
        return None
    return bytes(field.parts[field.data[0]]).decode("utf-8", errors="replace")
