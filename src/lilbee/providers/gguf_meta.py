"""GGUF metadata helpers: header reads, mmproj sidecar lookup, projector type.

Reads GGUF headers with the standalone ``gguf`` parser (no native binding), so
metadata is available to any provider without loading a model into the engine.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import NamedTuple, cast

from lilbee.catalog.header_probe import GGUF_ARCH_KEY
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
    # Embedding pooling the model was trained for; absent on most non-embedders.
    "pooling_type": "pooling_type",
    # Routed expert count; present only on MoE models, whose experts offload.
    "expert_count": "expert_count",
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


_METADATA_CACHE: dict[tuple[str, int, int], dict[str, str] | None] = {}
_METADATA_CACHE_LOCK = threading.Lock()

# Bump when the extracted field set changes, or when the reader does, so old
# entries are ignored rather than served stale. Version 2 reads through
# gguf-parser: entries written by version 1 recorded "no metadata" for files
# whose tensor type gguf-py did not know, and the key is (path, mtime, size),
# so an unchanged file would keep serving that answer after the upgrade.
_DISK_CACHE_VERSION = 2
_DISK_CACHE_DIRNAME = "gguf-meta"
# Distinguishes "no entry on disk" from a cached "this file has no metadata".
_DISK_MISS = object()


def _disk_cache_file(key: tuple[str, int, int]) -> Path | None:
    """Where *key*'s metadata is cached on disk, or None if no state dir works."""
    from lilbee.core.system import default_cache_dir

    digest = hashlib.sha256(
        "\0".join(str(part) for part in (_DISK_CACHE_VERSION, *key)).encode()
    ).hexdigest()
    try:
        # A cache dir, not the state dir: losing this costs a re-parse, never a
        # lost handle on a running engine.
        return default_cache_dir() / _DISK_CACHE_DIRNAME / f"{digest}.json"
    except OSError:  # pragma: no cover - unwritable/undiscoverable state dir
        return None


def _disk_cache_load(key: tuple[str, int, int]) -> object:
    """The cached metadata for *key*, or ``_DISK_MISS`` when not usable."""
    path = _disk_cache_file(key)
    if path is None:
        return _DISK_MISS
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _DISK_MISS
    if not isinstance(payload, dict) or "metadata" not in payload:
        return _DISK_MISS
    meta = payload["metadata"]
    if meta is None:
        return None
    if not isinstance(meta, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in meta.items()
    ):
        return _DISK_MISS
    return meta


def _disk_cache_store(key: tuple[str, int, int], result: dict[str, str] | None) -> None:
    """Persist *result* for *key*; best effort, a failure just costs a re-parse."""
    path = _disk_cache_file(key)
    if path is None:
        return
    with contextlib.suppress(OSError, TypeError, ValueError):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so a concurrent reader never sees a half-written file.
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"metadata": result}), encoding="utf-8")
        tmp.replace(path)


def read_gguf_metadata(model_path: Path) -> dict[str, str] | None:
    """Read header metadata from a GGUF file with the ``gguf`` parser.

    Cached by ``(path, mtime, size)`` in memory and on disk. ``GGUFReader`` parses
    the whole header -- every tensor descriptor and the large tokenizer arrays --
    to hand back a dozen scalar fields, which measured at ~60s for a 2.6GB model
    on a spinning-rust-era CPU. Planning reads the same model several times per
    fleet build, so the in-memory cache collapses those to one parse; the on-disk
    cache carries it across processes, which is what stops a relaunch paying that
    minute again while an already-warm engine sits idle waiting to be adopted.
    Keying on mtime and size means an edited or replaced file re-reads. Returns a
    copy so callers can't mutate the shared entry.
    """
    try:
        stat = model_path.stat()
        key: tuple[str, int, int] | None = (str(model_path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        key = None
    if key is not None:
        with _METADATA_CACHE_LOCK:
            if key in _METADATA_CACHE:
                cached = _METADATA_CACHE[key]
                return dict(cached) if cached is not None else None
        from_disk = _disk_cache_load(key)
        if from_disk is not _DISK_MISS:
            entry = cast("dict[str, str] | None", from_disk)
            with _METADATA_CACHE_LOCK:
                _METADATA_CACHE[key] = entry
            return dict(entry) if entry is not None else None
    result = _read_gguf_metadata_uncached(model_path)
    if key is not None:
        with _METADATA_CACHE_LOCK:
            _METADATA_CACHE[key] = result
        _disk_cache_store(key, result)
    return dict(result) if result is not None else None


# Array values report as ``{type, len, startOffset}`` rather than their contents,
# so a 151k-token vocabulary costs nothing to skip. Only scalars are read here.
_PARSER_ARRAY_VALUE_TYPE = 9
# GGUF string value type, in both readers' numbering.
_GGUF_STRING_VALUE_TYPE = 8
_PARSER_TIMEOUT_SECONDS = 60.0
_PARSER_KILL_WAIT_SECONDS = 5.0
_PARSER_LABEL = "gguf-parser"


class _Scalar(NamedTuple):
    """One scalar metadata value, and whether the file typed it as a string.

    A caller that wants text out of a field the file wrote as an integer is
    reading a malformed file, so the type travels with the value rather than
    being inferred from how the digits look.
    """

    text: str
    is_string: bool


def _kv_via_parser(model_path: Path) -> dict[str, _Scalar] | None:
    """Every scalar metadata key in *model_path*, read by the engine's own parser.

    Returns ``None`` when there is no parser to ask.

    gguf-parser is built from the pin that builds llama-server, so the tensor
    types it accepts are the ones the engine can decode. gguf-py carries its own
    table and trails llama.cpp: a Q2_0 file (tensor type 42) that the engine
    loads makes ``GGUFReader`` raise while it builds tensor descriptors nothing
    here reads, and every field in the file becomes unreadable with it.
    """
    from lilbee.providers.fleet.binary import resolve_gguf_parser
    from lilbee.providers.fleet.proc import run_bounded

    try:
        parser = resolve_gguf_parser()
    except Exception as exc:
        log.debug("No gguf-parser to read %s with: %s", model_path, exc)
        return None
    try:
        # merge_stderr stays off: the caller parses stdout as JSON.
        out, code = run_bounded(
            [str(parser), "--path", str(model_path), "--raw"],
            timeout_s=_PARSER_TIMEOUT_SECONDS,
            kill_wait_s=_PARSER_KILL_WAIT_SECONDS,
            label=_PARSER_LABEL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("gguf-parser did not run against %s: %s", model_path, exc)
        return None
    if code != 0:
        # The engine's own reader rejected the file. Report "no readable
        # metadata" rather than asking gguf-py for a second opinion the engine
        # would not honour anyway.
        log.warning("Could not parse GGUF metadata from %s: gguf-parser exit %d", model_path, code)
        return {}
    try:
        entries = json.loads(out)["header"]["metadataKV"]
    except (ValueError, KeyError, TypeError) as exc:
        log.warning("gguf-parser returned unreadable output for %s: %s", model_path, exc)
        return {}
    return {
        e["key"]: _Scalar(str(e["value"]), e.get("valueType") == _GGUF_STRING_VALUE_TYPE)
        for e in entries
        if e.get("valueType") != _PARSER_ARRAY_VALUE_TYPE and e.get("value") is not None
    }


def _scalars(model_path: Path) -> dict[str, _Scalar]:
    """Scalar metadata for *model_path*, or empty when there is no parser to ask.

    An install without the engine extra has no gguf-parser, and also no
    llama-server to load a model with, so the fields here have nothing left to
    size. Callers already treat absent metadata as a fallback case.
    """
    return _kv_via_parser(model_path) or {}


def _read_gguf_metadata_uncached(model_path: Path) -> dict[str, str] | None:
    kv = _scalars(model_path)
    result: dict[str, str] = {}

    arch_field = kv.get(GGUF_ARCH_KEY)
    if arch_field is not None:
        result["architecture"] = arch_field.text
    arch = arch_field.text if arch_field is not None else _DEFAULT_ARCH

    for suffix, out_key in _ARCH_FIELD_SUFFIXES.items():
        value = kv.get(f"{arch}.{suffix}")
        if value is not None:
            result[out_key] = value.text
    for raw_key, out_key in (
        (_CHAT_TEMPLATE_KEY, "chat_template"),
        (_FILE_TYPE_KEY, "file_type"),
        (_NAME_KEY, "name"),
    ):
        value = kv.get(raw_key)
        if value is not None:
            result[out_key] = value.text
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

    Resolution order: (1) the HuggingFace-cache ``snapshots/`` sibling of
    ``blobs/``, (2) same-directory glob for flat sideloaded layouts. Both look
    beside the model file itself, so a projector is never borrowed from another
    repo. Raises ``ProviderError`` if neither finds a file.
    """
    found = _find_mmproj_in_hf_snapshots(model_path.parent) or _find_mmproj_in_flat_dir(
        model_path.parent
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
    value = _scalars(mmproj_path).get(_CLIP_PROJECTOR_TYPE_KEY)
    return value.text if value is not None and value.is_string else None
