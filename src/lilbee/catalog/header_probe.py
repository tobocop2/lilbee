"""Range-GET a GGUF blob's header and extract general.architecture via gguf-py."""

from __future__ import annotations

import contextlib
import logging
import struct
import tempfile
from http import HTTPStatus
from pathlib import Path

import httpx
from gguf import GGUFReader, GGUFValueType

log = logging.getLogger(__name__)

GGUF_HEADER_PROBE_BYTES = 65536
GGUF_MAGIC = b"GGUF"
GGUF_ARCH_KEY = "general.architecture"
_PROBE_TIMEOUT_S = 10.0
_TENSOR_COUNT_OFFSET = 8
_TENSOR_COUNT_SIZE = 8


def probe_architecture(blob_url: str) -> str:
    """Return general.architecture from the GGUF header, or empty string on any failure."""
    try:
        headers = {"Range": f"bytes=0-{GGUF_HEADER_PROBE_BYTES - 1}"}
        resp = httpx.get(blob_url, headers=headers, timeout=_PROBE_TIMEOUT_S)
        if resp.status_code >= HTTPStatus.BAD_REQUEST:
            return ""
        return _parse_arch(resp.content)
    except httpx.HTTPError as exc:
        log.debug("GGUF header probe failed for %s: %s", blob_url, exc)
        return ""


def _parse_arch(blob: bytes) -> str:
    """Extract general.architecture from a (possibly truncated) GGUF header.

    Patches the tensor_count field to zero so ``gguf-py``'s ``GGUFReader``
    skips the tensor info table that would otherwise require the full
    file. Only the KV table needs to be intact for architecture lookup.
    """
    if len(blob) < _TENSOR_COUNT_OFFSET + _TENSOR_COUNT_SIZE or blob[:4] != GGUF_MAGIC:
        return ""
    patched = bytearray(blob)
    patched[_TENSOR_COUNT_OFFSET : _TENSOR_COUNT_OFFSET + _TENSOR_COUNT_SIZE] = struct.pack("<Q", 0)
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(bytes(patched))
        tmp = Path(f.name)
    try:
        return _read_architecture(tmp)
    except (ValueError, struct.error, IndexError, OSError, UnicodeDecodeError) as exc:
        log.debug("GGUFReader parse failed: %s", exc)
        return ""
    finally:
        # GGUFReader memory-maps the file; on Windows the mapping must be released
        # before the file can be deleted (a held mapping raises PermissionError /
        # WinError 32, which missing_ok does not cover). _read_architecture scopes
        # the reader so it is freed on return; suppress is a best-effort backstop.
        with contextlib.suppress(OSError):
            tmp.unlink()


def _read_architecture(path: Path) -> str:
    """Return general.architecture from the GGUF file at *path*, or empty string.

    Kept separate so the ``GGUFReader`` (and its memory map of *path*) is released
    when this returns, letting the caller delete the temp file on Windows.
    """
    reader = GGUFReader(str(path))
    field = reader.fields.get(GGUF_ARCH_KEY)
    if field is None or not field.data:
        return ""
    if not field.types or field.types[-1] != GGUFValueType.STRING:
        return ""
    return bytes(field.parts[field.data[0]]).decode("utf-8", errors="replace")
