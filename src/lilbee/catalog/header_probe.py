"""Range-GET a GGUF blob's header and extract general.architecture.

The architecture is read by walking the metadata KV table directly (see
``_parse_arch``) rather than via gguf-py's ``GGUFReader``, which needs the whole
KV table -- including a model's multi-megabyte tokenizer arrays -- present, and
so chokes on a Range-GET-truncated header.
"""

from __future__ import annotations

import logging
import struct
from http import HTTPStatus

import httpx
from gguf import GGUFValueType, ReaderField

log = logging.getLogger(__name__)

GGUF_HEADER_PROBE_BYTES = 65536
GGUF_MAGIC = b"GGUF"
GGUF_ARCH_KEY = "general.architecture"
_PROBE_TIMEOUT_S = 10.0


def probe_architecture(blob_url: str) -> str:
    """Return general.architecture from the GGUF header, or empty string on any failure."""
    try:
        headers = {"Range": f"bytes=0-{GGUF_HEADER_PROBE_BYTES - 1}"}
        # follow_redirects: a HuggingFace ``/resolve/`` URL for an LFS-backed GGUF
        # 302-redirects to the CDN; without following it the probe reads the tiny
        # redirect body, the GGUF magic check fails, and every arch verdict is
        # silently UNKNOWN (so the unsupported-arch guard never fires).
        resp = httpx.get(blob_url, headers=headers, timeout=_PROBE_TIMEOUT_S, follow_redirects=True)
        if resp.status_code >= HTTPStatus.BAD_REQUEST:
            return ""
        return _parse_arch(resp.content)
    except httpx.HTTPError as exc:
        log.debug("GGUF header probe failed for %s: %s", blob_url, exc)
        return ""


# GGUF metadata value-type tags (gguf spec) and the fixed byte sizes of the
# scalar ones. ARRAY/STRING carry their own length prefixes.
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
# magic(4) + version(4) + tensor_count(8) + kv_count(8)
_GGUF_HEADER_FIXED = 24


class _TruncatedHeaderError(Exception):
    """The probe window ended before the bytes a parse step needed."""


class _HeaderCursor:
    """Little-endian reader over the probed GGUF header bytes."""

    def __init__(self, blob: bytes) -> None:
        self._blob = blob
        self._pos = 0

    def take(self, n: int) -> bytes:
        end = self._pos + n
        if n < 0 or end > len(self._blob):
            raise _TruncatedHeaderError
        chunk = self._blob[self._pos : end]
        self._pos = end
        return chunk

    def u32(self) -> int:
        return int(struct.unpack_from("<I", self.take(4))[0])

    def u64(self) -> int:
        return int(struct.unpack_from("<Q", self.take(8))[0])

    def gguf_string(self) -> bytes:
        return self.take(self.u64())


def _skip_value(cur: _HeaderCursor, value_type: int) -> None:
    """Advance *cur* past one metadata value of *value_type*."""
    size = _GGUF_SCALAR_SIZES.get(value_type)
    if size is not None:
        cur.take(size)
        return
    if value_type == _GGUF_TYPE_STRING:
        cur.gguf_string()
        return
    if value_type == _GGUF_TYPE_ARRAY:
        elem_type = cur.u32()
        count = cur.u64()
        elem_size = _GGUF_SCALAR_SIZES.get(elem_type)
        if elem_size is not None:
            cur.take(elem_size * count)
        elif elem_type == _GGUF_TYPE_STRING:
            for _ in range(count):
                cur.gguf_string()
        else:
            raise _TruncatedHeaderError  # nested/unknown array element: stop parsing
        return
    raise _TruncatedHeaderError  # unknown value type: stop parsing


def _parse_arch(blob: bytes) -> str:
    """Extract general.architecture from a (possibly truncated) GGUF header.

    Walks the metadata KV table directly and returns as soon as it reaches
    ``general.architecture``, which GGUF writers emit among the first entries.
    This deliberately avoids gguf-py's ``GGUFReader``, which parses the entire KV
    table up front: a real model's multi-megabyte tokenizer arrays run past the
    Range-GET probe window, so ``GGUFReader`` raises on the truncation before any
    field can be read. Returns empty string on a malformed or too-short header.
    """
    if len(blob) < _GGUF_HEADER_FIXED or blob[:4] != GGUF_MAGIC:
        return ""
    cur = _HeaderCursor(blob)
    arch_key = GGUF_ARCH_KEY.encode("utf-8")
    try:
        cur.take(4)  # magic
        cur.u32()  # version
        cur.u64()  # tensor_count (the tensor-info table is never read)
        kv_count = cur.u64()
        for _ in range(kv_count):
            key = cur.gguf_string()
            value_type = cur.u32()
            if key == arch_key:
                if value_type != _GGUF_TYPE_STRING:
                    return ""
                return cur.gguf_string().decode("utf-8", errors="replace")
            _skip_value(cur, value_type)
    except _TruncatedHeaderError:
        return ""
    return ""


def gguf_scalar_str(field: ReaderField | None) -> str | None:
    """Render a scalar GGUF metadata field as a string, or ``None``.

    Mirrors what ``Llama(vocab_only=True).metadata`` produced: STRING fields
    decode their bytes; numeric scalars (UINT*/INT*/FLOAT*/BOOL) stringify
    their single value. ARRAY fields and empty fields return ``None`` (callers
    here only read scalars). This is the one place the gguf-py ReaderField shape
    is decoded, so a gguf-py layout change breaks here, not at every call site.
    """
    if field is None or not field.types or not field.data:
        return None
    value_type = field.types[-1]
    part = field.parts[field.data[0]]
    if value_type == GGUFValueType.STRING:
        return bytes(part).decode("utf-8", errors="replace")
    if value_type == GGUFValueType.ARRAY:
        return None
    scalar = part.tolist() if hasattr(part, "tolist") else part
    if isinstance(scalar, (list, tuple)):
        scalar = scalar[0] if scalar else None
    return None if scalar is None else str(scalar)
