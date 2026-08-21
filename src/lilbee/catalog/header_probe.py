"""Range-GET a GGUF blob's header and read the fields that identify the file.

The fields are read by walking the metadata KV table directly (see
``_parse_header``) rather than via gguf-py's ``GGUFReader``, which needs the whole
KV table -- including a model's multi-megabyte tokenizer arrays -- present, and
so chokes on a Range-GET-truncated header.

The bundled gguf-parser reports the same two fields and is the authority on
whether the engine can load a file, but it always reads through to the tensor
table, which sits past those tokenizer arrays. Measured on the same files it
costs 6.7s against this probe's 0.7s for a small model. Selection asks about
every candidate and needs only ``general.type``, the second key in the table, so
it reads a 64 KB window here; the load verdict is left to gguf-parser, once, in
``providers.fleet.loadability``.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from http import HTTPStatus

import httpx
from gguf import GGUFValueType, ReaderField

log = logging.getLogger(__name__)

GGUF_HEADER_PROBE_BYTES = 65536
GGUF_MAGIC = b"GGUF"
GGUF_ARCH_KEY = "general.architecture"
GGUF_TYPE_KEY = "general.type"

# ``general.type`` values that name something other than loadable model weights.
# A file that omits the key predates it and counts as a model, so the check only
# rejects a file whose header says outright that it is not one.
NON_MODEL_GGUF_TYPES = frozenset({"mmproj", "adapter"})

_PROBE_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class GgufHeader:
    """What a GGUF file says it is. Empty fields mean the header was unreadable."""

    architecture: str = ""
    file_type: str = ""

    @property
    def is_model(self) -> bool:
        """True unless the header identifies the file as a projector or adapter."""
        return self.file_type not in NON_MODEL_GGUF_TYPES


def probe_header(blob_url: str) -> GgufHeader:
    """Read a GGUF blob's identifying header fields, empty on any failure."""
    try:
        headers = {"Range": f"bytes=0-{GGUF_HEADER_PROBE_BYTES - 1}"}
        # follow_redirects: a HuggingFace ``/resolve/`` URL for an LFS-backed GGUF
        # 302-redirects to the CDN; without following it the probe reads the tiny
        # redirect body, the GGUF magic check fails, and every arch verdict is
        # silently UNKNOWN (so the unsupported-arch guard never fires).
        resp = httpx.get(blob_url, headers=headers, timeout=_PROBE_TIMEOUT_S, follow_redirects=True)
        if resp.status_code >= HTTPStatus.BAD_REQUEST:
            return GgufHeader()
        return _parse_header(resp.content)
    except httpx.HTTPError as exc:
        log.debug("GGUF header probe failed for %s: %s", blob_url, exc)
        return GgufHeader()


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


_WANTED_STRING_KEYS: dict[bytes, str] = {
    GGUF_ARCH_KEY.encode("utf-8"): "architecture",
    GGUF_TYPE_KEY.encode("utf-8"): "file_type",
}
_GGUF_MAGIC_LEN = 4


def _collect_string_fields(cur: _HeaderCursor, kv_count: int) -> dict[str, str]:
    """Walk the KV table and return the ``_WANTED_STRING_KEYS`` values it carries.

    Keeps what it read when the probe window ends mid-table, so a header whose
    wanted keys precede a huge tokenizer array still resolves.
    """
    found: dict[str, str] = {}
    try:
        for _ in range(kv_count):
            key = cur.gguf_string()
            value_type = cur.u32()
            field = _WANTED_STRING_KEYS.get(key)
            if field is not None and value_type == _GGUF_TYPE_STRING:
                found[field] = cur.gguf_string().decode("utf-8", errors="replace")
                if len(found) == len(_WANTED_STRING_KEYS):
                    break
            else:
                _skip_value(cur, value_type)
    except _TruncatedHeaderError:
        pass
    return found


def _parse_header(blob: bytes) -> GgufHeader:
    """Extract the identifying fields from a (possibly truncated) GGUF header.

    Walks the metadata KV table directly and stops once every wanted key is read;
    GGUF writers emit ``general.architecture`` and ``general.type`` among the
    first entries. This deliberately avoids gguf-py's ``GGUFReader``, which parses
    the entire KV table up front: a real model's multi-megabyte tokenizer arrays
    run past the Range-GET probe window, so ``GGUFReader`` raises on the
    truncation before any field can be read. A malformed or too-short header
    yields whatever was read before the truncation.
    """
    if len(blob) < _GGUF_HEADER_FIXED or blob[:_GGUF_MAGIC_LEN] != GGUF_MAGIC:
        return GgufHeader()
    # The length guard above covers exactly these reads, so none of them can
    # run off the end; only the KV walk that follows can truncate.
    cur = _HeaderCursor(blob)
    cur.take(_GGUF_MAGIC_LEN)
    cur.u32()  # version
    cur.u64()  # tensor_count (the tensor-info table is never read)
    return GgufHeader(**_collect_string_fields(cur, cur.u64()))


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
