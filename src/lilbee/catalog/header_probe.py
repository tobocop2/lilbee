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
import math
import os
import re
import struct
from dataclasses import dataclass
from functools import lru_cache
from http import HTTPStatus
from pathlib import Path
from typing import BinaryIO

import httpx

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


class _UnsupportedLayoutError(_TruncatedHeaderError):
    """The header holds a value type or layout this reader cannot walk."""


class _FileView:
    """Bounded reads of an open file, sliced like bytes; a short read is a truncation."""

    def __init__(self, handle: BinaryIO, size: int) -> None:
        self._handle = handle
        self._size = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, span: slice) -> bytes:
        self._handle.seek(span.start)
        chunk = self._handle.read(span.stop - span.start)
        if len(chunk) != span.stop - span.start:
            raise _TruncatedHeaderError
        return chunk


class _HeaderCursor:
    """Little-endian reader over the probed GGUF header bytes."""

    def __init__(self, blob: bytes | _FileView) -> None:
        self._blob = blob
        self._pos = 0

    @property
    def position(self) -> int:
        return self._pos

    def skip(self, n: int) -> None:
        """Move past *n* bytes without reading them; the bytes must lie inside the blob."""
        end = self._pos + n
        if n < 0 or end > len(self._blob):
            raise _TruncatedHeaderError
        self._pos = end

    def take(self, n: int) -> bytes:
        start = self._pos
        self.skip(n)
        return self._blob[start : self._pos]

    def skip_string(self) -> None:
        self.skip(self.u64())

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
        cur.skip(size)
        return
    if value_type == _GGUF_TYPE_STRING:
        cur.skip_string()
        return
    if value_type == _GGUF_TYPE_ARRAY:
        elem_type = cur.u32()
        count = cur.u64()
        elem_size = _GGUF_SCALAR_SIZES.get(elem_type)
        if elem_size is not None:
            cur.skip(elem_size * count)
        elif elem_type == _GGUF_TYPE_STRING:
            for _ in range(count):
                cur.skip_string()
        else:
            raise _UnsupportedLayoutError  # nested/unknown array element: stop parsing
        return
    raise _UnsupportedLayoutError  # unknown value type: stop parsing


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
    cur.skip(_GGUF_MAGIC_LEN)
    cur.u32()  # version
    cur.u64()  # tensor_count (the tensor-info table is never read)
    return GgufHeader(**_collect_string_fields(cur, cur.u64()))


_GGUF_ALIGNMENT_KEY = b"general.alignment"
_GGUF_TYPE_UINT32 = 4
_GGUF_DEFAULT_ALIGNMENT = 32
_GGUF_SIZED_VERSIONS = (2, 3)
_GGUF_CUT_CACHE_SIZE = 64
# llama.cpp split-GGUF shard naming ("%s-%05d-of-%05d.gguf").
SPLIT_GGUF_NAME = re.compile(r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$")


def _tensor_bytes(dims: list[int], tensor_type: int) -> int:
    """The bytes one tensor of *dims* and ggml *tensor_type* holds."""
    from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

    try:
        block, type_size = GGML_QUANT_SIZES[GGMLQuantizationType(tensor_type)]
    except (ValueError, KeyError) as exc:
        raise _UnsupportedLayoutError from exc
    return math.prod(dims) * type_size // block


def _declared_extent(blob: _FileView) -> int:
    """The file size a GGUF header declares: tensor data ends at the last tensor's end.

    Raises ``_TruncatedHeaderError`` when the bytes end inside the header, and
    ``_UnsupportedLayoutError`` when the header holds something this reader cannot size.
    """
    cur = _HeaderCursor(blob)
    cur.skip(_GGUF_MAGIC_LEN)
    if cur.u32() not in _GGUF_SIZED_VERSIONS:
        raise _UnsupportedLayoutError
    tensor_count = cur.u64()
    alignment = _GGUF_DEFAULT_ALIGNMENT
    for _ in range(cur.u64()):
        key = cur.gguf_string()
        value_type = cur.u32()
        if key == _GGUF_ALIGNMENT_KEY and value_type == _GGUF_TYPE_UINT32:
            alignment = cur.u32()
        else:
            _skip_value(cur, value_type)
    data_end = 0
    for _ in range(tensor_count):
        cur.skip_string()
        dims = [cur.u64() for _ in range(cur.u32())]
        tensor_type = cur.u32()
        offset = cur.u64()
        data_end = max(data_end, offset + _tensor_bytes(dims, tensor_type))
    if tensor_count == 0:
        return cur.position
    if alignment <= 0:
        raise _UnsupportedLayoutError
    return -(-cur.position // alignment) * alignment + data_end


def gguf_is_cut_off(path: Path) -> bool:
    """Whether *path* is a GGUF file whose header declares more bytes than the file holds.

    A file that is not a GGUF, is shorter than the fixed header, or has a header this reader
    cannot size, is not judged cut off.
    """
    try:
        stat = path.stat()
    except OSError:
        return False
    return _cut_off(str(path), stat.st_size, stat.st_mtime_ns)


def _present_shards(path: Path, match: re.Match[str]) -> dict[int, Path]:
    """The shards of the split set *path* names that sit beside it, by shard number."""
    return {
        int(shard["index"]): sibling
        for sibling in path.parent.iterdir()
        if (shard := SPLIT_GGUF_NAME.fullmatch(sibling.name)) is not None
        and shard["prefix"] == match["prefix"]
        and shard["total"] == match["total"]
    }


def gguf_is_incomplete(path: Path) -> bool:
    """Whether *path*, or any shard of the split set it names, is cut off or missing."""
    match = SPLIT_GGUF_NAME.fullmatch(path.name)
    if match is None:
        return gguf_is_cut_off(path)
    try:
        shards = _present_shards(path, match)
    except OSError:
        return False
    return shards.keys() != set(range(1, int(match["total"]) + 1)) or any(
        gguf_is_cut_off(sibling) for sibling in shards.values()
    )


def gguf_bytes_on_disk(path: Path) -> int:
    """The bytes *path* holds, or all the shards of its split set that sit beside it."""
    match = SPLIT_GGUF_NAME.fullmatch(path.name)
    try:
        files = [path] if match is None else list(_present_shards(path, match).values())
        return sum(file.stat().st_size for file in files)
    except OSError:
        return 0


@lru_cache(maxsize=_GGUF_CUT_CACHE_SIZE)
def _cut_off(path_str: str, size: int, _mtime_ns: int) -> bool:
    if size < _GGUF_HEADER_FIXED:
        return False
    try:
        with open(path_str, "rb") as handle:
            if handle.read(_GGUF_MAGIC_LEN) != GGUF_MAGIC:
                return False
            view = _FileView(handle, os.fstat(handle.fileno()).st_size)
            try:
                return _declared_extent(view) > size
            except _UnsupportedLayoutError:
                return False
            except _TruncatedHeaderError:
                return True
    except OSError:
        return False
