"""Range-GET a GGUF blob's header and extract general.architecture."""

from __future__ import annotations

import logging
import struct
from http import HTTPStatus

import httpx

log = logging.getLogger(__name__)

GGUF_HEADER_PROBE_BYTES = 65536
GGUF_MAGIC = b"GGUF"
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_ARCH_KEY = "general.architecture"
_PROBE_TIMEOUT_S = 10.0


_FIXED_SIZE_BY_TYPE: dict[int, int] = {
    0: 1,
    1: 1,
    2: 2,
    3: 2,
    4: 4,
    5: 4,
    6: 4,
    7: 1,
    10: 8,
    11: 8,
    12: 8,
}


def probe_architecture(blob_url: str) -> str:
    """Return general.architecture from the GGUF header, or empty string on any failure."""
    try:
        headers = {"Range": f"bytes=0-{GGUF_HEADER_PROBE_BYTES - 1}"}
        resp = httpx.get(blob_url, headers=headers, timeout=_PROBE_TIMEOUT_S)
        if resp.status_code >= HTTPStatus.BAD_REQUEST:
            return ""
        return _parse_arch(resp.content)
    except (httpx.HTTPError, struct.error, UnicodeDecodeError) as exc:
        log.debug("GGUF header probe failed for %s: %s", blob_url, exc)
        return ""


def _parse_arch(blob: bytes) -> str:
    cursor = 0
    if blob[cursor : cursor + 4] != GGUF_MAGIC:
        return ""
    cursor += 4
    _version = _read_u32(blob, cursor)
    cursor += 4
    _tensor_count = _read_u64(blob, cursor)
    cursor += 8
    kv_count = _read_u64(blob, cursor)
    cursor += 8

    for _ in range(kv_count):
        key, cursor = _read_string(blob, cursor)
        value_type = _read_u32(blob, cursor)
        cursor += 4
        if key == GGUF_ARCH_KEY:
            if value_type != GGUF_TYPE_STRING:
                return ""
            value, _ = _read_string(blob, cursor)
            return value
        cursor = _skip_value(blob, cursor, value_type)
    return ""


def _read_u32(blob: bytes, cursor: int) -> int:
    return struct.unpack_from("<I", blob, cursor)[0]


def _read_u64(blob: bytes, cursor: int) -> int:
    return struct.unpack_from("<Q", blob, cursor)[0]


def _read_string(blob: bytes, cursor: int) -> tuple[str, int]:
    length = _read_u64(blob, cursor)
    cursor += 8
    value = blob[cursor : cursor + length].decode("utf-8")
    return value, cursor + length


def _skip_value(blob: bytes, cursor: int, value_type: int) -> int:
    fixed = _FIXED_SIZE_BY_TYPE.get(value_type)
    if fixed is not None:
        return cursor + fixed
    if value_type == GGUF_TYPE_STRING:
        _, cursor = _read_string(blob, cursor)
        return cursor
    if value_type == GGUF_TYPE_ARRAY:
        elem_type = _read_u32(blob, cursor)
        cursor += 4
        count = _read_u64(blob, cursor)
        cursor += 8
        for _ in range(count):
            cursor = _skip_value(blob, cursor, elem_type)
        return cursor
    raise struct.error(f"unknown gguf value type {value_type}")
