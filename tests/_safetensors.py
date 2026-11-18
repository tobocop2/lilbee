"""Whole safetensors files of a stated size, for tests that need a known weight size."""

from __future__ import annotations

import json

_PREFIX_BYTES = 8
_TENSOR_HEADER_BYTES = 128
"""A header padded to this width holds one tensor whose data fills the rest of the file."""


def safetensors_header(size: int) -> bytes:
    """The length prefix and padded JSON header of a whole safetensors file of *size* bytes.

    A file of 136 bytes or more holds one tensor spanning the data after the header; a
    shorter one holds an empty header padded to the stated size.
    """
    if size < _PREFIX_BYTES + len("{}"):
        raise ValueError(f"a safetensors file needs at least 10 bytes, not {size}")
    if size < _PREFIX_BYTES + _TENSOR_HEADER_BYTES:
        header_bytes = size - _PREFIX_BYTES
        body = b"{}"
    else:
        header_bytes = _TENSOR_HEADER_BYTES
        data = size - _PREFIX_BYTES - header_bytes
        tensor = {"dtype": "U8", "shape": [data], "data_offsets": [0, data]}
        body = json.dumps({"weights": tensor}, separators=(",", ":")).encode()
    return header_bytes.to_bytes(_PREFIX_BYTES, "little") + body.ljust(header_bytes, b" ")


def safetensors_bytes(size: int) -> bytes:
    """A whole safetensors file of exactly *size* bytes."""
    header = safetensors_header(size)
    return header + b"\0" * (size - len(header))
