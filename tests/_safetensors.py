"""Whole safetensors files of a stated size, for tests that need a known weight size."""

from __future__ import annotations

import json
from collections.abc import Mapping

_PREFIX_BYTES = 8
_TENSOR_HEADER_BYTES = 128
"""A header padded to this width holds one tensor whose data fills the rest of the file."""


def safetensors_header(size: int, dtype: str = "U8") -> bytes:
    """The length prefix and padded JSON header of a whole safetensors file of *size* bytes.

    A file of 136 bytes or more holds one tensor of *dtype* spanning the data after the
    header; a shorter one holds an empty header padded to the stated size.
    """
    if size < _PREFIX_BYTES + len("{}"):
        raise ValueError(f"a safetensors file needs at least 10 bytes, not {size}")
    if size < _PREFIX_BYTES + _TENSOR_HEADER_BYTES:
        header_bytes = size - _PREFIX_BYTES
        body = b"{}"
    else:
        header_bytes = _TENSOR_HEADER_BYTES
        data = size - _PREFIX_BYTES - header_bytes
        tensor = {"dtype": dtype, "shape": [data], "data_offsets": [0, data]}
        body = json.dumps({"weights": tensor}, separators=(",", ":")).encode()
    return header_bytes.to_bytes(_PREFIX_BYTES, "little") + body.ljust(header_bytes, b" ")


def safetensors_bytes(size: int, dtype: str = "U8") -> bytes:
    """A whole safetensors file of exactly *size* bytes, its one tensor of *dtype*."""
    header = safetensors_header(size, dtype)
    return header + b"\0" * (size - len(header))


def safetensors_mixed_bytes(data_bytes: Mapping[str, int]) -> bytes:
    """A whole safetensors file with one tensor per dtype, each holding the stated data bytes.

    The tensors follow each other in the data, in the order of *data_bytes*.
    """
    tensors: dict[str, object] = {}
    start = 0
    for dtype, count in data_bytes.items():
        end = start + count
        tensors[f"{dtype.lower()}_weights"] = {
            "dtype": dtype,
            "shape": [count],
            "data_offsets": [start, end],
        }
        start = end
    body = json.dumps(tensors, separators=(",", ":")).encode()
    return len(body).to_bytes(_PREFIX_BYTES, "little") + body + b"\0" * start
