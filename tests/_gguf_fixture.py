"""Build minimal GGUF headers in-memory for probe tests."""

from __future__ import annotations

import struct

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_TYPE_STRING = 8


def _string_kv(key: bytes, value: str) -> bytes:
    """One GGUF metadata entry holding a string value."""
    encoded = value.encode("utf-8")
    return (
        struct.pack("<Q", len(key))
        + key
        + struct.pack("<I", GGUF_TYPE_STRING)
        + struct.pack("<Q", len(encoded))
        + encoded
    )


def make_minimal_gguf(architecture: str, file_type: str | None = None) -> bytes:
    """A minimal valid GGUF header carrying general.architecture and optionally general.type."""
    entries = [_string_kv(b"general.architecture", architecture)]
    if file_type is not None:
        entries.append(_string_kv(b"general.type", file_type))
    head = GGUF_MAGIC + struct.pack("<I", GGUF_VERSION) + struct.pack("<Q", 0)
    return head + struct.pack("<Q", len(entries)) + b"".join(entries)
