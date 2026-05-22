"""Build minimal GGUF headers in-memory for probe tests."""

from __future__ import annotations

import struct

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_TYPE_STRING = 8


def make_minimal_gguf(architecture: str) -> bytes:
    """Return a minimal valid GGUF header containing only general.architecture=<arch>."""
    buf = bytearray()
    buf += GGUF_MAGIC
    buf += struct.pack("<I", GGUF_VERSION)
    buf += struct.pack("<Q", 0)
    buf += struct.pack("<Q", 1)
    key = b"general.architecture"
    buf += struct.pack("<Q", len(key)) + key
    buf += struct.pack("<I", GGUF_TYPE_STRING)
    val = architecture.encode("utf-8")
    buf += struct.pack("<Q", len(val)) + val
    return bytes(buf)
