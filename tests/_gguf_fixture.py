"""Build minimal GGUF headers in-memory for probe tests."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from tests._stated_sizes import state_size

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


def cut_file_of(path: Path, declared_bytes: int, size: int) -> Path:
    """A one-tensor F32 GGUF that declares *declared_bytes* of data and states *size* bytes.

    Request the ``stated_sizes`` fixture first.
    """
    name = b"t0"
    infos = (
        struct.pack("<Q", len(name))
        + name
        + struct.pack("<I", 1)
        + struct.pack("<Q", declared_bytes // 4)
        + struct.pack("<I", 0)
        + struct.pack("<Q", 0)
    )
    head = GGUF_MAGIC + struct.pack("<IQQ", GGUF_VERSION, 1, 0) + infos
    path.write_bytes(head + b"\0" * (-len(head) % 32))
    state_size(path, size)
    return path


# Tests that hand real GGUF bytes to the parser carry this. The mapping from a
# parsed table onto lilbee's fields is covered against a stubbed parser, so a
# host with no parser still exercises it.
needs_gguf_parser = pytest.mark.usefixtures("real_gguf_parser")
