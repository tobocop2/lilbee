"""Wire-protocol message kinds shared by every worker subprocess.

Every per-role worker (embed, chat, rerank, vision) and the parent-side
``transport_pipe`` channel send/receive ``(kind, payload)`` tuples. The
``kind`` strings live here so a typo in one module (``"PING"`` vs
``"ping"``) becomes an :class:`AttributeError` instead of a silent
protocol mismatch that surfaces only when the worker actually crashes.

``WireKind`` is a :class:`enum.StrEnum`, so its members compare equal to
the underlying string and cross the pipe as plain ``str`` without the
parent and child needing matched class identity. ``test_wire_kinds.py``
enforces single-source-of-truth via a ``grep`` against literal
redefinitions in worker modules.
"""

from __future__ import annotations

from enum import StrEnum


class WireKind(StrEnum):
    """One enum for every kind that can appear on the worker wire."""

    # Generic envelope kinds.
    PING = "ping"
    PONG = "pong"
    SHUTDOWN = "shutdown"
    ACK = "ack"
    RESULT = "result"
    ERROR = "error"

    # Streaming kinds (chat).
    STREAM_CHUNK = "stream_chunk"
    STREAM_END = "stream_end"

    # Per-role request kinds.
    EMBED = "embed"
    CHAT = "chat"
    RERANK = "rerank"
    VISION = "vision_ocr"
    PDF_OCR = "pdf_ocr"


__all__ = ["WireKind"]
