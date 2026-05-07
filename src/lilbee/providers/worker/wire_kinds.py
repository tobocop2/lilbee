"""Wire-protocol message kinds shared by every worker subprocess.

Every per-role worker (embed, chat, rerank, vision) and the parent-side
``transport_pipe`` channel send/receive ``(kind, payload)`` tuples. The
``kind`` strings live here so a typo in one module (``"PING"`` vs
``"ping"``) becomes an :class:`ImportError` instead of a silent protocol
mismatch that surfaces only when the worker actually crashes.

Module-level constants (not an :class:`enum.Enum`) so the values cross
the pipe as plain ``str`` without the parent and child both having to
import the enum class. ``test_wire_kinds.py`` enforces single-source-of-
truth via a ``grep`` against literal redefinitions.
"""

from __future__ import annotations

# Generic envelope kinds.
PING_KIND = "ping"
PONG_KIND = "pong"
SHUTDOWN_KIND = "shutdown"
ACK_KIND = "ack"
RESULT_KIND = "result"
ERROR_KIND = "error"

# Streaming kinds (chat).
STREAM_CHUNK_KIND = "stream_chunk"
STREAM_END_KIND = "stream_end"

# Per-role request kinds.
EMBED_KIND = "embed"
CHAT_KIND = "chat"
RERANK_KIND = "rerank"
VISION_KIND = "vision_ocr"
PDF_OCR_KIND = "pdf_ocr"


__all__ = [
    "ACK_KIND",
    "CHAT_KIND",
    "EMBED_KIND",
    "ERROR_KIND",
    "PDF_OCR_KIND",
    "PING_KIND",
    "PONG_KIND",
    "RERANK_KIND",
    "RESULT_KIND",
    "SHUTDOWN_KIND",
    "STREAM_CHUNK_KIND",
    "STREAM_END_KIND",
    "VISION_KIND",
]
