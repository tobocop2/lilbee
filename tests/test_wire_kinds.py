"""Single-source-of-truth check for worker wire-protocol message kinds.

Every per-role worker (embed, chat, rerank, vision) and the parent-side
``transport_pipe`` channel exchange ``(kind, payload)`` tuples. The
``kind`` strings live in :mod:`lilbee.providers.worker.wire_kinds` so a
typo in one module does not silently break the protocol.

These tests enforce two invariants:

1. The string values themselves are stable. A rename in
   ``wire_kinds.py`` has to be reflected on both ends of the pipe at
   the same time (worker + parent), and that has to ride on a
   deliberate change to this test file.
2. No worker module redefines the constants as bare ``"ping"`` /
   ``"pong"`` / ... literals. A drift like ``"PING"`` in one module
   only would otherwise pass type-check and surface as a silent
   protocol mismatch at runtime.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lilbee.providers.worker import wire_kinds

_WIRE_KINDS_PATH = Path(wire_kinds.__file__).resolve()
_WORKER_DIR = _WIRE_KINDS_PATH.parent
_WORKER_MODULES = (
    "embed_worker.py",
    "chat_worker.py",
    "rerank_worker.py",
    "vision_worker.py",
    "transport_pipe.py",
)


_EXPECTED_VALUES = {
    "PING": "ping",
    "PONG": "pong",
    "SHUTDOWN": "shutdown",
    "ACK": "ack",
    "RESULT": "result",
    "ERROR": "error",
    "STREAM_CHUNK": "stream_chunk",
    "STREAM_END": "stream_end",
    "EMBED": "embed",
    "CHAT": "chat",
    "RERANK": "rerank",
    "VISION": "vision_ocr",
    "PDF_OCR": "pdf_ocr",
}


@pytest.mark.parametrize(("name", "value"), list(_EXPECTED_VALUES.items()))
def test_wire_kind_values_are_stable(name: str, value: str) -> None:
    assert wire_kinds.WireKind[name] == value


def test_no_worker_module_redefines_kinds_as_literals() -> None:
    """No worker file may bind ``_..._KIND = "..."`` itself.

    The single allowed definition site is :mod:`wire_kinds`; everywhere
    else must import. A grep-style assertion is intentional: the wire
    kinds are bare strings on the pipe, and a stray local constant of
    the same name with a typo would not be caught by mypy.
    """
    pattern = re.compile(r"^_?[A-Z_]+_KIND\s*=\s*[\"']", re.MULTILINE)
    offenders: list[str] = []
    for module_name in _WORKER_MODULES:
        text = (_WORKER_DIR / module_name).read_text()
        for line in pattern.findall(text):
            offenders.append(f"{module_name}: {line.strip()}")
    assert not offenders, (
        "Worker modules must import wire-kind constants from "
        "lilbee.providers.worker.wire_kinds, not redefine them. Offenders:\n" + "\n".join(offenders)
    )
