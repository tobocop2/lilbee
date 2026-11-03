"""Integration test: layout detection keeps column order and spaces in stored text.

Runs the real xberg call on a generated two-column PDF that holds a ruled table,
with the extraction config lilbee builds when layout detection is on. The layout
and table models load on first use, so the test is marked slow like the other
tests that need a model.

Run with:
    uv run pytest tests/integration/test_layout_extraction_integration.py -v -m slow
"""

from __future__ import annotations

import pytest

from lilbee.core.config import cfg
from lilbee.data.ingest import ExtractMode, extraction_config

pytestmark = pytest.mark.slow


def _text_object(x: int, y: int, text: str, size: int = 10) -> str:
    escaped = text.replace("(", "\\(").replace(")", "\\)")
    return f"BT /F1 {size} Tf {x} {y} Td ({escaped}) Tj ET\n"


def _page_stream() -> bytes:
    parts = [_text_object(72, 750, "Two column retrieval paper", 16)]
    y = 710
    for i in range(20):
        parts.append(_text_object(72, y, f"LEFT line {i} the corpus embedding query ranking."))
        y -= 14
    y = 710
    for i in range(20):
        parts.append(_text_object(330, y, f"RIGHT line {i} the index latency precision recall."))
        y -= 14
    cols = [72, 200, 330, 460]
    rows = [
        ["Model", "Recall", "Precision", "Latency"],
        ["alpha", "0.91", "0.88", "12ms"],
        ["beta", "0.85", "0.80", "9ms"],
    ]
    y = 400
    top = y + 12
    for row in rows:
        for x, cell in zip(cols, row, strict=True):
            parts.append(_text_object(x, y, cell, 11))
        y -= 18
    rules = [top - 18 * k + 4 for k in range(4)]
    for rule_y in rules:
        parts.append(f"68 {rule_y} m 560 {rule_y} l S\n")
    for rule_x in [*cols, 560]:
        parts.append(f"{rule_x - 4} {rules[0]} m {rule_x - 4} {rules[-1]} l S\n")
    return "".join(parts).encode()


def _build_pdf() -> bytes:
    stream = _page_stream()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
        b" /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(out)


def test_layout_extraction_keeps_column_order_and_table_spaces(monkeypatch, run_async):
    """The stored text reads the left column first and keeps the spaces in a table row."""
    from xberg import ExtractInput, ExtractInputKind, extract

    monkeypatch.setattr(cfg, "layout_detection", True)
    config = extraction_config(ExtractMode.PAGINATED)
    document = run_async(
        extract(
            ExtractInput(
                kind=ExtractInputKind.BYTES,
                bytes=_build_pdf(),
                mime_type="application/pdf",
                filename="two-column.pdf",
            ),
            config,
        )
    ).results[0]

    stored = "\n".join(chunk.content for chunk in document.chunks or [])
    assert stored.index("LEFT line 19") < stored.index("RIGHT line 0")
    assert "Precision Latency" in stored
    assert "0.88 12ms" in stored
    assert "PrecisionLatency" not in stored
    assert "paperLEFT" not in stored
