"""Tests for the PDF-extract subprocess entry point and parent wrapper."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

from lilbee.runtime import pdf_extract


def test_emit_progress_writes_to_stderr(capsys) -> None:
    pdf_extract._emit_progress(3, 7)
    captured = capsys.readouterr()
    assert "progress: page=3 total=7" in captured.err


def test_on_progress_emits_for_batch_event(capsys) -> None:
    from lilbee.runtime.progress import BatchProgressEvent, EventType

    pdf_extract._on_progress(
        EventType.BATCH_PROGRESS,
        BatchProgressEvent(file="x", status="rasterizing", current=2, total=5),
    )
    assert "progress: page=2 total=5" in capsys.readouterr().err


def test_on_progress_ignores_non_batch_events(capsys) -> None:
    from lilbee.runtime.progress import EventType

    pdf_extract._on_progress(EventType.SETUP_PROGRESS, object())
    assert capsys.readouterr().err == ""


def test_main_invalid_args_returns_error(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("not json"))
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    rc = pdf_extract.main()
    assert rc == 1
    out = sys.stdout.getvalue()
    assert "error" in json.loads(out)


def test_main_missing_required_field(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"path": "x"})))
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    rc = pdf_extract.main()
    assert rc == 1


def test_main_happy_path(monkeypatch) -> None:
    args = {
        "path": "/tmp/x.pdf",
        "vision_model": "model",
        "timeout": 5.0,
        "quiet": True,
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(args)))
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    with mock.patch(
        "lilbee.vision.extract_pdf_vision",
        return_value=[(1, "page 1 text"), (2, "page 2 text")],
    ):
        rc = pdf_extract.main()
    assert rc == 0
    payload = json.loads(sys.stdout.getvalue())
    assert payload["page_texts"] == [[1, "page 1 text"], [2, "page 2 text"]]


def test_main_extraction_error_returns_json_error(monkeypatch) -> None:
    args = {"path": "/tmp/x.pdf", "vision_model": "m", "timeout": None, "quiet": True}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(args)))
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    with mock.patch("lilbee.vision.extract_pdf_vision", side_effect=RuntimeError("boom")):
        rc = pdf_extract.main()
    assert rc == 1
    assert "boom" in json.loads(sys.stdout.getvalue())["error"]


class TestParentWrapper:
    """Tests for ``_extract_pdf_vision_in_subprocess`` and ``_pump_pdf_progress``."""

    async def test_pump_progress_parses_lines(self) -> None:
        from lilbee.data.ingest.extract import _pump_pdf_progress
        from lilbee.runtime.progress import EventType

        captured: list[tuple[str, object]] = []

        def on_progress(event_type: object, data: object) -> None:
            captured.append((event_type, data))

        class FakeReader:
            def __init__(self, lines: list[bytes]) -> None:
                self._lines = list(lines)

            async def readline(self) -> bytes:
                if not self._lines:
                    return b""
                return self._lines.pop(0)

        reader = FakeReader(
            [
                b"progress: page=1 total=10\n",
                b"progress: page=5 total=10\n",
                b"unrelated stderr line\n",
                b"progress: oops bad\n",
                b"",
            ]
        )
        await _pump_pdf_progress(reader, on_progress, Path("/tmp/x.pdf"))
        events = [c for c in captured if c[0] == EventType.BATCH_PROGRESS]
        assert len(events) == 2
        assert events[0][1].current == 1
        assert events[1][1].current == 5

    async def test_subprocess_wrapper_returns_page_texts(self) -> None:
        from lilbee.data.ingest.extract import _extract_pdf_vision_in_subprocess

        async def _noop(*_a, **_kw) -> None:
            return None

        class FakeProc:
            stdin = mock.MagicMock()
            stdout = mock.AsyncMock()
            stderr = mock.AsyncMock()
            returncode = 0

            async def communicate(self) -> tuple[bytes, bytes]:
                payload = json.dumps({"page_texts": [[1, "p1"], [2, "p2"]]}).encode()
                return payload, b""

            async def wait(self) -> int:
                return 0

            def kill(self) -> None: ...

        FakeProc.stdin.drain = mock.AsyncMock()
        FakeProc.stdin.write = mock.MagicMock()
        FakeProc.stdin.close = mock.MagicMock()

        with (
            mock.patch(
                "asyncio.create_subprocess_exec", new=mock.AsyncMock(return_value=FakeProc())
            ),
            mock.patch(
                "lilbee.data.ingest.extract._pump_pdf_progress",
                new=mock.AsyncMock(return_value=None),
            ),
        ):
            result = await _extract_pdf_vision_in_subprocess(
                Path("/tmp/x.pdf"),
                "model",
                timeout=5.0,
                quiet=True,
                on_progress=lambda *_: None,
            )
        assert result == [(1, "p1"), (2, "p2")]

    async def test_subprocess_wrapper_raises_on_error_payload(self) -> None:
        from lilbee.data.ingest.extract import _extract_pdf_vision_in_subprocess

        class FakeProc:
            stdin = mock.MagicMock()
            stdout = mock.AsyncMock()
            stderr = mock.AsyncMock()
            returncode = 1

            async def communicate(self) -> tuple[bytes, bytes]:
                return json.dumps({"error": "kaboom"}).encode(), b""

            async def wait(self) -> int:
                return 0

            def kill(self) -> None: ...

        FakeProc.stdin.drain = mock.AsyncMock()
        FakeProc.stdin.write = mock.MagicMock()
        FakeProc.stdin.close = mock.MagicMock()

        with (
            mock.patch(
                "asyncio.create_subprocess_exec", new=mock.AsyncMock(return_value=FakeProc())
            ),
            mock.patch(
                "lilbee.data.ingest.extract._pump_pdf_progress",
                new=mock.AsyncMock(return_value=None),
            ),
            pytest.raises(RuntimeError, match="kaboom"),
        ):
            await _extract_pdf_vision_in_subprocess(
                Path("/tmp/x.pdf"),
                "model",
                timeout=5.0,
                quiet=True,
                on_progress=lambda *_: None,
            )

    async def test_subprocess_wrapper_raises_on_invalid_json(self) -> None:
        from lilbee.data.ingest.extract import _extract_pdf_vision_in_subprocess

        class FakeProc:
            stdin = mock.MagicMock()
            stdout = mock.AsyncMock()
            stderr = mock.AsyncMock()
            returncode = 0

            async def communicate(self) -> tuple[bytes, bytes]:
                return b"not json", b""

            async def wait(self) -> int:
                return 0

            def kill(self) -> None: ...

        FakeProc.stdin.drain = mock.AsyncMock()
        FakeProc.stdin.write = mock.MagicMock()
        FakeProc.stdin.close = mock.MagicMock()

        with (
            mock.patch(
                "asyncio.create_subprocess_exec", new=mock.AsyncMock(return_value=FakeProc())
            ),
            mock.patch(
                "lilbee.data.ingest.extract._pump_pdf_progress",
                new=mock.AsyncMock(return_value=None),
            ),
            pytest.raises(RuntimeError, match="invalid JSON"),
        ):
            await _extract_pdf_vision_in_subprocess(
                Path("/tmp/x.pdf"),
                "model",
                timeout=5.0,
                quiet=True,
                on_progress=lambda *_: None,
            )

    async def test_subprocess_wrapper_kills_on_timeout(self) -> None:
        import asyncio as _asyncio

        from lilbee.data.ingest.extract import _extract_pdf_vision_in_subprocess

        kill_calls: list[bool] = []

        class FakeProc:
            stdin = mock.MagicMock()
            stdout = mock.AsyncMock()
            stderr = mock.AsyncMock()
            returncode = -9

            async def communicate(self) -> tuple[bytes, bytes]:
                raise TimeoutError

            async def wait(self) -> int:
                return -9

            def kill(self) -> None:
                kill_calls.append(True)

        FakeProc.stdin.drain = mock.AsyncMock()
        FakeProc.stdin.write = mock.MagicMock()
        FakeProc.stdin.close = mock.MagicMock()

        with (
            mock.patch(
                "asyncio.create_subprocess_exec", new=mock.AsyncMock(return_value=FakeProc())
            ),
            mock.patch(
                "lilbee.data.ingest.extract._pump_pdf_progress",
                new=mock.AsyncMock(return_value=None),
            ),
            mock.patch.object(
                _asyncio,
                "wait_for",
                new=mock.AsyncMock(side_effect=TimeoutError),
            ),
            pytest.raises(RuntimeError, match="timed out"),
        ):
            await _extract_pdf_vision_in_subprocess(
                Path("/tmp/x.pdf"),
                "model",
                timeout=5.0,
                quiet=True,
                on_progress=lambda *_: None,
            )
        assert kill_calls == [True]
