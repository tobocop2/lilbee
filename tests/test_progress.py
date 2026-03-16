"""Tests for the progress callback protocol module."""

from lilbee.progress import noop_callback


class TestNoopCallback:
    def test_noop_accepts_any_event(self) -> None:
        noop_callback("file_start", {"file": "test.txt"})

    def test_noop_returns_none(self) -> None:
        result = noop_callback("embed", {"chunk": 1, "total_chunks": 5})
        assert result is None
