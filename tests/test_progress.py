"""Tests for the progress callback protocol module."""

from lilbee.progress import (
    EventType,
    noop_callback,
)


class TestNoopCallback:
    def test_noop_accepts_any_event(self) -> None:
        noop_callback("file_start", {"file": "test.txt"})

    def test_noop_returns_none(self) -> None:
        result = noop_callback("embed", {"chunk": 1, "total_chunks": 5})
        assert result is None


class TestEventTypes:
    def test_core_event_types_exist(self) -> None:
        assert EventType.FILE_START == "file_start"
        assert EventType.FILE_DONE == "file_done"
        assert EventType.BATCH_PROGRESS == "batch_progress"
        assert EventType.DONE == "done"
        assert EventType.EMBED == "embed"
        assert EventType.EXTRACT == "extract"
