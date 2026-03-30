"""Tests for the progress callback protocol module."""

from lilbee.progress import (
    BatchProgressEvent,
    CrawlDoneEvent,
    CrawlPageEvent,
    CrawlStartEvent,
    EmbedEvent,
    EventType,
    ExtractEvent,
    FileDoneEvent,
    FileStartEvent,
    SseEvent,
    SyncDoneEvent,
    noop_callback,
)


class TestNoopCallback:
    def test_noop_accepts_model_event(self) -> None:
        ev = FileStartEvent(file="test.txt", total_files=1, current_file=1)
        noop_callback(EventType.FILE_START, ev)

    def test_noop_returns_none(self) -> None:
        result = noop_callback(EventType.EMBED, EmbedEvent(file="x", chunk=1, total_chunks=5))
        assert result is None


class TestEventTypes:
    def test_core_event_types_exist(self) -> None:
        assert EventType.FILE_START == "file_start"
        assert EventType.FILE_DONE == "file_done"
        assert EventType.BATCH_PROGRESS == "batch_progress"
        assert EventType.DONE == "done"
        assert EventType.EMBED == "embed"
        assert EventType.EXTRACT == "extract"

    def test_crawl_event_types(self) -> None:
        assert EventType.CRAWL_START == "crawl_start"
        assert EventType.CRAWL_PAGE == "crawl_page"
        assert EventType.CRAWL_DONE == "crawl_done"

    def test_crawl_error_removed(self) -> None:
        assert not hasattr(EventType, "CRAWL_ERROR")


class TestSseEvent:
    def test_sse_event_values(self) -> None:
        assert SseEvent.TOKEN == "token"
        assert SseEvent.REASONING == "reasoning"
        assert SseEvent.SOURCES == "sources"
        assert SseEvent.ERROR == "error"
        assert SseEvent.DONE == "done"
        assert SseEvent.PROGRESS == "progress"

    def test_sse_event_is_str(self) -> None:
        assert isinstance(SseEvent.TOKEN, str)


class TestEventModels:
    def test_file_start_event(self) -> None:
        ev = FileStartEvent(file="doc.pdf", total_files=10, current_file=3)
        assert ev.file == "doc.pdf"
        assert ev.total_files == 10
        assert ev.current_file == 3

    def test_file_done_event(self) -> None:
        ev = FileDoneEvent(file="doc.pdf", status="ok", chunks=5)
        assert ev.file == "doc.pdf"
        assert ev.status == "ok"
        assert ev.chunks == 5

    def test_batch_progress_event(self) -> None:
        ev = BatchProgressEvent(file="a.txt", status="ingested", current=3, total=10)
        assert ev.current == 3
        assert ev.total == 10

    def test_extract_event(self) -> None:
        ev = ExtractEvent(file="scan.pdf", page=2, total_pages=8)
        assert ev.page == 2
        assert ev.total_pages == 8

    def test_embed_event(self) -> None:
        ev = EmbedEvent(file="notes.md", chunk=5, total_chunks=20)
        assert ev.file == "notes.md"
        assert ev.chunk == 5
        assert ev.total_chunks == 20

    def test_crawl_start_event(self) -> None:
        ev = CrawlStartEvent(url="https://example.com", depth=2)
        assert ev.url == "https://example.com"
        assert ev.depth == 2

    def test_crawl_page_event(self) -> None:
        ev = CrawlPageEvent(url="https://example.com/page", current=3, total=10)
        assert ev.url == "https://example.com/page"
        assert ev.current == 3
        assert ev.total == 10

    def test_crawl_done_event(self) -> None:
        ev = CrawlDoneEvent(pages_crawled=5, files_written=3)
        assert ev.pages_crawled == 5
        assert ev.files_written == 3

    def test_sync_done_event(self) -> None:
        ev = SyncDoneEvent(added=2, updated=1, removed=0, failed=0)
        assert ev.added == 2
        assert ev.updated == 1

    def test_model_dump_roundtrip(self) -> None:
        ev = EmbedEvent(file="f.txt", chunk=1, total_chunks=10)
        dumped = ev.model_dump()
        assert dumped == {"file": "f.txt", "chunk": 1, "total_chunks": 10}
        restored = EmbedEvent(**dumped)
        assert restored == ev
