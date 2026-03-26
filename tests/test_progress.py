"""Tests for the progress callback protocol module."""

from lilbee.progress import (
    CrawlDoneEvent,
    CrawlErrorEvent,
    CrawlPageEvent,
    CrawlStartEvent,
    EventType,
    noop_callback,
)


class TestNoopCallback:
    def test_noop_accepts_any_event(self) -> None:
        noop_callback("file_start", {"file": "test.txt"})

    def test_noop_returns_none(self) -> None:
        result = noop_callback("embed", {"chunk": 1, "total_chunks": 5})
        assert result is None


class TestCrawlEventTypes:
    def test_crawl_event_types_exist(self) -> None:
        assert EventType.CRAWL_START == "crawl_start"
        assert EventType.CRAWL_PAGE == "crawl_page"
        assert EventType.CRAWL_DONE == "crawl_done"
        assert EventType.CRAWL_ERROR == "crawl_error"


class TestCrawlStartEvent:
    def test_creation(self) -> None:
        event = CrawlStartEvent(url="https://example.com", depth=2, max_pages=50)
        assert event.url == "https://example.com"
        assert event.depth == 2
        assert event.max_pages == 50


class TestCrawlPageEvent:
    def test_creation(self) -> None:
        event = CrawlPageEvent(url="https://example.com/p1", pages_crawled=3, pages_total=10)
        assert event.pages_crawled == 3
        assert event.pages_total == 10


class TestCrawlDoneEvent:
    def test_creation(self) -> None:
        event = CrawlDoneEvent(url="https://example.com", pages_crawled=5, files_written=5)
        assert event.files_written == 5


class TestCrawlErrorEvent:
    def test_creation(self) -> None:
        event = CrawlErrorEvent(url="https://example.com", error="Connection refused")
        assert event.error == "Connection refused"
