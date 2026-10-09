"""A stand-in crawler for the harness's tests: it fetches the seed and the pages it links."""

from __future__ import annotations

import html
import re
import sys
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urljoin, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "drivers"))

from _driver_io import (
    PageOut,
    RunOut,
    crawl_arguments,
    now,
    serve_conversions,
    thread_count,
    write_pages,
    write_run,
)

_LINK = re.compile(r'<a href="([^"#]+)"')
_BLOCK_END = re.compile(r"</(p|h1|h2|li|td|div)>")
_TAG = re.compile(r"<[^>]+>")


def to_markdown(source: str) -> str:
    """The text of *source*, one paragraph for each block element."""
    return html.unescape(_TAG.sub("", _BLOCK_END.sub("\n\n", source))).strip()


def convert_forever(drop_word: str = "") -> None:
    """Serve conversions on stdin and stdout; optionally drop a word from every answer."""
    serve_conversions(lambda source, _base_url: to_markdown(source).replace(drop_word, ""))


def fetch(url: str) -> tuple[int, str]:
    """Status and body text of *url*."""
    parts = urlsplit(url)
    connection = HTTPConnection(parts.hostname or "", parts.port, timeout=10)
    connection.request("GET", parts.path or "/")
    response = connection.getresponse()
    body = response.read().decode("utf-8", errors="replace")
    connection.close()
    return response.status, body


def crawl(drop_word: str = "", lose_path: str = "") -> int:
    """Crawl one level from the seed; optionally drop a word everywhere and lose one page."""
    args = crawl_arguments("fake crawler")
    record = RunOut(started=now(), versions={"fake": "1"}, threads_before=thread_count())
    record.crawl_started = now()
    _status, seed_body = fetch(args.seed)
    urls = [args.seed, *(urljoin(args.seed, link) for link in _LINK.findall(seed_body))]
    pages: list[PageOut] = []
    for url in dict.fromkeys(urls):
        status, body = fetch(url)
        markdown = to_markdown(body).replace(drop_word, "") if drop_word else to_markdown(body)
        if status != 200 or (lose_path and url.endswith(lose_path)):
            pages.append(PageOut(url, None, f"fake failure {status}"))
        else:
            pages.append(PageOut(url, markdown, saved_at=now()))
    record.crawl_ended = now()
    record.threads_after = thread_count()
    write_pages(args.out, pages)
    write_run(args.out, record)
    return 0
