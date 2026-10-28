"""The crawl events crawlberg streams, and lilbee's reading of them."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

from lilbee.crawler import url_filter
from lilbee.crawler.bootstrap import CrawlEngineRefusedError
from lilbee.crawler.messages import (
    BROWSER_FLAGS_REFUSED_MESSAGE,
    CRAWL_REFUSED_MESSAGE,
    EXCLUDE_PATTERN_REFUSED_MESSAGE,
    ROBOTS_REFUSED_MESSAGE,
    ROBOTS_UNREACHABLE_MESSAGE,
)
from lilbee.crawler.models import CancelToken

# crawlberg is the optional ``crawler`` extra and loads a native library, so each
# function that needs it imports it on call.
if TYPE_CHECKING:
    import crawlberg

_SEED_DEPTH = 0
_SSRF_ERROR_CODE = "ssrf_policy_violation"
# How crawlberg 1.12.3 starts its refusal of a setting, and the message that names that setting.
_REFUSED_SETTING_MESSAGES = {
    "invalid_config: browser.chrome_args": BROWSER_FLAGS_REFUSED_MESSAGE,
    "invalid_config: invalid exclude_path regex": EXCLUDE_PATTERN_REFUSED_MESSAGE,
}
# How crawlberg 1.12.3 starts the error of a page the site's robots.txt disallows, and of a
# page it refuses because the site's robots.txt could not be read.
_ROBOTS_DISALLOWED_START = "robots.txt disallows "
_ROBOTS_UNREACHABLE_START = "robots_unreachable: "


class _EventKind(StrEnum):
    """The ``type`` of a crawlberg crawl event lilbee reads; the end-of-crawl event is not one."""

    PAGE = "page"
    ERROR = "error"


_EVENT_KINDS = {kind.value: kind for kind in _EventKind}


@dataclass(frozen=True)
class _Event:
    """The fields lilbee reads from one crawlberg crawl event."""

    kind: _EventKind
    url: str
    depth: int = _SEED_DEPTH
    markdown: str = ""
    error: str = ""
    links: tuple[str, ...] = ()
    title: str = ""

    @property
    def refused_by_ssrf(self) -> bool:
        """True for an error event crawlberg raised because its SSRF policy refused the URL."""
        return self.kind is _EventKind.ERROR and self.error.startswith(_SSRF_ERROR_CODE)


def _refusal_message(reason: str) -> str:
    """crawlberg's *reason* for a refusal, with lilbee's setting named when the text tells it."""
    # crawlberg's error has no type or attribute for the field; only the start of its text names it.
    template = next(
        (
            message
            for start, message in _REFUSED_SETTING_MESSAGES.items()
            if reason.startswith(start)
        ),
        CRAWL_REFUSED_MESSAGE,
    )
    return template.format(reason=reason)


def _create_engine(config: crawlberg.CrawlConfig) -> crawlberg.CrawlEngineHandle:
    """The engine crawlberg builds for *config*.

    An exception of any type from ``crawlberg.create_engine`` is a
    :class:`CrawlEngineRefusedError` that carries crawlberg's reason. No other call is guarded.
    """
    import crawlberg

    try:
        return crawlberg.create_engine(config)
    except Exception as exc:
        raise CrawlEngineRefusedError(_refusal_message(str(exc))) from exc


def _error_text(error: str) -> str:
    """crawlberg's *error* for a page, in lilbee's words when robots.txt refused the page.

    A robots.txt that could not be read because the SSRF policy refuses its address is
    that SSRF refusal.
    """
    # crawlberg's error event has no code for a robots.txt refusal; only its text tells it.
    if error.startswith(_ROBOTS_DISALLOWED_START):
        return ROBOTS_REFUSED_MESSAGE
    if not error.startswith(_ROBOTS_UNREACHABLE_START):
        return error
    reason = error.removeprefix(_ROBOTS_UNREACHABLE_START)
    if reason.startswith(_SSRF_ERROR_CODE):
        return reason
    return ROBOTS_UNREACHABLE_MESSAGE.format(reason=reason)


def _parse_event(raw: object) -> _Event | None:
    """Read a crawlberg event, whose payload is only reachable as JSON; None for other kinds.

    Every address of the event is read without its username and password.
    """
    payload: dict[str, Any] = json.loads(str(raw))
    kind = _EVENT_KINDS.get(payload["type"])
    if kind is None:
        return None
    if kind is _EventKind.ERROR:
        url = url_filter.without_userinfo(payload["url"])
        return _Event(kind, url, error=_error_text(payload["error"]))
    result: dict[str, Any] = payload["result"]
    markdown: dict[str, Any] = result.get("markdown") or {}
    metadata: dict[str, Any] = result.get("metadata") or {}
    return _Event(
        kind,
        url_filter.without_userinfo(result["url"]),
        depth=result.get("depth", _SEED_DEPTH),
        markdown=markdown.get("content") or "",
        links=tuple(url_filter.without_userinfo(link["url"]) for link in result.get("links") or ()),
        title=metadata.get("title") or "",
    )


async def _events(
    config: crawlberg.CrawlConfig,
    seed_url: str,
    cancel: CancelToken | None,
) -> AsyncGenerator[_Event, None]:
    """Stream one crawl's page and error events until it completes or *cancel* is set.

    The engine is created and dropped inside this generator, so it never leaves the
    thread whose event loop runs the crawl.
    """
    import crawlberg

    engine = _create_engine(config)
    # crawl_stream is an async generator; its stub types it as an AsyncIterator.
    stream = cast(AsyncGenerator[object, None], crawlberg.crawl_stream(engine, seed_url))
    async with aclosing(stream):
        async for raw in stream:
            if cancel is not None and cancel.is_set():
                return
            event = _parse_event(raw)
            if event is not None:
                yield event
