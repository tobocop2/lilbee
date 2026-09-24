"""A stand-in ``crawlberg`` module that replays scripted crawl events.

The crawler extra is absent from the unit-test environment, so the adapter's
``import crawlberg`` resolves to this module through ``inject_modules``.
"""

from __future__ import annotations

import json
import types
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import Any

from tests._sys_modules import inject_modules

Payload = dict[str, Any]
Script = Callable[[], AsyncIterator[Payload]]


def page(url: str, markdown: str | None = "# Page", *, depth: int = 1) -> Payload:
    """A ``page`` event payload; ``markdown=None`` is a page with no markdown output."""
    content = None if markdown is None else {"content": markdown}
    return {"type": "page", "result": {"url": url, "depth": depth, "markdown": content}}


def error(url: str, message: str) -> Payload:
    """An ``error`` event payload."""
    return {"type": "error", "url": url, "error": message}


def complete(pages: int) -> Payload:
    """The end-of-crawl event payload."""
    return {"type": "complete", "pages_crawled": pages}


class StubEvent:
    """A crawl event whose payload, like crawlberg's, is reachable only as JSON."""

    def __init__(self, payload: Payload) -> None:
        self.type = payload["type"]
        self._payload = payload

    def __str__(self) -> str:
        return json.dumps(self._payload)


class Recorded:
    """A config object that keeps the keyword arguments it was built with."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _listed(payloads: Iterable[Payload]) -> Script:
    async def replay() -> AsyncIterator[Payload]:
        for payload in payloads:
            yield payload

    return replay


class StubCrawlberg:
    """Builds the stand-in module and records every engine config, seed and close."""

    def __init__(self, script: Iterable[Payload] | Script = ()) -> None:
        self._script: Script = script if callable(script) else _listed(list(script))
        self.configs: list[Recorded] = []
        self.seeds: list[str] = []
        self.closed = 0
        self.module = self._build()

    def _build(self) -> types.ModuleType:
        module = types.ModuleType("crawlberg")
        module.CrawlConfig = Recorded  # type: ignore[attr-defined]
        module.ContentConfig = Recorded  # type: ignore[attr-defined]
        module.BrowserConfig = Recorded  # type: ignore[attr-defined]
        module.SsrfPolicy = Recorded  # type: ignore[attr-defined]
        module.HostMatcher = types.SimpleNamespace(cidr=lambda value: ("cidr", value))  # type: ignore[attr-defined]
        module.create_engine = self._create_engine  # type: ignore[attr-defined]
        module.crawl_stream = self._crawl_stream  # type: ignore[attr-defined]
        return module

    def _create_engine(self, config: Recorded) -> object:
        self.configs.append(config)
        return object()

    async def _crawl_stream(self, engine: object, url: str) -> AsyncIterator[StubEvent]:
        self.seeds.append(url)
        try:
            async for payload in self._script():
                yield StubEvent(payload)
        finally:
            self.closed += 1

    @property
    def config(self) -> dict[str, Any]:
        """The keyword arguments of the last engine config."""
        return self.configs[-1].kwargs

    @contextmanager
    def installed(self) -> Iterator[StubCrawlberg]:
        """Make ``import crawlberg`` return the stand-in module."""
        with inject_modules({"crawlberg": self.module}):
            yield self
