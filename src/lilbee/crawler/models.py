"""Backend-agnostic value types crossing the runner/fetcher seam."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from lilbee.core.config.defaults import MAX_CRAWL_DEPTH, MAX_CRAWL_PAGES
from lilbee.core.config.model import CONTROL_CHARACTER
from lilbee.runtime.cancellation import CancelSignal

# Explicit "no page limit" for a crawl. Distinct from None, which means
# "unspecified, use the protective default cfg.crawl_safety_max_pages".
CRAWL_PAGES_UNLIMITED = 0

AUTHORIZATION_HEADER = "Authorization"
COOKIE_HEADER = "Cookie"
# The login fields sent as raw header values; crawlberg and httpx send ASCII only.
_HEADER_FIELDS = ("authorization", "cookie")


def crawl_limit_error(depth: int | None, max_pages: int | None) -> str | None:
    """The message for a per-crawl depth or page count outside the shared bounds, else None."""
    if depth is not None and not 0 <= depth <= MAX_CRAWL_DEPTH:
        return f"depth must be between 0 and {MAX_CRAWL_DEPTH} (omit it to crawl the whole site)"
    if max_pages is not None and not 0 <= max_pages <= MAX_CRAWL_PAGES:
        return f"max_pages must be between 0 and {MAX_CRAWL_PAGES} (0 = unlimited)"
    return None


@dataclass
class CrawlResult:
    """Outcome of crawling a single URL.

    This is the high-level result surfaced to lilbee callers
    (CLI, MCP, HTTP, TUI). The adapter produces ``FetchedPage``
    and the orchestration layer converts it to ``CrawlResult``
    when returning up to the caller.
    """

    url: str
    markdown: str = ""
    success: bool = True
    error: str | None = None
    title: str = ""

    def failure_reason(self) -> str | None:
        """Why this page has nothing to save, or None when it does.

        The save layer skips exactly these pages, so the event layer reports
        failures from the same predicate: the two cannot drift apart.
        """
        if not self.success:
            return self.error or "fetch failed"
        if not self.markdown.strip():
            return "page produced empty markdown"
        return None


@dataclass
class FetchedPage:
    """Single page produced by a ``WebFetcher`` backend.

    Distinct from :class:`CrawlResult` so the adapter surface
    stays narrow and neutral: just the bytes we needed out of
    the underlying SDK's response object.
    """

    url: str
    markdown: str = ""
    success: bool = True
    error: str | None = None
    links: list[str] = field(default_factory=list)
    title: str = ""


@dataclass(frozen=True)
class CrawlLogin:
    """The login one crawl sends to its seed host; no field shows in a repr.

    An empty field is unset. Raises ``ValueError`` naming the field, never its value.
    """

    authorization: str = field(default="", repr=False)
    username: str = field(default="", repr=False)
    password: str = field(default="", repr=False)
    cookie: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        values = {
            "authorization": self.authorization,
            "username": self.username,
            "password": self.password,
            "cookie": self.cookie,
        }
        for name, value in values.items():
            if CONTROL_CHARACTER.search(value):
                raise ValueError(f"the crawl login {name} holds a control character")
        for name in _HEADER_FIELDS:
            if not values[name].isascii():
                raise ValueError(
                    f"the crawl login {name} holds a character outside ASCII, "
                    "which an HTTP header cannot carry"
                )
        if self.password and not self.username:
            raise ValueError("the crawl login password needs a username")
        if self.authorization and self.username:
            raise ValueError("give the crawl login an authorization header or a username, not both")

    @property
    def basic_credentials(self) -> tuple[str, str] | None:
        """The username and password, or None when no username is set."""
        return (self.username, self.password) if self.username else None

    @property
    def cookie_header(self) -> dict[str, str]:
        """The Cookie request header, or no header when no cookie is set."""
        return {COOKIE_HEADER: self.cookie} if self.cookie else {}

    @property
    def authorization_header(self) -> dict[str, str]:
        """The Authorization request header, or no header when none is set."""
        return {AUTHORIZATION_HEADER: self.authorization} if self.authorization else {}


NO_LOGIN = CrawlLogin()


@dataclass
class ConcurrencySpec:
    """Backend-agnostic concurrency, pacing and retry settings for one crawl."""

    semaphore_count: int = 1
    mean_delay: float = 0.0
    max_delay_range: float = 0.0
    retry_on_rate_limit: bool = False
    retry_base_delay_min: float = 0.0
    retry_base_delay_max: float = 0.0
    retry_max_backoff: float = 0.0
    retry_max_attempts: int = 0


@dataclass
class FilterSpec:
    """Backend-agnostic filter settings applied to discovered links.

    Pure Python data; each adapter decides how to plug the settings
    into its own filter pipeline.
    """

    exclude_patterns: list[str] = field(default_factory=list)
    include_subdomains: bool = False


CancelToken: TypeAlias = CancelSignal
"""Cancellation handle the orchestration layer passes to a fetcher.

A token whose ``is_set()`` is True means "stop as soon as you can". A
fetcher checks it between pages.
"""
