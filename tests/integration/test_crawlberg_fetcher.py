"""Real crawlberg against a local site: SSRF, cancel, threads, limits, excludes, retries, Chrome."""

from __future__ import annotations

import asyncio
import base64
import datetime
import hashlib
import ipaddress
import json
import logging
import os
import re
import socket
import ssl
import stat
import threading
import time
from collections.abc import Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import click
import pytest
from click.testing import Result
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from litestar.testing import TestClient
from typer.testing import CliRunner

crawlberg = pytest.importorskip("crawlberg")

import lilbee.server.auth as auth_mod  # noqa: E402
from lilbee.cli.app import app  # noqa: E402
from lilbee.cli.commands.serve_logging import setup_server_log_file  # noqa: E402
from lilbee.core.config import Config, cfg  # noqa: E402
from lilbee.core.config.enums import CrawlRenderMode  # noqa: E402
from lilbee.core.web_names import page_stem  # noqa: E402
from lilbee.crawler import CrawlMeta, bootstrap, crawl_and_save, save, url_filter  # noqa: E402
from lilbee.crawler.bootstrap import CrawlEngineRefusedError  # noqa: E402
from lilbee.crawler.crawlberg_fetcher import (  # noqa: E402
    _SSRF_ERROR_CODE,
    CRAWLBERG_DENIED_NETWORKS,
    ROBOTS_REFUSED_MESSAGE,
    CrawlbergFetcher,
    _queued_url,
    admitted_networks,
)
from lilbee.crawler.models import ConcurrencySpec, FetchedPage, FilterSpec  # noqa: E402
from lilbee.crawler.task import CrawlTask, TaskStatus, get_task, run_crawl  # noqa: E402
from lilbee.mcp_server import add as mcp_add  # noqa: E402
from lilbee.mcp_server import crawl as mcp_crawl  # noqa: E402
from lilbee.runtime.progress import CrawlPageFailedEvent, EventType  # noqa: E402
from lilbee.server.app import create_app  # noqa: E402
from tests._crawl_meta import write_crawl_metadata  # noqa: E402
from tests._private_mode import posix_only  # noqa: E402
from tests._windows_loop import windows_proactor_loop  # noqa: E402
from tests.integration._crawl_site import (  # noqa: E402
    REAL_BROWSERS_PATH,
    full_crawl_only,
    require_chromium,
)

LOOPBACK = (ipaddress.ip_network("127.0.0.0/8"), ipaddress.ip_network("::1/128"))
BROWSER = CrawlRenderMode.BROWSER
HTTP = CrawlRenderMode.HTTP
# lilbee blocks the whole NAT64 prefix; crawlberg refuses a NAT64 address only when the
# IPv4 address it carries is refused.
NAT64 = ipaddress.ip_network("64:ff9b::/96")
# IPv6 literals that carry the private IPv4 address 10.0.0.1 or loopback.
IPV6_FORMS_OF_PRIVATE_IPV4 = {
    "6to4": "[2002:7f00:1::]",
    "ipv4-translated": "[::ffff:0:10.0.0.1]",
    "ipv4-compatible": "[::10.0.0.1]",
    "nat64": "[64:ff9b::a00:1]",
    "local-use-nat64": "[64:ff9b:1::a00:1]",
}
# 6to4 of the public address 8.8.8.8.
IPV6_FORM_OF_PUBLIC_IPV4 = "[2002:808:808::]"
SLOW_PAGES = 40
SLOW_DELAY_S = 0.3
CANCEL_BOUND_S = 2.0
# A request sent just before the crawl stops can reach the site just after it.
LATE_START_MARGIN_S = 0.1
SETTLE_S = 1.0
RETRY_ATTEMPTS = 2
# The most retries crawlberg 1.11.1 accepts.
CRAWLBERG_RETRY_LIMIT = 20
RETRY_DELAY_S = 0.05
RETRY_STATUSES = {
    "/retry/broken": HTTPStatus.INTERNAL_SERVER_ERROR,
    "/retry/busy": HTTPStatus.SERVICE_UNAVAILABLE,
}
# A page that answers its first request with 429 and every later one with the page.
RATE_LIMITED_ONCE = "/retry/once"
# A page a default crawl_exclude_patterns entry matches.
EXCLUDED_SEED = "/wiki/Special:Random"
ROOT_PATTERN = r":\d+/$"
ONE_PAGE_LIMITS = {"depth-0": {"depth": 0}, "one-page": {"depth": 1, "max_pages": 1}}
SITE_CRAWL_LIMITS = {
    "two-pages": {"depth": 1, "max_pages": 2},
    "no-page-limit": {"depth": 1, "max_pages": 0},
    "no-depth-limit": {"max_pages": 2},
}
SCOPE_LINKS = ("/scope/skip/a", "/scope/keep/b", "/scope/la/drop", "/scope/la/keepme")
# Links as a page writes them: a raw letter, a raw space, one already percent-encoded, a
# backslash, a raw letter before an escaped slash and before a slash, two control characters,
# a space that ends a segment, a colon and a Windows device name.
SPELL_LINKS = (
    "/spell/naïve",
    "/spell/raw space",
    "/spell/caf%C3%A9",
    "/spell/back\\slash",
    "/spell/naïve%2Fs",
    "/spell/naïve/s",
    "/spell/a\x01b",
    "/spell/e\x1b]0",
    "/spell/tail /x",
    "/spell/Special:Random",
    "/spell/con",
)
# The same addresses as crawlberg reports and requests them.
SPELL_REQUESTS = (
    "/spell/na%C3%AFve",
    "/spell/raw%20space",
    "/spell/caf%C3%A9",
    "/spell/back/slash",
    "/spell/na%C3%AFve%2Fs",
    "/spell/na%C3%AFve/s",
    "/spell/a%01b",
    "/spell/e%1B]0",
    "/spell/tail%20/x",
    "/spell/Special:Random",
    "/spell/con",
)
# The readable names lilbee saves them under: letters read as written, each a legal Windows name.
SPELL_FOLDERS = (
    "spell/naïve",
    "spell/raw-space",
    "spell/café",
    "spell/back/slash",
    "spell/naïve-s",
    "spell/naïve/s",
    "spell/a-b",
    "spell/e-0",
    "spell/tail/x",
    "spell/Special-Random",
    "spell/_con",
)
# Pairs of links a server answers with two pages: a re-crawl keeps each pair as two pages.
PAIR_LINKS = (
    ("/pair/ca^ret", "/pair/ca%5Eret"),
    ("/pair/pi|pe", "/pair/pi%7Cpe"),
    ("/pair/a+b", "/pair/a%20b"),
    ("/pair/e%2Fs", "/pair/e/s"),
    ("/pair/Case", "/pair/case"),
)
# Two listings that link one page twice, raw and encoded, in opposite orders.
ORDER_LISTINGS = {
    "/twice/": ("/twice/naïve", "/twice/na%C3%AFve"),
    "/twice/again": ("/twice/na%C3%AFve", "/twice/naïve"),
}
# A link with an escaped hyphen, and the equivalent address a library holds for it.
EQUIVALENT_LINK = "/same/x%2Dy"
EQUIVALENT_STORED = "/same/x-y"
# How the text of a stored copy starts before a crawl, and how a served page names its request.
STALE_TEXT = "stale text"
FRESH_TEXT = "Served"
# The links each listing page of written addresses holds.
LINK_LISTINGS = {
    "/spell/": SPELL_LINKS,
    "/pair/": tuple(link for pair in PAIR_LINKS for link in pair),
    "/same/": (EQUIVALENT_LINK,),
    **ORDER_LISTINGS,
}
# Flag lists crawlberg refuses, each with a part of the reason it gives.
REFUSED_FLAGS: tuple[tuple[list[str], str], ...] = (
    (["--headless=new"], "--headless"),
    (["--user-data-dir=/x"], "--user-data-dir"),
    (["disable-gpu"], "must start with --"),
    (["--Lang=fr"], "must name the flag in lowercase"),
    (["--disable-gpu", "--disable-gpu"], "duplicates an earlier flag"),
)
DEFAULT_FLAGS = ["--disable-dev-shm-usage", "--disable-gpu"]
# Patterns Python's ``re`` compiles, so lilbee stores them, and crawlberg refuses. The last
# one holds the name crawlberg gives the launch flags.
REFUSED_PATTERNS = (
    "(?P<n>a)(?(n)b|c)",
    "(?#comment)a",
    "(?P<n>browser.chrome_args)(?(n)b|c)",
)
FLAG_SETTING = "The crawl_browser_extra_args setting holds a launch flag that crawlberg refuses: "
PATTERN_SETTING = "The crawl_exclude_patterns setting holds a pattern that crawlberg refuses: "
NO_SETTING = "crawlberg refuses to start this crawl: "
# A timeout in seconds whose milliseconds pass crawlberg's integer width, and what it raises.
OVERSIZED_TIMEOUT = 10**17
OVERSIZED_REASON = "int too big to convert"


def _force_oversized_timeout() -> None:
    """Set the timeout past its bound, which no entry point lets through, to reach the engine."""
    object.__setattr__(cfg, "crawl_timeout", OVERSIZED_TIMEOUT)


# How the user agent of crawlberg's own HTTP client starts.
CRAWLBERG_USER_AGENT_START = "crawlberg/"
# The request a recursive crawl makes for its progress total, and how its client names itself.
SITEMAP_PATH = "/sitemap.xml"
HTTP_CLIENT_USER_AGENT_START = "python-httpx/"
# The username and password of a credentialed seed, and the header that carries them.
SEED_USER = "alice"
SEED_PASSWORD = "Sitemap-pw-7Kq"
SEED_BASIC_AUTH = "Basic " + base64.b64encode(f"{SEED_USER}:{SEED_PASSWORD}".encode()).decode()
# A Chrome user agent set by the crawl_user_agent setting, and the Chrome release it names.
OVERRIDE = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
OVERRIDE_MAJOR = "120"
# The Chrome release headless Chrome names in its own user agent.
HEADLESS_CHROME_RELEASE = re.compile(r"HeadlessChrome/(\d+)\.")
# The headers a browser sends with a page request that an HTTP crawl sends too.
BROWSER_ACCEPT_HEADERS = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "accept-language": "en-US,en;q=0.5",
    "upgrade-insecure-requests": "1",
}
# A page with a script and a link, the script, and the linked page.
HINT_PAGES = ("/hints/", "/hints/app.js", "/hints/next")
# Pages in the recursive browser crawl that must start Chromium once.
BROWSER_CRAWL_PAGES = 3
# A page whose image carries its own bytes in a ``data:`` address, here a one-pixel PNG.
INLINE_IMAGE_PAGE = "/inline/image"
INLINE_IMAGE_ALT = "Harbor lighthouse logo"
INLINE_IMAGE_PAYLOAD = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhf"
    "DwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
# A page that links to a second page, each with text hidden three ways beside visible text.
HIDDEN_PAGE = "/hidden/"
LATE_PAGE = "/late/"
LATER_PAGE = "/later/"
LATE_NEXT_PAGE = "/late/next"
# Documents with no body element: an SVG image and a frameset page.
SVG_PAGE = "/chart.svg"
FRAMESET_PAGE = "/frames/"
# Long enough for Chromium to start and load the page, far under a hang to the page timeout.
NO_BODY_TIMEOUT_S = 12
CONTENT_TYPES = {SVG_PAGE: "image/svg+xml"}
LATE_HEADING = "Harbor timetable"
LATE_TEXT = "The ferry leaves at noon"
# Within the default extra wait (200 ms) of the load, and a text later than it.
LATE_DELAY_MS = 100
LATER_DELAY_MS = 600
LATER_WAIT_MS = 1000
# A page timeout and a wait that alone outlasts two page timeouts.
LONG_WAIT_TIMEOUT_S = 3
LONG_WAIT_MS = 6300
HIDDEN_NEXT_PAGE = "/hidden/next"
COOKIE_START = "/cookie/"
COOKIE_NEED = "/cookie/need"
SESSION_COOKIE = "lilbeesession=ok"
COOKIE_TEXT = "Words behind the session cookie"
VISIBLE_TEXT = "Visible harbor words"
HIDDEN_TEXTS = {
    "display:none": "Styled away harbor words",
    "hidden attribute": "Attribute hidden harbor words",
    "aria-hidden": "Aria hidden harbor words",
}


QUERY_LINKS = (
    "/query/item?id=1",
    "/query/item?id=2",
    "/query/promo?utm_source=news",
    "/query/ad?gclid=abc&utm_term=spring",
    "/query/blog?p=42",
    "/query/branch?ref=main",
    "/moved",
)


# The links of the ``/defaults/`` listing on the site's own host, each with the request the
# crawl makes for it when no exclude pattern matches: a tracking parameter is not sent.
DEFAULTS_LINKS = {
    "/wiki/Help:Formatting": "/wiki/Help:Formatting",
    "/wiki/Manual:Installation_guide": "/wiki/Manual:Installation_guide",
    "/v/order": "/v/order",
    "/v/layout": "/v/layout",
    "/v/tilde?q=~x*y": "/v/tilde?q=%7Ex*y",
    "/blog/?utm_source=x": "/blog/",
    "/news/?msclkid=9": "/news/",
    "/wp-admin/": "/wp-admin/",
    "/defaults/feed/?utm_source=x": "/defaults/feed/",
    "/cart": "/cart",
    "/login?next=/": "/login?next=%2F",
}
# The requests of that table the default patterns exclude, each with the pattern that matches it.
DEFAULTS_EXCLUDED = {
    "/wp-admin/": r"/wp-admin/",
    "/defaults/feed/": r"/feed/?$",
    "/cart": r"^https?://[^/?#]+/cart(?:/|\?|#|$)",
    "/login?next=%2F": r"^https?://[^/?#]+/login(?:/|\?|#|$)",
}
# The requests a crawl of that listing makes before it follows a link: the listing, its sitemap
# and its robots.txt.
SEED_REQUESTS = {"/defaults/", "/sitemap.xml", "/robots.txt"}
ROBOTS_LISTING = "/robots/"
ROBOTS_DISALLOWED = "/robots/private/page"
ROBOTS_ALLOWED = "/robots/open"
ROBOTS_TXT = "User-agent: *\nDisallow: /robots/private/\n"
# A link of the same listing to a host the crawl does not follow, which a default pattern matches.
OTHER_HOST_LINK = "http://other.test/feed/"
# Where a served page that is not a listing starts, besides the listing prefixes.
PLAIN_PAGE_PREFIXES = (
    "/wiki/",
    "/hints/",
    "/v/",
    "/blog/",
    "/news/",
    "/wp-admin/",
    "/defaults/",
    "/cart",
    "/login",
)
FETCHER_LOGGER = "lilbee.crawler.crawlberg_fetcher"
LEFT_OUT = "Left out {url}: it matches {pattern}, a pattern of the crawl_exclude_patterns setting"


def _query_page(path: str, filler: str) -> str:
    """The query listing, or a page whose text names the path and query it was served for."""
    if path == "/query/":
        links = "".join(f'<a href="{link}">{link}</a> ' for link in QUERY_LINKS)
        links += '<a href="/query/pair?a=1&amp;b=2">pair</a> <A HREF="/query/upper">UPPER</A> '
        return f"<html><head><title>Listing</title></head><body>{links}{filler}</body></html>"
    if path in LINK_LISTINGS:
        links = "".join(f'<a href="{link}">{link}</a> ' for link in LINK_LISTINGS[path])
        head = '<head><meta charset="utf-8"><title>Spell</title></head>'
        return f"<html>{head}<body><h1>Index</h1>{links}{filler}</body></html>"
    served = f"<p>{FRESH_TEXT} {path}.</p>"
    return f"<html><head><title>T</title></head><body>{served}{filler}</body></html>"


def _scope_or_retry_page(path: str, filler: str) -> tuple[int, str]:
    """The ``/scope/`` and ``/retry/`` listings, the two failing retry pages, or a scope page."""
    listings = {"/scope/": SCOPE_LINKS, "/retry/": tuple(RETRY_STATUSES)}
    if path in listings:
        links = "".join(f'<a href="{link}">{link}</a> ' for link in listings[path])
        return 200, f"<html><body><h1>Index</h1>{links}{filler}</body></html>"
    if path in RETRY_STATUSES:
        return RETRY_STATUSES[path], "<h1>failed</h1>"
    return 200, _query_page(path, filler)


def _hidden_page(link: str) -> str:
    """A page with visible text, the three hidden texts, and *link*."""
    hidden = (
        f'<div style="display:none">{HIDDEN_TEXTS["display:none"]}</div>'
        f"<div hidden>{HIDDEN_TEXTS['hidden attribute']}</div>"
        f'<div aria-hidden="true">{HIDDEN_TEXTS["aria-hidden"]}</div>'
    )
    return f"<html><body><p>{VISIBLE_TEXT}</p>{hidden}{link}</body></html>"


def _late_page(link: str, delay_ms: int = LATE_DELAY_MS) -> str:
    """A page whose script writes its main text *delay_ms* after the load, and *link*."""
    write = (
        'window.addEventListener("load", () => setTimeout(() => {'
        f'document.getElementById("late").textContent = "{LATE_TEXT}";'
        f"}}, {delay_ms}));"
    )
    return (
        f'<html><body><h1>{LATE_HEADING}</h1><p id="late"></p>{link}'
        f"<script>{write}</script></body></html>"
    )


def _fixed_pages(filler: str, robots_status: int) -> dict[str, tuple[int, str]]:
    """The status and body, or redirect target, of each page served at one exact path."""
    image = f"data:image/png;base64,{INLINE_IMAGE_PAYLOAD}"
    logo = f'<img alt="{INLINE_IMAGE_ALT}" src="{image}">'
    listed = (*DEFAULTS_LINKS, OTHER_HOST_LINK)
    defaults = "".join(f'<a href="{link}">{link}</a> ' for link in listed)
    return {
        "/defaults/": (
            HTTPStatus.OK,
            f"<html><body><h1>Index</h1>{defaults}{filler}</body></html>",
        ),
        "/moved": (HTTPStatus.MOVED_PERMANENTLY, "/query/target"),
        "/robots.txt": (robots_status, ROBOTS_TXT),
        ROBOTS_LISTING: (
            HTTPStatus.OK,
            f'<html><body><h1>Robots</h1><a href="{ROBOTS_DISALLOWED}">private</a> '
            f'<a href="{ROBOTS_ALLOWED}">open</a>{filler}</body></html>',
        ),
        ROBOTS_DISALLOWED: (HTTPStatus.OK, f"<html><body><h1>Private</h1>{filler}</body></html>"),
        ROBOTS_ALLOWED: (HTTPStatus.OK, f"<html><body><h1>Open</h1>{filler}</body></html>"),
        HIDDEN_PAGE: (HTTPStatus.OK, _hidden_page(f'<a href="{HIDDEN_NEXT_PAGE}">next</a>')),
        HIDDEN_NEXT_PAGE: (HTTPStatus.OK, _hidden_page("")),
        LATE_PAGE: (HTTPStatus.OK, _late_page(f'<a href="{LATE_NEXT_PAGE}">next</a>')),
        LATE_NEXT_PAGE: (HTTPStatus.OK, _late_page("")),
        LATER_PAGE: (HTTPStatus.OK, _late_page("", LATER_DELAY_MS)),
        SVG_PAGE: (
            HTTPStatus.OK,
            '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="60">'
            f'<text x="5" y="30">{LATE_HEADING}</text></svg>',
        ),
        FRAMESET_PAGE: (
            HTTPStatus.OK,
            f'<html><head><title>Frames</title></head><frameset><frame src="{LATE_NEXT_PAGE}">'
            "</frameset></html>",
        ),
        COOKIE_START: (
            HTTPStatus.OK,
            f'<html><body><h1>Start</h1><a href="{COOKIE_NEED}">need</a>{filler}</body></html>',
        ),
        COOKIE_NEED: (HTTPStatus.OK, f"<html><body><p>{COOKIE_TEXT}</p>{filler}</body></html>"),
        INLINE_IMAGE_PAGE: (
            HTTPStatus.OK,
            f"<html><body><h1>Logo</h1>{logo}{filler}</body></html>",
        ),
        "/hints/": (
            HTTPStatus.OK,
            '<html><head><script src="/hints/app.js"></script></head><body><h1>Hints</h1>'
            f'<a href="/hints/next">next</a>{filler}</body></html>',
        ),
    }


class _Site:
    """A threaded local site: a wide listing, a slow listing, and a special page."""

    def __init__(self, tls: ssl.SSLContext | None = None) -> None:
        self.requests: list[tuple[float, str]] = []
        self.agents: list[tuple[str, str]] = []
        self.hints: list[tuple[str, dict[str, str]]] = []
        self.accept_headers: list[tuple[str, dict[str, str]]] = []
        self.cookies: list[tuple[str, str]] = []
        self.authorizations: list[tuple[str, str]] = []
        # The Authorization header the sitemap needs; blank serves no sitemap.
        self.sitemap_authorization = ""
        self.robots_status = HTTPStatus.OK
        self._answered: set[str] = set()
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._server.daemon_threads = True
        self._scheme = "http"
        if tls is not None:
            self._server.socket = tls.wrap_socket(self._server.socket, server_side=True)
            self._scheme = "https"
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def url(self, path: str) -> str:
        return f"{self._scheme}://127.0.0.1:{self.port}{path}"

    def paths_since(self, since: float, prefix: str) -> list[str]:
        with self._lock:
            return [path for at, path in self.requests if at >= since and path.startswith(prefix)]

    def agents_for(self, prefix: str) -> list[str]:
        """The user agent of each request for a path under *prefix*."""
        with self._lock:
            return [agent for path, agent in self.agents if path.startswith(prefix)]

    def hints_for(self, prefix: str) -> dict[str, list[dict[str, str]]]:
        """The client hint headers of each request for a path under *prefix*, by path."""
        with self._lock:
            found: dict[str, list[dict[str, str]]] = {}
            for path, hints in self.hints:
                if path.startswith(prefix):
                    found.setdefault(path, []).append(hints)
            return found

    def accept_headers_for(self, prefix: str) -> list[dict[str, str]]:
        """The headers of :data:`BROWSER_ACCEPT_HEADERS` each request under *prefix* carried."""
        with self._lock:
            return [sent for path, sent in self.accept_headers if path.startswith(prefix)]

    def cookies_for(self, path: str) -> list[str]:
        """The Cookie header of each request for exactly *path*."""
        with self._lock:
            return [cookie for at, cookie in self.cookies if at == path]

    def authorizations_for(self, path: str) -> list[str]:
        """The Authorization header of each request for exactly *path*."""
        with self._lock:
            return [sent for at, sent in self.authorizations if at == path]

    def reply(self, path: str, cookie: str, authorization: str = "") -> tuple[int, str]:
        """The status and body for *path*; the cookie page refuses a request without the cookie."""
        if path == SITEMAP_PATH and self.sitemap_authorization:
            return self._sitemap(authorization)
        if path == COOKIE_NEED and SESSION_COOKIE not in cookie:
            return HTTPStatus.FORBIDDEN, "<h1>forbidden</h1>"
        if path == RATE_LIMITED_ONCE and self._first_request(path):
            return HTTPStatus.TOO_MANY_REQUESTS, "<h1>slow down</h1>"
        return self._body(path)

    def _sitemap(self, authorization: str) -> tuple[int, str]:
        """A sitemap of the wide listing behind basic auth."""
        if authorization != self.sitemap_authorization:
            return HTTPStatus.UNAUTHORIZED, "<h1>log in</h1>"
        locs = "".join(f"<url><loc>{self.url(f'/wide/p{n}')}</loc></url>" for n in range(3))
        return HTTPStatus.OK, f"<urlset>{locs}</urlset>"

    def _first_request(self, path: str) -> bool:
        """True the first time *path* is answered."""
        with self._lock:
            first = path not in self._answered
            self._answered.add(path)
            return first

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _body(self, path: str) -> tuple[int, str]:
        filler = "<p>" + "Ordinary prose for a real page. " * 6 + "</p>"
        fixed = _fixed_pages(filler, self.robots_status)
        if path in fixed:
            return fixed[path]
        if path.startswith(("/query/", *LINK_LISTINGS)):
            return 200, _query_page(path, filler)
        if path.startswith(("/scope/", "/retry/")):
            return _scope_or_retry_page(path, filler)
        if path in ("/wide/", "/slow/"):
            links = "".join(f'<a href="{path}p{n}">p{n}</a> ' for n in range(SLOW_PAGES))
            special = '<a href="/wiki/Special:Random">random</a><a href="/wiki/Home">home</a>'
            return 200, f"<html><body><h1>Index</h1>{links}{special}{filler}</body></html>"
        if path.startswith(("/wide/p", "/slow/p", *PLAIN_PAGE_PREFIXES)):
            if path.startswith("/slow/p"):
                time.sleep(SLOW_DELAY_S)
            return 200, f"<html><body><h1>{path}</h1>{filler}</body></html>"
        return 404, "<h1>missing</h1>"

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        site = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                """Silence the access log."""

            def do_GET(self) -> None:
                with site._lock:
                    site.requests.append((time.monotonic(), self.path))
                    site.agents.append((self.path, self.headers.get("User-Agent", "")))
                    hints = {
                        name.lower(): value
                        for name, value in self.headers.items()
                        if name.lower().startswith("sec-ch-")
                    }
                    site.hints.append((self.path, hints))
                    sent = {
                        name.lower(): value
                        for name, value in self.headers.items()
                        if name.lower() in BROWSER_ACCEPT_HEADERS
                    }
                    site.accept_headers.append((self.path, sent))
                    site.cookies.append((self.path, self.headers.get("Cookie", "")))
                    authorization = self.headers.get("Authorization", "")
                    site.authorizations.append((self.path, authorization))
                status, body = site.reply(self.path, self.headers.get("Cookie", ""), authorization)
                if status == HTTPStatus.MOVED_PERMANENTLY:
                    self.send_response(status)
                    self.send_header("Location", body)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                data = body.encode()
                self.send_response(status)
                self.send_header("Content-Type", CONTENT_TYPES.get(self.path, "text/html"))
                if self.path == COOKIE_START:
                    self.send_header("Set-Cookie", f"{SESSION_COOKIE}; Path=/")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler


@pytest.fixture
def site() -> Iterator[_Site]:
    running = _Site()
    yield running
    running.close()


def _self_signed_context(folder: Path) -> ssl.SSLContext:
    """A server context whose certificate for 127.0.0.1 is signed by its own key."""
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = datetime.datetime.now(datetime.UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_path, key_path = folder / "cert.pem", folder / "key.pem"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    return context


@pytest.fixture
def self_signed_site(tmp_path) -> Iterator[_Site]:
    """The site over HTTPS with a certificate no authority signed."""
    running = _Site(tls=_self_signed_context(tmp_path))
    yield running
    running.close()


@pytest.fixture
def allow_loopback(monkeypatch):
    """The same loopback carve-out every local crawl test uses, read by both SSRF layers."""
    allowed = tuple(n for n in url_filter.get_blocked_networks() if n not in LOOPBACK)
    monkeypatch.setattr(url_filter, "get_blocked_networks", lambda: allowed)


@pytest.fixture
def isolated_env(tmp_path):
    snapshot = cfg.model_copy()
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir()
    cfg.data_dir = tmp_path / "data"
    cfg.data_dir.mkdir()
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.crawl_sync_interval = 0
    cfg.crawl_mean_delay = 0.0
    from lilbee.app.services import reset_services

    reset_services()
    yield tmp_path
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))
    reset_services()


def _representative(network: ipaddress.IPv4Network | ipaddress.IPv6Network) -> str:
    address = network[1] if network.num_addresses > 1 else network[0]
    return f"[{address}]" if address.version == 6 else str(address)


async def _first_error(policy: object, host: str) -> str:
    """The error crawlberg reports for a one-page crawl of *host* under *policy*, or ''."""
    config = crawlberg.CrawlConfig(
        max_depth=0, max_pages=1, request_timeout=1000, retry_count=0, ssrf=policy
    )
    engine = crawlberg.create_engine(config)
    async for event in crawlberg.crawl_stream(engine, f"http://{host}:9/"):
        payload = json.loads(str(event))
        if payload["type"] == "error":
            return str(payload["error"])
    return ""


class TestSsrfBlocklistsAgree:
    def test_lilbee_blocks_nothing_crawlberg_misses_but_nat64(self):
        uncovered = [
            net
            for net in url_filter.get_blocked_networks()
            if not any(
                net.version == denied.version and net.subnet_of(denied)  # type: ignore[arg-type]
                for denied in CRAWLBERG_DENIED_NETWORKS
            )
        ]
        assert uncovered == [NAT64]

    @pytest.mark.parametrize("network", CRAWLBERG_DENIED_NETWORKS, ids=str)
    async def test_crawlberg_refuses_each_network_in_the_copied_list(self, network):
        policy = crawlberg.SsrfPolicy(deny_private=True)
        assert (await _first_error(policy, _representative(network))).startswith(_SSRF_ERROR_CODE)

    @pytest.mark.parametrize(
        "network",
        [n for n in url_filter.get_blocked_networks() if n != NAT64],
        ids=str,
    )
    async def test_built_policy_refuses_every_network_lilbee_blocks(self, network):
        from lilbee.crawler.crawlberg_fetcher import _ssrf_policy

        error = await _first_error(_ssrf_policy(), _representative(network))
        assert error.startswith(_SSRF_ERROR_CODE), error

    @pytest.mark.parametrize(
        "host", IPV6_FORMS_OF_PRIVATE_IPV4.values(), ids=list(IPV6_FORMS_OF_PRIVATE_IPV4)
    )
    async def test_fetch_refuses_an_ipv6_address_that_carries_a_private_ipv4(self, host: str):
        fetcher = CrawlbergFetcher(render_mode=CrawlRenderMode.HTTP)
        fetched = await fetcher.fetch_single(
            f"http://{host}:9/", timeout=1, concurrency=ConcurrencySpec()
        )
        assert fetched.success is False
        assert fetched.error.startswith(_SSRF_ERROR_CODE), fetched.error

    async def test_fetch_tries_an_ipv6_address_that_carries_a_public_ipv4(self):
        fetcher = CrawlbergFetcher(render_mode=CrawlRenderMode.HTTP)
        fetched = await fetcher.fetch_single(
            f"http://{IPV6_FORM_OF_PUBLIC_IPV4}:9/", timeout=1, concurrency=ConcurrencySpec()
        )
        assert fetched.success is False
        assert fetched.error
        assert not fetched.error.startswith(_SSRF_ERROR_CODE), fetched.error

    async def test_built_policy_admits_what_lilbee_admits(self):
        from lilbee.crawler.crawlberg_fetcher import _ssrf_policy

        assert admitted_networks(url_filter.get_blocked_networks()) == [
            ipaddress.ip_network("224.0.0.0/4")
        ]
        error = await _first_error(_ssrf_policy(), "224.0.0.1")
        assert not error.startswith(_SSRF_ERROR_CODE), error


def _stream(fetcher: CrawlbergFetcher, url: str, **kwargs):
    return fetcher.fetch_recursive(
        url,
        depth=kwargs.get("depth", 1),
        max_pages=kwargs.get("max_pages"),
        timeout=10,
        concurrency=ConcurrencySpec(semaphore_count=3),
        filters=FilterSpec(),
        cancel=kwargs.get("cancel"),
    )


@pytest.mark.usefixtures("allow_loopback")
class TestCancel:
    async def test_cancel_stops_the_stream_and_no_request_starts_after_it(self, site):
        cancel = threading.Event()
        received: list[FetchedPage] = []
        fetcher = CrawlbergFetcher(render_mode=CrawlRenderMode.HTTP)
        cancelled_at = 0.0
        async for fetched in _stream(fetcher, site.url("/slow/"), cancel=cancel):
            received.append(fetched)
            if len(received) == 3:
                cancel.set()
                cancelled_at = time.monotonic()
        stopped_at = time.monotonic()
        assert stopped_at - cancelled_at < CANCEL_BOUND_S
        await asyncio.sleep(SETTLE_S)
        assert site.paths_since(stopped_at + LATE_START_MARGIN_S, "/slow/p") == []
        assert site.paths_since(0, "/slow/p")
        assert len(received) <= 4


@pytest.mark.usefixtures("allow_loopback")
class TestWorkerThread:
    def test_crawl_runs_on_a_worker_thread_with_its_own_loop(self, site):
        box: dict[str, object] = {}

        def worker() -> None:
            async def crawl() -> list[FetchedPage]:
                fetcher = CrawlbergFetcher(render_mode=CrawlRenderMode.HTTP)
                return [f async for f in _stream(fetcher, site.url("/wide/"), max_pages=3)]

            try:
                box["pages"] = asyncio.run(crawl())
            except BaseException as exc:  # PanicException derives from BaseException
                box["error"] = exc

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert "error" not in box, box.get("error")
        assert len(box["pages"]) == 3  # type: ignore[arg-type]


@pytest.mark.usefixtures("allow_loopback", "isolated_env")
class TestCrawlAndSave:
    async def test_max_pages_saves_exactly_that_many_pages(self, site):
        paths = await crawl_and_save(site.url("/wide/"), depth=1, max_pages=5)
        assert len(paths) == 5

    async def test_a_one_page_crawl_requests_its_page_and_no_sitemap(self, site):
        paths = await crawl_and_save(site.url("/wide/"), depth=1, max_pages=1)
        one_page = [path for path in site.paths_since(0, "/") if path != "/robots.txt"]
        two_pages_from = time.monotonic()
        await crawl_and_save(site.url("/wide/"), depth=1, max_pages=2)
        assert "/sitemap.xml" in site.paths_since(two_pages_from, "/")
        assert one_page == ["/wide/"]
        assert len(paths) == 1

    @pytest.mark.parametrize("limit", ONE_PAGE_LIMITS.values(), ids=list(ONE_PAGE_LIMITS))
    async def test_a_one_page_crawl_of_a_seed_an_exclude_pattern_matches_saves_it(
        self, site, limit
    ):
        seed = site.url(EXCLUDED_SEED)
        assert any(re.search(pattern, seed) for pattern in cfg.crawl_exclude_patterns)
        paths = await crawl_and_save(seed, render_mode=HTTP, **limit)
        assert site.paths_since(0, EXCLUDED_SEED) == [EXCLUDED_SEED]
        assert len(paths) == 1

    @pytest.mark.parametrize("limit", SITE_CRAWL_LIMITS.values(), ids=list(SITE_CRAWL_LIMITS))
    async def test_a_site_crawl_of_a_seed_an_exclude_pattern_matches_names_the_pattern(
        self, site, caplog, limit
    ):
        seed = site.url(EXCLUDED_SEED)
        pattern = next(p for p in cfg.crawl_exclude_patterns if re.search(p, seed))
        paths = await crawl_and_save(seed, render_mode=HTTP, **limit)
        warnings = [
            record.getMessage()
            for record in caplog.records
            if record.name == FETCHER_LOGGER and record.levelname == "WARNING"
        ]
        # crawlberg 1.11.1 leaves out a seed an exclude pattern matches, at any limit.
        assert site.paths_since(0, EXCLUDED_SEED) == []
        assert paths == []
        assert warnings == [LEFT_OUT.format(url=seed, pattern=pattern)]

    @pytest.mark.parametrize("limit", SITE_CRAWL_LIMITS.values(), ids=list(SITE_CRAWL_LIMITS))
    async def test_a_seed_typed_without_a_path_is_named_when_a_root_pattern_excludes_it(
        self, site, caplog, limit
    ):
        seed = site.url("")
        cfg.crawl_exclude_patterns = [ROOT_PATTERN]
        assert re.search(ROOT_PATTERN, seed) is None
        paths = await crawl_and_save(seed, render_mode=HTTP, **limit)
        warnings = [
            record.getMessage()
            for record in caplog.records
            if record.name == FETCHER_LOGGER and record.levelname == "WARNING"
        ]
        assert site.paths_since(0, "/") == ["/sitemap.xml"]
        assert paths == []
        assert warnings == [LEFT_OUT.format(url=site.url("/"), pattern=ROOT_PATTERN)]

    async def test_a_seed_with_a_username_and_password_is_named_as_crawlberg_matches_it(
        self, site, caplog
    ):
        address = site.url(EXCLUDED_SEED)
        pattern = f"^{re.escape(address)}$"
        cfg.crawl_exclude_patterns = [pattern]
        seed = address.replace("://", "://user:pw@", 1)
        assert re.search(pattern, seed) is None
        paths = await crawl_and_save(seed, render_mode=HTTP, depth=1, max_pages=2)
        warnings = [
            record.getMessage()
            for record in caplog.records
            if record.name == FETCHER_LOGGER and record.levelname == "WARNING"
        ]
        # crawlberg 1.11.1 matches a pattern against the address without its userinfo.
        assert site.paths_since(0, EXCLUDED_SEED) == []
        assert paths == []
        assert warnings == [LEFT_OUT.format(url=address, pattern=pattern)]

    async def test_default_exclude_patterns_all_apply(self, site):
        """``/wiki/Special:`` is a default pattern that is not a regex anchor or digit class."""
        paths = await crawl_and_save(site.url("/wide/"), depth=1, max_pages=0)
        saved = {p.relative_to(cfg.documents_dir).as_posix() for p in paths}
        assert f"_web/{save.url_to_filename(site.url('/wiki/Home'))}" in saved, saved
        assert not any("Special" in name for name in saved), saved

    async def test_an_inline_image_is_saved_as_its_alt_text_without_its_encoded_data(self, site):
        assert INLINE_IMAGE_PAYLOAD in site._body(INLINE_IMAGE_PAGE)[1]
        paths = await crawl_and_save(site.url(INLINE_IMAGE_PAGE), depth=0)
        assert len(paths) == 1
        text = paths[0].read_text(encoding="utf-8")
        assert "Ordinary prose for a real page." in text
        assert INLINE_IMAGE_ALT in text
        assert INLINE_IMAGE_PAYLOAD not in text
        assert "base64" not in text


def _assert_hidden_text_kept(paths: list[Path], pages: int) -> None:
    """Each saved page holds its visible text and each of its hidden texts."""
    assert len(paths) == pages
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert VISIBLE_TEXT in text, text
        for way, hidden in HIDDEN_TEXTS.items():
            assert hidden in text, f"{way}: {text}"


def _assert_late_text(paths: list[Path], pages: int, *, kept: bool) -> None:
    """Each saved page holds its heading, and holds its late text exactly when *kept*."""
    assert len(paths) == pages
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert LATE_HEADING in text, text
        assert (LATE_TEXT in text) is kept, text


@pytest.mark.usefixtures("allow_loopback", "isolated_env")
class TestHiddenText:
    async def test_a_single_page_http_crawl_keeps_hidden_text(self, site):
        paths = await crawl_and_save(site.url(HIDDEN_PAGE), depth=0, render_mode=HTTP)
        _assert_hidden_text_kept(paths, 1)

    async def test_a_recursive_http_crawl_keeps_hidden_text(self, site):
        paths = await crawl_and_save(site.url(HIDDEN_PAGE), depth=1, render_mode=HTTP)
        _assert_hidden_text_kept(paths, 2)


@pytest.mark.usefixtures("allow_loopback", "isolated_env")
class TestDefaultExcludePatterns:
    async def _requested(self, site: _Site) -> set[str]:
        await crawl_and_save(site.url("/defaults/"), depth=1, max_pages=0)
        return set(site.paths_since(0, "/")) - SEED_REQUESTS

    async def test_content_pages_are_fetched_and_scaffolding_is_not(self, site):
        assert await self._requested(site) == {
            "/wiki/Help:Formatting",
            "/wiki/Manual:Installation_guide",
            "/v/order",
            "/v/layout",
            "/v/tilde?q=%7Ex*y",
            "/blog/",
            "/news/",
        }

    def test_lilbee_computes_for_each_link_the_request_the_crawl_makes(self, site):
        """The two crawl tests of this class hold the site to the same request table."""
        for link, request in DEFAULTS_LINKS.items():
            assert _queued_url(site.url(link)) == site.url(request)

    async def test_each_fetched_page_is_saved(self, site):
        paths = await crawl_and_save(site.url("/defaults/"), depth=1, max_pages=0)
        saved = "\n".join(p.read_text(encoding="utf-8") for p in paths)
        for path in ("/wiki/Help:Formatting", "/v/order", "/v/layout", "/blog/", "/news/"):
            assert f"# {path}\n" in saved, path
        for heading in ("# /wp-admin/\n", "# /cart\n", "# /login"):
            assert heading not in saved, heading

    async def test_the_log_names_exactly_the_addresses_the_site_never_received(self, site, caplog):
        requested = await self._requested(site)
        warnings = sorted(
            record.getMessage()
            for record in caplog.records
            if record.name == FETCHER_LOGGER and record.levelname == "WARNING"
        )
        never_received = set(DEFAULTS_LINKS.values()) - requested
        assert never_received == set(DEFAULTS_EXCLUDED)
        assert warnings == sorted(
            LEFT_OUT.format(url=site.url(path), pattern=DEFAULTS_EXCLUDED[path])
            for path in never_received
        )


@pytest.mark.usefixtures("allow_loopback", "isolated_env")
class TestQueryUrlsAndRedirects:
    async def _saved_text(self, site: _Site) -> str:
        paths = await crawl_and_save(site.url("/query/"), depth=1, max_pages=0)
        return "\n".join(p.read_text(encoding="utf-8") for p in paths)

    async def test_pages_that_differ_only_by_query_are_each_saved(self, site):
        saved = await self._saved_text(site)
        assert "Served /query/item?id=1." in saved
        assert "Served /query/item?id=2." in saved

    async def test_tracking_parameters_are_stripped_before_the_fetch(self, site):
        saved = await self._saved_text(site)
        assert "Served /query/promo." in saved
        assert site.paths_since(0, "/query/promo") == ["/query/promo"]

    async def test_gclid_and_utm_term_are_stripped(self, site):
        saved = await self._saved_text(site)
        assert "Served /query/ad." in saved
        assert site.paths_since(0, "/query/ad") == ["/query/ad"]

    async def test_a_ref_parameter_is_kept_in_the_fetch(self, site):
        saved = await self._saved_text(site)
        assert "Served /query/branch?ref=main." in saved
        assert site.paths_since(0, "/query/branch") == ["/query/branch?ref=main"]

    async def test_exclude_patterns_see_the_query(self, site):
        saved = await self._saved_text(site)
        assert "/query/blog?p=42." not in saved
        assert site.paths_since(0, "/query/blog") == []

    async def test_a_discovered_link_that_redirects_saves_its_target(self, site):
        saved = await self._saved_text(site)
        assert "Served /query/target." in saved

    async def test_a_link_with_an_encoded_ampersand_is_crawled_at_the_decoded_url(self, site):
        saved = await self._saved_text(site)
        assert "Served /query/pair?a=1&b=2." in saved
        assert site.paths_since(0, "/query/pair") == ["/query/pair?a=1&b=2"]

    async def test_a_link_in_uppercase_markup_is_followed(self, site):
        saved = await self._saved_text(site)
        assert "Served /query/upper." in saved

    async def test_saved_markdown_has_no_frontmatter(self, site):
        paths = await crawl_and_save(site.url("/query/"), depth=0)
        text = paths[0].read_text(encoding="utf-8")
        assert not text.startswith("---")
        assert "title: Listing" not in text


@pytest.mark.usefixtures("allow_loopback", "isolated_env")
class TestWholeUrlExcludePatterns:
    async def _crawled(self, site: _Site, patterns: list[str]) -> set[str]:
        cfg.crawl_exclude_patterns = patterns
        await crawl_and_save(site.url("/scope/"), depth=1, max_pages=0)
        return set(site.paths_since(0, "/scope/"))

    async def test_a_pattern_anchored_on_the_scheme_and_host_drops_its_pages(self, site):
        crawled = await self._crawled(site, [r"^http://127\.0\.0\.1:\d+/scope/skip/"])
        assert crawled == {"/scope/", "/scope/keep/b", "/scope/la/drop", "/scope/la/keepme"}

    async def test_a_look_ahead_pattern_keeps_the_page_it_names(self, site):
        crawled = await self._crawled(site, [r"^https?://[^/]+/scope/la/(?!keep)"])
        assert crawled == {"/scope/", "/scope/skip/a", "/scope/keep/b", "/scope/la/keepme"}

    async def test_a_pattern_for_another_host_drops_nothing(self, site):
        crawled = await self._crawled(site, [r"^https?://example\.com/scope/"])
        assert crawled == {"/scope/", *SCOPE_LINKS}


def _earlier_name(url: str) -> str:
    """A name an earlier name rule gave *url*: one per page, never its name today."""
    digest = hashlib.sha256(save.canonical_address(url).encode("utf-8")).hexdigest()[:8]
    return f"earlier/{digest}/index.md"


def _library_with(*urls: str) -> list[Path]:
    """An earlier library that holds each of *urls* under an earlier name, with its own text."""
    entries = {}
    paths = []
    for url in urls:
        name = _earlier_name(url)
        path = cfg.documents_dir / "_web" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        text = f"{STALE_TEXT} of {url}"
        path.write_text(text, encoding="utf-8")
        entries[url] = CrawlMeta(
            file=name, content_hash=save.content_hash(text), crawled_at="2026-01-01T00:00:00+00:00"
        )
        paths.append(path)
    write_crawl_metadata(entries)
    return paths


def _pair_library(site: _Site) -> list[tuple[Path, Path]]:
    """An earlier library that holds both addresses of each pair of PAIR_LINKS."""
    copies = _library_with(*(site.url(link) for pair in PAIR_LINKS for link in pair))
    assert len(set(copies)) == len(copies)
    return list(zip(copies[::2], copies[1::2], strict=True))


def _current_file(url: str, earlier: Path) -> Path:
    """The page of *url* at its name when the crawl saved it, else its *earlier* copy."""
    page = cfg.documents_dir / "_web" / save.url_to_filename(url)
    return page if page.is_file() else earlier


def _pair_texts(
    site: _Site, library: list[tuple[Path, Path]]
) -> dict[tuple[str, str], tuple[str, str]]:
    """The text of the two pages of each pair of PAIR_LINKS, saved or still earlier copies."""
    return {
        (one, other): (
            _current_file(site.url(one), one_copy).read_text(encoding="utf-8"),
            _current_file(site.url(other), other_copy).read_text(encoding="utf-8"),
        )
        for (one, other), (one_copy, other_copy) in zip(PAIR_LINKS, library, strict=True)
    }


def _assert_each_pair_is_two_pages(site: _Site, library: list[tuple[Path, Path]]) -> None:
    """Each pair has a page the crawl saved, no pair holds one text twice, each save recorded."""
    texts = _pair_texts(site, library)
    assert [pair for pair, both in texts.items() if FRESH_TEXT not in "".join(both)] == []
    assert [pair for pair, (one, other) in texts.items() if one == other] == []
    assert [pair for pair, both in texts.items() if not _own_pages(pair, both)] == []
    saved = [
        link
        for pair in PAIR_LINKS
        for link in pair
        if (cfg.documents_dir / "_web" / save.url_to_filename(site.url(link))).is_file()
    ]
    assert set(save.load_crawl_record().pages) == {
        save.canonical_address(site.url("/pair/")),
        *(save.canonical_address(site.url(link)) for link in saved),
    }


def _own_pages(pair: tuple[str, str], texts: tuple[str, str]) -> bool:
    """Whether each text is the stale copy, or the page the site serves for its own address."""
    return all(
        text.startswith(STALE_TEXT) or f"{FRESH_TEXT} {link}." in text
        for link, text in zip(pair, texts, strict=True)
    )


def _assert_saved_in_readable_folders(site: _Site) -> None:
    """The ``/spell/`` pages are stored as crawlberg requested them and saved in SPELL_FOLDERS."""
    assert set(site.paths_since(0, "/spell/")) == {"/spell/", *SPELL_REQUESTS}
    meta = save.load_crawl_record().pages
    stored = [save.canonical_address(site.url(path)) for path in SPELL_REQUESTS]
    assert set(meta) == {save.canonical_address(site.url("/spell/")), *stored}
    web = cfg.documents_dir / "_web"
    files = [web / meta[url].file for url in stored]
    readable = [file.parent.relative_to(web) / page_stem(file.name) for file in files]
    assert [name.as_posix().split("/", 1)[1] for name in readable] == list(SPELL_FOLDERS)
    assert [file for file in files if not file.is_file()] == []


@pytest.mark.usefixtures("allow_loopback", "isolated_env")
class TestStoredSpellingOfAnAddress:
    async def test_a_page_stored_under_an_equivalent_address_replaces_its_earlier_copy(self, site):
        stored = site.url(EQUIVALENT_STORED)
        (copy,) = _library_with(stored)
        await crawl_and_save(site.url("/same/"), depth=1, max_pages=0)
        assert site.paths_since(0, "/same/x") == [EQUIVALENT_LINK]
        page = cfg.documents_dir / "_web" / save.url_to_filename(stored)
        assert f"{FRESH_TEXT} {EQUIVALENT_LINK}." in page.read_text(encoding="utf-8")
        assert not copy.exists()
        pages = save.load_crawl_record().pages
        assert set(pages) == {save.canonical_address(url) for url in (site.url("/same/"), stored)}
        assert len(list((cfg.documents_dir / "_web").rglob("*.md"))) == len(pages)

    async def test_a_page_stored_under_two_equivalent_addresses_keeps_one_entry(self, site):
        stored, as_written = site.url(EQUIVALENT_STORED), site.url(EQUIVALENT_LINK)
        other = site.url("/elsewhere/page")
        plain_copy, escaped_copy, other_copy = _library_with(stored, as_written, other)
        assert plain_copy == escaped_copy
        await crawl_and_save(site.url("/same/"), depth=1, max_pages=0)
        page = cfg.documents_dir / "_web" / save.url_to_filename(stored)
        assert f"{FRESH_TEXT} {EQUIVALENT_LINK}." in page.read_text(encoding="utf-8")
        assert not plain_copy.exists()
        assert other_copy.read_text(encoding="utf-8") == f"{STALE_TEXT} of {other}"
        pages = save.load_crawl_record().pages
        assert set(pages) == {save.canonical_address(url) for url in (site.url("/same/"), stored)}

    async def test_two_stored_pages_stay_two_pages_after_an_http_recrawl(self, site):
        library = _pair_library(site)
        await crawl_and_save(site.url("/pair/"), depth=1, max_pages=0)
        _assert_each_pair_is_two_pages(site, library)

    async def test_a_new_page_is_saved_in_a_folder_that_reads_as_its_link(self, site):
        await crawl_and_save(site.url("/spell/"), depth=1, max_pages=0)
        _assert_saved_in_readable_folders(site)

    async def test_a_page_linked_raw_and_encoded_keeps_one_copy_in_either_link_order(self, site):
        for listing in ORDER_LISTINGS:
            await crawl_and_save(site.url(listing), depth=1, max_pages=0)
        assert site.paths_since(0, "/twice/na").count("/twice/na%C3%AFve") == len(ORDER_LISTINGS)
        meta = save.load_crawl_record().pages
        assert set(meta) == {
            *(save.canonical_address(site.url(listing)) for listing in ORDER_LISTINGS),
            save.canonical_address(site.url("/twice/na%C3%AFve")),
        }
        leaf_entry = meta[save.canonical_address(site.url("/twice/na%C3%AFve"))]
        leaf = cfg.documents_dir / "_web" / leaf_entry.file
        assert f"{FRESH_TEXT} /twice/na%C3%AFve." in leaf.read_text(encoding="utf-8")
        saved = sorted(page_stem(path.name) for path in leaf.parent.rglob("*.md"))
        assert saved == ["again", "naïve"]


@pytest.mark.usefixtures("allow_loopback", "isolated_env")
class TestRetriedStatuses:
    async def test_only_a_rate_limit_status_is_retried(self, site):
        cfg.crawl_retry_on_rate_limit = True
        cfg.crawl_retry_max_attempts = RETRY_ATTEMPTS
        cfg.crawl_retry_base_delay_min = RETRY_DELAY_S
        cfg.crawl_retry_base_delay_max = RETRY_DELAY_S
        cfg.crawl_retry_max_backoff = RETRY_DELAY_S
        await crawl_and_save(site.url("/retry/"), depth=1, max_pages=0)
        assert len(site.paths_since(0, "/retry/busy")) == 1 + RETRY_ATTEMPTS
        assert site.paths_since(0, "/retry/broken") == ["/retry/broken"]

    async def test_a_retry_count_over_crawlbergs_limit_crawls_with_the_limit(self, site):
        cfg.crawl_retry_on_rate_limit = True
        cfg.crawl_retry_max_attempts = CRAWLBERG_RETRY_LIMIT + 5
        cfg.crawl_retry_base_delay_min = RETRY_DELAY_S
        cfg.crawl_retry_base_delay_max = RETRY_DELAY_S
        cfg.crawl_retry_max_backoff = RETRY_DELAY_S
        await crawl_and_save(site.url("/retry/"), depth=1, max_pages=0)
        assert len(site.paths_since(0, "/retry/busy")) == 1 + CRAWLBERG_RETRY_LIMIT

    @pytest.mark.parametrize("limit", ONE_PAGE_LIMITS.values(), ids=list(ONE_PAGE_LIMITS))
    async def test_a_one_page_crawl_retries_a_rate_limited_page_and_saves_it(self, site, limit):
        cfg.crawl_retry_on_rate_limit = True
        cfg.crawl_retry_max_attempts = RETRY_ATTEMPTS
        cfg.crawl_retry_base_delay_min = RETRY_DELAY_S
        cfg.crawl_retry_base_delay_max = RETRY_DELAY_S
        cfg.crawl_retry_max_backoff = RETRY_DELAY_S
        paths = await crawl_and_save(site.url(RATE_LIMITED_ONCE), render_mode=HTTP, **limit)
        assert site.paths_since(0, RATE_LIMITED_ONCE) == [RATE_LIMITED_ONCE] * 2
        assert len(paths) == 1

    @pytest.mark.parametrize("limit", ONE_PAGE_LIMITS.values(), ids=list(ONE_PAGE_LIMITS))
    async def test_a_one_page_crawl_with_retries_off_asks_once_and_saves_nothing(self, site, limit):
        cfg.crawl_retry_on_rate_limit = False
        paths = await crawl_and_save(site.url(RATE_LIMITED_ONCE), render_mode=HTTP, **limit)
        assert site.paths_since(0, RATE_LIMITED_ONCE) == [RATE_LIMITED_ONCE]
        assert paths == []


@pytest.mark.usefixtures("allow_loopback", "isolated_env")
class TestRefusedLaunchFlags:
    """crawlberg checks the flags when it builds the engine, before any Chrome starts."""

    @pytest.fixture(autouse=True)
    def _a_shell_that_is_never_started(self, monkeypatch, tmp_path):
        shell = tmp_path / "chrome-headless-shell"
        shell.write_text("", encoding="utf-8")
        monkeypatch.setattr(bootstrap, "chromium_installed", lambda: True)
        monkeypatch.setattr(bootstrap, "headless_shell_executable", lambda: shell)

    @pytest.mark.parametrize(("flags", "reason"), REFUSED_FLAGS, ids=repr)
    @pytest.mark.parametrize("depth", [0, 1], ids=["single-page", "recursive"])
    async def test_a_browser_crawl_names_the_setting_and_gives_crawlbergs_reason(
        self, site, flags: list[str], reason: str, depth: int
    ):
        cfg.crawl_browser_extra_args = flags
        with pytest.raises(CrawlEngineRefusedError) as refused:
            await crawl_and_save(site.url("/wiki/Home"), depth=depth, render_mode=BROWSER)
        message = str(refused.value)
        assert message.startswith("The crawl_browser_extra_args setting")
        assert "invalid_config: browser.chrome_args" in message
        assert reason in message
        assert site.paths_since(0, "/wiki/Home") == []

    def test_the_default_flags_build_an_engine(self):
        assert Config().crawl_browser_extra_args == DEFAULT_FLAGS
        browser = crawlberg.BrowserConfig(mode="always", chrome_args=DEFAULT_FLAGS)
        assert crawlberg.create_engine(crawlberg.CrawlConfig(browser=browser)) is not None

    async def test_an_http_crawl_with_a_refused_flag_fetches_the_page(self, site):
        cfg.crawl_browser_extra_args = ["--headless=new"]
        paths = await crawl_and_save(site.url("/wiki/Home"), depth=0, render_mode=HTTP)
        assert len(paths) == 1


@pytest.mark.usefixtures("allow_loopback", "isolated_env")
class TestShellThatCannotRun:
    """A headless shell path that cannot run is installed, and crawlberg refuses it."""

    def _shell_dir(self, tmp_path: Path) -> Path:
        """The folder of a finished headless shell install at revision 1, with no shell in it."""
        directory = tmp_path / f"{bootstrap._HEADLESS_SHELL_DIR_PREFIX}1"
        directory.mkdir()
        (directory / bootstrap._INSTALL_COMPLETE_MARKER).write_bytes(b"")
        return directory

    @posix_only
    @pytest.mark.parametrize("depth", [0, 1], ids=["single-page", "recursive"])
    @pytest.mark.parametrize("shape", ["dangling-link", "directory"])
    async def test_a_shell_path_that_is_no_file_is_named_and_starts_no_install(
        self, site, monkeypatch, tmp_path, depth: int, shape: str
    ):
        shell = self._shell_dir(tmp_path) / "chrome-headless-shell"
        if shape == "directory":
            shell.mkdir()
        else:
            shell.symlink_to(tmp_path / "no-such-target")
        monkeypatch.setattr(bootstrap, "_browsers_cache_path", lambda: tmp_path)
        monkeypatch.setattr(bootstrap, "_expected_chromium_revision", lambda: "1")
        installs: list[object] = []

        async def install(on_progress: object) -> None:
            installs.append(on_progress)

        monkeypatch.setattr(bootstrap, "_install_chromium", install)
        with pytest.raises(CrawlEngineRefusedError) as refused:
            await crawl_and_save(site.url("/wiki/Home"), depth=depth, render_mode=BROWSER)
        assert installs == []
        assert str(refused.value).startswith(f"{NO_SETTING}invalid_config: browser.chrome_path")
        assert f"'{shell}'" in str(refused.value)
        assert site.paths_since(0, "/wiki/Home") == []

    @posix_only
    @pytest.mark.parametrize("depth", [0, 1], ids=["single-page", "recursive"])
    async def test_a_browser_crawl_names_the_shell_and_starts_no_install(
        self, site, monkeypatch, tmp_path, depth: int
    ):
        shell = self._shell_dir(tmp_path) / "chrome-headless-shell"
        shell.write_text("#!/bin/sh\n", encoding="utf-8")
        shell.chmod(stat.S_IRUSR | stat.S_IWUSR)
        monkeypatch.setattr(bootstrap, "_browsers_cache_path", lambda: tmp_path)
        monkeypatch.setattr(bootstrap, "_expected_chromium_revision", lambda: "1")
        installs: list[object] = []

        async def install(on_progress: object) -> None:
            installs.append(on_progress)

        monkeypatch.setattr(bootstrap, "_install_chromium", install)
        with pytest.raises(CrawlEngineRefusedError) as refused:
            await crawl_and_save(site.url("/wiki/Home"), depth=depth, render_mode=BROWSER)
        assert installs == []
        assert str(refused.value) == (
            f"{NO_SETTING}invalid_config: browser.chrome_path '{shell}' is not executable"
        )
        assert site.paths_since(0, "/wiki/Home") == []


@pytest.mark.usefixtures("allow_loopback", "isolated_env")
class TestRefusedExcludePatterns:
    """crawlberg parses the exclude patterns when it builds the engine, before it fetches a page."""

    @pytest.mark.parametrize("pattern", REFUSED_PATTERNS)
    @pytest.mark.parametrize("render_mode", [HTTP, BROWSER])
    async def test_a_recursive_crawl_names_the_setting_and_gives_crawlbergs_reason(
        self, site, monkeypatch, tmp_path, pattern: str, render_mode: CrawlRenderMode
    ):
        shell = tmp_path / "chrome-headless-shell"
        monkeypatch.setattr(bootstrap, "chromium_installed", lambda: True)
        monkeypatch.setattr(bootstrap, "headless_shell_executable", lambda: shell)
        assert Config(crawl_exclude_patterns=[pattern]).crawl_exclude_patterns == [pattern]
        cfg.crawl_exclude_patterns = [r"/scope/skip/", pattern]
        with pytest.raises(CrawlEngineRefusedError) as refused:
            await crawl_and_save(site.url("/scope/"), depth=1, max_pages=0, render_mode=render_mode)
        message = str(refused.value)
        assert message.startswith(PATTERN_SETTING + "invalid_config: invalid exclude_path regex")
        assert pattern in message
        assert site.paths_since(0, "/scope/") == []

    async def test_a_single_page_crawl_sends_no_pattern_and_fetches_the_page(self, site):
        cfg.crawl_exclude_patterns = [REFUSED_PATTERNS[0]]
        paths = await crawl_and_save(site.url("/wiki/Home"), depth=0, render_mode=HTTP)
        assert len(paths) == 1

    async def test_the_same_crawl_without_the_pattern_reaches_the_site(self, site):
        cfg.crawl_exclude_patterns = [r"/scope/skip/"]
        await crawl_and_save(site.url("/scope/"), depth=1, max_pages=0, render_mode=HTTP)
        assert "/scope/" in site.paths_since(0, "/scope/")


@pytest.mark.usefixtures("allow_loopback", "isolated_env")
class TestRefusedEngineOnTheCli:
    """``lilbee add`` ends a crawl crawlberg refuses with an error and a non-zero exit."""

    @pytest.fixture(autouse=True)
    def _a_shell_that_is_never_started(self, monkeypatch, tmp_path):
        shell = tmp_path / "chrome-headless-shell"
        shell.write_text("", encoding="utf-8")
        monkeypatch.setattr(bootstrap, "chromium_installed", lambda: True)
        monkeypatch.setattr(bootstrap, "headless_shell_executable", lambda: shell)

    def _add(self, *args: str) -> Result:
        return CliRunner().invoke(app, [*args, "--crawl", "--depth", "1"])

    def test_a_refused_exclude_pattern_exits_nonzero_and_names_the_setting(self, site):
        cfg.crawl_exclude_patterns = [REFUSED_PATTERNS[0]]
        cfg.crawl_render_mode = HTTP
        result = self._add("add", site.url("/scope/"))
        assert result.exit_code == 1
        assert "Error: " + PATTERN_SETTING + "invalid_config: invalid exclude_path" in result.output
        assert "Traceback" not in result.output
        assert "Crawled" not in result.output
        assert site.paths_since(0, "/scope/") == []

    def test_a_refused_flag_exits_nonzero_and_names_the_setting(self, site):
        cfg.crawl_browser_extra_args = ["--headless=new"]
        cfg.crawl_render_mode = BROWSER
        result = self._add("add", site.url("/scope/"))
        assert result.exit_code == 1
        assert "Error: " + FLAG_SETTING + "invalid_config: browser.chrome_args" in result.output
        assert "Traceback" not in result.output
        assert site.paths_since(0, "/scope/") == []

    def test_json_output_is_one_error_object(self, site):
        cfg.crawl_exclude_patterns = [REFUSED_PATTERNS[0]]
        cfg.crawl_render_mode = HTTP
        result = self._add("--json", "add", site.url("/scope/"))
        assert result.exit_code == 1
        objects = [json.loads(line) for line in result.output.splitlines() if line.startswith("{")]
        assert [list(found) for found in objects] == [["error"]]
        assert objects[0]["error"].startswith(PATTERN_SETTING)

    def test_a_depth_above_the_engine_limit_is_refused_before_any_request(self, site):
        cfg.crawl_render_mode = HTTP
        result = CliRunner().invoke(app, ["add", site.url("/scope/"), "--crawl", "--depth", "101"])
        assert result.exit_code == 2
        refusal = click.unstyle(result.output)
        assert "--depth" in refusal
        assert "100" in refusal
        assert "Traceback" not in result.output
        assert "Crawled" not in result.output
        assert site.paths_since(0, "/scope/") == []

    def test_a_number_too_large_for_crawlberg_exits_nonzero_with_crawlbergs_reason(self, site):
        _force_oversized_timeout()
        cfg.crawl_render_mode = HTTP
        result = self._add("add", site.url("/scope/"))
        assert result.exit_code == 1
        assert f"Error: {NO_SETTING}{OVERSIZED_REASON}" in result.output
        assert "Traceback" not in result.output
        assert "Crawled" not in result.output
        assert site.paths_since(0, "/scope/") == []

    @pytest.mark.parametrize(
        ("timeout", "depth"),
        [(30, 0), (OVERSIZED_TIMEOUT, 0), (OVERSIZED_TIMEOUT, 1), (30, 101)],
        ids=["single-page", "timeout-single-page", "timeout", "depth"],
    )
    async def test_a_task_cancelled_before_the_fetch_ends_cancelled_not_failed(
        self, site, timeout: int, depth: int
    ):
        if timeout == OVERSIZED_TIMEOUT:
            _force_oversized_timeout()
        else:
            cfg.crawl_timeout = timeout
        task = CrawlTask(
            task_id="t1", url=site.url("/scope/"), depth=depth, max_pages=5, render_mode=HTTP
        )
        task.cancel.set()
        await run_crawl(task)
        assert task.status is TaskStatus.CANCELLED
        assert task.error is None
        assert task.pages_crawled == 0
        assert site.paths_since(0, "/scope/") == []

    @pytest.mark.parametrize(
        ("timeout", "depth"),
        [(OVERSIZED_TIMEOUT, 0), (OVERSIZED_TIMEOUT, 1), (30, 101)],
        ids=["timeout-single-page", "timeout", "depth"],
    )
    async def test_the_same_task_without_the_cancel_fails_with_crawlbergs_reason(
        self, site, timeout: int, depth: int
    ):
        if timeout == OVERSIZED_TIMEOUT:
            _force_oversized_timeout()
        else:
            cfg.crawl_timeout = timeout
        task = CrawlTask(
            task_id="t1", url=site.url("/scope/"), depth=depth, max_pages=5, render_mode=HTTP
        )
        await run_crawl(task)
        assert task.status is TaskStatus.FAILED
        assert task.error is not None
        assert task.error.startswith(NO_SETTING)

    @pytest.mark.parametrize("depth", [0, 1], ids=["single-page", "recursive"])
    async def test_crawlberg_raises_no_runtime_error_for_the_number_and_the_crawl_still_fails(
        self, site, depth: int
    ):
        _force_oversized_timeout()
        with pytest.raises(CrawlEngineRefusedError) as refused:
            await crawl_and_save(site.url("/wiki/Home"), depth=depth, render_mode=HTTP)
        assert str(refused.value) == NO_SETTING + OVERSIZED_REASON
        assert type(refused.value.__cause__) is OverflowError
        assert site.paths_since(0, "/wiki/Home") == []


def _logging_shell(directory: Path, log: Path) -> Path:
    """A script that records its arguments in *log*, then runs the real headless shell."""
    real = bootstrap.headless_shell_executable()
    assert real is not None
    script = directory / "chrome-headless-shell"
    script.write_text(f'#!/bin/sh\necho "$@" >> "{log}"\nexec "{real}" "$@"\n', encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


@full_crawl_only
@pytest.mark.usefixtures("allow_loopback", "isolated_env")
class TestBrowserLaunch:
    @pytest.fixture(autouse=True)
    def _real_browsers(self, monkeypatch):
        require_chromium()
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", REAL_BROWSERS_PATH)

    def _crawl(self, url: str, depth: int = 0, max_pages: int | None = None) -> list[Path]:
        with windows_proactor_loop():
            return asyncio.run(
                crawl_and_save(url, depth=depth, max_pages=max_pages, render_mode=BROWSER)
            )

    @posix_only
    def test_two_stored_pages_stay_two_pages_after_a_browser_recrawl(self, site):
        library = _pair_library(site)
        self._crawl(site.url("/pair/"), depth=1, max_pages=0)
        _assert_each_pair_is_two_pages(site, library)

    def test_a_new_page_is_saved_in_a_folder_that_reads_as_its_link(self, site):
        self._crawl(site.url("/spell/"), depth=1, max_pages=0)
        _assert_saved_in_readable_folders(site)

    def test_a_chrome_variable_that_names_no_binary_does_not_fail_the_crawl(
        self, site, monkeypatch, tmp_path
    ):
        missing = tmp_path / "no-such-chrome"
        monkeypatch.setenv("CHROME", str(missing))
        paths = self._crawl(site.url("/wiki/Home"))
        assert len(paths) == 1
        assert "/wiki/Home" in paths[0].read_text(encoding="utf-8")
        assert os.environ["CHROME"] == str(missing)

    @posix_only
    def test_chromium_starts_from_the_shell_lilbee_names_with_the_extra_flags(
        self, site, monkeypatch, tmp_path
    ):
        log = tmp_path / "launch.log"
        script = _logging_shell(tmp_path, log)
        monkeypatch.setattr(bootstrap, "headless_shell_executable", lambda: script)
        monkeypatch.setenv("CHROME", str(tmp_path / "no-such-chrome"))
        cfg.crawl_browser_extra_args = ["--lang=fr"]
        paths = self._crawl(site.url("/wiki/Home"))
        assert len(paths) == 1
        launch = log.read_text(encoding="utf-8")
        assert "--headless" in launch
        assert "--lang=fr" in launch
        assert "--lang=en_US" not in launch

    @posix_only
    def test_chromium_starts_with_the_default_flags(self, site, monkeypatch, tmp_path):
        log = tmp_path / "launch.log"
        script = _logging_shell(tmp_path, log)
        monkeypatch.setattr(bootstrap, "headless_shell_executable", lambda: script)
        cfg.crawl_browser_extra_args = Config().crawl_browser_extra_args
        paths = self._crawl(site.url("/wiki/Home"))
        assert len(paths) == 1
        launch = log.read_text(encoding="utf-8")
        for flag in DEFAULT_FLAGS:
            assert flag in launch

    @posix_only
    def test_a_recursive_crawl_of_several_pages_starts_chromium_once(
        self, site, monkeypatch, tmp_path
    ):
        log = tmp_path / "launch.log"
        script = _logging_shell(tmp_path, log)
        monkeypatch.setattr(bootstrap, "headless_shell_executable", lambda: script)
        paths = self._crawl(site.url("/wide/"), depth=1, max_pages=BROWSER_CRAWL_PAGES)
        assert len(paths) == BROWSER_CRAWL_PAGES
        assert len(log.read_text(encoding="utf-8").splitlines()) == 1

    def test_a_single_page_browser_crawl_keeps_hidden_text(self, site):
        _assert_hidden_text_kept(self._crawl(site.url(HIDDEN_PAGE)), 1)

    def test_a_recursive_browser_crawl_keeps_hidden_text(self, site):
        _assert_hidden_text_kept(self._crawl(site.url(HIDDEN_PAGE), depth=1), 2)

    def test_a_single_page_browser_crawl_keeps_text_written_after_the_load(self, site):
        cfg.crawl_browser_extra_wait_ms = Config().crawl_browser_extra_wait_ms
        _assert_late_text(self._crawl(site.url(LATE_PAGE)), 1, kept=True)

    def test_a_recursive_browser_crawl_keeps_text_written_after_the_load(self, site):
        cfg.crawl_browser_extra_wait_ms = Config().crawl_browser_extra_wait_ms
        _assert_late_text(self._crawl(site.url(LATE_PAGE), depth=1), 2, kept=True)

    def test_a_wait_near_the_fetch_deadline_still_saves_the_page(self, site):
        cfg.crawl_timeout = LONG_WAIT_TIMEOUT_S
        cfg.crawl_browser_extra_wait_ms = LONG_WAIT_MS
        _assert_late_text(self._crawl(site.url(LATE_PAGE)), 1, kept=True)

    def test_with_no_extra_wait_text_written_after_the_default_wait_is_missing(self, site):
        cfg.crawl_browser_extra_wait_ms = 0
        _assert_late_text(self._crawl(site.url(LATER_PAGE)), 1, kept=False)

    @pytest.mark.parametrize("page", [SVG_PAGE, FRAMESET_PAGE])
    def test_a_page_with_no_body_ends_before_the_page_timeout(self, site, page):
        cfg.crawl_timeout = NO_BODY_TIMEOUT_S
        start = time.monotonic()
        self._crawl(site.url(page))
        assert time.monotonic() - start < NO_BODY_TIMEOUT_S

    def test_a_longer_extra_wait_keeps_text_written_after_the_default_wait(self, site):
        cfg.crawl_browser_extra_wait_ms = LATER_WAIT_MS
        _assert_late_text(self._crawl(site.url(LATER_PAGE)), 1, kept=True)

    def test_with_no_override_chrome_names_itself_and_its_hints_agree(self, site):
        paths = self._crawl(site.url("/hints/"), depth=1)
        assert len(paths) == 2
        agents = set(site.agents_for("/hints/"))
        assert len(agents) == 1, agents
        release = HEADLESS_CHROME_RELEASE.search(agents.pop())
        assert release is not None
        hints = site.hints_for("/hints/")
        assert sorted(hints) == sorted(HINT_PAGES)
        brand = f'"Chromium";v="{release.group(1)}"'
        for page in HINT_PAGES:
            assert all(brand in sent["sec-ch-ua"] for sent in hints[page]), (page, hints[page])

    def test_a_browser_page_request_carries_chromes_own_accept_header(self, site):
        paths = self._crawl(site.url("/wiki/Home"))
        assert len(paths) == 1
        (sent,) = site.accept_headers_for("/wiki/Home")
        assert sent["accept"] != BROWSER_ACCEPT_HEADERS["accept"]
        assert sent["accept"].startswith("text/html,")

    def test_an_override_names_every_browser_request_and_its_hints_follow(self, site):
        cfg.crawl_user_agent = OVERRIDE
        paths = self._crawl(site.url("/hints/"), depth=1)
        assert len(paths) == 2
        assert set(site.agents_for("/hints/")) == {OVERRIDE}
        assert site.agents_for(SITEMAP_PATH) == [OVERRIDE]
        hints = site.hints_for("/hints/")
        assert sorted(hints) == sorted(HINT_PAGES)
        brand = f'"Chromium";v="{OVERRIDE_MAJOR}"'
        for page in HINT_PAGES:
            assert all(brand in sent["sec-ch-ua"] for sent in hints[page]), (page, hints[page])


@pytest.mark.usefixtures("allow_loopback", "isolated_env")
class TestUserAgent:
    async def test_with_no_override_http_requests_name_crawlberg(self, site):
        paths = await crawl_and_save(site.url("/wide/"), depth=1, max_pages=3, render_mode=HTTP)
        assert len(paths) == 3
        agents = site.agents_for("/wide/")
        assert len(agents) >= 3
        assert all(agent.startswith(CRAWLBERG_USER_AGENT_START) for agent in agents), agents
        sitemap_agents = site.agents_for(SITEMAP_PATH)
        assert len(sitemap_agents) == 1
        assert sitemap_agents[0].startswith(HTTP_CLIENT_USER_AGENT_START), sitemap_agents

    async def test_an_override_names_every_request_of_a_recursive_http_crawl(self, site):
        cfg.crawl_user_agent = OVERRIDE
        paths = await crawl_and_save(site.url("/wide/"), depth=1, max_pages=3, render_mode=HTTP)
        assert len(paths) == 3
        agents = site.agents_for("/wide/")
        assert len(agents) >= 3
        assert set(agents) == {OVERRIDE}
        assert site.agents_for(SITEMAP_PATH) == [OVERRIDE]

    async def test_an_override_names_a_single_page_http_request(self, site):
        cfg.crawl_user_agent = OVERRIDE
        await crawl_and_save(site.url("/wiki/Home"), depth=0, render_mode=HTTP)
        assert site.agents_for("/wiki/Home") == [OVERRIDE]

    async def test_each_request_of_a_recursive_http_crawl_carries_the_browser_accept_headers(
        self, site
    ):
        paths = await crawl_and_save(site.url("/wide/"), depth=1, max_pages=3, render_mode=HTTP)
        assert len(paths) == 3
        sent = site.accept_headers_for("/wide/")
        assert len(sent) >= 3
        assert sent == [BROWSER_ACCEPT_HEADERS] * len(sent)

    async def test_a_single_page_http_request_carries_the_browser_accept_headers(self, site):
        await crawl_and_save(site.url("/wiki/Home"), depth=0, render_mode=HTTP)
        assert site.accept_headers_for("/wiki/Home") == [BROWSER_ACCEPT_HEADERS]

    async def test_http_requests_carry_no_client_hints(self, site):
        cfg.crawl_user_agent = OVERRIDE
        await crawl_and_save(site.url("/hints/"), depth=1, render_mode=HTTP)
        assert site.hints_for("/hints/") == {"/hints/": [{}], "/hints/next": [{}]}
        assert set(site.agents_for("/hints/")) == {OVERRIDE}


@pytest.fixture
def server_log(isolated_env) -> Iterator[Path]:
    """server.log under the data root, installed as serve installs it, taken out afterwards."""
    root = logging.getLogger()
    root_level = root.level
    handler_levels = [(handler, handler.level) for handler in root.handlers]
    # Importing litellm raises the httpx logger to WARNING; the test reads its request line.
    httpx_logger = logging.getLogger("httpx")
    httpx_level = httpx_logger.level
    httpx_logger.setLevel(logging.INFO)
    cfg.data_root = isolated_env
    yield setup_server_log_file()
    httpx_logger.setLevel(httpx_level)
    earlier = [handler for handler, _ in handler_levels]
    for handler in [h for h in root.handlers if h not in earlier]:
        root.removeHandler(handler)
        handler.close()
    for handler, level in handler_levels:
        handler.setLevel(level)
    root.setLevel(root_level)


def _credentialed(site: _Site, path: str) -> str:
    """The address of *path* on *site* with the seed username and password in it."""
    return site.url(path).replace("://", f"://{SEED_USER}:{SEED_PASSWORD}@", 1)


@pytest.mark.usefixtures("allow_loopback", "isolated_env")
class TestSeedPasswordInTheServerLog:
    @pytest.fixture(autouse=True)
    def _sitemap_behind_basic_auth(self, site):
        site.sitemap_authorization = SEED_BASIC_AUTH

    @staticmethod
    def _assert_sitemap_logged_without_password(site: _Site, log_path: Path) -> None:
        for handler in logging.getLogger().handlers:
            handler.flush()
        logged = log_path.read_text(encoding="utf-8")
        assert f"HTTP Request: GET {site.url(SITEMAP_PATH)}" in logged
        assert SEED_PASSWORD not in logged
        assert site.authorizations_for(SITEMAP_PATH) == [SEED_BASIC_AUTH]

    def test_a_rest_site_crawl_logs_no_password(self, site, server_log):
        auth_mod.session_manager.disable()
        try:
            server = create_app()
            # Building the app replaces the root handlers; serve installs the file log after it.
            assert setup_server_log_file() == server_log
            response = TestClient(server).post(
                "/api/crawl",
                json={
                    "url": _credentialed(site, "/wide/"),
                    "depth": 1,
                    "max_pages": 2,
                    "render_mode": HTTP,
                },
            )
        finally:
            auth_mod.session_manager.cleanup()
        assert response.status_code == HTTPStatus.CREATED
        assert b"files_written" in response.content
        self._assert_sitemap_logged_without_password(site, server_log)

    async def test_an_mcp_site_crawl_logs_no_password(self, site, server_log):
        with mock.patch("lilbee.data.ingest.sync", new_callable=mock.AsyncMock):
            started = await mcp_crawl(
                url=_credentialed(site, "/wide/"), depth=1, max_pages=2, render_mode=HTTP
            )
            task = get_task(started["task_id"])
            assert task is not None and task._async_task is not None
            await asyncio.wait_for(task._async_task, timeout=60)
        assert task.status is TaskStatus.DONE
        self._assert_sitemap_logged_without_password(site, server_log)


def _closed_port() -> int:
    """A loopback port nothing listens on."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _failure_reasons(events: list[tuple[EventType, object]]) -> list[str]:
    return [
        data.reason
        for kind, data in events
        if kind is EventType.CRAWL_PAGE_FAILED and isinstance(data, CrawlPageFailedEvent)
    ]


async def _robots_site_crawl(site: _Site, mode: CrawlRenderMode) -> list[Path]:
    return await crawl_and_save(site.url(ROBOTS_LISTING), depth=1, max_pages=0, render_mode=mode)


async def _page_crawl(
    site: _Site, path: str, mode: CrawlRenderMode
) -> tuple[list[Path], list[tuple[EventType, object]]]:
    """The saved paths and the progress events of a one-page crawl of *path*."""
    events: list[tuple[EventType, object]] = []
    paths = await crawl_and_save(
        site.url(path),
        depth=0,
        render_mode=mode,
        on_progress=lambda kind, data: events.append((kind, data)),
    )
    return paths, events


async def _robots_single_crawl(
    site: _Site, mode: CrawlRenderMode
) -> tuple[list[Path], list[tuple[EventType, object]]]:
    return await _page_crawl(site, ROBOTS_DISALLOWED, mode)


def _assert_site_crawl_honours_robots(site: _Site, paths: list[Path]) -> None:
    assert site.paths_since(0, ROBOTS_DISALLOWED) == []
    assert site.paths_since(0, ROBOTS_ALLOWED) == [ROBOTS_ALLOWED]
    assert site.paths_since(0, "/robots.txt")
    assert len(paths) == 2


def _assert_single_crawl_honours_robots(
    site: _Site, paths: list[Path], events: list[tuple[EventType, object]]
) -> None:
    assert site.paths_since(0, ROBOTS_DISALLOWED) == []
    assert site.paths_since(0, "/robots.txt")
    assert paths == []
    assert _failure_reasons(events) == [ROBOTS_REFUSED_MESSAGE]


def _assert_robots_ignored(site: _Site, site_paths: list[Path], single_paths: list[Path]) -> None:
    assert site.paths_since(0, ROBOTS_DISALLOWED) == [ROBOTS_DISALLOWED] * 2
    assert site.paths_since(0, "/robots.txt") == []
    assert len(site_paths) == 3
    assert single_paths != []


@pytest.mark.usefixtures("allow_loopback", "isolated_env")
class TestRobotsTxt:
    async def test_a_site_crawl_never_requests_a_page_robots_txt_disallows(self, site):
        _assert_site_crawl_honours_robots(site, await _robots_site_crawl(site, HTTP))

    async def test_a_disallowed_single_page_is_not_requested_and_the_crawl_says_why(self, site):
        paths, events = await _robots_single_crawl(site, HTTP)
        _assert_single_crawl_honours_robots(site, paths, events)

    async def test_with_the_setting_off_the_disallowed_page_is_crawled(self, site):
        cfg.crawl_respect_robots_txt = False
        site_paths = await _robots_site_crawl(site, HTTP)
        single_paths, _ = await _robots_single_crawl(site, HTTP)
        _assert_robots_ignored(site, site_paths, single_paths)

    @pytest.mark.parametrize(
        "status", [HTTPStatus.SERVICE_UNAVAILABLE, HTTPStatus.REQUEST_TIMEOUT], ids=["503", "408"]
    )
    async def test_a_robots_txt_that_cannot_be_read_skips_the_page_and_says_why(self, site, status):
        site.robots_status = status
        events: list[tuple[EventType, object]] = []
        paths = await crawl_and_save(
            site.url(ROBOTS_ALLOWED),
            depth=0,
            render_mode=HTTP,
            on_progress=lambda kind, data: events.append((kind, data)),
        )
        assert site.paths_since(0, ROBOTS_ALLOWED) == []
        assert site.paths_since(0, "/robots.txt")
        assert paths == []
        reasons = _failure_reasons(events)
        assert len(reasons) == 1
        assert reasons[0].startswith("The site's robots.txt could not be read ("), reasons
        assert "crawl_respect_robots_txt" in reasons[0]

    async def test_mcp_add_of_a_site_whose_robots_txt_cannot_be_read_names_the_setting(self, site):
        site.robots_status = HTTPStatus.SERVICE_UNAVAILABLE
        result = await mcp_add(paths=[site.url(ROBOTS_ALLOWED)], render_mode=HTTP)
        assert site.paths_since(0, "/robots.txt")
        assert site.paths_since(0, ROBOTS_ALLOWED) == []
        assert "The site's robots.txt could not be read (" in result["error"], result
        assert "crawl_respect_robots_txt" in result["error"], result

    async def test_a_site_that_does_not_answer_gets_the_robots_txt_message_with_the_reason(self):
        events: list[tuple[EventType, object]] = []
        paths = await crawl_and_save(
            f"http://127.0.0.1:{_closed_port()}/page",
            depth=0,
            render_mode=HTTP,
            on_progress=lambda kind, data: events.append((kind, data)),
        )
        result = await mcp_add(paths=[f"http://127.0.0.1:{_closed_port()}/page"], render_mode=HTTP)
        assert paths == []
        reasons = [*_failure_reasons(events), result["error"]]
        assert len(reasons) == 2
        for reason in reasons:
            assert "The site's robots.txt could not be read (connection: " in reason, reason
            assert "Set crawl_respect_robots_txt to false" in reason, reason


@full_crawl_only
@pytest.mark.usefixtures("allow_loopback", "isolated_env")
class TestRobotsTxtInBrowserMode:
    @pytest.fixture(autouse=True)
    def _real_browsers(self, monkeypatch):
        require_chromium()
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", REAL_BROWSERS_PATH)

    def test_a_site_crawl_never_requests_a_page_robots_txt_disallows(self, site):
        with windows_proactor_loop():
            paths = asyncio.run(_robots_site_crawl(site, BROWSER))
        _assert_site_crawl_honours_robots(site, paths)

    def test_a_disallowed_single_page_is_not_requested_and_the_crawl_says_why(self, site):
        with windows_proactor_loop():
            paths, events = asyncio.run(_robots_single_crawl(site, BROWSER))
        _assert_single_crawl_honours_robots(site, paths, events)

    def test_with_the_setting_off_the_disallowed_page_is_crawled(self, site):
        cfg.crawl_respect_robots_txt = False
        with windows_proactor_loop():
            site_paths = asyncio.run(_robots_site_crawl(site, BROWSER))
            single_paths, _ = asyncio.run(_robots_single_crawl(site, BROWSER))
        _assert_robots_ignored(site, site_paths, single_paths)

    def test_a_certificate_only_chrome_accepts_names_the_setting_that_crawls_the_page(
        self, self_signed_site
    ):
        """crawlberg reads robots.txt without Chrome, so it refuses the certificate Chrome takes."""
        with windows_proactor_loop():
            refused, events = asyncio.run(_page_crawl(self_signed_site, ROBOTS_ALLOWED, BROWSER))
            cfg.crawl_respect_robots_txt = False
            crawled, _ = asyncio.run(_page_crawl(self_signed_site, ROBOTS_ALLOWED, BROWSER))
        reasons = _failure_reasons(events)
        assert refused == []
        assert len(reasons) == 1
        assert reasons[0].startswith("The site's robots.txt could not be read (ssl: "), reasons
        assert "crawl_respect_robots_txt" in reasons[0]
        assert len(crawled) == 1
        assert self_signed_site.paths_since(0, ROBOTS_ALLOWED) == [ROBOTS_ALLOWED]


async def _cookie_crawl(site: _Site, mode: CrawlRenderMode) -> list[Path]:
    return await crawl_and_save(site.url(COOKIE_START), depth=1, max_pages=0, render_mode=mode)


def _assert_cookie_page_saved(site: _Site, paths: list[Path]) -> None:
    """The page behind the cookie was asked for with it and saved with its text."""
    assert site.cookies_for(COOKIE_START) != []
    assert set(site.cookies_for(COOKIE_NEED)) == {SESSION_COOKIE}
    assert len(paths) == 2
    texts = [path.read_text(encoding="utf-8") for path in paths]
    assert sum(COOKIE_TEXT in text for text in texts) == 1, texts


@pytest.mark.usefixtures("allow_loopback", "isolated_env")
class TestCookies:
    async def test_a_cookie_a_page_sets_is_sent_to_the_next_page(self, site):
        _assert_cookie_page_saved(site, await _cookie_crawl(site, HTTP))


@full_crawl_only
@pytest.mark.usefixtures("allow_loopback", "isolated_env")
class TestCookiesInBrowserMode:
    @pytest.fixture(autouse=True)
    def _real_browsers(self, monkeypatch):
        require_chromium()
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", REAL_BROWSERS_PATH)

    def test_a_cookie_a_page_sets_is_sent_to_the_next_page(self, site):
        with windows_proactor_loop():
            paths = asyncio.run(_cookie_crawl(site, BROWSER))
        _assert_cookie_page_saved(site, paths)
