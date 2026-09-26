"""A local fixture website for the end-to-end crawl tests, served from a thread."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import socket
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from typer.testing import CliRunner

from lilbee.app.services import reset_services
from lilbee.catalog import download_model
from lilbee.cli.app import app
from lilbee.core.config import cfg
from lilbee.core.config.enums import CrawlRenderMode
from lilbee.core.system import canonical_models_dir
from lilbee.crawler import (
    chromium_installed,
    crawler_available,
    crawler_browsers_path,
    url_filter,
)
from lilbee.runtime import asyncio_loop
from tests.integration.conftest import EMBED_ENTRY, EMBEDDING_DIM, _resolve_installed_ref

LOOPBACK = "127.0.0.1"
NAMED_HOST = "site.test"
SUB_HOST = f"docs.{NAMED_HOST}"

TREE_PATHS = ("/a/", "/a/b/", "/a/b/c/", "/a/b/c/d/")
WIDE_PAGE_COUNT = 30
SLOW_PAGE_COUNT = 40
SLOW_PAGE_DELAY_S = 0.4
POOL_PAGE_COUNT = 12
POOL_PAGE_DELAY_S = 0.3
PACED_PAGE_COUNT = 5
STALL_DELAY_S = 6.0
RATE_LIMITED_ATTEMPTS = 2
FULL_CRAWL_ENV = "LILBEE_TEST_FULL_CRAWL"
# Read at import, before the per-test fixture points the browser cache at an empty directory.
REAL_BROWSERS_PATH = os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or str(crawler_browsers_path())
_LOOPBACK_NETWORKS = (ipaddress.ip_network("127.0.0.0/8"), ipaddress.ip_network("::1/128"))

needs_crawler = pytest.mark.skipif(
    not crawler_available(), reason="the crawler extra is not installed"
)


def _resolves_to_loopback(name: str) -> bool:
    try:
        infos = socket.getaddrinfo(name, None)
    except OSError:
        return False
    return all(ipaddress.ip_address(info[4][0]).is_loopback for info in infos)


needs_named_hosts = pytest.mark.skipif(
    not all(_resolves_to_loopback(name) for name in (NAMED_HOST, SUB_HOST)),
    reason=f"{NAMED_HOST} and {SUB_HOST} do not resolve to loopback; CI maps them in hosts",
)
full_crawl_only = pytest.mark.skipif(
    os.environ.get(FULL_CRAWL_ENV) != "1",
    reason=f"browser-mode and TUI crawl tests run in one CI cell per OS; set {FULL_CRAWL_ENV}=1",
)

HOME_TEXT = "The home page describes the amber lighthouse keeper."
NAV_TEXT = "Home nav"
TABLE_CELL_NAME = "Cobalt"
TABLE_CELL_VALUE = "4217"
CODE_LINE = 'return "kestrel-marker"'
TREE_TEXT = "Tree level text for {path}."
CHILD_TEXT = "The relative child page talks about velvet orchards."
SIBLING_TEXT = "The relative sibling page talks about granite harbors."
BASE_TEXT = "The base href page explains copper lanterns."
LEAF_TEXT = "The leaf reached through the base href mentions saffron kites."
LEAF_WITHOUT_BASE_TEXT = "This leaf is only reachable when the base href is ignored."
TARGET_TEXT = "The redirect target page mentions indigo glaciers."
NEXT_TEXT = "The page next to the redirect target mentions teal meadows."
SUB_TEXT = "The subdomain docs page mentions walnut observatories."
WIDE_TEXT = "Wide page number {n}."
SLOW_TEXT = "Slow page number {n}."
EXCLUDED_TEXT = "A query page that the default exclude patterns drop."
PDF_TEXT = "PDF body text"
SCRIPT_TEXT = "Script painted text"
RATE_LIMITED_TEXT = "The rate limited page answered after its retries."
STALL_TEXT = "The stalled page answered too late."
TIMEOUT_NEIGHBOUR_TEXT = "Timeout neighbour page {name}."
FILTER_TEXT = "Filter fixture page {name}."
QUERY_TEXT = "Filter fixture query {query}."

FILLER = (
    "<p>This fixture page carries enough ordinary prose to read as a real document. "
    "The quick brown fox jumps over the lazy dog while the librarian files each record "
    "in its drawer. Rivers carry silt to the delta, where farmers plant rice in the rich "
    "soil every spring. The committee meets on Tuesdays to review the budget.</p>"
)

_HTML = "text/html; charset=utf-8"
_MINI_PDF = (
    b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]/Contents 4 0 R>>endobj\n"
    b"4 0 obj<</Length 44>>stream\nBT /F1 12 Tf 10 100 Td (PDF body text) Tj ET\n"
    b"endstream endobj\ntrailer<</Root 1 0 R>>\n%%EOF\n"
)


@dataclass(frozen=True)
class Response:
    """One HTTP response the fixture site sends."""

    status: int
    body: bytes = b""
    content_type: str = _HTML
    location: str = ""
    delay_s: float = 0.0


@dataclass
class SiteRequest:
    """One request the fixture site received, with when it started and finished."""

    at: float
    host: str
    path: str
    end: float = float("inf")


def _page(title: str, body: str, head: str = "") -> Response:
    html = (
        f"<!doctype html><html><head><title>{title}</title>{head}</head>"
        f"<body>{body}{FILLER}</body></html>"
    )
    return Response(200, html.encode())


def _redirect(location: str, status: int = 301) -> Response:
    return Response(status, location=location)


def _home(port: int) -> Response:
    links = "".join(
        f'<li><a href="{href}">{label}</a></li>'
        for href, label in (
            ("/a/", "Tree root"),
            ("rel/child.html", "Relative child"),
            ("/docs/base/page.html", "Base href page"),
            ("/redirect-me", "Redirected link"),
            ("/file.pdf", "A PDF asset"),
            ("/missing", "A missing page"),
            ("/blog?p=42", "An excluded query page"),
            (f"http://{SUB_HOST}:{port}/", "Subdomain docs"),
        )
    )
    return _page(
        "Fixture Home",
        f'<nav><a href="/">{NAV_TEXT}</a> | <a href="/a/">Tree nav</a></nav>'
        f"<main><h1>Fixture Home</h1><p>{HOME_TEXT}</p>"
        f"<table><thead><tr><th>Name</th><th>Value</th></tr></thead><tbody>"
        f"<tr><td>{TABLE_CELL_NAME}</td><td>{TABLE_CELL_VALUE}</td></tr></tbody></table>"
        f'<pre><code class="language-python">def marker():\n    {CODE_LINE}\n</code></pre>'
        f'<form action="/search" method="get"><label>Search label</label>'
        f'<input type="text" name="q"><button type="submit">Go</button></form>'
        f"<ul>{links}</ul></main>",
    )


def _tree(path: str) -> Response:
    index = TREE_PATHS.index(path)
    deeper = TREE_PATHS[index + 1] if index + 1 < len(TREE_PATHS) else "/"
    text = TREE_TEXT.format(path=path)
    return _page(f"Tree {path}", f'<h1>Tree {path}</h1><p>{text}</p><a href="{deeper}">deeper</a>')


def _listing(prefix: str, count: int) -> Response:
    links = " ".join(f'<a href="{prefix}p{n}">p{n}</a>' for n in range(count))
    return _page(f"Index {prefix}", f"<h1>Index {prefix}</h1>{links}")


def _main_routes(origin: str, port: int) -> dict[str, Callable[[], Response]]:
    return {
        "/": lambda: _home(port),
        "/rel/child.html": lambda: _page(
            "Child", f'<h1>Child</h1><p>{CHILD_TEXT}</p><a href="sibling.html">sibling</a>'
        ),
        "/rel/sibling.html": lambda: _page("Sibling", f"<h1>Sibling</h1><p>{SIBLING_TEXT}</p>"),
        "/docs/base/page.html": lambda: _page(
            "Base",
            f'<h1>Base</h1><p>{BASE_TEXT}</p><a href="leaf.html">leaf via base</a>',
            head=f'<base href="{origin}/other/">',
        ),
        "/other/leaf.html": lambda: _page("Leaf", f"<h1>Leaf</h1><p>{LEAF_TEXT}</p>"),
        "/docs/base/leaf.html": lambda: _page("Wrong", f"<p>{LEAF_WITHOUT_BASE_TEXT}</p>"),
        "/redirect-me": lambda: _redirect("/landing/target.html"),
        "/seed-redirect": lambda: _redirect("/landing/target.html", status=302),
        "/landing/target.html": lambda: _page(
            "Target", f'<h1>Target</h1><p>{TARGET_TEXT}</p><a href="next.html">next page</a>'
        ),
        "/landing/next.html": lambda: _page("Next", f"<h1>Next</h1><p>{NEXT_TEXT}</p>"),
        "/file.pdf": lambda: Response(200, _MINI_PDF, content_type="application/pdf"),
        "/blog": lambda: _page("Blog", f"<p>{EXCLUDED_TEXT}</p>"),
        "/js/": lambda: _page(
            "Script",
            '<h1>Script</h1><p id="painted"></p>'
            "<script>document.getElementById('painted').textContent = "
            "'Script ' + 'painted text';</script>",
        ),
        "/wide/": lambda: _listing("/wide/", WIDE_PAGE_COUNT),
        "/slow/": lambda: _listing("/slow/", SLOW_PAGE_COUNT),
    }


def _setting_routes() -> dict[str, Callable[[], Response]]:
    neighbour = TIMEOUT_NEIGHBOUR_TEXT.format
    return {
        "/pool/": lambda: _listing("/pool/", POOL_PAGE_COUNT),
        "/paced/": lambda: _listing("/paced/", PACED_PAGE_COUNT),
        "/timeout/": lambda: _page(
            "Timeout",
            '<a href="fast-a">a</a> <a href="stall">stall</a> <a href="fast-b">b</a> '
            '<a href="gone">gone</a>',
        ),
        "/timeout/fast-a": lambda: _page("A", f"<p>{neighbour(name='a')}</p>"),
        "/timeout/fast-b": lambda: _page("B", f"<p>{neighbour(name='b')}</p>"),
        "/timeout/stall": lambda: Response(
            200, _page("Stall", f"<p>{STALL_TEXT}</p>").body, delay_s=STALL_DELAY_S
        ),
        "/filters/": lambda: _page(
            "Filters",
            '<a href="keep.html">keep</a> <a href="drop-me.html">drop</a> '
            '<a href="item?lang=7">lang</a> <a href="item?page=ok">page</a>',
        ),
        "/filters/keep.html": lambda: _page("Keep", f"<p>{FILTER_TEXT.format(name='keep')}</p>"),
        "/filters/drop-me.html": lambda: _page("Drop", f"<p>{FILTER_TEXT.format(name='drop')}</p>"),
    }


def _rate_limited(path: str, attempt: int) -> Response | None:
    """``/flaky/<tag>/`` links to ``p``, which answers 429 for its first attempts."""
    parts = path.split("/")
    if len(parts) != 4 or parts[1] != "flaky":
        return None
    if parts[3] == "":
        return _page("Flaky", '<h1>Flaky</h1><a href="p">rate limited page</a>')
    if attempt <= RATE_LIMITED_ATTEMPTS:
        return Response(429, b"<html><body>Too many requests</body></html>")
    return _page("Flaky page", f"<p>{RATE_LIMITED_TEXT}</p>")


def _numbered(path: str) -> Response | None:
    for prefix, text, delay in (
        ("/wide/p", WIDE_TEXT, 0.0),
        ("/slow/p", SLOW_TEXT, SLOW_PAGE_DELAY_S),
        ("/pool/p", WIDE_TEXT, POOL_PAGE_DELAY_S),
        ("/paced/p", WIDE_TEXT, 0.0),
    ):
        if path.startswith(prefix) and path[len(prefix) :].isdigit():
            n = path[len(prefix) :]
            page = _page(f"Page {n}", f"<h1>Page {n}</h1><p>{text.format(n=n)}</p>")
            return Response(200, page.body, delay_s=delay)
    return None


def route(host_header: str, path: str, query: str, port: int, attempt: int) -> Response:
    """Return the response for *path* on the host named by *host_header*.

    *attempt* counts this request among all requests for the same host and path.
    """
    hostname = host_header.rsplit(":", 1)[0]
    if hostname == SUB_HOST:
        if path == "/":
            return _page("Sub Docs", f"<h1>Sub docs</h1><p>{SUB_TEXT}</p>")
        return Response(404, b"<h1>Not found</h1>")
    if path in TREE_PATHS:
        return _tree(path)
    if path == "/filters/item":
        return _page("Query", f"<p>{QUERY_TEXT.format(query=query)}</p>")
    routes = {**_main_routes(f"http://{host_header}", port), **_setting_routes()}
    handler = routes.get(path)
    if handler is not None:
        return handler()
    found = _rate_limited(path, attempt) or _numbered(path)
    return found or Response(404, b"<html><body><h1>404 Not Found</h1></body></html>")


@dataclass
class CrawlSite:
    """A running fixture site and the requests it has received."""

    port: int
    requests: list[SiteRequest] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def url(self, path: str = "/", host: str = LOOPBACK) -> str:
        """Absolute URL of *path* on *host* at this site's port."""
        return f"http://{host}:{self.port}{path}"

    def record(self, host: str, path: str) -> tuple[SiteRequest, int]:
        """Log one received request; return it and its attempt number for this host and path."""
        with self.lock:
            request = SiteRequest(time.monotonic(), host, path)
            self.requests.append(request)
            attempt = sum(1 for r in self.requests if (r.host, r.path) == (host, path))
            return request, attempt

    def finish(self, request: SiteRequest) -> None:
        """Mark *request* as answered."""
        with self.lock:
            request.end = time.monotonic()

    def requests_under(self, prefix: str, since: float = 0.0) -> list[SiteRequest]:
        """Requests whose path starts with *prefix*, made at or after *since*."""
        with self.lock:
            return [r for r in self.requests if r.path.startswith(prefix) and r.at >= since]

    def peak_in_flight(self, prefix: str, since: float = 0.0) -> int:
        """The most requests under *prefix* that the site was answering at one time."""
        edges = sorted(
            edge for r in self.requests_under(prefix, since) for edge in ((r.at, 1), (r.end, -1))
        )
        peak = current = 0
        for _at, step in edges:
            current += step
            peak = max(peak, current)
        return peak

    def requested_paths(self, prefix: str = "/", since: float = 0.0) -> list[str]:
        """Paths requested under *prefix* at or after the monotonic time *since*."""
        return [r.path for r in self.requests_under(prefix, since)]


def _handler_for(site: CrawlSite) -> type[BaseHTTPRequestHandler]:
    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: object) -> None:
            """Silence the default stderr access log."""

        def do_GET(self) -> None:
            host = (self.headers.get("Host") or LOOPBACK).lower()
            parts = urlsplit(self.path)
            request, attempt = site.record(host, self.path)
            try:
                self._answer(route(host, parts.path, parts.query, site.port, attempt))
            finally:
                site.finish(request)

        def _answer(self, response: Response) -> None:
            if response.delay_s:
                time.sleep(response.delay_s)
            self.send_response(response.status)
            self.send_header("Content-Type", response.content_type)
            if response.location:
                self.send_header("Location", response.location)
            self.send_header("Content-Length", str(len(response.body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(response.body)

        def do_HEAD(self) -> None:
            self.do_GET()

    return _Handler


class RunningSite:
    """Context manager that serves a :class:`CrawlSite` on a free loopback port."""

    def __init__(self) -> None:
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> CrawlSite:
        site = CrawlSite(port=0)
        self._server = ThreadingHTTPServer((LOOPBACK, 0), _handler_for(site))
        self._server.daemon_threads = True
        site.port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="crawl-fixture-site", daemon=True
        )
        self._thread.start()
        return site

    def __exit__(self, *exc: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=10)


def _configure_embedding() -> None:
    cfg.models_dir = canonical_models_dir()
    download_model(EMBED_ENTRY)
    cfg.embedding_model = _resolve_installed_ref(EMBED_ENTRY.hf_repo)
    cfg.embedding_dim = EMBEDDING_DIM


@contextmanager
def crawl_sandbox(
    root: Path, render_mode: CrawlRenderMode = CrawlRenderMode.HTTP
) -> Iterator[Path]:
    """Point lilbee at *root* with loopback crawling allowed and a real embedding model.

    Yields the canonical data root, which is what ``--data-dir`` resolves *root* to.
    """
    snapshot = cfg.model_copy()
    allowed = tuple(n for n in url_filter.get_blocked_networks() if n not in _LOOPBACK_NETWORKS)
    root = root.resolve()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(url_filter, "get_blocked_networks", lambda: allowed)
        mp.setenv("LILBEE_SKIP_TOML_CONFIG", "1")
        mp.setenv("LILBEE_DATA", str(root))
        mp.setenv("PLAYWRIGHT_BROWSERS_PATH", REAL_BROWSERS_PATH)
        cfg.data_root = root
        cfg.documents_dir = root / "documents"
        cfg.data_dir = root / "data"
        cfg.lancedb_dir = root / "data" / "lancedb"
        cfg.documents_dir.mkdir(parents=True, exist_ok=True)
        _configure_embedding()
        cfg.wiki = False
        cfg.concept_graph = False
        cfg.hyde = False
        cfg.query_expansion_count = 0
        cfg.crawl_mean_delay = 0.0
        cfg.crawl_max_delay_range = 0.0
        cfg.crawl_sync_interval = 0
        cfg.crawl_render_mode = render_mode
        reset_services()
        try:
            yield root
        finally:
            reset_services()
            for name in type(cfg).model_fields:
                setattr(cfg, name, getattr(snapshot, name))


@contextmanager
def windows_proactor_loop() -> Iterator[None]:
    """On Windows, run the block's crawl loops on the proactor loop, which can start Chromium.

    Covers loops made with ``asyncio.new_event_loop()`` inside the block and the TUI's
    background loop, which is made here so a test's own loop policy cannot replace it.
    """
    if sys.platform != "win32":
        yield
        return
    previous = asyncio.get_event_loop_policy()
    asyncio_loop.shutdown()
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    try:
        asyncio_loop.get_loop()
        yield
    finally:
        asyncio_loop.shutdown()
        asyncio.set_event_loop_policy(previous)


def require_chromium() -> None:
    """Fail, never skip, when a browser-mode test finds no Chromium in the real browser cache."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PLAYWRIGHT_BROWSERS_PATH", REAL_BROWSERS_PATH)
        assert chromium_installed(), "browser-mode crawl tests need Chromium: lilbee setup crawler"


def saved_pages(root: Path, host: str = LOOPBACK) -> dict[str, str]:
    """Saved markdown under the documents ``_web`` dir for *host*, keyed by relative name."""
    host_dir = root / "documents" / "_web" / host
    if not host_dir.is_dir():
        return {}
    return {
        p.relative_to(host_dir).as_posix(): p.read_text(encoding="utf-8")
        for p in sorted(host_dir.rglob("*.md"))
    }


def all_saved_text(root: Path) -> str:
    """Every file saved under the documents ``_web`` dir, for any host, read as text."""
    web = root / "documents" / "_web"
    files = sorted(p for p in web.rglob("*") if p.is_file()) if web.is_dir() else []
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in files)


@dataclass(frozen=True)
class CrawlRun:
    """What one ``lilbee add`` crawl reported and left on disk."""

    report: dict[str, object]
    pages: dict[str, str]


def cli_json(output: str) -> dict[str, object]:
    """The JSON object the CLI printed, ignoring any engine log lines."""
    for line in output.splitlines():
        if line.startswith("{"):
            return dict(json.loads(line))
    raise AssertionError(f"no JSON object in CLI output:\n{output}")


def run_cli(*args: str) -> str:
    """Run the ``lilbee`` CLI in-process with ``--json`` and return its output."""
    result = CliRunner().invoke(app, ["--json", *args])
    assert result.exit_code == 0, result.output
    return result.output


def run_add(root: Path, url: str, *flags: str, host: str = LOOPBACK) -> CrawlRun:
    """Run ``lilbee --json add URL`` with *flags* and collect what it saved for *host*."""
    output = run_cli("add", url, "--data-dir", str(root), *flags)
    return CrawlRun(cli_json(output), saved_pages(root, host))
