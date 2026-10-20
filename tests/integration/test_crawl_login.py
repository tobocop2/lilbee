"""A crawl reads pages behind each kind of login, and the secret goes nowhere else."""

from __future__ import annotations

import base64
import logging
import threading
from collections.abc import Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from litestar.testing import TestClient

crawlberg = pytest.importorskip("crawlberg")

import lilbee.server.auth as auth_mod  # noqa: E402
from lilbee.core.config import cfg  # noqa: E402
from lilbee.core.config.enums import CrawlRenderMode  # noqa: E402
from lilbee.crawler import url_filter  # noqa: E402
from lilbee.crawler.models import CrawlLogin  # noqa: E402
from lilbee.crawler.runner import crawl_and_save  # noqa: E402
from lilbee.server.app import create_app  # noqa: E402

LOOPBACK = ("127.0.0.0/8", "::1/128")
USER = "alice"
PASSWORD = "Pw-basic-9Zt"
TOKEN = "Bearer Tok-bearer-4Qm"
COOKIE = "sid=Ck-cookie-7Wx"
BASIC = "Basic " + base64.b64encode(f"{USER}:{PASSWORD}".encode()).decode()
SECRETS = (PASSWORD, "Tok-bearer-4Qm", "Ck-cookie-7Wx", BASIC.split()[1])
# Each guarded page, the login it needs, and the text it shows once logged in.
GUARDED = {
    "basic": ("/basic/", CrawlLogin(username=USER, password=PASSWORD), "BASIC-PAGE-OK"),
    "bearer": ("/bearer/", CrawlLogin(authorization=TOKEN), "BEARER-PAGE-OK"),
    "cookie": ("/cookie/", CrawlLogin(cookie=COOKIE), "COOKIE-PAGE-OK"),
}
LANDED = "OTHER-HOST-LANDED"


class _Site:
    """A loopback site whose guarded pages each need one kind of login."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, str, str]] = []
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def url(self, path: str, host: str = "127.0.0.1") -> str:
        return f"http://{host}:{self.port}{path}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def seen(self, path: str) -> list[tuple[str, str, str]]:
        """The host, Authorization and Cookie header of each request for *path*."""
        with self._lock:
            return [(host, auth, cookie) for at, host, auth, cookie in self.requests if at == path]

    def reply(self, path: str, auth: str, cookie: str) -> tuple[int, dict[str, str], str]:
        if path == "/away":
            return HTTPStatus.FOUND, {"Location": self.url("/landing", "localhost")}, ""
        if path == "/landing":
            return HTTPStatus.OK, {}, f"<h1>{LANDED}</h1><p>{LANDED} text.</p>"
        for prefix, (_, _, text) in zip(GUARDED, GUARDED.values(), strict=True):
            if path == f"/{prefix}/":
                ok = {"basic": auth == BASIC, "bearer": auth == TOKEN, "cookie": COOKIE in cookie}
                if not ok[prefix]:
                    return HTTPStatus.UNAUTHORIZED, {}, "<h1>log in</h1>"
                return HTTPStatus.OK, {}, f"<h1>{text}</h1><p>{text} body.</p>"
        return HTTPStatus.NOT_FOUND, {}, "<h1>missing</h1>"

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        site = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                return None

            def do_GET(self) -> None:
                auth = self.headers.get("Authorization", "")
                cookie = self.headers.get("Cookie", "")
                with site._lock:
                    site.requests.append((self.path, self.headers.get("Host", ""), auth, cookie))
                status, headers, body = site.reply(self.path, auth, cookie)
                data = body.encode()
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler


@pytest.fixture
def site() -> Iterator[_Site]:
    running = _Site()
    yield running
    running.close()


@pytest.fixture(autouse=True)
def allow_loopback(monkeypatch):
    allowed = tuple(n for n in url_filter.get_blocked_networks() if str(n) not in LOOPBACK)
    monkeypatch.setattr(url_filter, "get_blocked_networks", lambda: allowed)


@pytest.fixture(autouse=True)
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


def _texts(paths: list[Path]) -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def _files_under(folder: Path) -> str:
    """Every byte lilbee wrote under *folder*, decoded loosely, for a leak grep."""
    return "\n".join(
        path.read_bytes().decode("utf-8", "replace") for path in folder.rglob("*") if path.is_file()
    )


@pytest.mark.parametrize("mode", list(CrawlRenderMode))
@pytest.mark.parametrize("case", GUARDED.values(), ids=list(GUARDED))
class TestEachLogin:
    async def test_the_page_is_read_with_the_login(self, site, case, mode):
        path, login, text = case
        paths = await crawl_and_save(site.url(path), depth=0, render_mode=mode, login=login)
        assert text in _texts(paths)

    async def test_the_page_is_refused_without_the_login(self, site, case, mode):
        path, _, text = case
        paths = await crawl_and_save(site.url(path), depth=0, render_mode=mode)
        assert text not in _texts(paths)
        assert site.seen(path)


@pytest.mark.parametrize("mode", list(CrawlRenderMode))
class TestOtherHost:
    async def test_a_redirect_to_another_host_carries_no_login(self, site, mode):
        login = CrawlLogin(authorization=TOKEN, cookie=COOKIE)
        await crawl_and_save(site.url("/away"), depth=0, render_mode=mode, login=login)
        assert site.seen("/away") == [(f"127.0.0.1:{site.port}", TOKEN, COOKIE)]
        landed = site.seen("/landing")
        assert [host for host, _, _ in landed] == [f"localhost:{site.port}"] * len(landed)
        assert landed
        assert all(TOKEN not in auth and COOKIE not in cookie for _, auth, cookie in landed)


class _Collected(logging.Handler):
    """Every record any logger emits, formatted with its arguments."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(f"{record.name} {record.getMessage()} {record.exc_text or ''}")


@pytest.fixture
def every_log_line() -> Iterator[_Collected]:
    root = logging.getLogger()
    httpx_logger = logging.getLogger("httpx")
    collected = _Collected()
    level, httpx_level = root.level, httpx_logger.level
    root.setLevel(logging.DEBUG)
    # The request lines are the log's positive control; other imports can raise this logger's level.
    httpx_logger.setLevel(logging.INFO)
    root.addHandler(collected)
    yield collected
    root.removeHandler(collected)
    httpx_logger.setLevel(httpx_level)
    root.setLevel(level)


class TestNoLeak:
    def test_logs_metadata_and_api_output_hold_no_secret(self, site, isolated_env, every_log_line):
        body = {
            "url": site.url("/basic/"),
            "depth": 1,
            "max_pages": 3,
            "render_mode": "http",
            "login": {"username": USER, "password": PASSWORD, "cookie": COOKIE},
        }
        auth_mod.session_manager.disable()
        try:
            server = create_app()
            logging.getLogger().addHandler(every_log_line)
            resp = TestClient(server).post("/api/crawl", json=body)
        finally:
            auth_mod.session_manager.cleanup()
        api = resp.text
        logs = "\n".join(every_log_line.lines)
        saved = _files_under(isolated_env)
        # Positive controls: the crawl ran, logged, saved and answered about this seed.
        assert resp.status_code == HTTPStatus.CREATED
        assert "BASIC-PAGE-OK" in saved
        assert str(site.port) in logs
        assert "files_written" in api
        assert site.seen("/sitemap.xml")[0][1] == BASIC
        for secret in SECRETS:
            assert secret not in api
            assert secret not in logs
            assert secret not in saved


WRONG_PASSWORD = "Pw-wrong-2Lk"
WRONG_BASIC = base64.b64encode(f"{USER}:{WRONG_PASSWORD}".encode()).decode()
ORPHAN_PASSWORD = "Pw-orphan-8Ve"
CONTROL_PASSWORD = "Pw-ctl-3Nd\nInjected: 1"


async def _status_after_the_crawl(site, **login) -> dict:
    """Start a crawl through the MCP tool, wait for it to end and return its status answer."""
    import asyncio

    from lilbee.crawler.task import clear_tasks
    from lilbee.mcp_server import crawl, crawl_status

    clear_tasks()
    started = await crawl(site.url("/basic/"), depth=0, render_mode=CrawlRenderMode.HTTP, **login)
    assert started["status"] == "started"
    for _ in range(200):
        answer = crawl_status(started["task_id"])
        if answer["status"] not in ("pending", "running"):
            return {"started": started, "answer": answer}
        await asyncio.sleep(0.05)
    raise AssertionError("the crawl did not end")


class TestNoLeakInStatusAndErrors:
    async def test_the_status_of_a_finished_crawl_holds_no_secret(self, site, every_log_line):
        result = await _status_after_the_crawl(
            site, username=USER, password=PASSWORD, cookie=COOKIE
        )
        shown = f"{result['started']} {result['answer']}"
        # Positive controls: the crawl logged in, ended and the answer names this seed.
        assert result["answer"]["status"] == "done"
        assert str(site.port) in result["answer"]["url"]
        for secret in SECRETS:
            assert secret not in shown

    async def test_the_status_of_a_refused_login_holds_no_secret(self, site, every_log_line):
        result = await _status_after_the_crawl(site, username=USER, password=WRONG_PASSWORD)
        shown = f"{result['started']} {result['answer']}"
        logs = "\n".join(every_log_line.lines)
        # Positive control: the site refused the login and the answer says the crawl got nothing.
        assert site.seen("/basic/")
        assert result["answer"]["pages_failed"] + (result["answer"]["status"] == "failed") >= 1
        for secret in (WRONG_PASSWORD, WRONG_BASIC):
            assert secret not in shown
            assert secret not in logs

    @pytest.mark.parametrize(
        "login",
        [
            {"password": ORPHAN_PASSWORD},
            {"username": USER, "password": CONTROL_PASSWORD},
            {"authorization": TOKEN, "username": USER, "password": PASSWORD},
        ],
        ids=["password-without-username", "control-character", "header-and-username"],
    )
    def test_a_refused_login_is_never_echoed(self, site, every_log_line, login):
        auth_mod.session_manager.disable()
        try:
            server = create_app()
            logging.getLogger().addHandler(every_log_line)
            resp = TestClient(server).post(
                "/api/crawl", json={"url": site.url("/basic/"), "login": login}
            )
        finally:
            auth_mod.session_manager.cleanup()
        shown = resp.text + "\n".join(every_log_line.lines)
        # Positive control: the request was refused with a message about the login.
        assert resp.status_code == HTTPStatus.BAD_REQUEST
        assert "login" in resp.text
        for secret in (ORPHAN_PASSWORD, "Pw-ctl-3Nd", PASSWORD, "Tok-bearer-4Qm"):
            assert secret not in shown

    @pytest.mark.parametrize(
        "login", [{"password": ORPHAN_PASSWORD}, {"username": USER, "password": CONTROL_PASSWORD}]
    )
    async def test_a_refused_login_is_never_echoed_by_the_mcp_tool(self, site, login):
        from lilbee.mcp_server import crawl

        answer = await crawl(site.url("/basic/"), **login)
        assert "login" in str(answer)
        assert ORPHAN_PASSWORD not in str(answer)
        assert "Pw-ctl-3Nd" not in str(answer)

    def test_a_refused_login_is_never_echoed_by_the_command_line(self, site):
        from typer.testing import CliRunner

        from lilbee.cli import app

        result = CliRunner().invoke(
            app, ["add", site.url("/basic/"), "--password", ORPHAN_PASSWORD]
        )
        assert result.exit_code != 0
        assert "login" in result.output
        assert ORPHAN_PASSWORD not in result.output
