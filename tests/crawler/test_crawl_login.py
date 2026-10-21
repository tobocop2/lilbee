"""The per-crawl login: its checks, and where crawlberg and the sitemap lookup get it."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from lilbee.core.config.enums import CrawlRenderMode
from lilbee.crawler import sitemap
from lilbee.crawler import task as task_mod
from lilbee.crawler.backends.crawlberg import fetcher as fetcher_mod
from lilbee.crawler.backends.crawlberg.fetcher import CrawlbergFetcher
from lilbee.crawler.models import NO_LOGIN, CrawlLogin
from tests._crawlberg_stub import StubCrawlberg, complete, page

SEED = "https://example.com/"
TOKEN = "Bearer tok-5Qz"
COOKIE = "sid=ck-8Wm"
USER = "alice"
PASSWORD = "pw-3Lx"
ACCEPT = fetcher_mod._BROWSER_ACCEPT_HEADERS


@pytest.fixture(autouse=True)
def _admit_every_url(monkeypatch):
    monkeypatch.setattr(fetcher_mod.url_filter, "validate_crawl_url", lambda url: None)


class TestCrawlLogin:
    @pytest.mark.parametrize("name", ["authorization", "username", "password", "cookie"])
    def test_a_control_character_is_refused_by_field_name_only(self, name: str):
        values = {"username": USER} if name == "password" else {}
        with pytest.raises(ValueError, match=f"login {name} holds a control") as caught:
            CrawlLogin(**{**values, name: "sec\r\nret"})
        assert "sec" not in str(caught.value)

    @pytest.mark.parametrize("name", ["authorization", "cookie"])
    @pytest.mark.parametrize("value", ["tok=caféX", "tok=\u0080X", "tok=\U0001f600X"])
    def test_a_header_value_outside_ascii_is_refused_by_field_name_only(self, name, value):
        with pytest.raises(
            ValueError, match=f"login {name} holds a character outside ASCII"
        ) as caught:
            CrawlLogin(**{name: value})
        assert "tok" not in str(caught.value)

    def test_a_username_and_password_outside_ascii_are_kept(self):
        login = CrawlLogin(username="josé", password="pässwort")
        assert login.basic_credentials == ("josé", "pässwort")

    def test_a_password_needs_a_username(self):
        with pytest.raises(ValueError, match="password needs a username"):
            CrawlLogin(password=PASSWORD)

    def test_an_authorization_header_and_a_username_are_refused_together(self):
        with pytest.raises(ValueError, match="not both"):
            CrawlLogin(authorization=TOKEN, username=USER)

    def test_no_value_shows_in_the_repr(self):
        shown = repr(CrawlLogin(username=USER, password=PASSWORD, cookie=COOKIE))
        assert "CrawlLogin" in shown
        assert not {USER, PASSWORD, COOKIE} & set(shown.replace("'", " ").split())
        assert PASSWORD not in shown

    def test_an_empty_login_sends_nothing(self):
        assert NO_LOGIN.basic_credentials is None
        assert NO_LOGIN.authorization_header == {}
        assert NO_LOGIN.cookie_header == {}


LOGINS = {
    "basic": (CrawlLogin(username=USER, password=PASSWORD), ("basic", USER, PASSWORD), {}),
    "header": (CrawlLogin(authorization=TOKEN), ("header", "Authorization", TOKEN), {}),
    "cookie": (CrawlLogin(cookie=COOKIE), None, {"Cookie": COOKIE}),
    "none": (NO_LOGIN, None, {}),
}


class TestFetcherLogin:
    @pytest.mark.parametrize("mode", list(CrawlRenderMode))
    @pytest.mark.parametrize("case", LOGINS.values(), ids=list(LOGINS))
    async def test_the_login_reaches_the_crawlberg_config(
        self,
        case: tuple[CrawlLogin, Any, dict[str, str]],
        mode: CrawlRenderMode,
        monkeypatch,
        tmp_path,
    ):
        login, auth, cookie = case
        monkeypatch.setattr(
            fetcher_mod.bootstrap, "headless_shell_executable", lambda: tmp_path / "shell"
        )
        stub = StubCrawlberg([page(SEED, "# Home", depth=0), complete(1)])
        with stub.installed():
            await CrawlbergFetcher(render_mode=mode, login=login).fetch_single(
                SEED, timeout=5, concurrency=fetcher_mod.ConcurrencySpec()
            )
        accept = ACCEPT if mode is CrawlRenderMode.HTTP else {}
        assert stub.config["auth"] == auth
        assert stub.config["custom_headers"] == {**accept, **cookie}


class _Answer:
    status_code = 404
    text = ""


class TestSitemapLogin:
    def _asked(self, monkeypatch, url: str, login: CrawlLogin) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []

        def fake_get(target: str, **kwargs: Any) -> _Answer:
            calls.append({"url": target, **kwargs})
            return _Answer()

        monkeypatch.setattr(httpx, "get", fake_get)
        monkeypatch.setattr(sitemap, "require_valid_crawl_url", lambda url: None)
        sitemap._count_sitemap_urls(
            url, include_subdomains=False, respect_robots_txt=True, login=login
        )
        return calls

    def test_the_login_goes_with_the_robots_and_sitemap_requests(self, monkeypatch):
        login = CrawlLogin(username=USER, password=PASSWORD, cookie=COOKIE)
        calls = self._asked(monkeypatch, SEED, login)
        assert [call["url"] for call in calls] == [
            "https://example.com/robots.txt",
            "https://example.com/sitemap.xml",
        ]
        assert {call["auth"] for call in calls} == {(USER, PASSWORD)}
        assert [call["headers"] for call in calls] == [{"Cookie": COOKIE}] * 2

    def test_the_authorization_header_goes_as_a_header(self, monkeypatch):
        calls = self._asked(monkeypatch, SEED, CrawlLogin(authorization=TOKEN))
        assert [call["headers"] for call in calls] == [{"Authorization": TOKEN}] * 2
        assert {call["auth"] for call in calls} == {None}

    def test_the_seed_username_and_password_win_over_the_login(self, monkeypatch):
        login = CrawlLogin(username=USER, password=PASSWORD)
        calls = self._asked(monkeypatch, "https://bob:pw@example.com/", login)
        assert {call["auth"] for call in calls} == {("bob", "pw")}


class TestTaskLogin:
    async def test_a_finished_task_keeps_no_login(self, monkeypatch):
        sent: list[CrawlLogin] = []

        async def fake_crawl(url: str, **kwargs: Any) -> list[Any]:
            sent.append(kwargs["login"])
            return []

        async def no_sync(**kwargs: Any) -> None:
            return None

        monkeypatch.setattr(task_mod, "crawl_and_save", fake_crawl)
        monkeypatch.setattr("lilbee.data.ingest.sync", no_sync)
        login = CrawlLogin(cookie=COOKIE)
        task_id = task_mod.start_crawl(SEED, depth=0, login=login)
        task = task_mod.get_task(task_id)
        assert task is not None
        assert task.login == login
        running = task._async_task
        assert running is not None
        await asyncio.wait_for(running, timeout=5)
        assert sent == [login]
        assert task.login == NO_LOGIN
        assert COOKIE not in repr(task)
