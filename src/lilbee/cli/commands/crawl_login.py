"""The login `lilbee add` sends: one host per command, and the settings name their host."""

from __future__ import annotations

from lilbee.cli import theme
from lilbee.crawler.models import CrawlLogin
from lilbee.crawler.stored_login import STORED_LOGIN_NOT_SENT, stored_login_withheld
from lilbee.crawler.stored_login import crawl_login as resolve_crawl_login
from lilbee.runtime.console import PlainConsole


def crawl_login(urls: list[str], typed: CrawlLogin) -> CrawlLogin:
    """The login for the crawl of *urls*; a stored login that is not sent gets a note."""
    login = resolve_crawl_login(urls, typed)
    if stored_login_withheld(urls):
        PlainConsole(stderr=True).print(STORED_LOGIN_NOT_SENT, style=theme.WARNING)
    return login
