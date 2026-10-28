"""The login the crawl settings store, and the one login a crawl sends."""

from __future__ import annotations

from dataclasses import asdict

from lilbee.core.config import cfg
from lilbee.core.config.login_host import login_host
from lilbee.crawler.models import NO_LOGIN, CrawlLogin
from lilbee.crawler.url_filter import parsed_url

ONE_HOST_MESSAGE = (
    "a crawl login goes to one host, and these URLs name {count}; "
    "add the URLs of each host in their own command"
)
STORED_LOGIN_NOT_SENT = (
    "The stored crawl login was not sent: crawl_login_host "
    "(LILBEE_CRAWL_LOGIN_HOST) does not name the one host of these URLs."
)


def url_hosts(urls: list[str]) -> set[str]:
    """The hosts of *urls* as the URL standard reads them; a URL that does not parse is its own."""
    return {parsed.hostname if (parsed := parsed_url(url)) else url for url in urls}


def _stored_fields() -> dict[str, str]:
    """The login fields the settings hold, whatever host they name."""
    stored = {
        "authorization": cfg.crawl_authorization,
        "username": cfg.crawl_username,
        "password": cfg.crawl_password,
        "cookie": cfg.crawl_cookie,
    }
    return {name: value for name, value in stored.items() if value}


def stored_login_withheld(urls: list[str]) -> bool:
    """Whether the settings hold a login that `crawl_login_host` does not send to *urls*."""
    return bool(_stored_fields()) and url_hosts(urls) != {login_host(cfg.crawl_login_host)}


def crawl_login(urls: list[str], typed: CrawlLogin = NO_LOGIN) -> CrawlLogin:
    """The login for the crawl of *urls*: *typed* over the stored one, all for one host.

    Raises ``ValueError`` when a typed login meets URLs on more than one host.
    """
    hosts = url_hosts(urls)
    typed_fields = {name: value for name, value in asdict(typed).items() if value}
    if typed_fields and len(hosts) > 1:
        raise ValueError(ONE_HOST_MESSAGE.format(count=len(hosts)))
    stored = {} if stored_login_withheld(urls) else _stored_fields()
    return CrawlLogin(**{**stored, **typed_fields})
