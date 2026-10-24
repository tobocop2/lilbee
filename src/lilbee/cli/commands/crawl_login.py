"""The login `lilbee add` sends: one host per command, and the environment names its host."""

from __future__ import annotations

import os
from dataclasses import asdict

from lilbee.cli import theme
from lilbee.crawler.models import CrawlLogin
from lilbee.crawler.url_filter import parsed_url
from lilbee.runtime.console import PlainConsole

LOGIN_HOST_ENV = "LILBEE_CRAWL_LOGIN_HOST"
LOGIN_ENV = {
    "authorization": "LILBEE_CRAWL_AUTHORIZATION",
    "password": "LILBEE_CRAWL_PASSWORD",
    "cookie": "LILBEE_CRAWL_COOKIE",
}
ONE_HOST_MESSAGE = (
    "a crawl login goes to one host, and these URLs name {count}; "
    "add the URLs of each host in their own command"
)
ENVIRONMENT_LOGIN_NOT_SENT = (
    f"The login in the LILBEE_CRAWL_* variables was not sent: {LOGIN_HOST_ENV} "
    "does not name the one host of these URLs."
)


def url_hosts(urls: list[str]) -> set[str]:
    """The hosts of *urls* as the URL standard reads them; a URL that does not parse is its own."""
    return {parsed.hostname if (parsed := parsed_url(url)) else url for url in urls}


def _login_host() -> str | None:
    """The host LILBEE_CRAWL_LOGIN_HOST names, read as a URL host, or None."""
    parsed = parsed_url(f"http://{os.environ.get(LOGIN_HOST_ENV, '')}/")
    return parsed.hostname if parsed else None


def _environment_fields(hosts: set[str]) -> dict[str, str]:
    """The login fields set in the environment, when LILBEE_CRAWL_LOGIN_HOST names *hosts*."""
    fields = {name: os.environ[var] for name, var in LOGIN_ENV.items() if os.environ.get(var)}
    if fields and hosts != {_login_host()}:
        PlainConsole(stderr=True).print(ENVIRONMENT_LOGIN_NOT_SENT, style=theme.WARNING)
        return {}
    return fields


def crawl_login(urls: list[str], typed: CrawlLogin) -> CrawlLogin:
    """The login for the crawl of *urls*: *typed* over the environment's, all for one host.

    Raises ``ValueError`` when a typed login meets URLs on more than one host.
    """
    hosts = url_hosts(urls)
    typed_fields = {name: value for name, value in asdict(typed).items() if value}
    if typed_fields and len(hosts) > 1:
        raise ValueError(ONE_HOST_MESSAGE.format(count=len(hosts)))
    return CrawlLogin(**{**_environment_fields(hosts), **typed_fields})
