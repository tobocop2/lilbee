"""The one rule for the host a stored crawl login names, shared by every source of the setting."""

from __future__ import annotations

LOGIN_HOST_NOT_A_HOST = "crawl_login_host must be a host name, such as wiki.example.com"


def login_host(value: str) -> str | None:
    """*value* read as a bare host with an optional port, or None when it is anything else.

    A scheme, a path, a query or a user name makes the value a URL, not a host. ada-url is in
    the optional ``crawler`` extra, so this imports it on call.
    """
    import ada_url

    try:
        parsed = ada_url.URL(f"http://{value}/")
    except ValueError:
        return None
    if parsed.pathname != "/" or parsed.search or parsed.hash:
        return None
    if parsed.username or parsed.password:
        return None
    return parsed.hostname or None


def refuse_unless_host(value: str) -> None:
    """Raise ``ValueError`` when a non-empty *value* is not a host name.

    Without ada-url no crawl runs, so nothing can use the host and the value passes.
    """
    if not value:
        return
    try:
        host = login_host(value)
    except ModuleNotFoundError:
        return
    if host is None:
        raise ValueError(LOGIN_HOST_NOT_A_HOST)
