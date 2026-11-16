"""The one place that removes home paths, the hostname and URL secrets from a recording."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

QUERY_TOKEN = "<query>"
USERINFO_TOKEN = "<userinfo>"
UNPARSED_TOKEN = "<url>"
_URL = re.compile(r"https?://[^\s\"'<>]+")


def _scrub_url(match: re.Match[str]) -> str:
    candidate = match.group()
    try:
        parts = urlsplit(candidate)
    except ValueError:
        return UNPARSED_TOKEN
    userinfo, at, host = parts.netloc.rpartition("@")
    secret = f"{USERINFO_TOKEN}@" if userinfo else at
    mark = "?" if "?" in candidate.partition("#")[0] else ""
    query = f"?{QUERY_TOKEN}" if parts.query else mark
    fragment = f"#{parts.fragment}" if "#" in candidate else ""
    return f"{parts.scheme}://{secret}{host}{parts.path}{query}{fragment}"


def scrub_urls(text: str) -> str:
    """The text with the userinfo and the query string of each URL replaced by tokens."""
    return _URL.sub(_scrub_url, text)


def map_strings(value: Any, scrub: Callable[[str], str]) -> Any:
    """The decoded JSON value with scrub applied to each string, key and item in it."""
    if isinstance(value, dict):
        return {scrub(key): map_strings(item, scrub) for key, item in value.items()}
    if isinstance(value, list):
        return [map_strings(item, scrub) for item in value]
    if isinstance(value, str):
        return scrub(value)
    return value


HOME_TOKEN = "<home>"
_HOME_DIR = re.compile(r"(?<![^\s\"'=])/(?:home|Users)/[^/\s\"'\\]+")


def scrub_home(text: str) -> str:
    """The text with each home directory, such as /home/alice, replaced by the home token."""
    return _HOME_DIR.sub(HOME_TOKEN, text)


HOST_TOKEN = "<host>"


def scrub_host(text: str, hostname: str) -> str:
    """The text with the hostname and its short form replaced by the host token."""
    names = {hostname, hostname.split(".")[0]} - {""}
    if not names:
        return text
    pattern = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    return re.sub(rf"(?<![\w-])(?:{pattern})(?![\w-])", HOST_TOKEN, text, flags=re.IGNORECASE)


def scrub_recording(value: Any, hostname: str) -> Any:
    """The value with each home directory, the hostname and each URL secret replaced by tokens."""
    return map_strings(value, lambda s: scrub_urls(scrub_host(scrub_home(s), hostname)))
