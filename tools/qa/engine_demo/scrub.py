"""The one place that removes secrets from the URLs of a recording."""

from __future__ import annotations

import re

QUERY_TOKEN = "<query>"
USERINFO_TOKEN = "<userinfo>"
_URL_QUERY = re.compile(r"(https?://[^\s\"'\\?#<>]+)\?[^\s\"'\\#<>]+")
_URL_USERINFO = re.compile(r"(https?://)[^\s\"'\\/?#<>@]+@")


def scrub_urls(text: str) -> str:
    """The text with the userinfo and the query string of each URL replaced by tokens."""
    return _URL_USERINFO.sub(rf"\1{USERINFO_TOKEN}@", _URL_QUERY.sub(rf"\1?{QUERY_TOKEN}", text))
