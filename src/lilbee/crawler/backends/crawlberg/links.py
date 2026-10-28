"""The links a crawl follows or leaves out, as crawlberg queues them."""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Sequence
from urllib.parse import parse_qsl, quote_plus, urlsplit, urlunsplit

from lilbee.crawler import url_filter
from lilbee.crawler.backends.crawlberg.config import _TRACKING_PARAMS, _CrawlSpec
from lilbee.crawler.backends.crawlberg.events import _Event

log = logging.getLogger(__name__)

_PREFIX_MARK = "*"
# What crawlberg's query encoder leaves as is besides letters, digits and ``-._``, and the one
# character Python's encoder leaves as is that crawlberg's does not.
_FORM_SAFE = "*"
_TILDE = "~"
_TILDE_ENCODED = "%7E"
_LEFT_OUT = "Left out %s: it matches %s, a pattern of the crawl_exclude_patterns setting"


def _is_tracking_param(name: str) -> bool:
    """True when crawlberg strips the query parameter *name* from a discovered link."""
    return any(
        name.startswith(param.removesuffix(_PREFIX_MARK))
        if param.endswith(_PREFIX_MARK)
        else name == param
        for param in _TRACKING_PARAMS
    )


def _form_encoded(text: str) -> str:
    """*text* as crawlberg writes it in a query: only letters, digits and ``*-._`` stay as is."""
    return quote_plus(text, safe=_FORM_SAFE).replace(_TILDE, _TILDE_ENCODED)


def _queued_url(link: str) -> str:
    """*link* as crawlberg queues it: no fragment, no tracking parameter, a form-encoded query."""
    parts = urlsplit(link)
    query = "&".join(
        f"{_form_encoded(name)}={_form_encoded(value)}"
        for name, value in parse_qsl(parts.query, keep_blank_values=True)
        if not _is_tracking_param(name)
    )
    return urlunsplit(parts._replace(query=query, fragment=""))


def _python_patterns(patterns: Sequence[str]) -> tuple[re.Pattern[str], ...]:
    """The *patterns* Python compiles; crawlberg also accepts some that Python refuses.

    The log names an excluded address through Python. A pattern only crawlberg compiles
    still excludes its addresses and is left out of the log.
    """
    compiled = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern))
        except re.error:
            log.debug(
                "The exclude pattern %s is not a Python pattern; the log cannot name it", pattern
            )
    return tuple(compiled)


class _ExcludedLinks:
    """The addresses one crawl leaves out because an exclude pattern matches them.

    crawlberg counts the addresses it excludes and names none. This makes crawlberg's
    decision again for each link of a page, so the log can name the address and the pattern.
    """

    def __init__(self, spec: _CrawlSpec, seed_url: str) -> None:
        self._seed_host = urlsplit(seed_url).hostname or ""
        self._depth = spec.depth
        self._include_subdomains = spec.filters.include_subdomains
        self._patterns = _python_patterns(spec.filters.exclude_patterns)
        self._seen_links: set[str] = set()
        self._checked: set[str] = set()
        self._left_out: Counter[str] = Counter()

    def record_seed(self, seed_url: str) -> bool:
        """Warn about the seed and return True when a pattern matches it.

        crawlberg fetches a seed that an exclude pattern matches, so the crawl of a matched
        seed does not start. The seed is matched as crawlberg's URL parser writes it; a seed
        that does not parse is one crawlberg refuses, so no pattern applies to it.
        """
        address = url_filter.parsed_address(seed_url)
        if address is None:
            return False
        url = _queued_url(address)
        self._checked.add(url)
        matched = self._matching(url)
        if matched is None:
            return False
        log.warning(_LEFT_OUT, url, matched)
        return True

    def record(self, event: _Event) -> None:
        """Count each link of *event* that the crawl leaves out, the first time a page holds it.

        An exclude pattern leaves out links as a matter of course, so each one is a debug line
        and ``summarize`` reports the total.
        """
        if self._depth is not None and event.depth >= self._depth:
            return
        new_links = [link for link in event.links if link not in self._seen_links]
        self._seen_links.update(new_links)
        for url in map(_queued_url, new_links):
            if url not in self._checked and self._in_scope(url):
                self._checked.add(url)
                matched = self._matching(url)
                if matched is not None:
                    self._left_out[matched] += 1
                    log.debug(_LEFT_OUT, url, matched)

    def summarize(self) -> None:
        """Log one line with the number of addresses each pattern left out, when any."""
        total = self._left_out.total()
        if not total:
            return
        by_pattern = ", ".join(f"{pattern} {count}" for pattern, count in self._left_out.items())
        log.info(
            "Crawl left out %d %s by crawl_exclude_patterns: %s",
            total,
            "address" if total == 1 else "addresses",
            by_pattern,
        )

    def _in_scope(self, url: str) -> bool:
        host = urlsplit(url).hostname or ""
        return url_filter.host_in_scope(
            host, self._seed_host, include_subdomains=self._include_subdomains
        )

    def _matching(self, url: str) -> str | None:
        """The first exclude pattern that matches *url*, or None."""
        return next((p.pattern for p in self._patterns if p.search(url)), None)
