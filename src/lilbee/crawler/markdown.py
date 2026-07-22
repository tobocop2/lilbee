"""HTML to markdown conversion, owned by lilbee rather than by the fetcher backend.

Fetching a page and converting it are separate jobs. Keeping the conversion here
means a fetcher backend only has to return HTML, and the conversion runs where
lilbee can await it instead of inside the backend's own call stack.
"""

from __future__ import annotations

import re

# Mirrors crawl4ai's own <base href> derivation (async_webcrawler) verbatim so a
# silenced re-conversion resolves relative links the way an un-silenced crawl would.
# A real HTML parser here would resolve differently and diverge from the backend;
# keep this in sync with crawl4ai instead.
_BASE_HREF = re.compile(r"<base\s[^>]*href\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)


def base_url_for(html: str, url: str, redirected_url: str | None = None) -> str:
    """The URL relative links resolve against: a ``<base href>`` if the page sets one."""
    match = _BASE_HREF.search(html)
    if match:
        return match.group(1)
    return redirected_url or url


def html_to_markdown(html: str, base_url: str) -> str:
    """Convert *html* to markdown, resolving relative links against *base_url*.

    Imports its backend on call: this is the one place lilbee depends on a
    third-party HTML-to-markdown implementation.
    """
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

    result = DefaultMarkdownGenerator().generate_markdown(html, base_url=base_url)
    return str(result.raw_markdown or "")
