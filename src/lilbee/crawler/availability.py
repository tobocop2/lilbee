"""Whether the ``crawler`` extra is installed."""

from __future__ import annotations

import functools
import importlib.util

# The modules of the crawler extra a crawl imports: the crawler, and the URL parser of saved names.
_CRAWLER_MODULES = ("crawlberg", "ada_url")
CRAWLER_MISSING_MESSAGE = "Web crawling requires: pip install 'lilbee[crawler]'"


@functools.cache
def crawler_available() -> bool:
    """Whether every module of the crawler extra is installed, checked without importing one."""
    return all(importlib.util.find_spec(module) is not None for module in _CRAWLER_MODULES)
