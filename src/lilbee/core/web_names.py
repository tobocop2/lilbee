"""The layout of the names crawled pages are saved under in the documents directory."""

from __future__ import annotations

# The folder under the documents directory that holds every crawled page.
WEB_DIR = "_web"
# The start of every crawled page's source, a path relative to the documents directory.
WEB_PREFIX = f"{WEB_DIR}/"
# Separates a page's readable name from the hash of its address in the file stem.
PAGE_HASH_MARK = "~"
# The readable name of a page whose address ends in a slash or has no path.
INDEX_STEM = "index"
PAGE_SUFFIX = ".md"


def is_web_source(source: str) -> bool:
    """Whether *source*, a path relative to the documents directory, is a crawled page."""
    return source.startswith(WEB_PREFIX)


def page_stem(file_name: str) -> str:
    """The readable part of a saved page's *file_name*: no suffix and no address hash."""
    stem = file_name.removesuffix(PAGE_SUFFIX)
    readable, mark, _ = stem.rpartition(PAGE_HASH_MARK)
    return readable if mark else stem
