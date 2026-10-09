"""Builders the harness's tests share."""

from __future__ import annotations

from tools.qa.crawl_parity.model import Page
from tools.qa.crawl_parity.thresholds import ParityLimits
from tools.qa.crawl_parity.truth import Capture, Truth, truth_from_capture

STRICT = ParityLimits(
    pages_lost_max=0,
    pages_extra_unreadable_max=0,
    words_lost_per_page_max=0,
    words_added_per_page_max=0,
    invisible_text_lost_counts=False,
    symbols_changed_per_page_max=0,
    structure_changed_per_page_max=0,
)


def truth(
    path: str,
    nodes: list[tuple[str, bool, str]],
    attributes: list[str] | None = None,
    status: int = 200,
) -> Truth:
    """A truth record as the browser would return it: (context, rendered, text) for each node."""
    capture: Capture = {
        "text": "\n".join(text for _context, shown, text in nodes if shown),
        "nodes": [list(node) for node in nodes],
        "attributes": attributes or [],
    }
    return truth_from_capture(path, status, True, capture)


def pages(**markdown_by_name: str | None) -> dict[str, Page]:
    """Pages by path ``/<name>``; None is a page with no markdown and the error ``boom 42``."""
    return {
        f"/{name}": Page(f"http://127.0.0.1:1/{name}", markdown, None if markdown else "boom 42")
        for name, markdown in markdown_by_name.items()
    }
