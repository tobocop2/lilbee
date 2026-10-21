"""Which crawl exclude patterns crawlberg refuses."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lilbee.core.config.defaults import DEFAULT_CRAWL_EXCLUDE_PATTERNS, exclude_pattern_lines

# The line of crawlberg's regex refusal that says what is wrong with the pattern.
_REGEX_ERROR_LINE = "error: "


def exclude_patterns_refusal(value: Any) -> str | None:
    """The message that names every pattern in *value* crawlberg refuses, or None.

    A settings write calls this. Loading the config does not, because the check loads crawlberg.
    The shipped list is not checked: a test holds it to crawlberg's rules.
    """
    patterns = exclude_pattern_lines(value)
    if not isinstance(patterns, list) or patterns == list(DEFAULT_CRAWL_EXCLUDE_PATTERNS):
        return None
    bad = [
        f"[{index}] {patterns[index]!r}: {reason}"
        for index, reason in refused_exclude_patterns(patterns)
    ]
    return "invalid regex in crawl_exclude_patterns: " + "; ".join(bad) if bad else None


def refused_exclude_patterns(patterns: Sequence[str]) -> list[tuple[int, str]]:
    """Each of *patterns* that crawlberg refuses, as its index and crawlberg's reason.

    crawlberg compiles an exclude pattern when it builds the engine, with its own regex rules
    and not Python's. The binding has no call that only compiles, so this builds an engine
    config for each pattern. Without the crawler extra nothing is refused here: no crawl can
    start, and the engine check runs when one does.
    """
    try:
        import crawlberg
    except ImportError:
        return []
    refused: list[tuple[int, str]] = []
    for index, pattern in enumerate(patterns):
        try:
            crawlberg.create_engine(crawlberg.CrawlConfig(exclude_paths=[pattern]))
        except Exception as exc:
            refused.append((index, _compile_reason(str(exc))))
    return refused


def _compile_reason(error: str) -> str:
    """The last ``error:`` line of crawlberg's regex refusal, or the whole text without one."""
    for line in reversed(error.splitlines()):
        if line.startswith(_REGEX_ERROR_LINE):
            return line.removeprefix(_REGEX_ERROR_LINE)
    return error
