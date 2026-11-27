"""Whether the ``crawler`` extra is installed and loads."""

from __future__ import annotations

import functools
import importlib
import importlib.util

# The modules of the crawler extra a crawl imports: the crawler, and the URL parser of saved names.
_CRAWLER_MODULES = ("crawlberg", "ada_url")
CRAWLER_MISSING_MESSAGE = "Web crawling requires: pip install 'lilbee[crawler]'"
# ``{reason}`` is the text of the error the module raised as it loaded.
CRAWLER_BROKEN_MESSAGE = (
    "The crawler is installed but cannot load on this computer: {reason}. "
    "Reinstall it for this platform with: pip install --force-reinstall 'lilbee[crawler]'"
)


def crawler_installed() -> bool:
    """Whether every module of the crawler extra is installed, without importing any.

    The check reads the import path only, so a screen can call it on the UI thread.
    A broken install counts as installed: a crawl runs :func:`crawler_available`
    and shows the reinstall message.
    """
    return all(importlib.util.find_spec(module) is not None for module in _CRAWLER_MODULES)


@functools.cache
def crawler_unavailable_reason() -> str | None:
    """The message for why the crawler cannot run, or ``None`` when every module of it loads.

    A module that is not installed gives the install message. A module that is installed but
    raises ``ImportError`` or ``OSError`` as it loads (a broken native library, a wheel for
    another platform) gives the reinstall message with the error text.
    """
    for module in _CRAWLER_MODULES:
        if importlib.util.find_spec(module) is None:
            return CRAWLER_MISSING_MESSAGE
        try:
            importlib.import_module(module)
        except (ImportError, OSError) as exc:
            return CRAWLER_BROKEN_MESSAGE.format(reason=exc)
    return None


def crawler_available() -> bool:
    """Whether every module of the crawler extra is installed and loads."""
    return crawler_unavailable_reason() is None


def crawler_unavailable_message() -> str:
    """The words that tell a user the crawler cannot run, with the fix."""
    return crawler_unavailable_reason() or CRAWLER_MISSING_MESSAGE
