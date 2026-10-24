"""The stored titles of sources, as every surface that names a source shows them."""

from __future__ import annotations

from collections.abc import Iterable

from lilbee.app.services import get_services
from lilbee.core.web_names import is_web_source


def stored_web_titles(names: Iterable[str]) -> dict[str, str]:
    """The stored title of each crawled page among *names* that has one.

    A list with no crawled page reads nothing, so it never starts the services.
    """
    pages = [name for name in names if is_web_source(name)]
    return get_services().store.web_titles(pages) if pages else {}
