"""The Sources list of a saved chat turn, as every session surface returns it."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NotRequired
from urllib.parse import urlparse
from urllib.request import url2pathname

from typing_extensions import TypedDict

from lilbee.app.titles import stored_web_titles
from lilbee.core import settings
from lilbee.core.config import active_config
from lilbee.data.store import is_memory_source
from lilbee.retrieval.query.formatting import (
    FILE_LINK_RE,
    SourceRef,
    is_locator,
    source_label,
    split_sources_list,
)
from lilbee.sessions import MessageRole, SessionMessage, StoredSourceRef, message_with_refs


class SourceRefInput(TypedDict):
    """One entry of a turn's Sources list as a client saves it."""

    source: str
    # The ``[n]`` the answer cites the source by; absent or null when unknown.
    citation_number: NotRequired[int | None]
    # The page or line span the answer read, such as "page 3"; absent when none.
    locator: NotRequired[str]


LOCATOR_ERROR = (
    "locator {locator!r} is not a page or line span: "
    "use 'page N', 'pages A-B', 'line N' or 'lines A-B', or leave it out"
)
# One numbered line of a stored Sources list; a hard-broken line ends in a backslash.
_LIST_LINE_RE = re.compile(r"^(?P<number>\d+)\\?\. (?P<item>.*?)\\?$")


class LocatorError(ValueError):
    """A posted Sources ref names a locator that lilbee never writes."""


def message_sources(message: SessionMessage) -> tuple[str, list[SourceRef]]:
    """*message*'s content without lilbee's Sources list, and that list as refs.

    A turn saved with ``source_refs`` gives its content whole and them, each
    labelled from the current stored title. An older turn whose content ends in
    lilbee's Sources list gives the refs read from that list: its numbers and its
    labels as written. Any other older turn gives its ``sources`` with no number,
    as their positions are not the answer's citation numbers.
    """
    if message.source_refs is not None:
        return message.content, labelled(message.source_refs)
    text, listed = split_sources_list(message.content)
    refs = _listed_refs(listed)
    if refs:
        return text, refs
    return message.content, labelled(tuple(StoredSourceRef(s) for s in message.sources))


def message_payload(message: SessionMessage) -> dict[str, Any]:
    """*message* as the session surfaces send it, its Sources list in ``source_refs``."""
    content, refs = message_sources(message)
    return {
        "role": message.role.value,
        "content": content,
        "sources": list(message.sources),
        "ts": message.ts,
        "source_refs": [
            {
                "source": r.source,
                "label": r.label,
                "citation_number": r.citation_number,
                "locator": r.locator,
            }
            for r in refs
        ],
    }


def posted_message(
    role: MessageRole,
    content: str,
    sources: Sequence[str],
    refs: Sequence[StoredSourceRef] | None,
) -> SessionMessage:
    """The turn a client appends: with its Sources list *refs* when given, ``sources``
    then derived from it; else *sources* alone.

    Raises LocatorError when a ref's locator is not one lilbee writes.
    """
    if refs is None:
        return SessionMessage(role=role, content=content, sources=tuple(sources))
    for ref in refs:
        if not is_locator(ref.locator):
            raise LocatorError(LOCATOR_ERROR.format(locator=ref.locator))
    return message_with_refs(role, content, tuple(refs))


def labelled(refs: tuple[StoredSourceRef, ...]) -> list[SourceRef]:
    """*refs* with the label of each source from its current stored title."""
    titles = stored_web_titles({r.source for r in refs})
    return [
        SourceRef(
            r.source, source_label(r.source, titles.get(r.source, "")), r.citation_number, r.locator
        )
        for r in refs
    ]


def _listed_refs(listed: str) -> list[SourceRef]:
    """The refs of a stored Sources list, one per numbered line."""
    refs = []
    for line in listed.split("\n"):
        match = _LIST_LINE_RE.match(line)
        if match is None:
            continue
        item = match["item"]
        link = FILE_LINK_RE.search(item)
        locator = ""
        if link is not None:
            label, source = link["label"], _source_of(link["url"])
            # A linked line's locator follows its link: "[label](url), page 3".
            locator = item[link.end() :].removeprefix(", ")
        else:
            label, source = item, item if is_memory_source(item) else ""
        refs.append(SourceRef(source, label, int(match["number"]), locator))
    return refs


def _source_of(url: str) -> str:
    """The source key of the file *url* names, or '' when no known root holds it."""
    path = Path(url2pathname(urlparse(url).path))
    config = active_config()
    settings.refresh_registry(config)
    # The reverse of ``resolve_source_path``: registered roots own their label's keys.
    roots: list[tuple[str, str | Path]] = [*config.linked_roots.items(), ("", config.documents_dir)]
    for label, root in roots:
        try:
            relative = path.relative_to(Path(root).resolve()).as_posix()
        except (OSError, ValueError):
            continue
        return "/".join(part for part in (label, relative) if part and part != ".")
    return ""
