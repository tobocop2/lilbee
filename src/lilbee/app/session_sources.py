"""The Sources list of a saved chat turn, as every session surface returns it."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NotRequired

from typing_extensions import TypedDict

from lilbee.app.titles import stored_web_titles
from lilbee.retrieval.query.formatting import (
    SourceRef,
    is_locator,
    source_label,
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
    'locator "{locator}" is not a page or line span: '
    "use 'page N', 'pages A-B', 'line N' or 'lines A-B', or leave it out"
)


class LocatorError(ValueError):
    """A posted Sources ref names a locator that lilbee never writes."""


def message_sources(message: SessionMessage) -> tuple[str, list[SourceRef]]:
    """*message*'s content and its Sources list as refs.

    A turn saved with ``source_refs`` gives them, each labelled from the current
    stored title. A turn saved without them gives its content as saved and no refs.
    """
    if message.source_refs is None:
        return message.content, []
    return message.content, labelled(message.source_refs)


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
    """The turn a client appends, with its Sources list *refs*, or else built from
    *sources* with no citation numbers; ``sources`` is derived from it.

    Raises LocatorError when a ref's locator is not one lilbee writes.
    """
    if refs is None:
        return message_with_refs(role, content, tuple(StoredSourceRef(s) for s in sources))
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
