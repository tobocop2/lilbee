"""Render a saved chat session as a markdown document and write it to disk."""

from __future__ import annotations

import os
import re
from functools import cache
from pathlib import Path

import yaml
from markdown_it import MarkdownIt

from lilbee.core.security import write_private_text
from lilbee.core.text import collapse_whitespace, make_slug
from lilbee.retrieval.query.formatting import (
    FILE_LINK_RE,
    SOURCES_BLOCK_MARKER,
    with_sources_block,
)
from lilbee.sessions import MessageRole, Session, SessionMessage, SessionMeta

_EXPORT_SUFFIX = ".md"
SLUG_MAX_LEN = 60
_FALLBACK_STEM = "chat"
_ID_PREFIX_LEN = 8
_FRONT_MATTER_FENCE = "---"
_ROLE_HEADINGS: dict[MessageRole, str] = {
    MessageRole.USER: "User",
    MessageRole.ASSISTANT: "Assistant",
}
# Parsed after a message body; if it lands inside a fence, the body left that fence open.
_PROBE_HEADING = "\n\n# probe"
_ATX_MARKER = "#"
_TURN_HEADING_LEVEL = 2
_MAX_HEADING_LEVEL = 6
_ESCAPE = "\\"
_LINE_BREAK_RE = re.compile(r"\r\n?")


def session_markdown(session: Session) -> str:
    """The session as markdown: front matter, a title heading, one section per turn."""
    parts = [_front_matter(session.meta), f"# {collapse_whitespace(session.meta.title)}"]
    parts.extend(_message_section(message) for message in session.messages)
    return "\n\n".join(parts) + "\n"


def _front_matter(meta: SessionMeta) -> str:
    fields = {
        "title": meta.title,
        "session": meta.id,
        "model": meta.model_ref,
        "created": meta.created_at,
        "updated": meta.updated_at,
    }
    if meta.forked_from:
        fields["forked_from"] = meta.forked_from
    dumped = yaml.safe_dump(fields, sort_keys=False, allow_unicode=True)
    return f"{_FRONT_MATTER_FENCE}\n{dumped}{_FRONT_MATTER_FENCE}"


def _message_section(message: SessionMessage) -> str:
    body = _close_open_fence(_LINE_BREAK_RE.sub("\n", message.content).rstrip())
    if message.sources:
        body = with_sources_block(body, message.sources)
    return f"## {_ROLE_HEADINGS[message.role]}\n\n{_nest_headings(_unlink_sources(body))}"


def _unlink_sources(text: str) -> str:
    """*text* with the file links in its Sources list reduced to their labels."""
    head, marker, block = text.partition(SOURCES_BLOCK_MARKER)
    return head + marker + FILE_LINK_RE.sub(r"\1", block)


def _nest_headings(text: str) -> str:
    """*text* with no heading at or above the turn level.

    The parser finds the headings, so a ``#`` in code or a ``#tag`` stays as
    written. A ``#`` heading drops two levels, to at most six; an underlined
    heading cannot go below level two, so its underline is escaped to text.
    """
    lines = text.split("\n")
    for token in _commonmark().parse(text):
        if token.type == "heading_open" and token.map:
            _nest_heading(lines, token.markup, *token.map)
    return "\n".join(lines)


def _nest_heading(lines: list[str], markup: str, start: int, end: int) -> None:
    """Rewrite the heading spanning lines *start* to *end* (exclusive) in place."""
    if markup.startswith(_ATX_MARKER):
        level = min(len(markup) + _TURN_HEADING_LEVEL, _MAX_HEADING_LEVEL)
        lines[start] = lines[start].replace(markup, _ATX_MARKER * level, 1)
    else:
        underline = end - 1
        lines[underline] = lines[underline].replace(markup, _ESCAPE + markup, 1)


@cache
def _commonmark() -> MarkdownIt:
    return MarkdownIt("commonmark")


def _close_open_fence(text: str) -> str:
    """*text* with a code fence it leaves open closed, so it cannot swallow what follows.

    The parser decides: when a heading placed after *text* ends up inside a fence,
    that fence is closed. A fence in a list item never does, because the heading
    ends the item, so only a top-level fence is closed, and at column 0.
    """
    last = _commonmark().parse(text + _PROBE_HEADING)[-1]
    return f"{text}\n{last.markup}" if last.type == "fence" else text


def default_export_name(meta: SessionMeta) -> str:
    """``<title-slug>-<id prefix>.md``, or ``chat-<id prefix>.md`` when the title has no slug."""
    slug = make_slug(meta.title)[:SLUG_MAX_LEN].strip("-") or _FALLBACK_STEM
    return f"{slug}-{meta.id[:_ID_PREFIX_LEN]}{_EXPORT_SUFFIX}"


def write_session_markdown(session: Session, destination: str) -> Path:
    """Write *session* owner-only to *destination*, or to its default name inside
    it when it is a directory or ends in a separator. Returns the absolute path."""
    target = Path(destination).expanduser()
    if target.is_dir() or destination.endswith(("/", os.sep)):
        target = target / default_export_name(session.meta)
    target = target.resolve()
    write_private_text(target, session_markdown(session))
    return target
