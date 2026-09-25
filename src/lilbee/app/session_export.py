"""Render a saved chat session as a markdown document and write it to disk."""

from __future__ import annotations

import os
import re
import string
from pathlib import Path

import yaml
from markdown_it.token import Token

from lilbee.core.security import write_private_text
from lilbee.core.text import collapse_whitespace, make_slug
from lilbee.retrieval.query.formatting import (
    FILE_LINK_RE,
    SOURCES_BLOCK_MARKER,
    close_open_fence,
    commonmark,
    open_code_fence,
    source_label,
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
_ATX_MARKER = "#"
_TURN_HEADING_LEVEL = 2
_MAX_HEADING_LEVEL = 6
_ESCAPE = "\\"
_LINE_BREAK_RE = re.compile(r"\r\n?")
_ASCII_PUNCTUATION = frozenset(string.punctuation)
# An HTML heading tag, open or close, any case, with or without attributes:
# <h2>, </h2>, <H3 class="x">. ``header`` and ``hr`` never match, because a
# digit must follow the ``h`` directly.
_HTML_HEADING_RE = re.compile(r"<(?P<slash>/?)(?P<tag>[Hh])(?P<level>[1-6])(?P<rest>[^>]*)>")
# An ordered-list marker at the start of a name, as in ``2. notes.md``.
_LIST_NUMBER_RE = re.compile(r"^(\d+)([.)])(?=\s|$)")


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
    """One turn: the message with its headings nested, then its Sources list as plain names."""
    content = _LINE_BREAK_RE.sub("\n", message.content).rstrip()
    text, stored_sources = _split_sources_list(content)
    body = _contained(text.rstrip())
    if stored_sources:
        body += _contained(FILE_LINK_RE.sub(_plain_link, stored_sources))
    elif message.sources:
        body = with_sources_block(body, message.sources, render=_plain_source)
    return f"## {_ROLE_HEADINGS[message.role]}\n\n{body}"


def _split_sources_list(content: str) -> tuple[str, str]:
    """*content* split before its last Sources list, unless that list is quoted in a code block.

    A list after a code block the answer never closed (a cut-off answer) is lilbee's;
    a list inside a code block that closes after it is pasted text.
    """
    text, marker, stored_sources = content.rpartition(SOURCES_BLOCK_MARKER)
    if not marker:
        return content, ""
    fence = open_code_fence(text)
    if fence is None or _never_closed(fence, content):
        return text, marker + stored_sources
    return content, ""


def _never_closed(fence: Token, content: str) -> bool:
    """Whether *fence*, opened in a prefix of *content*, is still open at its end."""
    end = open_code_fence(content)
    return (
        end is not None
        and end.map is not None
        and fence.map is not None
        and end.map[0] == fence.map[0]
    )


def _contained(text: str) -> str:
    """*text* with open fences closed and headings nested, so it stays inside its turn."""
    return _nest_headings(close_open_fence(text))


def _plain_source(source: str) -> str:
    return _plain_name(source_label(source))


def _plain_link(link: re.Match[str]) -> str:
    return _plain_name(link["label"])


def _plain_name(name: str) -> str:
    """*name* on one line, escaped so it cannot start a heading, list, quote or other block."""
    name = collapse_whitespace(name)
    if name[:1] in _ASCII_PUNCTUATION:
        return _ESCAPE + name
    return _LIST_NUMBER_RE.sub(r"\1\\\2", name, count=1)


def _nest_headings(text: str) -> str:
    """*text* with no heading, markdown or HTML, at or above the turn level.

    The parser finds the headings, so a ``#`` in code or a ``#tag`` stays as
    written, and an HTML heading tag inside a fence or an inline code span
    stays as written too. A ``#`` heading drops two levels, to at most six;
    an underlined heading cannot go below level two, so its underline is
    escaped to text. An HTML heading tag, open or close, drops the same two
    levels, independently of whether it is ever closed.
    """
    lines = text.split("\n")
    for token in commonmark().parse(text):
        if token.type == "heading_open" and token.map:
            _nest_heading(lines, token.markup, *token.map)
        elif token.type == "html_block" and token.map:
            _demote_html_block(lines, *token.map)
        elif token.type == "inline" and token.map:
            _demote_html_inline(lines, token.children, *token.map)
    return "\n".join(lines)


def _nest_heading(lines: list[str], markup: str, start: int, end: int) -> None:
    """Rewrite the heading spanning lines *start* to *end* (exclusive) in place."""
    if markup.startswith(_ATX_MARKER):
        level = min(len(markup) + _TURN_HEADING_LEVEL, _MAX_HEADING_LEVEL)
        lines[start] = lines[start].replace(markup, _ATX_MARKER * level, 1)
    else:
        underline = end - 1
        lines[underline] = lines[underline].replace(markup, _ESCAPE + markup, 1)


def _demote_html_tag(match: re.Match[str]) -> str:
    """*match*, an HTML heading tag, with its level raised by the turn offset."""
    level = min(int(match["level"]) + _TURN_HEADING_LEVEL, _MAX_HEADING_LEVEL)
    return f"<{match['slash']}{match['tag']}{level}{match['rest']}>"


def _demote_html_block(lines: list[str], start: int, end: int) -> None:
    """Rewrite any HTML heading tag in the raw HTML block spanning *start* to *end*."""
    joined = "\n".join(lines[start:end])
    lines[start:end] = _HTML_HEADING_RE.sub(_demote_html_tag, joined).split("\n")


def _demote_html_inline(
    lines: list[str], children: list[Token] | None, start: int, end: int
) -> None:
    """Rewrite each HTML heading tag among *children* within the span *start* to *end*.

    Only tags the parser tokenized as real HTML are candidates, so the same
    text inside a code span, which the parser tokenizes separately, is never
    a match. Children are walked in source order and each tag is located from
    where the previous one ended, so two identical tags in one span each
    rewrite their own occurrence.
    """
    tags = [child.content for child in children or [] if child.type == "html_inline"]
    if not tags:
        return
    joined = "\n".join(lines[start:end])
    cursor = 0
    pieces: list[str] = []
    for tag in tags:
        index = joined.index(tag, cursor)
        pieces.append(joined[cursor:index])
        pieces.append(_HTML_HEADING_RE.sub(_demote_html_tag, tag))
        cursor = index + len(tag)
    pieces.append(joined[cursor:])
    lines[start:end] = "".join(pieces).split("\n")


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
