"""Render a saved chat session as a markdown document and write it to disk."""

from __future__ import annotations

import os
import re
import string
from bisect import bisect_right
from functools import cache
from pathlib import Path

import yaml
from markdown_it import MarkdownIt
from markdown_it.rules_inline.html_inline import html_inline as _stock_html_inline
from markdown_it.rules_inline.state_inline import StateInline
from markdown_it.token import Token

from lilbee.core.security import write_private_text
from lilbee.core.text import collapse_whitespace, make_slug
from lilbee.retrieval.query.formatting import (
    FILE_LINK_RE,
    SOURCES_BLOCK_MARKER,
    close_open_fence,
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
# An HTML heading's ``<h``/``</h`` plus level; a boundary after the digit (or the
# text simply ending there) is what excludes ``<header>``, ``<hr>`` and ``<h2o>``.
_HTML_HEADING_RE = re.compile(r"<(?P<slash>/?)(?P<tag>[Hh])(?P<level>[1-6])(?=[\s/>]|$)")
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


def _html_inline_with_span(state: StateInline, silent: bool) -> bool:
    """The stock ``html_inline`` rule, plus the ``state.src`` span it matched on ``token.meta``."""
    start = state.pos
    matched = _stock_html_inline(state, silent)
    if matched and not silent and state.tokens and state.tokens[-1].type == "html_inline":
        state.tokens[-1].meta = {"start": start, "end": state.pos}
    return matched


@cache
def _heading_parser() -> MarkdownIt:
    """A CommonMark parser whose ``html_inline`` tokens carry the source span they matched."""
    md = MarkdownIt("commonmark")
    md.inline.ruler.at("html_inline", _html_inline_with_span)
    return md


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
    for token in _heading_parser().parse(text):
        if token.type == "heading_open" and token.map:
            _nest_heading(lines, token.markup, *token.map)
        elif token.type == "html_block" and token.map:
            _demote_html_block(lines, *token.map)
        elif token.type == "inline" and token.map:
            _demote_html_inline(lines, token.content, token.children, *token.map)
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
    """*match*, an HTML heading tag's opening chars, with its level raised by the turn offset."""
    level = min(int(match["level"]) + _TURN_HEADING_LEVEL, _MAX_HEADING_LEVEL)
    return f"<{match['slash']}{match['tag']}{level}"


def _demote_html_block(lines: list[str], start: int, end: int) -> None:
    """Rewrite any HTML heading tag's level in the raw HTML block spanning *start* to *end*."""
    joined = "\n".join(lines[start:end])
    lines[start:end] = _HTML_HEADING_RE.sub(_demote_html_tag, joined).split("\n")


def _demote_html_inline(
    lines: list[str], content: str, children: list[Token] | None, start: int, end: int
) -> None:
    """Rewrite each real HTML heading tag's level within the inline span *start* to *end*."""
    spans = [
        child.meta
        for child in children or ()
        if child.type == "html_inline"
        and child.meta
        and _HTML_HEADING_RE.search(content[child.meta["start"] : child.meta["end"]])
    ]
    if not spans:
        return
    raw_lines = lines[start:end]
    content_lines = content.split("\n")
    mapping = _content_to_raw_map(raw_lines, content_lines)
    if mapping is None:
        # A fake turn-level heading is worse than over-demoting, so every heading
        # look-alike in the span is demoted directly in the raw text, including one
        # inside a code span or an escaped string, when a construct such as a
        # tab-expanded continuation line keeps this mapping from placing it exactly.
        joined = "\n".join(raw_lines)
        lines[start:end] = _HTML_HEADING_RE.sub(_demote_html_tag, joined).split("\n")
        return
    content_starts, deltas = mapping
    raw = "\n".join(raw_lines)
    pieces: list[str] = []
    cursor = 0
    for span in spans:
        raw_start = _map_content_pos(span["start"], content_starts, deltas)
        raw_end = _map_content_pos(span["end"], content_starts, deltas)
        pieces.append(raw[cursor:raw_start])
        pieces.append(_HTML_HEADING_RE.sub(_demote_html_tag, raw[raw_start:raw_end]))
        cursor = raw_end
    pieces.append(raw[cursor:])
    lines[start:end] = "".join(pieces).split("\n")


def _content_to_raw_map(
    raw_lines: list[str], content_lines: list[str]
) -> tuple[list[int], list[int]] | None:
    """Each content line's start offset and its raw-offset delta, or None if a line isn't mapped."""
    if len(raw_lines) != len(content_lines):
        return None
    content_starts = []
    deltas = []
    content_pos = 0
    raw_pos = 0
    for raw_line, content_line in zip(raw_lines, content_lines, strict=True):
        width = _line_prefix_width(raw_line, content_line)
        if width is None:
            return None
        content_starts.append(content_pos)
        deltas.append(raw_pos + width - content_pos)
        content_pos += len(content_line) + 1
        raw_pos += len(raw_line) + 1
    return content_starts, deltas


def _line_prefix_width(raw_line: str, content_line: str) -> int | None:
    """*content_line*'s leading offset into *raw_line*, front-trimmed and, since a
    container also drops trailing spaces and tabs, optionally back-trimmed too."""
    if raw_line.endswith(content_line):
        return len(raw_line) - len(content_line)
    trimmed = raw_line.rstrip(" \t")
    if trimmed.endswith(content_line):
        return len(trimmed) - len(content_line)
    return None


def _map_content_pos(pos: int, content_starts: list[int], deltas: list[int]) -> int:
    """*pos*, an offset into the content lines joined by newlines, as the same offset
    into the raw lines joined the same way, found in one line's constant offset."""
    line = bisect_right(content_starts, pos) - 1
    return pos + deltas[line]


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
