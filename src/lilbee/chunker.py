"""Token-based recursive text chunking with markdown-aware heading splitting.

Heading path prepending inspired by Anthropic's Contextual Retrieval
research (2024), which showed that adding document context to chunks
reduces retrieval failures by 49%. Our approach prepends the heading
hierarchy (e.g. "# Setup > ## Install") so the LLM knows each
chunk's section context without requiring an extra LLM call per chunk.
"""

import re

import tiktoken

from lilbee.config import cfg

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

_enc = tiktoken.get_encoding("cl100k_base")

# Separators tried in order from coarsest to finest
_SEPARATORS = ("\n\n", ". ", " ")


def _token_len(text: str) -> int:
    return len(_enc.encode(text))


def _split_nonempty(text: str, sep: str) -> list[str]:
    """Split text on separator, dropping empty/whitespace-only parts."""
    return [p for p in text.split(sep) if p.strip()]


def _split_to_segments(text: str, max_tokens: int) -> list[str]:
    """Recursively split text into segments within max_tokens.

    Tries separators coarsest-first: paragraphs, sentences, words.
    """
    if _token_len(text) <= max_tokens:
        return [text]

    for sep in _SEPARATORS:
        parts = _split_nonempty(text, sep)
        if len(parts) > 1:
            return [seg for part in parts for seg in _split_to_segments(part, max_tokens)]

    return hard_split_words(text, max_tokens)


def hard_split_words(text: str, max_tokens: int) -> list[str]:
    """Last-resort split by individual words."""
    words = text.split()
    segments: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    for word in words:
        wt = _token_len(word + " ")
        if buf_tokens + wt > max_tokens and buf:
            segments.append(" ".join(buf))
            buf, buf_tokens = [], 0
        buf.append(word)
        buf_tokens += wt

    if buf:
        segments.append(" ".join(buf))
    return segments


def _tail_overlap(segments: list[str], max_tokens: int) -> list[str]:
    """Take trailing segments that fit within the overlap token budget."""
    result: list[str] = []
    tokens = 0
    for seg in reversed(segments):
        seg_t = _token_len(seg)
        if tokens + seg_t > max_tokens:
            break
        result.insert(0, seg)
        tokens += seg_t
    return result


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Split text into overlapping token-sized chunks.

    Strategy: recursively split on paragraph/sentence/word boundaries,
    then merge segments into target-sized chunks with overlap.
    """
    if chunk_size is None:
        chunk_size = cfg.chunk_size
    if chunk_overlap is None:
        chunk_overlap = cfg.chunk_overlap
    if not text or not text.strip():
        return []

    segments = _split_to_segments(text, chunk_size)
    if not segments:
        return []

    chunks: list[str] = []
    pending_segments: list[str] = []
    pending_tokens = 0

    for seg in segments:
        seg_t = _token_len(seg)
        if pending_tokens + seg_t > chunk_size and pending_segments:
            chunks.append("\n\n".join(pending_segments))
            pending_segments = _tail_overlap(pending_segments, chunk_overlap)
            pending_tokens = sum(_token_len(s) for s in pending_segments)
        pending_segments.append(seg)
        pending_tokens += seg_t

    if pending_segments:
        chunks.append("\n\n".join(pending_segments))

    return chunks


def _split_by_headings(text: str) -> list[tuple[list[str], str]]:
    """Split markdown into sections, each with its heading hierarchy.

    Returns a list of (heading_path, section_body) tuples.
    heading_path is the current heading stack, e.g. ["# Setup", "## Install"].
    """
    sections: list[tuple[list[str], str]] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []

    for line in text.split("\n"):
        m = _HEADING_RE.match(line)
        if m:
            # Flush current section
            body = "\n".join(current_lines).strip()
            if body:
                path = [h for _, h in heading_stack]
                sections.append((list(path), body))
            current_lines = []

            level = len(m.group(1))
            heading_text = f"{m.group(1)} {m.group(2)}"
            # Pop headings at same or deeper level
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_text))
        else:
            current_lines.append(line)

    # Flush final section
    body = "\n".join(current_lines).strip()
    if body:
        path = [h for _, h in heading_stack]
        sections.append((list(path), body))

    return sections


def chunk_markdown(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Split markdown into chunks respecting heading boundaries.

    Each chunk is prefixed with its heading hierarchy so the LLM
    knows the section context. Falls back to plain chunk_text()
    for non-markdown or text with no headings.
    """
    if not text or not text.strip():
        return []

    sections = _split_by_headings(text)
    if not sections:
        return chunk_text(text, chunk_size, chunk_overlap)

    chunks: list[str] = []
    for path, body in sections:
        prefix = " > ".join(path)
        sub_chunks = chunk_text(body, chunk_size, chunk_overlap)
        for sub in sub_chunks:
            if prefix:
                chunks.append(f"{prefix}\n\n{sub}")
            else:
                chunks.append(sub)

    return chunks if chunks else chunk_text(text, chunk_size, chunk_overlap)
