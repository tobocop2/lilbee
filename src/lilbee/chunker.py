"""Token-based recursive text chunking for PDFs, markdown, HTML, plain text."""

import logging
from dataclasses import dataclass

import tiktoken

from lilbee.config import CHUNK_OVERLAP, CHUNK_SIZE

log = logging.getLogger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")

# Max characters used to locate a chunk's position in the source text.
# Longer prefixes are more accurate but slower; 200 chars is well above
# the typical paragraph-boundary overlap.
_CHUNK_POSITION_PREFIX_LEN = 200

# Separators tried in order from coarsest to finest
_SEPARATORS = ("\n\n", ". ", " ")


@dataclass
class PageChunk:
    """A chunk of text with page location metadata."""

    chunk: str
    page_start: int
    page_end: int
    chunk_index: int


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

    return _hard_split_words(text, max_tokens)


def _hard_split_words(text: str, max_tokens: int) -> list[str]:
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
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping token-sized chunks.

    Strategy: recursively split on paragraph/sentence/word boundaries,
    then merge segments into target-sized chunks with overlap.
    """
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


def _pages_for_range(
    char_start: int,
    char_end: int,
    boundaries: list[tuple[int, int, int]],
) -> tuple[int, int]:
    """Determine which pages a character range spans."""
    page_start = boundaries[0][2]
    page_end = page_start
    for bstart, bend, pnum in boundaries:
        if bstart < char_end and bend > char_start:
            page_start = min(page_start, pnum)
            page_end = max(page_end, pnum)
    return page_start, page_end


def chunk_pages(pages: list[dict]) -> list[PageChunk]:
    """Chunk page-structured PDF output into PageChunks with page tracking.

    Input: list of {"page": int, "text": str} dicts (from pymupdf4llm).
    """
    if not pages:
        return []

    parts: list[str] = []
    boundaries: list[tuple[int, int, int]] = []
    offset = 0

    for p in pages:
        text = p["text"] if p["text"].endswith("\n\n") else p["text"] + "\n\n"
        parts.append(text)
        boundaries.append((offset, offset + len(text), p["page"]))
        offset += len(text)

    full_text = "".join(parts)
    raw_chunks = chunk_text(full_text)

    results: list[PageChunk] = []
    search_from = 0

    for idx, chunk in enumerate(raw_chunks):
        pos = full_text.find(chunk[:_CHUNK_POSITION_PREFIX_LEN], search_from)
        if pos == -1:
            log.debug("Chunk position fallback at index %d, search_from=%d", idx, search_from)
            pos = search_from

        ps, pe = _pages_for_range(pos, pos + len(chunk), boundaries)
        results.append(PageChunk(chunk=chunk, page_start=ps, page_end=pe, chunk_index=idx))
        search_from = max(search_from, pos + 1)

    return results
