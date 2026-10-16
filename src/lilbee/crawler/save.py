"""URL-to-filename mapping, metadata I/O, and per-page save-to-disk."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import string
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlparse

from lilbee.core.config import cfg
from lilbee.core.security import validate_path_within
from lilbee.crawler.models import CrawlResult

log = logging.getLogger(__name__)

# Maximum UTF-8 length of a saved path before truncation (most filesystems cap a name at 255 bytes)
_MAX_FILENAME_BYTES = 200

# Sentinel for index pages (trailing slash or empty path)
_INDEX_FILENAME = "index.md"

# Length of the query-string hash folded into a filename to keep distinct
# queries on the same path from mapping to (and overwriting) the same file.
_QUERY_HASH_LEN = 8

# How often the crawl metadata JSON is rewritten during a streaming crawl.
# Markdown files are durable per-page; metadata batches to keep write volume
# bounded. Worst-case loss on crash is N-1 entries, recoverable from the files.
METADATA_FLUSH_INTERVAL = 10

# The characters RFC 3986 section 2.3 calls unreserved: an escape of one names the character.
_UNRESERVED = frozenset(string.ascii_letters + string.digits + "-._~")
_PERCENT_ESCAPE = re.compile(r"%([0-9A-Fa-f]{2})")
_HEX = 16
# Every printable ASCII character but the space: the encoded form leaves these as written.
_URI_CHARACTERS = string.digits + string.ascii_letters + string.punctuation
# A run of escapes, or one other character, of a path segment in its equivalent form.
_SEGMENT_PART = re.compile(r"(?:%[0-9A-Fa-f]{2})+|.", re.DOTALL)
_PERCENT = "%"
_SPACE = " "
# Characters a Windows file name may not hold; a segment never holds "/" or "?".
_WINDOWS_RESERVED = frozenset('<>:"\\|*')
# Characters Windows strips from the end of a file name.
_WINDOWS_TRAILING = " ."
# Names Windows gives to devices, with or without an extension, in any letter case.
_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"{port}{n}" for port in ("COM", "LPT") for n in [*string.digits, "¹", "²", "³"]}
)
_EXTENSION_MARK = "."


def _normalized_escape(escape: re.Match[str]) -> str:
    """The unreserved character *escape* names, or *escape* with upper-case hex digits."""
    char = chr(int(escape.group(1), _HEX))
    return char if char in _UNRESERVED else escape.group().upper()


def _encoded_form(url: str) -> str:
    """*url* with each space, control and non-ASCII character percent-encoded as UTF-8."""
    return quote(url, safe=_URI_CHARACTERS)


def equivalent_form(url: str) -> str:
    """The encoded form of *url* with each escape normalized as RFC 3986 section 6.2.2 defines."""
    return _PERCENT_ESCAPE.sub(_normalized_escape, _encoded_form(url))


def _escaped(text: str) -> str:
    """*text* as upper-case percent escapes of its UTF-8 bytes, a byte that is not UTF-8 as is."""
    return "".join(f"%{byte:02X}" for byte in text.encode("utf-8", errors="surrogateescape"))


def _named_char(char: str) -> str:
    """One character an escape names, as a file name holds it.

    A space and a printable non-ASCII character read as written. The escape of a character
    Windows refuses, or of a percent sign, escapes its own percent sign, so it never reads
    like the escape a raw reserved character becomes.
    """
    if char == _SPACE or (not char.isascii() and char.isprintable()):
        return char
    if char in _WINDOWS_RESERVED or char == _PERCENT:
        return _escaped(_PERCENT) + _escaped(char).removeprefix(_PERCENT)
    return _escaped(char)


def _named_part(part: str) -> str:
    """One run of escapes, or one raw character, of a segment, as a file name holds it."""
    if len(part) > 1:
        named = bytes.fromhex(part.replace(_PERCENT, "")).decode("utf-8", "surrogateescape")
        return "".join(map(_named_char, named))
    return _escaped(part) if part in _WINDOWS_RESERVED else part


def _segment_name(segment: str) -> str:
    """The file name for one path segment of an address in its equivalent form.

    The trailing spaces and dots Windows strips, and the first letter of a device name, become
    escapes, so the name maps back to exactly one segment.
    """
    name = "".join(map(_named_part, _SEGMENT_PART.findall(segment)))
    body = name.rstrip(_WINDOWS_TRAILING)
    name = body + _escaped(name[len(body) :])
    if name.split(_EXTENSION_MARK)[0].rstrip(_SPACE).upper() in _DEVICE_NAMES:
        name = _escaped(name[0]) + name[1:]
    return name


def stored_spellings(meta: dict[str, CrawlMeta]) -> dict[str, list[str]]:
    """The URLs *meta* holds, grouped by their equivalent form."""
    spellings: dict[str, list[str]] = {}
    for url in meta:
        spellings.setdefault(equivalent_form(url), []).append(url)
    return spellings


def url_to_filename(url: str) -> str:
    """Convert a URL to a safe filesystem path ending in .md.

    Examples:
        https://docs.python.org/3/tutorial/ → docs.python.org/3/tutorial/index.md
        https://example.com/page?q=1        → example.com/page/index_q<hash>.md
        https://example.com/                → example.com/index.md

    A distinct query string folds a short hash of the query into the filename so
    two URLs differing only in their query do not map to the same file and
    overwrite each other. The fragment is client-side and intentionally ignored.
    Every spelling of one address in :func:`equivalent_form` maps to one path, and each
    segment is named by :func:`_segment_name`.
    """
    canonical = equivalent_form(url)
    parsed = urlparse(canonical)
    host = _segment_name(parsed.hostname or "unknown")
    path = "/".join(map(_segment_name, parsed.path.strip("/").split("/")))

    if not path:
        rel = _INDEX_FILENAME
    else:
        last_segment = path.rsplit("/", 1)[-1]
        if re.search(r"\.[^./]+$", last_segment):
            rel = re.sub(r"\.[^./]+$", ".md", path)  # real extension: swap it for .md
        else:
            # No real extension (incl. a bare "." segment) -> treat as a directory.
            # This guarantees rel ends in .md so the query suffix below always applies.
            rel = f"{path}/{_INDEX_FILENAME}"

    if parsed.query:
        query_hash = hashlib.sha256(parsed.query.encode()).hexdigest()[:_QUERY_HASH_LEN]
        rel = re.sub(r"\.md$", f"_q{query_hash}.md", rel)

    full = f"{host}/{rel}"

    # Truncate if too long, preserving .md extension
    encoded = full.encode()
    if len(encoded) > _MAX_FILENAME_BYTES:
        url_hash = hashlib.sha256(canonical.encode()).hexdigest()[:12]
        head = encoded[: _MAX_FILENAME_BYTES - 16].decode(errors="ignore")
        full = f"{head}_{url_hash}.md"

    return full


def _web_dir() -> Path:
    """Return the _web/ subdirectory under documents."""
    return cfg.documents_dir / "_web"


def _crawl_meta_path() -> Path:
    """Path to the crawl metadata sidecar JSON."""
    return cfg.data_dir / "crawl_meta.json"


@dataclass
class CrawlMeta:
    """Metadata for a single crawled URL."""

    file: str
    content_hash: str
    crawled_at: str


def load_crawl_metadata() -> dict[str, CrawlMeta]:
    """Load URL→metadata mapping from the JSON sidecar."""
    path = _crawl_meta_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        # A corrupt sidecar means every known URL looks new and gets re-crawled;
        # surface why rather than silently discarding the whole mapping.
        log.warning("Discarding unreadable crawl metadata sidecar %s: %s", path, exc)
        return {}
    result: dict[str, CrawlMeta] = {}
    for url, data in raw.items():
        try:
            result[url] = CrawlMeta(**data)
        except (TypeError, KeyError):
            log.warning("Skipping malformed crawl metadata entry: %s", url)
    return result


def save_crawl_metadata(meta: dict[str, CrawlMeta]) -> None:
    """Persist URL→metadata mapping to the JSON sidecar (atomic write)."""
    path = _crawl_meta_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        url: {"file": m.file, "content_hash": m.content_hash, "crawled_at": m.crawled_at}
        for url, m in meta.items()
    }
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
            tmp_name = tmp.name
            tmp.write(json.dumps(serializable, indent=2).encode("utf-8"))
        Path(tmp_name).replace(path)
    except BaseException:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)
        raise


def content_hash(text: str) -> str:
    """SHA-256 hex digest of text content."""
    return hashlib.sha256(text.encode()).hexdigest()


# Reference-style nested-bracket links, e.g. Wikipedia footnote markers like
# ``[[1]](https://en.wikipedia.org/wiki/Foo#cite_note-1)``. The inner brackets
# make this a normal Markdown link with the text ``[1]``, but readers that
# treat ``[[...]]`` as a wikilink (Obsidian) mis-parse it as a broken wikilink
# followed by the literal URL.
_REFERENCE_LINK_RE = re.compile(r"\[\[([^\]]*)\]\]\(([^)]*)\)")


def normalize_crawled_markdown(markdown: str) -> str:
    """Collapse reference-style ``[[N]](url)`` links to plain ``[N](url)``.

    This fixes the double-bracket/wikilink collision without dropping the
    link text or URL. Ordinary single-bracket links are left untouched.
    """
    return _REFERENCE_LINK_RE.sub(r"[\1](\2)", markdown)


@dataclass(frozen=True)
class SaveOutcome:
    """Return value of ``_save_single_result``: written path and the hash/filename used."""

    path: Path
    filename: str
    content_hash: str


def _save_single_result(result: CrawlResult, meta: dict[str, CrawlMeta]) -> SaveOutcome | None:
    """Write one crawl result to disk if it's new or changed.

    Returns the outcome (written path plus reusable filename/hash), or
    None if skipped (failure, empty markdown, unchanged hash with file
    on disk, or blocked by path traversal).
    """
    if result.failure_reason() is not None:
        return None
    markdown = normalize_crawled_markdown(result.markdown)
    filename = url_to_filename(result.url)
    web_dir = _web_dir()
    file_path = web_dir / filename
    resolved_web_dir = web_dir.resolve()
    try:
        validate_path_within(file_path, resolved_web_dir)
    except ValueError:
        log.warning("Path traversal blocked: %s -> %s", result.url, file_path)
        return None
    new_hash = content_hash(markdown)
    prev = meta.get(result.url)
    if prev is not None and prev.content_hash == new_hash and file_path.exists():
        log.info("Content unchanged, skipping save: %s", result.url)
        return None
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(markdown, encoding="utf-8")
    _retire_earlier_copy(result.url, file_path, meta)
    return SaveOutcome(path=file_path, filename=filename, content_hash=new_hash)


def _retire_earlier_copy(url: str, file_path: Path, meta: dict[str, CrawlMeta]) -> None:
    """Remove the copy of *url* saved under an earlier name, unless another address holds it."""
    prev = meta.get(url)
    if prev is None or any(m.file == prev.file for other, m in meta.items() if other != url):
        return
    web_dir = _web_dir()
    try:
        earlier = validate_path_within(web_dir / prev.file, web_dir.resolve())
    except ValueError:
        return
    if earlier.is_file() and not earlier.samefile(file_path):
        earlier.unlink()
        log.info("Removed the copy of %s saved under its earlier name %s", url, prev.file)


def _update_single_metadata(
    meta: dict[str, CrawlMeta],
    url: str,
    outcome: SaveOutcome,
    now: str,
) -> None:
    """Update the metadata dict in place with a previously-computed outcome."""
    meta[url] = CrawlMeta(
        file=outcome.filename,
        content_hash=outcome.content_hash,
        crawled_at=now,
    )
