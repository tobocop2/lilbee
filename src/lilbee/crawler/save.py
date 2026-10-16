"""Page addresses and saved names, metadata I/O, and per-page save-to-disk."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import string
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote

from lilbee.core.config import cfg
from lilbee.core.security import validate_path_within
from lilbee.core.web_names import INDEX_STEM, PAGE_HASH_MARK, PAGE_SUFFIX, WEB_DIR
from lilbee.crawler.models import CrawlResult

# ada-url is in the optional ``crawler`` extra, so each function that needs it imports it on call.
if TYPE_CHECKING:
    import ada_url

log = logging.getLogger(__name__)

# How often the crawl metadata JSON is rewritten during a streaming crawl.
# Markdown files are durable per-page; metadata batches to keep write volume
# bounded. Worst-case loss on crash is N-1 entries, recoverable from the files.
METADATA_FLUSH_INTERVAL = 10

# The characters RFC 3986 section 2.3 calls unreserved: an escape of one names the character.
_UNRESERVED = frozenset(string.ascii_letters + string.digits + "-._~")
_PERCENT_ESCAPE = re.compile(r"%([0-9A-Fa-f]{2})")
_HEX = 16
# A WHATWG parser escapes a raw "^" in a path; the url crate 2.5.8 crawlberg 1.11.1 uses keeps it.
_CARET = "^"
_CARET_ESCAPE = "%5E"
_CARET_ESCAPE_LOWER = "%5e"
_ANY_CARET_ESCAPE = re.compile(_CARET_ESCAPE, re.IGNORECASE)
_QUERY_MARK = "?"
_QUERY_SEPARATOR = "&"
_QUERY_VALUE_MARK = "="
# Hex digits of the sha256 of a page's canonical address in its file name: 64 bits.
_HASH_HEX_DIGITS = 16
_MAX_SLUG_BYTES = 80
_MAX_PATH_BYTES = 200
# A slug keeps letters, digits and these; every other run of characters becomes one fill.
_SLUG_KEPT = frozenset("-_.")
_SLUG_FILL = "-"
_SLUG_FILL_RUN = re.compile(f"{_SLUG_FILL}{{2,}}")
_SLUG_ENDS = ".- "
_EMPTY_SLUG = "_"
_DEVICE_MARK = "_"
_EXTENSION_MARK = "."
_PORT_MARK = ":"
_UNKNOWN_HOST = "unknown"
# Names Windows gives to devices, with or without an extension, in any letter case.
_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"{port}{n}" for port in ("COM", "LPT") for n in [*string.digits, "¹", "²", "³"]}
)


def _parsed(url: str) -> ada_url.URL | None:
    """*url* parsed by the WHATWG URL standard, or None when it does not parse."""
    import ada_url

    try:
        return ada_url.URL(url)
    except ValueError:
        return None


def _normalized_escape(escape: re.Match[str]) -> str:
    """The unreserved character *escape* names, or *escape* with upper-case hex digits."""
    char = chr(int(escape.group(1), _HEX))
    return char if char in _UNRESERVED else escape.group().upper()


def _parameter_name(parameter: str) -> str:
    """The name of one query parameter: the text before its first ``=``."""
    return parameter.partition(_QUERY_VALUE_MARK)[0]


def _sorted_query(address: str) -> str:
    """*address* with its query parameters sorted by name, stable, each written as it was."""
    before, mark, query = address.partition(_QUERY_MARK)
    if not mark:
        return address
    parameters = sorted(query.split(_QUERY_SEPARATOR), key=_parameter_name)
    return f"{before}{_QUERY_MARK}{_QUERY_SEPARATOR.join(parameters)}"


def canonical_address(url: str) -> str:
    """The key crawlberg 1.11.1 gives the page at *url*, or *url* when it does not parse.

    The steps of crawlberg's ``normalize_url`` (``normalize.rs`` at v1.11.1): the WHATWG
    parse, no fragment, the escapes of RFC 3986 section 6.2.2, the query sorted by name.
    """
    parsed = _parsed(_ANY_CARET_ESCAPE.sub(_CARET_ESCAPE_LOWER, url))
    if parsed is None:
        return url
    parsed.hash = ""
    address = parsed.href.replace(_CARET_ESCAPE, _CARET)
    return _sorted_query(_PERCENT_ESCAPE.sub(_normalized_escape, address))


def _cut(text: str, limit: int) -> str:
    """*text* cut to at most *limit* UTF-8 bytes, on a character boundary."""
    return text.encode("utf-8")[:limit].decode("utf-8", errors="ignore")


def _slug(part: str) -> str:
    """A legal, readable file name for one escaped part of an address; lossy by design."""
    text = unicodedata.normalize("NFC", unquote(part, errors="replace"))
    kept = "".join(c if c.isalnum() or c in _SLUG_KEPT else _SLUG_FILL for c in text)
    slug = _cut(_SLUG_FILL_RUN.sub(_SLUG_FILL, kept), _MAX_SLUG_BYTES).strip(_SLUG_ENDS)
    slug = slug or _EMPTY_SLUG
    if slug.split(_EXTENSION_MARK)[0].upper() in _DEVICE_NAMES:
        return _DEVICE_MARK + slug
    return slug


def _address_parts(address: str) -> tuple[str, list[str]]:
    """The host, in Unicode and with any port, and the path segments of *address*."""
    import ada_url

    parsed = _parsed(address)
    if parsed is None:
        return _UNKNOWN_HOST, [""]
    host = ada_url.idna_to_unicode(parsed.hostname)
    if parsed.port:
        host = f"{host}{_PORT_MARK}{parsed.port}"
    return host, parsed.pathname.removeprefix("/").split("/")


def page_name(address: str) -> str:
    """The path under ``_web`` of the page at canonical *address*.

    Readable folders, then the page's readable name, ``~`` and the start of the sha256 of
    *address*. The hash keeps every page in a file of its own on any disk.
    """
    host, segments = _address_parts(address)
    *folders, last = segments
    digest = hashlib.sha256(address.encode("utf-8")).hexdigest()[:_HASH_HEX_DIGITS]
    file_name = f"{_slug(last) if last else INDEX_STEM}{PAGE_HASH_MARK}{digest}{PAGE_SUFFIX}"
    head = [_slug(host), *(_slug(folder) for folder in folders if folder)]
    while len("/".join([*head, file_name]).encode("utf-8")) > _MAX_PATH_BYTES:
        head.pop()
    return "/".join([*head, file_name])


def url_to_filename(url: str) -> str:
    """The path under ``_web`` that the page at *url* is saved under."""
    return page_name(canonical_address(url))


def web_dir() -> Path:
    """The folder under the documents directory that holds every crawled page."""
    return cfg.documents_dir / WEB_DIR


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
    """What ``_save_single_result`` wrote: the path, the page's key, its file name and hash."""

    path: Path
    address: str
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
    address = canonical_address(result.url)
    filename = page_name(address)
    root = web_dir()
    file_path = root / filename
    try:
        validate_path_within(file_path, root.resolve())
    except ValueError:
        log.warning("Path traversal blocked: %s -> %s", result.url, file_path)
        return None
    new_hash = content_hash(markdown)
    prev = meta.get(address)
    if prev is not None and prev.content_hash == new_hash and file_path.exists():
        log.info("Content unchanged, skipping save: %s", result.url)
        return None
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(markdown, encoding="utf-8")
    return SaveOutcome(path=file_path, address=address, filename=filename, content_hash=new_hash)


def _update_single_metadata(meta: dict[str, CrawlMeta], outcome: SaveOutcome, now: str) -> None:
    """Record *outcome* in *meta* under the page's canonical address."""
    meta[outcome.address] = CrawlMeta(
        file=outcome.filename,
        content_hash=outcome.content_hash,
        crawled_at=now,
    )
