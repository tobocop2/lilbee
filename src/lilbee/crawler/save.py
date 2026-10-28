"""Page addresses and saved names, and the per-page save to disk under the crawl lock."""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import os
import re
import string
import tempfile
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

from lilbee.core.config import cfg
from lilbee.core.security import validate_path_within
from lilbee.core.web_names import INDEX_STEM, PAGE_HASH_MARK, PAGE_SUFFIX, WEB_DIR
from lilbee.crawler.models import CrawlResult
from lilbee.crawler.url_filter import parsed_address, parsed_url
from lilbee.data.store import CrawlPage, CrawlPages, Store

log = logging.getLogger(__name__)

_NOT_SAVED = "Page %s is not saved: %s holds a file or folder of its own; move or delete it"

# The characters RFC 3986 section 2.3 calls unreserved: an escape of one names the character.
_UNRESERVED = frozenset(string.ascii_letters + string.digits + "-._~")
_PERCENT_ESCAPE = re.compile(r"%([0-9A-Fa-f]{2})")
_MALFORMED_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_HEX = 16
_QUERY_MARK = "?"
_QUERY_SEPARATOR = "&"
_QUERY_VALUE_MARK = "="
# Hex digits of the sha256 of a page's canonical address in its file name: 64 bits.
_HASH_HEX_DIGITS = 16
_MAX_SLUG_BYTES = 80
_MAX_PATH_BYTES = 200
# The bytes a readable name may take: the path less a full host folder, "/", "~", hash and suffix.
_MAX_NAME_BYTES = (
    _MAX_PATH_BYTES
    - _MAX_SLUG_BYTES
    - len("/")
    - len(PAGE_HASH_MARK)
    - _HASH_HEX_DIGITS
    - len(PAGE_SUFFIX)
)
# A slug keeps letters, digits and these; every other run of characters becomes one fill.
_WORD_BREAKS = frozenset("-_.")
_SLUG_KEPT = _WORD_BREAKS | {"@"}
_SLUG_FILL = "-"
_SLUG_FILL_RUN = re.compile(f"{_SLUG_FILL}{{2,}}")
_SLUG_ENDS = ".- "
_EMPTY_SLUG = "_"
_DEVICE_MARK = "_"
_EXTENSION_MARK = "."
_PORT_MARK = ":"
# An IPv6 host is written in brackets, its groups apart by colons.
_IPV6_OPEN = "["
_IPV6_CLOSE = "]"
_IPV6_GROUP_MARK = ":"
_UNKNOWN_HOST = "unknown"
# Names Windows gives to devices, with or without an extension, in any letter case.
_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"{port}{n}" for port in ("COM", "LPT") for n in [*string.digits, "¹", "²", "³"]}
)


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
    """The key crawlberg 1.12.3's ``normalize_url`` gives the page at *url*, or *url* unparsed.

    An address with a malformed escape keeps all its escapes as written: decoding the digits
    after a bad ``%`` could make a valid escape the next time the key is made.
    """
    address = parsed_address(url)
    if address is None:
        return url
    if _MALFORMED_ESCAPE.search(address):
        return _sorted_query(address)
    return _sorted_query(_PERCENT_ESCAPE.sub(_normalized_escape, address))


def _cut(text: str, limit: int) -> str:
    """*text* cut to at most *limit* UTF-8 bytes, on a character boundary."""
    return text.encode("utf-8")[:limit].decode("utf-8", errors="ignore")


def _filled(part: str) -> str:
    """One escaped part of an address, decoded, NFC, with every other run of characters as ``-``."""
    text = unicodedata.normalize("NFC", unquote(part, errors="replace"))
    kept = "".join(c if c.isalnum() or c in _SLUG_KEPT else _SLUG_FILL for c in text)
    return _SLUG_FILL_RUN.sub(_SLUG_FILL, kept)


def _legal(text: str) -> str:
    """*text* with legal ends, never empty, and a mark before a Windows device name."""
    slug = text.strip(_SLUG_ENDS) or _EMPTY_SLUG
    if slug.split(_EXTENSION_MARK)[0].upper() in _DEVICE_NAMES:
        return _DEVICE_MARK + slug
    return slug


def _slug(part: str) -> str:
    """A legal, readable file name for one escaped part of an address; lossy by design."""
    return _legal(_cut(_filled(part), _MAX_SLUG_BYTES))


def _without_suffix(text: str) -> str:
    """*text* without its last ``.`` and the text after it, when text stands on both sides."""
    stem, mark, suffix = text.rpartition(_EXTENSION_MARK)
    return stem if stem and mark and suffix else text


def _cut_at_word(text: str, limit: int) -> str:
    """*text* cut to at most *limit* UTF-8 bytes, at the last separator when a word would break."""
    head = _cut(text, limit)
    if len(head) == len(text) or text[len(head)] in _WORD_BREAKS:
        return head
    end = max(head.rfind(c) for c in _WORD_BREAKS)
    return head[:end] if end > 0 else head


def _page_slug(segment: str) -> str:
    """The readable name of the page whose last path segment is *segment*, without its suffix."""
    marked = _legal(_without_suffix(_filled(segment)))
    return _legal(_cut_at_word(marked, _MAX_NAME_BYTES))


def _folder_slug(part: str) -> str:
    """The slug of a folder; one ending in the page suffix ends in ``-md``, never a file's name."""
    slug = _slug(part)
    if not slug.lower().endswith(PAGE_SUFFIX):
        return slug
    mark = len(slug) - len(PAGE_SUFFIX)
    return f"{slug[:mark]}{_SLUG_FILL}{slug[mark + 1 :]}"


def _readable_host(hostname: str) -> str:
    """*hostname* in Unicode, or an IPv6 address as its eight groups joined by ``-``."""
    import ada_url

    if not hostname.startswith(_IPV6_OPEN):
        return ada_url.idna_to_unicode(hostname)
    address = ipaddress.IPv6Address(hostname.removeprefix(_IPV6_OPEN).removesuffix(_IPV6_CLOSE))
    groups = address.exploded.split(_IPV6_GROUP_MARK)
    return _SLUG_FILL.join(f"{int(group, _HEX):x}" for group in groups)


def _address_parts(address: str) -> tuple[str, list[str]]:
    """The host, readable and with any port, and the path segments of *address*."""
    parsed = parsed_url(address)
    if parsed is None:
        return _UNKNOWN_HOST, [""]
    host = _readable_host(parsed.hostname)
    if parsed.port:
        host = f"{host}{_PORT_MARK}{parsed.port}"
    return host, parsed.pathname.removeprefix("/").split("/")


def page_name(address: str) -> str:
    """The path under ``_web`` of canonical *address*: readable folders and name, ``~``, hash.

    The name comes from the last non-empty segment of the path, less its suffix.
    """
    host, segments = _address_parts(address)
    named = [segment for segment in segments if segment]
    *folders, last = named or [""]
    digest = hashlib.sha256(address.encode("utf-8")).hexdigest()[:_HASH_HEX_DIGITS]
    readable = _page_slug(last) if last else INDEX_STEM
    file_name = f"{readable}{PAGE_HASH_MARK}{digest}{PAGE_SUFFIX}"
    head = [_slug(host), *map(_folder_slug, folders)]
    while len("/".join([*head, file_name]).encode("utf-8")) > _MAX_PATH_BYTES:
        head.pop()
    return "/".join([*head, file_name])


def url_to_filename(url: str) -> str:
    """The path under ``_web`` that the page at *url* is saved under."""
    return page_name(canonical_address(url))


def web_dir() -> Path:
    """The folder under the documents directory that holds every crawled page."""
    return cfg.documents_dir / WEB_DIR


def _replace_atomically(path: Path, text: str) -> None:
    """Write *text* to a temporary file beside *path*, sync it, then put it in place of *path*."""
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
            tmp_name = tmp.name
            tmp.write(text.encode("utf-8"))
            tmp.flush()
            os.fsync(tmp.fileno())
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
# The same link with its inner brackets escaped: ``[\[1\]](url)``.
_ESCAPED_REFERENCE_LINK = r"[\\[\1\\]](\2)"


def normalize_crawled_markdown(markdown: str) -> str:
    """Give the page LF line ends and escape the inner brackets of ``[[N]](url)`` links.

    LF line ends keep the saved bytes equal to the text a universal-newline read
    gives back. ``[[N]](url)`` becomes ``[\\[N\\]](url)``: no ``[[`` is left for a
    wikilink reader, and the link text still reads ``[N]``, so "word[1]" does not
    index as "word1". Ordinary single-bracket links stay.
    """
    lf = markdown.replace("\r\n", "\n").replace("\r", "\n")
    return _REFERENCE_LINK_RE.sub(_ESCAPED_REFERENCE_LINK, lf)


@dataclass
class CrawlRecord:
    """The crawl page table one crawl saves into, and the files it warned about."""

    pages: CrawlPages
    # Files standing where a folder of a page belongs, each warned about once per crawl.
    blocked: set[Path] = field(default_factory=set)


def load_crawl_record(store: Store) -> CrawlRecord:
    """The crawl page table of *store*, each earlier copy keyed by its canonical address."""
    pages = store.crawl_pages
    with pages.held():
        pages.rekey(canonical_address, _holds_text)
    return CrawlRecord(pages)


def finish_crawl(record: CrawlRecord) -> None:
    """Fold the commits the crawl's saves made into one version of the table."""
    with record.pages.held():
        record.pages.compact()


def _same_file(one: str, other: str) -> bool:
    """Whether two names under ``_web`` are one file as the disk resolves them."""
    try:
        return os.path.samefile(web_dir() / one, web_dir() / other)
    except OSError:
        return False


def _holds_text(copy: CrawlPage) -> bool:
    """Whether the file of the earlier *copy* is under ``_web`` and holds the copy's text."""
    path = web_dir() / copy.file
    try:
        validate_path_within(path, web_dir().resolve())
    except ValueError:
        return False
    if not path.is_file():
        return False
    return content_hash(path.read_text(encoding="utf-8", errors="replace")) == copy.content_hash


def _needed_by_twin(copy: CrawlPage, pages: CrawlPages) -> bool:
    """Whether another page, not yet saved at its own name, has the file of *copy* as its copy."""
    return any(
        twin.address != copy.address
        and pages.current(twin.address) is None
        and _same_file(twin.file, copy.file)
        for twin in pages.copies_of(copy.content_hash)
    )


def _remove_empty_folders(folder: Path) -> None:
    """Remove *folder* and each parent that is left empty, up to ``_web``."""
    root = web_dir()
    while folder != root and folder.is_relative_to(root) and not any(folder.iterdir()):
        folder.rmdir()
        folder = folder.parent


def _disk_name(name: str) -> str:
    """*name* as the disk spells it, which a case-folding disk may spell differently."""
    folder, parts = web_dir(), []
    for part in PurePosixPath(name).parts:
        target = folder / part
        found = (e.name for e in os.scandir(folder) if os.path.samefile(e.path, target))
        parts.append(next(found, part))
        folder = folder / parts[-1]
    return "/".join(parts)


def _retire_earlier(
    address: str, pages: CrawlPages, remove_sources: Callable[[list[str]], object]
) -> None:
    """Delete each earlier copy of the page at *address* that holds its text and no other's.

    The caller holds the crawl lock. The file goes before its index source, so a sync
    flush that runs between the two finds no file and adds no source. The row of a copy
    goes with its file, or at once when its file is gone, changed, or another page's own.
    """
    for copy in pages.earlier(address):
        if pages.owners(copy.file) or not _holds_text(copy):
            pages.forget(copy)
            continue
        if _needed_by_twin(copy, pages):
            continue
        source = PurePosixPath(WEB_DIR, _disk_name(copy.file)).as_posix()
        (web_dir() / copy.file).unlink()
        remove_sources([source])
        _remove_empty_folders((web_dir() / copy.file).parent)
        pages.forget(copy)


def _blocking_file(path: Path) -> Path | None:
    """The file that stands where a folder of *path* under ``_web`` belongs, or None."""
    root = web_dir()
    return next(
        (
            folder
            for folder in path.parents
            if folder.is_relative_to(root) and os.path.lexists(folder) and not folder.is_dir()
        ),
        None,
    )


def _folders_are_free(path: Path, record: CrawlRecord) -> bool:
    """Whether the folders of *path* can be made; a file in their way is warned about once."""
    blocker = _blocking_file(path)
    if blocker is None:
        return True
    if blocker not in record.blocked:
        record.blocked.add(blocker)
        log.warning("Pages under %s are not saved: it is a file; move or delete it", blocker)
    return False


def _page_path(result: CrawlResult, address: str, record: CrawlRecord) -> Path | None:
    """Where the page at *address* is saved, or None when its name is not free for it."""
    root = web_dir()
    path = root / page_name(address)
    try:
        validate_path_within(path, root.resolve())
    except ValueError:
        log.warning("Path traversal blocked: %s -> %s", result.url, path)
        return None
    if not _folders_are_free(path, record):
        return None
    if os.path.lexists(path) and (path.is_symlink() or not path.is_file()):
        log.warning(_NOT_SAVED, result.url, path)
        return None
    return path


def _held_text(path: Path) -> str | None:
    """The hash of the text the file at *path* holds, or None when there is no file.

    The file is read with universal newlines: an older lilbee wrote pages in text
    mode, so on Windows a page it saved holds CRLF line ends its record never hashed.
    """
    if not path.is_file():
        return None
    return content_hash(path.read_text(encoding="utf-8", errors="replace"))


def _may_write(prev: CrawlPage | None, held: str | None, new_hash: str) -> bool:
    """Whether the file at the page's name holds nothing, the new text, or a text it claimed."""
    owned = {new_hash} if prev is None else {new_hash, prev.content_hash, prev.replaces}
    return held is None or held in owned


def _unchanged(prev: CrawlPage | None, held: str | None, new_hash: str, title: str) -> bool:
    """Whether the row and the file already hold this text, and the row this title."""
    if prev is None or held != new_hash or prev.content_hash != new_hash:
        return False
    return prev.title == title


def _kept_title(result: CrawlResult, prev: CrawlPage | None) -> str:
    """The title the crawl reports, else the one the row holds; no title never clears one."""
    return result.title or (prev.title if prev is not None else "")


def _write(
    result: CrawlResult,
    address: str,
    path: Path,
    markdown: str,
    held: str | None,
    prev: CrawlPage | None,
    pages: CrawlPages,
) -> Path | None:
    """Claim *markdown* for the page, then write it at *path*; None when the file holds it.

    The claim commits before the file is written, so a file exists only after its claim.
    """
    new_hash = content_hash(markdown)
    title = _kept_title(result, prev)
    if _unchanged(prev, held, new_hash, title):
        log.info("Content unchanged, skipping save: %s", result.url)
        return None
    now = datetime.now(UTC).isoformat()
    replaces = "" if held is None else held
    pages.claim(CrawlPage(address, page_name(address), new_hash, now, replaces, title))
    if held == new_hash:
        log.info("Recorded the page %s at %s, which already holds its text", result.url, path)
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    _replace_atomically(path, markdown)
    return path


def save_page(
    result: CrawlResult, record: CrawlRecord, remove_sources: Callable[[list[str]], object]
) -> Path | None:
    """Write one crawled page at its name; the path, or None when nothing was written.

    The check, the claim, the file write and the retire of earlier copies run under one
    hold of the crawl lock, so the last claim of a page always describes its file. The
    save waits for the lock without a deadline: every holder's work is bounded, and a
    page skipped for a lock timeout would be a page lost.
    """
    if result.failure_reason() is not None:
        return None
    address = canonical_address(result.url)
    with record.pages.held(timeout=None):
        path = _page_path(result, address, record)
        if path is None:
            return None
        markdown = normalize_crawled_markdown(result.markdown)
        held = _held_text(path)
        prev = record.pages.current(address)
        if not _may_write(prev, held, content_hash(markdown)):
            log.warning(_NOT_SAVED, result.url, path)
            return None
        written = _write(result, address, path, markdown, held, prev, record.pages)
        _retire_earlier(address, record.pages, remove_sources)
        record.pages.compact_when_due()
    return written
