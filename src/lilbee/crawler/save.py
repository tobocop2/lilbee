"""Page addresses and saved names, metadata I/O, and per-page save-to-disk."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import os
import re
import string
import tempfile
import unicodedata
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

from lilbee.core.config import cfg
from lilbee.core.security import file_lock_or_warn, validate_path_within
from lilbee.core.web_names import INDEX_STEM, PAGE_HASH_MARK, PAGE_SUFFIX, WEB_DIR
from lilbee.crawler.models import CrawlResult
from lilbee.crawler.url_filter import parsed_url

log = logging.getLogger(__name__)

_CRAWL_META_FILE = "crawl_meta.json"
_CRAWL_PAGES_FILE = "crawl_pages.jsonl"
# The field of a page record line that holds the page's canonical address.
_RECORD_ADDRESS_FIELD = "url"
_RECORD_LOCK_TIMEOUT_S = 10.0
_NOT_SAVED = "Page %s is not saved: %s holds a file or folder of its own; move or delete it"

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
_SLUG_KEPT = frozenset("-_.")
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


def parsed_address(url: str) -> str | None:
    """*url* without its fragment as crawlberg 1.11.1's URL parser writes it; None if it fails."""
    parsed = parsed_url(_ANY_CARET_ESCAPE.sub(_CARET_ESCAPE_LOWER, url))
    if parsed is None:
        return None
    parsed.hash = ""
    return parsed.href.replace(_CARET_ESCAPE, _CARET)


def canonical_address(url: str) -> str:
    """The key crawlberg 1.11.1's ``normalize_url`` gives the page at *url*, or *url* unparsed."""
    address = parsed_address(url)
    if address is None:
        return url
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
    if len(head) == len(text) or text[len(head)] in _SLUG_KEPT:
        return head
    end = max(head.rfind(c) for c in _SLUG_KEPT)
    return head[:end] if end > 0 else head


def _page_slug(segment: str) -> str:
    """The readable name of the page whose last path segment is *segment*, without its suffix."""
    return _legal(_cut_at_word(_without_suffix(_filled(segment)), _MAX_NAME_BYTES))


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


def _crawl_meta_path() -> Path:
    """Path to the crawl metadata sidecar JSON."""
    return cfg.data_dir / _CRAWL_META_FILE


@dataclass
class CrawlMeta:
    """Metadata for a single crawled URL."""

    file: str
    content_hash: str
    crawled_at: str
    # The hash of the text the file held when this entry was recorded; empty for a new file.
    replaces: str = ""


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


def normalize_crawled_markdown(markdown: str) -> str:
    """Collapse reference-style ``[[N]](url)`` links to plain ``[N](url)``.

    This fixes the double-bracket/wikilink collision without dropping the
    link text or URL. Ordinary single-bracket links are left untouched.
    """
    return _REFERENCE_LINK_RE.sub(r"[\1](\2)", markdown)


def _pages_path() -> Path:
    """The page record: one JSON line per save, the last line for an address counts."""
    return cfg.data_dir / _CRAWL_PAGES_FILE


@dataclass
class CrawlRecord:
    """The pages saved at their names, by address, and the copies an earlier name rule saved."""

    pages: dict[str, CrawlMeta] = field(default_factory=dict)
    claimed: set[str] = field(default_factory=set)
    earlier: dict[str, list[CrawlMeta]] = field(default_factory=dict)
    earlier_by_text: dict[str, list[tuple[str, CrawlMeta]]] = field(default_factory=dict)
    # Files standing where a folder of a page belongs, each warned about once per crawl.
    blocked: set[Path] = field(default_factory=set)

    def add_earlier(self, key: str, entry: CrawlMeta) -> None:
        """Record *entry*, saved under an earlier name, as a copy of the page *key* names."""
        address = canonical_address(key)
        self.earlier.setdefault(address, []).append(entry)
        self.earlier_by_text.setdefault(entry.content_hash, []).append((address, entry))

    def retire_line(self, address: str, previous: CrawlMeta) -> None:
        """Make *previous*, the earlier line of the page at *address*, a copy to retire."""
        self.earlier.setdefault(address, []).append(previous)
        self.earlier_by_text.setdefault(previous.content_hash, []).append((address, previous))
        if all(entry.file != previous.file for entry in self.pages.values()):
            self.claimed.discard(previous.file)


def _record_line(address: str, entry: CrawlMeta) -> str:
    """The page record line of *entry* for the page at *address*."""
    return json.dumps({_RECORD_ADDRESS_FIELD: address, **asdict(entry)}, ensure_ascii=False)


def _read_pages(lines: list[str]) -> dict[str, CrawlMeta]:
    """The last entry of each address in the page record *lines*; a malformed line is skipped."""
    pages: dict[str, CrawlMeta] = {}
    for line in filter(None, lines):
        try:
            data = json.loads(line)
            address = data.pop(_RECORD_ADDRESS_FIELD)
            pages[address] = CrawlMeta(**data)
        except (json.JSONDecodeError, TypeError, KeyError, AttributeError):
            log.warning("Skipping malformed crawl record line in %s", _pages_path())
    return pages


def _load_pages() -> dict[str, CrawlMeta]:
    """The page record, rewritten with one line per page when it holds more lines than pages."""
    path = _pages_path()
    if not path.exists():
        return {}
    with file_lock_or_warn(path, _RECORD_LOCK_TIMEOUT_S):
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        pages = _read_pages(lines)
        if len(pages) < len(list(filter(None, lines))):
            text = "".join(f"{_record_line(a, e)}\n" for a, e in pages.items())
            _replace_atomically(path, text)
    return pages


def load_crawl_record() -> CrawlRecord:
    """The page record, with the earlier crawl metadata as copies to retire."""
    record = CrawlRecord(pages=_load_pages())
    for key, entry in load_crawl_metadata().items():
        record.add_earlier(key, entry)
    record.claimed = {entry.file for entry in record.pages.values()}
    return record


def _claim(address: str, entry: CrawlMeta, record: CrawlRecord) -> None:
    """Append *entry* to the page record and sync it to disk; a cut line stays alone."""
    path = _pages_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock_or_warn(path, _RECORD_LOCK_TIMEOUT_S), path.open("a", encoding="utf-8") as out:
        out.write(f"\n{_record_line(address, entry)}\n")
        out.flush()
        os.fsync(out.fileno())
    previous = record.pages.get(address)
    record.pages[address] = entry
    record.claimed.add(entry.file)
    if previous is not None and previous.file != entry.file:
        record.retire_line(address, previous)


def _same_file(one: str, other: str) -> bool:
    """Whether two names under ``_web`` are one file as the disk resolves them."""
    try:
        return os.path.samefile(web_dir() / one, web_dir() / other)
    except OSError:
        return False


def _holds_only(address: str, entry: CrawlMeta, record: CrawlRecord) -> bool:
    """Whether *entry*'s file holds its page's text, and no page without its own name needs it."""
    path = web_dir() / entry.file
    try:
        validate_path_within(path, web_dir().resolve())
    except ValueError:
        return False
    if not path.is_file():
        return False
    if content_hash(path.read_text(encoding="utf-8", errors="replace")) != entry.content_hash:
        return False
    twins = record.earlier_by_text.get(entry.content_hash, [])
    return not any(
        other != address and other not in record.pages and _same_file(twin.file, entry.file)
        for other, twin in twins
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
    address: str, record: CrawlRecord, remove_sources: Callable[[list[str]], object]
) -> None:
    """Delete each earlier copy of the page at *address* that holds its text and no other's.

    The file goes before its index source, so a sync that runs between the two finds no file
    to index again. Each copy is checked and deleted under the record lock, so two crawls never
    delete one copy.
    """
    for entry in record.earlier.pop(address, []):
        if entry.file in record.claimed:
            continue
        with file_lock_or_warn(_pages_path(), _RECORD_LOCK_TIMEOUT_S):
            if not _holds_only(address, entry, record):
                continue
            source = PurePosixPath(WEB_DIR, _disk_name(entry.file)).as_posix()
            (web_dir() / entry.file).unlink()
            remove_sources([source])
            _remove_empty_folders((web_dir() / entry.file).parent)


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
    """The hash of the text the file at *path* holds, or None when there is no file."""
    if not path.is_file():
        return None
    return content_hash(path.read_bytes().decode("utf-8", errors="replace"))


def _may_write(address: str, held: str | None, new_hash: str, record: CrawlRecord) -> bool:
    """Whether the file at the page's name holds nothing, the new text, or a text it recorded."""
    entry = record.pages.get(address)
    owned = {new_hash} if entry is None else {new_hash, entry.content_hash, entry.replaces}
    return held is None or held in owned


def _owns_name(address: str, held: str | None, new_hash: str, record: CrawlRecord) -> bool:
    """Whether the page may write at its name; a refusal first reads the record from disk again."""
    if _may_write(address, held, new_hash, record):
        return True
    latest = _load_pages().get(address)
    if latest is None:
        return False
    record.pages[address] = latest
    record.claimed.add(latest.file)
    return _may_write(address, held, new_hash, record)


def _write(
    result: CrawlResult,
    address: str,
    path: Path,
    markdown: str,
    held: str | None,
    record: CrawlRecord,
) -> Path | None:
    """Record and write *markdown* at *path*, whose file holds *held*; None when it holds it."""
    new_hash = content_hash(markdown)
    prev = record.pages.get(address)
    if held == new_hash and prev is not None and prev.content_hash == new_hash:
        log.info("Content unchanged, skipping save: %s", result.url)
        return None
    now = datetime.now(UTC).isoformat()
    replaces = "" if held is None else held
    _claim(address, CrawlMeta(page_name(address), new_hash, now, replaces), record)
    if held == new_hash:
        log.info("Recorded the page %s at %s, which already holds its text", result.url, path)
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    _replace_atomically(path, markdown)
    return path


def save_page(
    result: CrawlResult, record: CrawlRecord, remove_sources: Callable[[list[str]], object]
) -> Path | None:
    """Write one crawled page at its name; the path, or None when nothing was written."""
    if result.failure_reason() is not None:
        return None
    address = canonical_address(result.url)
    path = _page_path(result, address, record)
    if path is None:
        return None
    markdown = normalize_crawled_markdown(result.markdown)
    held = _held_text(path)
    if not _owns_name(address, held, content_hash(markdown), record):
        log.warning(_NOT_SAVED, result.url, path)
        return None
    written = _write(result, address, path, markdown, held, record)
    _retire_earlier(address, record, remove_sources)
    return written
