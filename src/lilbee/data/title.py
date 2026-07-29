"""Document title and source-level metadata derivation for ingest."""

from __future__ import annotations

import re
from pathlib import PurePath
from typing import Protocol

from lilbee.data.store import SourceMeta

# Filename-stem separators flattened to spaces when no extracted title exists.
_STEM_SEPARATOR_RE = re.compile(r"[_\-\s]+")

# Stems with no searchable words: camera/scanner counters, generic names, bare
# numbers/dates, hex ids. Indexing these gives the title arm noise at full weight.
_COUNTER_STEM_RE = re.compile(
    r"(?:img|image|dsc[nf]?|pxl|mvimg|vid|video|scan|screenshot|photo|pic|picture"
    r"|untitled|unnamed|noname|new|document|doc|file|page)?(?:\s*\d+)*",
    re.IGNORECASE,
)
_NUMERIC_STEM_RE = re.compile(r"[\d\s.]+")
_HEX_ID_RE = re.compile(r"[0-9a-f]{8,}", re.IGNORECASE)

# Below this many characters a stem cannot form a searchable word.
_MIN_TITLE_CHARS = 3


def is_junk_stem(stem: str) -> bool:
    """True when a filename stem carries no searchable title words."""
    flat = _STEM_SEPARATOR_RE.sub(" ", stem).strip()
    if len(flat) < _MIN_TITLE_CHARS:
        return True
    if _NUMERIC_STEM_RE.fullmatch(flat) or _COUNTER_STEM_RE.fullmatch(flat):
        return True
    return bool(_HEX_ID_RE.fullmatch(flat.replace(" ", "")))


class ExtractionMetadata(Protocol):
    """xberg metadata fields, typed ``object``: xberg annotates but does not enforce
    them (a PDF /Author arrives as a bare str; a non-str title is accepted)."""

    @property
    def title(self) -> object: ...

    @property
    def authors(self) -> object: ...

    @property
    def created_at(self) -> object: ...


def derive_title(source_name: str, metadata_title: object = None) -> str:
    """Human-readable document title: the extracted title, else the cleaned filename stem.

    The stem cleanup flattens underscore/hyphen separators to spaces so BM25
    tokenizes ``survey_214.pdf`` into the same terms a query would use. Junk
    stems (``IMG 1234``, bare numbers, hex ids) yield "" so no title is stored.
    """
    if isinstance(metadata_title, str) and metadata_title.strip():
        return metadata_title.strip()
    stem = PurePath(source_name).stem
    if is_junk_stem(stem):
        return ""
    return _STEM_SEPARATOR_RE.sub(" ", stem).strip()


def source_meta_from_extraction(
    metadata: ExtractionMetadata | None, source_name: str
) -> SourceMeta:
    """Fold xberg extraction metadata into a :class:`SourceMeta`.

    The title falls back to the filename stem; authors and creation date stay
    empty (persisted NULL) when the extractor reports none. xberg annotates these
    fields but does not enforce them: a PDF ``/Author`` arrives as a bare ``str``
    where ``list[str]`` is declared, so a string is treated as one author rather
    than split into its characters, and non-string entries are coerced.
    """
    raw_authors = metadata.authors if metadata is not None else None
    if isinstance(raw_authors, str):
        authors: list[str] = [raw_authors]
    elif isinstance(raw_authors, (list, tuple)):
        authors = [str(a) for a in raw_authors if a]
    else:
        authors = []
    return SourceMeta(
        title=derive_title(source_name, metadata.title if metadata is not None else None),
        authors=", ".join(a for a in authors if a),
        created_at=str((metadata.created_at if metadata is not None else None) or ""),
    )
